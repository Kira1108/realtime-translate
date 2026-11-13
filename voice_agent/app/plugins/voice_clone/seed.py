"""Volcengine (Doubao) voice cloning, API v3: upload a voice sample, poll until the voice is ready.

    POST https://openspeech.bytedance.com/api/v3/tts/voice_clone   train a voice from a sample
    POST https://openspeech.bytedance.com/api/v3/tts/get_voice     query its status / demo audio

Docs: https://docs.volcengine.com/docs/doubao-voice/tone-training-http
      https://docs.volcengine.com/docs/doubao-voice/tone-query-http

Two kinds of voice:
  - prepaid: a slot bought in the console, with an id like "S_xxxxxxxx"
  - postpaid: a name of your own ("custom_speaker_id"); training and listening to the demo are free, the
    slot fee is charged the first time it is used for synthesis, so check the demo audio first

A ready voice (status SUCCESS / ACTIVE) is used with VolTTS as `speaker=<speaker id>` and a voice clone
resource id, e.g. VolTTS(speaker="S_xxx", resource_id="seed-icl-2.0").

    async with SeedVoiceClone() as vc:
        await vc.clone("me.wav", speaker_id="S_xxx", demo_text="你好，这是我复刻的声音。")
        voice = await vc.wait_until_ready(speaker_id="S_xxx")
        print(voice.demo_audio)

Command line (from voice_agent/):
    python -m app.plugins.voice_clone.seed me.wav --speaker-id S_xxx --demo-text "你好，这是我复刻的声音。"
    python -m app.plugins.voice_clone.seed me.wav --custom-speaker-id myVoice01 --denoise
    python -m app.plugins.voice_clone.seed --speaker-id S_xxx            # only query the status
"""
import asyncio
import base64
import re
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from uuid import uuid4

import httpx
from loguru import logger as _logger

from app.config import get_config

logger = _logger.bind(component="SeedVoiceClone")

BASE_URL = "https://openspeech.bytedance.com/api/v3/tts"
CUSTOM_SPEAKER = "custom_speaker_id"  # the fixed speaker_id that goes with a custom_speaker_id
MAX_AUDIO_BYTES = 10 * 1024 * 1024
AUDIO_FORMATS = {"wav", "mp3", "ogg", "m4a", "aac", "pcm"}

LANGUAGES = {
    "cn": 0, "en": 1, "ja": 2, "es": 3, "id": 4, "pt": 5, "de": 6, "fr": 7, "ko": 8, "it": 9,
    "th": 10, "vi": 11, "ru": 12, "fil": 13, "ms": 14, "ar": 15, "mx": 16, "pt-br": 17,
    "pl": 19, "tr": 20, "sv": 21,
}

# names the service rejects (the regex from the docs): official prefixes / suffixes, bad characters,
# wrong length, a non-letter first character, '-' or '_' last
_RESERVED_NAME = re.compile(
    r"^((?i:S_|ICL_|MIX_|DiT_|BV)|[a-z]{2}_|(?i:(wvae|moon|mercury|venus|earth|mars|jupiter|saturn|uranus"
    r"|neptune|pluto|umm)_)).*|.*_(?i:bigtts|bigtts_cc|tob|cs_tob|streaming)$|^[^a-zA-Z]|.*[-_]$|^.{0,7}$"
    r"|^.{257,}$|.*[^a-zA-Z0-9_-].*"
)


class VoiceCloneError(RuntimeError):
    def __init__(self, code: int | None, message: str, logid: str | None = None):
        super().__init__(f"火山声音复刻失败 ({code}): {message}" + (f" [logid {logid}]" if logid else ""))
        self.code = code
        self.logid = logid


class VoiceStatusCode(IntEnum):
    NOT_FOUND = 0
    TRAINING = 1
    SUCCESS = 2
    FAILED = 3
    ACTIVE = 4


@dataclass
class VoiceStatus:
    speaker_id: str
    status: VoiceStatusCode
    available_training_times: int | None = None
    create_time: int | None = None  # ms timestamp
    language: int | None = None
    demo_audio: str | None = None   # URL, valid for an hour after training
    model_type: int | None = None   # 5 = voice clone 2.0
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def ready(self) -> bool:
        """Usable for synthesis."""
        return self.status in (VoiceStatusCode.SUCCESS, VoiceStatusCode.ACTIVE)

    @classmethod
    def from_response(cls, data: dict) -> "VoiceStatus":
        speaker_status = (data.get("speaker_status") or [{}])[0]
        try:
            status = VoiceStatusCode(data.get("status", 0))
        except ValueError:
            status = VoiceStatusCode.NOT_FOUND
        return cls(
            speaker_id=data.get("speaker_id", ""),
            status=status,
            available_training_times=data.get("available_training_times"),
            create_time=data.get("create_time"),
            language=data.get("language"),
            demo_audio=speaker_status.get("demo_audio") or data.get("demo_audio"),
            model_type=speaker_status.get("model_type"),
            raw=data,
        )


def check_custom_speaker_id(name: str) -> str:
    """Raises ValueError for a postpaid voice name the service would reject (naming rules in the docs)."""
    if _RESERVED_NAME.match(name):
        raise ValueError(
            f"custom_speaker_id {name!r} 不合法：8~256 位，只能有字母、数字、- 和 _，以字母开头，"
            f"不能以 - 或 _ 结尾，也不能用官方音色的前缀（S_、ICL_、MIX_、两个小写字母加 _ ...）或后缀（_bigtts、_tob ...）"
        )
    return name


def _speaker_fields(speaker_id: str | None, custom_speaker_id: str | None) -> dict:
    if custom_speaker_id:
        return {"speaker_id": CUSTOM_SPEAKER, "custom_speaker_id": check_custom_speaker_id(custom_speaker_id)}
    if not speaker_id:
        raise ValueError("需要 speaker_id（预付费音色，如 S_xxx）或 custom_speaker_id（后付费自定义音色）")
    return {"speaker_id": speaker_id}


def _language_code(language: int | str) -> int:
    if isinstance(language, int):
        return language
    try:
        return LANGUAGES[language.lower()]
    except KeyError:
        raise ValueError(f"unknown language {language!r}, one of: {', '.join(LANGUAGES)}") from None


def _load_audio(audio: str | Path | bytes, audio_format: str | None) -> tuple[bytes, str | None]:
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        data = path.read_bytes()
        audio_format = audio_format or path.suffix.lstrip(".").lower() or None
    else:
        data = audio
    if audio_format is not None and audio_format not in AUDIO_FORMATS:
        raise ValueError(f"unsupported audio format {audio_format!r}, one of: {', '.join(sorted(AUDIO_FORMATS))}")
    if not data:
        raise ValueError("audio is empty")
    if len(data) > MAX_AUDIO_BYTES:
        raise ValueError(f"audio is {len(data) / 1024 / 1024:.1f} MB, the limit is 10 MB")
    return data, audio_format


class SeedVoiceClone:
    """Client for Volcengine voice cloning v3. The API key comes from VOLC_TTS_API_KEY (environment or ~/.env)."""

    def __init__(self, api_key: str | None = None, timeout: float = 120.0):
        self.api_key = api_key or get_config().volc_tts_api_key
        if not self.api_key:
            raise RuntimeError("缺少火山 API Key，请设置环境变量 VOLC_TTS_API_KEY")
        # training with a long demo_text takes a while: generous read timeout
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        await self._client.aclose()

    async def _post(self, path: str, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "X-Api-Request-Id": str(uuid4()),
        }
        response = await self._client.post(f"{BASE_URL}/{path}", headers=headers, json=payload)
        logid = response.headers.get("X-Tt-Logid")
        try:
            data = response.json()
        except ValueError:
            data = {}
        code = data.get("code")
        if response.status_code != 200 or code not in (None, 0):
            message = data.get("message") or response.text[:500] or f"HTTP {response.status_code}"
            raise VoiceCloneError(code if code is not None else response.status_code, message, logid)
        logger.debug(f"{path} ok [logid {logid}]")
        return data

    async def clone(
        self,
        audio: str | Path | bytes,
        speaker_id: str | None = None,
        custom_speaker_id: str | None = None,
        *,
        audio_format: str | None = None,
        text: str | None = None,
        language: int | str = 0,
        demo_text: str | None = None,
        denoise: bool = False,
        disable_volume_normalization: bool = False,
    ) -> VoiceStatus:
        """Upload a voice sample to train `speaker_id` (prepaid, "S_xxx") or `custom_speaker_id` (postpaid).

        audio:      a file path (format taken from the suffix) or raw bytes; wav / mp3 / ogg / m4a / aac /
                    pcm (24 kHz mono), at most 10 MB; pcm and m4a need `audio_format`
        text:       what the sample says; if the speech differs too much the training fails (45001109 WERError)
        language:   of the sample, an int or a code such as "cn", "en"
        demo_text:  4-300 characters for a demo clip (in `language`); longer text makes training slower
        denoise:    for noisy samples; costs some detail on clean ones
        disable_volume_normalization: keeps the sample's loudness, closer to it but less uniform

        Returns the status right after the request (may still be TRAINING): see `wait_until_ready`.
        """
        data, audio_format = _load_audio(audio, audio_format)
        if audio_format is None and isinstance(audio, bytes):
            raise ValueError("audio given as bytes: pass audio_format")
        if demo_text is not None and not 4 <= len(demo_text) <= 300:
            raise ValueError("demo_text must be 4-300 characters")

        audio_field = {"data": base64.b64encode(data).decode()}
        if audio_format:
            audio_field["format"] = audio_format
        payload = {
            **_speaker_fields(speaker_id, custom_speaker_id),
            "audio": audio_field,
            "language": _language_code(language),
        }
        if text:
            payload["text"] = text
        extra = {}
        if demo_text:
            extra["demo_text"] = demo_text
        if denoise:
            extra["enable_audio_denoise"] = True
        if disable_volume_normalization:
            extra["disable_volume_normalization"] = True
        if extra:
            payload["extra_params"] = extra

        name = custom_speaker_id or speaker_id
        started = time.perf_counter()
        logger.info(f"cloning {name} from {len(data) / 1024:.0f} KB of {audio_format or 'audio'}")
        status = VoiceStatus.from_response(await self._post("voice_clone", payload))
        logger.info(
            f"clone request for {name} done in {time.perf_counter() - started:.1f}s: {status.status.name}, "
            f"{status.available_training_times} training(s) left"
        )
        return status

    async def get_voice(self, speaker_id: str | None = None, custom_speaker_id: str | None = None) -> VoiceStatus:
        """Current status of a voice, with the demo audio URL once trained."""
        return VoiceStatus.from_response(await self._post("get_voice", _speaker_fields(speaker_id, custom_speaker_id)))

    async def wait_until_ready(
        self,
        speaker_id: str | None = None,
        custom_speaker_id: str | None = None,
        *,
        interval: float = 2.0,
        timeout: float = 180.0,
    ) -> VoiceStatus:
        """Poll `get_voice` until training is over. Raises VoiceCloneError if it failed, TimeoutError if too slow."""
        deadline = time.monotonic() + timeout
        name = custom_speaker_id or speaker_id
        while True:
            status = await self.get_voice(speaker_id, custom_speaker_id)
            if status.ready:
                logger.info(f"{name} is ready ({status.status.name})")
                return status
            if status.status == VoiceStatusCode.FAILED:
                raise VoiceCloneError(status.status.value, f"{name} 训练失败")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{name} still {status.status.name} after {timeout:.0f}s")
            await asyncio.sleep(interval)

    async def download_demo(self, status: VoiceStatus, path: str | Path) -> Path:
        """Save the demo audio (its URL expires an hour after training)."""
        if not status.demo_audio:
            raise ValueError(f"{status.speaker_id or 'voice'} has no demo audio yet ({status.status.name})")
        response = await self._client.get(status.demo_audio)
        response.raise_for_status()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return path


async def _main(args):
    async with SeedVoiceClone() as vc:
        ids = dict(speaker_id=args.speaker_id, custom_speaker_id=args.custom_speaker_id)
        if args.audio:
            await vc.clone(
                args.audio, **ids, audio_format=args.format, text=args.text, language=args.language,
                demo_text=args.demo_text, denoise=args.denoise,
                disable_volume_normalization=args.no_volume_normalization,
            )
            status = await vc.wait_until_ready(**ids, timeout=args.timeout)
        else:
            status = await vc.get_voice(**ids)
        print(f"speaker:        {status.speaker_id}")
        print(f"status:         {status.status.name}")
        print(f"trainings left: {status.available_training_times}")
        if status.demo_audio:
            name = args.custom_speaker_id or args.speaker_id
            out = get_config().PROJECT_PATH.parent / "output" / "voice_clone" / f"{name}_demo.wav"
            try:
                print(f"demo:           {await vc.download_demo(status, out)}")
            except Exception as e:
                print(f"demo:           {status.demo_audio} (download failed: {e})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clone a voice with Volcengine (Doubao), or query its status.")
    parser.add_argument("audio", nargs="?", help="voice sample (wav/mp3/ogg/m4a/aac/pcm, <= 10 MB); omit to only query")
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--speaker-id", help="prepaid voice slot, e.g. S_xxxxxxxx")
    who.add_argument("--custom-speaker-id", help="postpaid voice name of your own, e.g. myVoice01")
    parser.add_argument("--format", help="audio format, needed for pcm / m4a or when the file has no suffix")
    parser.add_argument("--text", help="what the sample says (training fails if the speech differs too much)")
    parser.add_argument("--language", default="cn", help="language of the sample: cn, en, ja, ... (default cn)")
    parser.add_argument("--demo-text", help="4-300 characters to synthesize as a demo")
    parser.add_argument("--denoise", action="store_true", help="denoise a noisy sample")
    parser.add_argument("--no-volume-normalization", action="store_true", help="keep the sample's loudness")
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds to wait for training (default 180)")
    asyncio.run(_main(parser.parse_args()))
