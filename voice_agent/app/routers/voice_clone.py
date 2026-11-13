"""Voice cloning endpoints (Volcengine / Doubao), on top of app/plugins/voice_clone/seed.py.

    POST /voice_clone          upload a voice sample (multipart form) and train a voice
    GET  /voice_clone/status   query a voice's status and demo audio

Example:
    curl -F file=@me.wav -F speaker_id=S_xxxxxxxx -F demo_text="你好，这是我复刻的声音。" \
         http://localhost:8000/voice_clone
    curl "http://localhost:8000/voice_clone/status?speaker_id=S_xxxxxxxx"
"""
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.plugins.voice_clone.seed import (
    AUDIO_FORMATS,
    MAX_AUDIO_BYTES,
    SeedVoiceClone,
    VoiceCloneError,
    VoiceStatus,
)

router = APIRouter(
    prefix="/voice_clone",
    tags=["voice_clone"],
)


class VoiceCloneResult(BaseModel):
    speaker_id: str
    status: str                             # NOT_FOUND / TRAINING / SUCCESS / FAILED / ACTIVE
    ready: bool                             # SUCCESS or ACTIVE: usable for synthesis
    available_training_times: int | None    # trainings left on this voice
    create_time: int | None                 # ms timestamp
    language: int | None
    demo_audio: str | None                  # demo clip URL, valid for an hour after training
    model_type: int | None                  # 5 = voice clone 2.0

    @classmethod
    def from_status(cls, status: VoiceStatus) -> "VoiceCloneResult":
        return cls(
            speaker_id=status.speaker_id,
            status=status.status.name,
            ready=status.ready,
            available_training_times=status.available_training_times,
            create_time=status.create_time,
            language=status.language,
            demo_audio=status.demo_audio,
            model_type=status.model_type,
        )


def _http_error(e: Exception) -> HTTPException:
    if isinstance(e, VoiceCloneError):
        # the service's own error code and logid, for a support ticket
        return HTTPException(502, {"code": e.code, "message": str(e), "logid": e.logid})
    if isinstance(e, TimeoutError):
        return HTTPException(504, str(e))
    return HTTPException(400, str(e))


@router.post("", response_model=VoiceCloneResult)
async def clone_voice(
    # the voice sample: wav / mp3 / ogg / m4a / aac / pcm (pcm: 24 kHz mono 16-bit), at most 10 MB.
    # 10-30 s of clean speech from one person works best.
    file: UploadFile = File(..., description="声音样本:wav/mp3/ogg/m4a/aac/pcm(pcm 须 24kHz 单声道),≤10MB"),

    # which voice to train, exactly one of the two:
    #   speaker_id         a prepaid slot bought in the console, e.g. "S_xxxxxxxx"
    #   custom_speaker_id  a postpaid name of your own (needs postpaid cloning enabled); 8-256 letters,
    #                      digits, '-' or '_', starting with a letter; free to train and preview,
    #                      the slot fee is charged on its first synthesis
    speaker_id: str | None = Form(None, description="预付费音色 ID,如 S_xxxxxxxx(与 custom_speaker_id 二选一)"),
    custom_speaker_id: str | None = Form(None, description="后付费自定义音色名,如 myVoice01(首次合成时收槽位费)"),

    # the audio format; taken from the file name when omitted, required for pcm and m4a
    audio_format: str | None = Form(None, description="音频格式,不填按文件后缀判断;pcm、m4a 必填"),

    # what the sample says; if the speech differs too much the training fails (45001109 WERError)
    text: str | None = Form(None, description="样本的朗读文本,和录音差异过大会训练失败"),

    # language of the sample: cn, en, ja, es, id, pt, de, fr, ko, it, th, vi, ru, fil, ms, ar, mx, pt-br, pl, tr, sv
    language: str = Form("cn", description="样本语种,如 cn、en、ja"),

    # 4-300 characters to synthesize as a demo clip, in `language`; longer text makes training slower
    demo_text: str | None = Form(None, description="试听文本,4~300 字,语种需与 language 一致"),

    # denoise a noisy sample; costs some detail on clean ones
    denoise: bool = Form(False, description="样本噪声大时开启降噪"),

    # keep the sample's loudness: closer to the sample, but less uniform volume
    disable_volume_normalization: bool = Form(False, description="关闭音量归一化,音量更贴近样本"),

    # wait for training to finish (polling the status) instead of returning right away
    wait: bool = Form(True, description="是否等训练完成再返回"),

    # how long to wait when `wait` is on, in seconds
    timeout: float = Form(180.0, description="等待训练完成的最长秒数"),
):
    if bool(speaker_id) == bool(custom_speaker_id):
        raise HTTPException(400, "speaker_id 和 custom_speaker_id 必须且只能填一个")

    audio_format = audio_format or (Path(file.filename or "").suffix.lstrip(".").lower() or None)
    if audio_format and audio_format not in AUDIO_FORMATS:
        raise HTTPException(400, f"不支持的音频格式 {audio_format!r},支持:{', '.join(sorted(AUDIO_FORMATS))}")
    data = await file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "音频超过 10MB")

    ids = dict(speaker_id=speaker_id or None, custom_speaker_id=custom_speaker_id or None)
    try:
        async with SeedVoiceClone() as vc:
            status = await vc.clone(
                data, **ids,
                audio_format=audio_format,
                text=text or None,
                language=language,
                demo_text=demo_text or None,
                denoise=denoise,
                disable_volume_normalization=disable_volume_normalization,
            )
            if wait and not status.ready:
                status = await vc.wait_until_ready(**ids, timeout=timeout)
    except (VoiceCloneError, TimeoutError, ValueError) as e:
        raise _http_error(e)
    return VoiceCloneResult.from_status(status)


@router.get("/status", response_model=VoiceCloneResult)
async def voice_status(
    # exactly one of the two, as for training
    speaker_id: str | None = Query(None, description="预付费音色 ID,如 S_xxxxxxxx"),
    custom_speaker_id: str | None = Query(None, description="后付费自定义音色名"),
):
    if bool(speaker_id) == bool(custom_speaker_id):
        raise HTTPException(400, "speaker_id 和 custom_speaker_id 必须且只能填一个")
    try:
        async with SeedVoiceClone() as vc:
            status = await vc.get_voice(speaker_id=speaker_id, custom_speaker_id=custom_speaker_id)
    except (VoiceCloneError, ValueError) as e:
        raise _http_error(e)
    return VoiceCloneResult.from_status(status)
