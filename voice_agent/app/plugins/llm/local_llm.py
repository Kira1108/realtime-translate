
from pydantic_ai.models.openai import ModelSettings, OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
import uuid

def create_pydantic_ai_model(
    **kwargs):
    """
    Create openai like pydantic-compatible model.
    Tested with sglang deployed models
    """
    
    session_id = str(uuid.uuid4())

    default_headers  = {
        "AccountId": "Algorithm",
        "AccountToken": "3fc524bd5e1a6f7fe8c9480b74cc2bc3",
        "AppId": "2041330513012068352",
        "TenantId": "YIXIN",
        "TokenKey": "w0p795gnp9htvfp0l7ntbaty5jzk2yr4",
        "RequestId": "test_1234567890",
        "TransId": "TEST_GEMMA"
    }
    default_headers.update(
        {
            "TransId": session_id,
            'UserSessionId': session_id
        }
    )
    
    client = AsyncOpenAI(
        base_url="https://vesta2.yxqiche.com/vesta/v2/engine_online/flow_schedule/schedule_run_openai/v1",
        api_key="EMPTY",
        default_headers=default_headers,
    )
    
    model_settings = ModelSettings(
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
                "separate_reasoning": False
            }
        },
        max_tokens=16384,
        **kwargs
    ) 
        
    return OpenAIChatModel(
        model_name = "gemma-4-31B-it",
        provider = OpenAIProvider(openai_client=client),
        settings = model_settings
        
    )