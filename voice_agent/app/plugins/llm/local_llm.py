
from pydantic_ai.models.openai import ModelSettings, OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
import uuid
from app.config import get_config

def create_pydantic_ai_model(
    **kwargs):
    """
    Create openai like pydantic-compatible model.
    Tested with sglang deployed models
    """
    
    config = get_config()
    
    session_id = str(uuid.uuid4())

    default_headers  = {
        "AccountId": "Algorithm",
        "AccountToken": config.private_model.AccountToken,
        "AppId": config.private_model.AppId,
        "TenantId": config.private_model.TenantId,
        "TokenKey": config.private_model.TokenKey,
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
        base_url=config.private_model.URL,
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