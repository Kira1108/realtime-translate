from dotenv import load_dotenv
from pathlib import Path
import logging
import os
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider


from app.config import get_config

logger = logging.getLogger(__name__)

def create_pydantic_ai_model(
    model_name:str = "google/gemma-4-31b-it",
    api_key:str = None):
    
    config = get_config()
    api_key = api_key or config.openrouter_api_key
    
    return OpenRouterModel(
        model_name = model_name,
        provider=OpenRouterProvider(api_key=api_key),
    )


if __name__ == "__main__":
    model = create_pydantic_ai_model()
    agent = Agent(model=model)
    response = agent.run_sync("Hello, how are you?")
    print(response)