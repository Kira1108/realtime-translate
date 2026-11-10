from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import applications
from app.logging_config import setup_logging

setup_logging()

from app.routers import test
from app.routers import voice_changer
from app.routers import translator
from app.routers import chatbot

def swagger_monkey_patch(*args, **kwargs):
    return get_swagger_ui_html(
        *args, **kwargs,
        swagger_js_url="https://cdn.staticfile.net/swagger-ui/5.1.0/swagger-ui-bundle.min.js",
        swagger_css_url="https://cdn.staticfile.net/swagger-ui/5.1.0/swagger-ui.min.css"
    )
    
applications.get_swagger_ui_html = swagger_monkey_patch


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(test.router)
app.include_router(voice_changer.router)
app.include_router(translator.router)
app.include_router(chatbot.router)