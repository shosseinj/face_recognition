# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "0.0.0.0"

LOCAL_IP = get_local_ip()

app = FastAPI(
    title="Face Detection API",
    description="API for face detection and recognition",
    version="1.0.0",
    servers=[
        {"url": f"http://{LOCAL_IP}:8000", "description": "Local Network"},
        {"url": "http://0.0.0.0:8000", "description": "Localhost"},
    ],
    docs_url=None,  # Disable auto-generated docs
    redoc_url=None,  # Disable auto-generated redoc
)

# MOUNT STATIC FILES - MUST BE BEFORE ROUTERS!
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include your API router
from .api import router
app.include_router(router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Face Detection API is running"}

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    Serve Swagger UI from local files (works offline!)
    """
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Face Detection API - Swagger UI (Offline)",
        swagger_js_url="/static/swagger-ui-bundle.js",  # Local file
        swagger_css_url="/static/swagger-ui.css",       # Local file
        swagger_favicon_url="/static/favicon.png",      # Local file
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_json():
    openapi_schema = app.openapi()
    openapi_schema["servers"] = [
        {"url": f"http://{LOCAL_IP}:8000", "description": "Local Network"},
        {"url": "http://localhost:8000", "description": "Localhost"},
        {"url": "http://0.0.0.0:8000", "description": "All Interfaces"},
    ]
    return openapi_schema

# CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)