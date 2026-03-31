import os

import uvicorn
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_fastapi_app

from environment import DataJanitorEnvironment
from models import DataJanitorAction, DataJanitorObservation

app = create_fastapi_app(DataJanitorEnvironment, DataJanitorAction, DataJanitorObservation)

# Allow embedding in HF Spaces iframe and cross-origin API calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_IFRAME_HEADERS = {
    "X-Frame-Options": "ALLOWALL",
    "Content-Security-Policy": "frame-ancestors *",
    "Cache-Control": "no-store",
}


def _serve_dashboard():
    with open(os.path.join(_STATIC, "index.html"), encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, headers=_IFRAME_HEADERS)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_root():
    return _serve_dashboard()


# HF Spaces App tab hits /web when ENABLE_WEB_INTERFACE=true is set
@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_web():
    return _serve_dashboard()


@app.get("/web/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_web_slash():
    return _serve_dashboard()


def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
