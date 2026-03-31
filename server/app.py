import os

import uvicorn
from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_fastapi_app

from environment import DataJanitorEnvironment
from models import DataJanitorAction, DataJanitorObservation

app = create_fastapi_app(DataJanitorEnvironment, DataJanitorAction, DataJanitorObservation)

_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    with open(os.path.join(_STATIC, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
