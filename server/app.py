import uvicorn
from openenv.core.env_server import create_fastapi_app
from environment import DataJanitorEnvironment
from models import DataJanitorAction, DataJanitorObservation

app = create_fastapi_app(DataJanitorEnvironment, DataJanitorAction, DataJanitorObservation)


def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
