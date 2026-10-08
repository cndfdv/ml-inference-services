import uvicorn

from .settings import Settings


def main():
    settings = Settings.from_env()
    from .prepare import ensure_prepared

    ensure_prepared(settings)
    uvicorn.run("service.app:app", host="0.0.0.0", port=8000, workers=settings.workers)


if __name__ == "__main__":
    main()
