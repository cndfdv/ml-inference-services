"""Run the standalone CPU service with configurable process workers."""
import os
import uvicorn


def main() -> None:
    workers = int(os.environ.get("EASYOCR_WORKERS", "1"))
    if workers < 1:
        raise ValueError("WORKERS must be a positive integer")
    thread_count = os.environ.get("EASYOCR_OMP_NUM_THREADS")
    if thread_count:
        os.environ.setdefault("OMP_NUM_THREADS", thread_count)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=workers)


if __name__ == "__main__":
    main()
