import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "inference.app:app", host="0.0.0.0", port=8000, workers=int(os.getenv("WORKERS", "1"))
    )
