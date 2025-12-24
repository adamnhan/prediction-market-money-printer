import os
from dotenv import load_dotenv
import uvicorn


def main():
    load_dotenv()
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    reload = os.getenv("APP_RELOAD", "").lower() in {"1", "true", "yes"}

    # Single worker to avoid splitting in-memory engine state across processes
    uvicorn.run(
        "trading_engine.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=1,
    )


if __name__ == "__main__":
    main()
