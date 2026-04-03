import os


class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    USE_VLLM = os.getenv("USE_VLLM", "0") == "1"
    VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
    VLLM_MODEL = os.getenv("VLLM_MODEL", "local-model")

    CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
    CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
    CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
    CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
    CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "gold")