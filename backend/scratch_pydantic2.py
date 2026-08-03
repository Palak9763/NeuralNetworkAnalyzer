from pydantic_settings import BaseSettings
from pydantic import field_validator
import os

os.environ["NNA_CORS_ORIGINS"] = '["https://neural-network-analyzer.vercel.app", "http://localhost:5173"]'

class Settings(BaseSettings):
    cors_origins: list[str] = ["http://localhost:5173"]

    class Config:
        env_prefix = "NNA_"

try:
    s = Settings()
    print("Success:", s.cors_origins)
except Exception as e:
    print("Error:", e)
