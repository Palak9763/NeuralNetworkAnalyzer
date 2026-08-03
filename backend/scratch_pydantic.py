from pydantic_settings import BaseSettings
from pydantic import field_validator
import os

os.environ["NNA_CORS_ORIGINS"] = "https://neural-network-analyzer.vercel.app,http://localhost:5173"

class Settings(BaseSettings):
    cors_origins: list[str] = ["http://localhost:5173"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        print(f"Parsing: {v}")
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    class Config:
        env_prefix = "NNA_"

try:
    s = Settings()
    print("Success:", s.cors_origins)
except Exception as e:
    print("Error:", e)
