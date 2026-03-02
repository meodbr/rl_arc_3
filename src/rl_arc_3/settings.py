from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    LOGGING_CONFIG: str = "config/logging.json"

settings = Settings()