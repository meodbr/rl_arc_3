from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    LOGGING_CONFIG: str = "config/logging.json"
    CHECKPOINT_DIR: str = "data/checkpoints"

settings = Settings()