from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    LOGGING_CONFIG: str = "config/logging.json"
    CHECKPOINT_DIR_NAME: str = "checkpoints"
    METRICS_DIR_NAME: str = "metrics"

settings = Settings()