import os

from rl_arc_3.base.trainer import TrainingArgs
from rl_arc_3.settings import settings

def checkpoint_dir(config: TrainingArgs):
    return os.path.join(config.output_dir, settings.CHECKPOINT_DIR_NAME)

def output_model_path(config: TrainingArgs):
    return os.path.join(config.output_dir, "model.pth")