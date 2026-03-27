import os
from rl_arc_3.settings import settings

def compute_run_name(output_dir: str) -> str:
    metric_dir = os.path.join(output_dir, settings.METRICS_DIR_NAME)
    os.makedirs(metric_dir, exist_ok=True)
    subdirs = [path.split("/")[-1] for path in os.listdir(metric_dir) if not os.path.isfile(path)]

    for i in range(len(subdirs) + 1):
        new_name = f"run{i+1:04d}"
        if new_name not in subdirs:
            os.makedirs(os.path.join(metric_dir, new_name))
            return new_name
    
    raise RuntimeError("Failed to assign run name automatically, you should set it manually (training_args.run)")
