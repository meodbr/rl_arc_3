import os
import logging

import pandas as pd
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from rl_arc_3.base.trainer import BaseMetricHub
from rl_arc_3.settings import settings

logger = logging.getLogger(__name__)


def get_metric_hub(name: str, output_dir: str) -> BaseMetricHub:
    hub_class_register = {"csv": CSVMetricHub}
    if name in hub_class_register:
        return hub_class_register[name](output_dir=output_dir)

    logger.error(
        "Wrong metric hub name : %s, should be in %s, defaulting to 'csv'",
        name,
        hub_class_register.keys(),
    )
    return CSVMetricHub(output_dir=output_dir)


class CSVMetricHub(BaseMetricHub):
    TYPE_MAP = {
        "worker": "w--",
        "learner": "l--",
        "memory": "m--",
    }

    def __init__(self, output_dir: str):
        self.base_dir = os.path.join(output_dir, settings.METRICS_DIR_NAME)

        os.makedirs(self.base_dir, exist_ok=True)
    
    def validate_run(self, run: str):
        assert run is not None

    def save(self, data: dict, run: str, emitter: str) -> None:
        self.validate_run(run)
        filename = self._get_csv(
            run, emitter_name=mp.current_process().name, emitter_type=emitter
        )
        data_for_df = {key: [val] for key, val in data.items()}
        new_df = pd.DataFrame(data_for_df)

        if os.path.exists(filename):
            df = pd.read_csv(filename, sep=";")
            new_df = pd.concat([df, new_df], ignore_index=True)

        new_df.to_csv(filename, sep=";", index=False)

    def get(self, run: str) -> pd.DataFrame:
        self.validate_run(run)
        df = self._get_global_dataframe(run)
        return df.sort_values(by="global_step").reset_index(drop=True)

    def plot(self, run: str, metric: str) -> None:
        self.validate_run(run)
        df = self.get(run)
        assert metric in df.columns

        x_label = "global_step"
        y_label = metric

        df = df.dropna(subset=[x_label, y_label])

        x = df[x_label]
        y = df[y_label]

        fig, ax = plt.subplots()

        ax.plot(x, y)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        plt.show()

    def _get_dataframes_by_type(self, run: str):
        dir_path = os.path.join(self.base_dir, run)
        result = {key: [] for key in self.TYPE_MAP}

        for name in os.listdir(dir_path):
            full_path = os.path.join(dir_path, name)

            if not os.path.isfile(full_path):
                continue

            if not name.endswith(".csv"):
                continue

            for key, prefix in self.TYPE_MAP.items():
                if name.startswith(prefix):
                    df = pd.read_csv(full_path, sep=";")
                    df["_source_file"] = name
                    result[key].append(df)
                    break  # stop once matched

        return result

    def _get_global_dataframe(self, run: str):
        df_dict = self._get_dataframes_by_type(run)
        df_list = []

        for emitter_type, processes_df_list in df_dict.items():
            for p_df in processes_df_list:
                p_df["source_type"] = emitter_type

                df_list.append(p_df)

        df = pd.concat(df_list)
        return df

    def _get_csv(self, run: str, emitter_name: str, emitter_type):
        run_dir = os.path.join(self.base_dir, run)
        os.makedirs(run_dir, exist_ok=True)
        if emitter_type not in self.TYPE_MAP:
            raise RuntimeError(
                f"Emitter should be in {self.TYPE_MAP.keys()}, got {emitter_type}"
            )
        csv_file = os.path.join(
            run_dir, f"{self.TYPE_MAP[emitter_type]}{emitter_name}.csv"
        )
        return csv_file
