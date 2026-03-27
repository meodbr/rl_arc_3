import sys

from rl_arc_3.trainer.metric_hubs import get_metric_hub

import argparse

def run(model_name: str, run_name: str, metric: str, metric_hub_method: str):
    """
    Plot a metric
    
    Parameters:
        model_name (str): Name of the model
        run_name (str): Name of the run
        metric (str): Metric to compute/use
        metric_hub_method (str): Method used (default: csv)
    """
    print("Plotting with args :")
    print(f"  model_name={model_name}")
    print(f"  run_name={run_name}")
    print(f"  metric={metric}")
    print(f"  metric_hub_method={metric_hub_method}")

    metric_hub = get_metric_hub(metric_hub_method, output_dir=model_name)
    metric_hub.plot(run=run_name, metric=metric)


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for running model metrics"
    )

    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument("run_name", type=str, help="Run identifier")
    parser.add_argument("metric", type=str, help="Metric to use")

    parser.add_argument(
        "--metric-hub-method",
        type=str,
        default="csv",
        help="Metric hub method (default: csv)",
    )

    args = parser.parse_args()

    run(
        model_name=args.model_name,
        run_name=args.run_name,
        metric=args.metric,
        metric_hub_method=args.metric_hub_method,
    )


if __name__ == "__main__":
    main()