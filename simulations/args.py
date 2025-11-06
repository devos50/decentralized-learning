import argparse


def get_args(dataset: str):
    parser = argparse.ArgumentParser()

    # Learning settings
    parser.add_argument('--client-learning-rate', type=float, default=4e-4)
    parser.add_argument('--client-optimizer', type=str, default="adamW", choices=["adamW"])
    parser.add_argument('--server-learning-rate', type=float, default=0.7)
    parser.add_argument('--server-optimizer', type=str, default="nesterov", choices=["nesterov"])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--local-steps', type=int, default=5)

    # Accuracy testing
    parser.add_argument('--dl-test-mode', type=str, default="local")
    parser.add_argument('--accuracy-logging-interval', type=int, default=5)
    parser.add_argument('--accuracy-logging-interval-is-in-sec', action=argparse.BooleanOptionalAction)
    parser.add_argument('--dl-accuracy-method', type=str, default="individual")  # individual or aggregate

    # Traces
    parser.add_argument('--availability-traces', type=str, default=None)
    parser.add_argument('--traces', type=str, default="none", choices=["none", "fedscale", "diablo"])
    parser.add_argument('--seed', type=int, default=42)

    # Other settings
    parser.add_argument('--log-level', type=str, default="INFO")
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--dataset-base-path', type=str, default=None)
    parser.add_argument('--duration', type=int, default=3600)  # Set to 0 to run forever
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--model', type=str, default="roberta-base", choices=["roberta-base", "google/vit-base-patch16-224", "gpt2"])
    parser.add_argument('--partitioner', type=str, default="uniform", choices=["uniform", "dirichlet"])
    parser.add_argument('--peers', type=int, default=10)
    parser.add_argument('--active-participants', type=str, default=None)
    parser.add_argument('--checkpoint-interval', type=int, default=None)
    parser.add_argument('--bypass-model-transfers', action=argparse.BooleanOptionalAction)
    parser.add_argument('--bypass-training', action=argparse.BooleanOptionalAction)
    parser.add_argument('--store-best-models', action=argparse.BooleanOptionalAction)
    parser.add_argument('--profile', action=argparse.BooleanOptionalAction)
    parser.add_argument('--log-events', action=argparse.BooleanOptionalAction)
    parser.add_argument('--el', action=argparse.BooleanOptionalAction, help="Uses Epidemic Learning (topology randomization each round)")
    parser.add_argument('--topology', type=str, default="k-regular")
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--latencies-file', type=str, default="data/latencies.txt")
    parser.add_argument('--fix-aggregator', action=argparse.BooleanOptionalAction)
    parser.add_argument('--success-fraction', type=float, default=1.0)
    parser.add_argument('--liveness-success-fraction', type=float, default=0.4)
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--num-aggregators', type=int, default=1)
    parser.add_argument('--aggregation-timeout', type=float, default=300)
    parser.add_argument('--activity-log-interval', type=int, default=None)
    parser.add_argument('--flush-statistics-interval', type=int, default=600)
    parser.add_argument('--write-view-histories', action=argparse.BooleanOptionalAction)
    parser.add_argument('--aggregate', type=str, default="fedavg", choices=["fedavg", "fedadam", "fednesterov"])
    parser.add_argument('--device', type=str, default=None)

    return parser.parse_args()
