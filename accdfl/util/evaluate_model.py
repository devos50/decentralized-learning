import json
import sys
import time
from binascii import unhexlify

import torch

from accdfl.core.model_evaluator import ModelEvaluator

torch.set_num_threads(16)

from accdfl.core.model import create_model

if __name__ == "__main__":
    with open("time_stats.txt", "a") as time_stats:
        time_stats.write("start,%f\n" % time.time())

    model_id = int(sys.argv[1])
    model_path = sys.argv[2]
    datadir = sys.argv[3]
    parameters = json.loads(unhexlify(sys.argv[4]))
    model = create_model(parameters["dataset"])

    with open("time_stats.txt", "a") as time_stats:
        time_stats.write("model_created,%f\n" % time.time())

    model.load_state_dict(torch.load(model_path))

    with open("time_stats.txt", "a") as time_stats:
        time_stats.write("model_loaded,%f\n" % time.time())

    evaluator = ModelEvaluator(datadir, parameters)
    acc, loss = evaluator.evaluate_accuracy(model)

    with open("time_stats.txt", "a") as time_stats:
        time_stats.write("model_evaluated,%f\n" % time.time())

    # Save the accuracies
    with open("%d_results.csv", "w") as out_file:
        out_file.write("%s,%s\n" % (acc, loss))

    with open("accs.txt", "a") as out_file:
        out_file.write("%s,%s\n" % (acc, loss))
