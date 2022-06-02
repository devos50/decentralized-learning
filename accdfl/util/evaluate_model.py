import json
import os
import sys

import torch

from accdfl.core.model import create_model
from accdfl.core.model_evaluator import ModelEvaluator

if __name__ == "__main__":
    work_dir = sys.argv[1]
    model_id = int(sys.argv[2])
    model_name = "%d.model" % model_id
    datadir = sys.argv[3]
    num_threads = int(sys.argv[4])
    torch.set_num_threads(num_threads)

    model_path = os.path.join(work_dir, model_name)
    if not os.path.exists(model_path):
        raise RuntimeError("Model %s does not exist!" % model_path)

    with open(os.path.join(work_dir, "parameters.json")) as in_file:
        parameters = json.loads(in_file.read())

    model = create_model(parameters["dataset"])
    model.load_state_dict(torch.load(model_path))

    evaluator = ModelEvaluator(datadir, parameters)
    acc, loss = evaluator.evaluate_accuracy(model)

    # Save the accuracies
    with open("%d_results.csv" % model_id, "w") as out_file:
        out_file.write("%s,%s\n" % (acc, loss))
