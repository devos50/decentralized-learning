"""
Standalone script to evaluate the accuracy of one or more models, given a settings file.
"""
import os
import sys

import torch

from accdfl.core.models import create_model
from accdfl.core.model_evaluator import ModelEvaluator
from accdfl.core.session_settings import SessionSettings

if __name__ == "__main__":
    work_dir = sys.argv[1]
    model_ids = [int(model_id) for model_id in sys.argv[2].split(",")]
    datadir = sys.argv[3]

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"

    with open(os.path.join(work_dir, "settings.json")) as in_file:
        settings: SessionSettings = SessionSettings.from_json(in_file.read())

    evaluator = ModelEvaluator(datadir, settings)

    for model_id in model_ids:
        model_name = "%d.model" % model_id
        model = create_model(settings.dataset, architecture=settings.model)
        model.to(device)

        model_path = os.path.join(work_dir, model_name)
        if not os.path.exists(model_path):
            raise RuntimeError("Model %s does not exist!" % model_path)

        model.load_state_dict(torch.load(model_path))
        acc, loss = evaluator.evaluate_accuracy(model, device_name=device)

        # Save the accuracies
        with open(os.path.join(work_dir, "%d_results.csv" % model_id), "w") as out_file:
            out_file.write("%f,%f\n" % (acc, loss))
