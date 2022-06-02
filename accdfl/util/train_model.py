import json
import os
import sys

import torch

from accdfl.core.model import create_model
from accdfl.core.model_trainer import ModelTrainer

if __name__ == "__main__":
    work_dir = sys.argv[1]
    model_name = sys.argv[2]
    datadir = sys.argv[3]
    participant_index = int(sys.argv[4])
    torch.set_num_threads(int(sys.argv[5]))

    model_path = os.path.join(work_dir, model_name)
    if not os.path.exists(model_path):
        raise RuntimeError("Model %s does not exist!" % model_path)

    with open(os.path.join(work_dir, "parameters.json")) as in_file:
        parameters = json.loads(in_file.read())

    model = create_model(parameters["dataset"])
    model.load_state_dict(torch.load(model_path))

    trainer = ModelTrainer(datadir, parameters, participant_index)
    trainer.train(model)

    # Save the model to the file
    torch.save(model.state_dict(), model_path)
