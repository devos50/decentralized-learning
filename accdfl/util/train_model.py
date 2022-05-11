import json
import sys
from binascii import unhexlify

import torch

from accdfl.core.model import create_model
from accdfl.core.model_trainer import ModelTrainer

if __name__ == "__main__":
    model_path = sys.argv[1]
    datadir = sys.argv[2]
    parameters = json.loads(unhexlify(sys.argv[3]))
    participant_index = int(sys.argv[4])
    model = create_model(parameters["dataset"])

    model.load_state_dict(torch.load(model_path))
    trainer = ModelTrainer(datadir, parameters, participant_index)
    trainer.train(model)

    # Save the model to the file
    torch.save(model.state_dict(), model_path)
