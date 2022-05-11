import json
import sys
import time
from binascii import unhexlify

import torch
torch.set_num_threads(16)

from accdfl.core.model import create_model
from accdfl.core.model_trainer import ModelTrainer

if __name__ == "__main__":
    with open("time_stats.txt", "a") as time_stats:
        time_stats.write("start,%f\n" % time.time())

    model_path = sys.argv[1]
    datadir = sys.argv[2]
    parameters = json.loads(unhexlify(sys.argv[3]))
    participant_index = int(sys.argv[4])
    model = create_model(parameters["dataset"])

    with open("time_stats.txt", "a") as time_stats:
        time_stats.write("model_created,%f\n" % time.time())

    model.load_state_dict(torch.load(model_path))

    with open("time_stats.txt", "a") as time_stats:
        time_stats.write("model_loaded,%f\n" % time.time())

    trainer = ModelTrainer(datadir, parameters, participant_index)
    trainer.train(model)

    with open("time_stats.txt", "a") as time_stats:
        time_stats.write("model_trained,%f\n" % time.time())

    # Save the model to the file
    torch.save(model.state_dict(), model_path)

    with open("time_stats.txt", "a") as time_stats:
        time_stats.write("model_saved,%f\n" % time.time())