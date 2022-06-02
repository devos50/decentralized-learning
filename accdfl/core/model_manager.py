import asyncio
import copy
import json
import logging
import os
import random
import sys
import time
from typing import Dict, Optional, List

import torch
import torch.nn as nn

from accdfl.core.model_trainer import ModelTrainer


class ModelManager:
    """
    This class manages the current ML model and training.
    """

    def __init__(self, model, parameters, participant_index: int):
        self.model = model
        self.parameters = parameters
        self.participant_index = participant_index
        self.logger = logging.getLogger(self.__class__.__name__)
        self.training_times: List[float] = []

        if self.parameters["dataset"] in ["cifar10", "cifar10_niid", "mnist"]:
            self.data_dir = os.path.join(os.environ["HOME"], "dfl-data")
        else:
            # The LEAF dataset
            self.data_dir = os.path.join(os.environ["HOME"], "leaf", self.parameters["dataset"])

        self.model_trainer = None  # Only used when not training in a subprocess (e.g., in the unit tests)

        # Keeps track of the incoming trained models as aggregator
        self.incoming_trained_models: Dict[bytes, nn.Module] = {}

    def process_incoming_trained_model(self, peer_pk: bytes, incoming_model: nn.Module):
        if peer_pk in self.incoming_trained_models:
            # We already processed this model
            return

        self.incoming_trained_models[peer_pk] = incoming_model

    def reset_incoming_trained_models(self):
        self.incoming_trained_models = {}

    def has_enough_trained_models(self) -> bool:
        return len(self.incoming_trained_models) >= (self.parameters["sample_size"] * self.parameters["success_fraction"])

    def average_trained_models(self) -> Optional[nn.Module]:
        models = [model for model in self.incoming_trained_models.values()]
        return self.average_models(models)

    def dump_parameters(self):
        """
        Dump the experiment parameters if they do not exist yet.
        """
        parameters_file_path = os.path.join(self.parameters["work_dir"], "parameters.json")
        if not os.path.exists(parameters_file_path):
            with open(parameters_file_path, "w") as parameters_file:
                parameters_file.write(json.dumps(self.parameters))

    async def train(self, in_subprocess: bool = True):
        if in_subprocess:
            # Dump the model and parameters to a file
            model_file_name = "%d.model" % random.randint(1, 1000000)
            model_path = os.path.join(self.parameters["work_dir"], model_file_name)
            torch.save(self.model.state_dict(), model_path)
            self.dump_parameters()

            # Get full path to the script
            import accdfl.util as autil
            script_dir = os.path.join(os.path.abspath(os.path.dirname(autil.__file__)), "train_model.py")
            cmd = "%s %s %s %s %s %s %d" % (sys.executable, script_dir, self.parameters["work_dir"], model_file_name,
                                            self.data_dir, self.participant_index, torch.get_num_threads())
            train_start_time = time.time()
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)

            stdout, stderr = await proc.communicate()
            self.logger.info(f'Training exited with {proc.returncode}]')

            self.training_times.append(time.time() - train_start_time)

            if proc.returncode != 0:
                if stdout:
                    self.logger.error(f'[stdout]\n{stdout.decode()}')
                if stderr:
                    self.logger.error(f'[stderr]\n{stderr.decode()}')
                raise RuntimeError("Training subprocess exited with non-zero exit code %d: %s" %
                                   (proc.returncode, stderr.decode()))

            # Read the new model and adopt it
            self.model.load_state_dict(torch.load(model_path))
            os.unlink(model_path)
        else:
            if not self.model_trainer:
                # Lazy initialize the model trainer
                self.model_trainer = ModelTrainer(self.data_dir, self.parameters, self.participant_index)
            self.model_trainer.train(self.model)

    @staticmethod
    def average_models(models: List[nn.Module]) -> nn.Module:
        with torch.no_grad():
            weights = [float(1. / len(models)) for _ in range(len(models))]
            center_model = copy.deepcopy(models[0])
            for p in center_model.parameters():
                p.mul_(0)
            for m, w in zip(models, weights):
                for c1, p1 in zip(center_model.parameters(), m.parameters()):
                    c1.add_(w * p1)
            return center_model

    async def compute_accuracy(self, model: nn.Module):
        """
        Compute the accuracy/loss of the current model.
        Optionally, one can provide a custom iterator to compute the accuracy on a custom dataset.
        """
        self.logger.info("Computing accuracy of model")

        # Dump the model and parameters to a file
        model_id = random.randint(1, 1000000)
        model_file_name = "%d.model" % model_id
        model_path = os.path.join(self.parameters["work_dir"], model_file_name)
        torch.save(model.state_dict(), model_path)
        self.dump_parameters()

        # Get full path to the script
        import accdfl.util as autil
        script_dir = os.path.join(os.path.abspath(os.path.dirname(autil.__file__)), "evaluate_model.py")
        self.logger.error(script_dir)
        cmd = "%s %s %s %d %s %d" % (sys.executable, script_dir, self.parameters["work_dir"], model_id,
                                     self.data_dir, torch.get_num_threads())
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)

        stdout, stderr = await proc.communicate()
        self.logger.info(f'Accuracy evaluator exited with {proc.returncode}]')

        if proc.returncode != 0:
            if stdout:
                self.logger.error(f'[stdout]\n{stdout.decode()}')
            if stderr:
                self.logger.error(f'[stderr]\n{stderr.decode()}')
            raise RuntimeError("Accuracy evaluation subprocess exited with non-zero exit code %d: %s" %
                               (proc.returncode, stderr.decode()))

        os.unlink(model_path)

        # Read the accuracy and the loss from the file
        results_file = os.path.join(os.getcwd(), "%d_results.csv" % model_id)
        with open(results_file) as in_file:
            content = in_file.read().strip().split(",")

        os.unlink(results_file)

        return float(content[0]), float(content[1])
