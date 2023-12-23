from accdfl.core.datasets.stackoverflow import stackoverflow

train_dataset = stackoverflow("/var/scratch/spandey/dfl-data/stackoverflow", train=True)
test_dataset = stackoverflow("/var/scratch/spandey/dfl-data/stackoverflow", train=False)