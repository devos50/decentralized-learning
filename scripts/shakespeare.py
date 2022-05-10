from accdfl.core.datasets.Shakespeare import Shakespeare, LSTM
from accdfl.core.mappings import Linear

train_dir = "/Users/martijndevos/Documents/leaf/shakespeare/per_user_data/train"
test_dir = "/Users/martijndevos/Documents/leaf/shakespeare/data/test"

mapping = Linear(1, 100)
s = Shakespeare(0, 0, mapping, train_dir=train_dir, test_dir=test_dir)
print(s.get_trainset(batch_size=20, shuffle=True))
print(s.get_testset())

# Model
model = LSTM()
print(model)