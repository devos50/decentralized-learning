import torch
import torch.nn as nn
import torchvision.models as models

# Initialize the model
model = models.mobilenet_v3_small(weights=None)

# Collect all parameters into a single flat tensor
def get_flat_params(model):
    param_tensors = [param.data.view(-1) for param in model.parameters()]
    flat_params = torch.cat(param_tensors)
    return flat_params

flat_params = get_flat_params(model)
print(flat_params[0])

print(len(flat_params))

num_nodes = 100
total_elements = flat_params.numel()
chunk_size = total_elements // num_nodes

# Create chunks
chunks = [flat_params[i * chunk_size: (i + 1) * chunk_size] for i in range(num_nodes)]

# Handle any remaining elements
if total_elements % num_nodes != 0:
    remaining = flat_params[num_nodes * chunk_size:]
    chunks[-1] = torch.cat([chunks[-1], remaining])

print(len(chunks[-1]))
