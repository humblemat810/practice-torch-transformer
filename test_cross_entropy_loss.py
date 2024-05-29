import torch
import torch.nn as nn

# Example predictions (logits) and targets for a batch of 3 samples and 4 classes
predictions = torch.tensor([[1.2, 0.9, 0.3, 2.1], 
                            [1.1, 2.3, 0.4, 0.7], 
                            [0.5, 1.4, 2.2, 1.0]], dtype=torch.float)

# Targets corresponding to the correct class indices
targets = torch.tensor([3, 1, 2], dtype=torch.long)

# Define the CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# Compute the loss
loss = criterion(predictions, targets)

print("Loss:", loss.item())

import torch
import torch.nn as nn

# Perfect prediction
predictions = torch.tensor([[0.0, 10000.0, 0.0]], dtype=torch.float)
# True labels
targets = torch.tensor([1], dtype=torch.long)

# Define the CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# Compute the loss
loss = criterion(predictions, targets)

print("Loss:", loss.item()) 