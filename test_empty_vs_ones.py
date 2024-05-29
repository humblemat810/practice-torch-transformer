import torch, torch.nn as nn

import time
t0 = time.time()

for _ in range(100):
    
    nn.Parameter(torch.empty(1000, dtype = torch.float32).fill_(1)).to('cuda')


t1 = time.time()
# Using log and exp
for _ in range(100):
    nn.Parameter(torch.ones(1000, dtype = torch.float32)).to('cuda')

    
    
t2 = time.time()
print(t1-t0, t2-t1)
