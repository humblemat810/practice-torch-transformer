import torch

a = torch.tensor([2.0000000000001], dtype=torch.float32, device='cuda')
b = torch.tensor([100.0], dtype=torch.float32, device='cuda')

# Using torch.pow
import time
for i in range(10000):
    pow_result = torch.pow(a, b)
    log_exp_result = torch.exp(b * torch.log(a))
t0 = time.time()
for i in range(10000):
    log_exp_result = torch.exp(b * torch.log(a))
t1 = time.time()
# Using log and exp
for i in range(10000):
    pow_result = torch.pow(a, b)
    
    
t2 = time.time()
print(f"torch.pow(2.0, 10.0): {pow_result.item()}")
print(f"exp(10.0 * log(2.0)): {log_exp_result.item()}")

print(t1-t0, t2-t1)