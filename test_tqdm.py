from tqdm import tqdm
import time

# Number of iterations
num_iterations = 10

# Initialize a progress bar
with tqdm(total=num_iterations) as pbar:
    for i in range(num_iterations):
        # Simulate some work by sleeping for 0.1 seconds
        time.sleep(0.1)
        
        # Update the progress bar
        pbar.update(1)
        
        # Update postfix with additional information
        pbar.set_postfix(loss=0.1234 * (i+1), accuracy=0.85 + 0.01 * i)