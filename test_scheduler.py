import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR

# Create a simple model to attach the optimizer to
model = torch.nn.Linear(2, 2)

# Setup parameters
initial_lr = 0.1
total_epochs = 100
iters_per_epoch = 100
total_steps = total_epochs * iters_per_epoch

# Function to collect learning rates over epochs
def get_learning_rates(scheduler, optimizer, total_steps):
    lrs = []
    for step in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    return lrs

# Test different schedulers
def plot_schedulers():
    plt.figure(figsize=(15, 10))
    
    # 1. CosineAnnealingLR
    optimizer = SGD(model.parameters(), lr=initial_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)
    lrs = get_learning_rates(scheduler, optimizer, total_steps)
    plt.plot(lrs, label='CosineAnnealingLR')
    
    # 2. CosineAnnealingWarmRestarts
    optimizer = SGD(model.parameters(), lr=initial_lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=1e-4)
    lrs = get_learning_rates(scheduler, optimizer, total_steps)
    plt.plot(lrs, label='CosineAnnealingWarmRestarts')
    
    # 3. OneCycleLR
    optimizer = SGD(model.parameters(), lr=initial_lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=initial_lr,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1e4
    )
    lrs = get_learning_rates(scheduler, optimizer, total_steps)
    plt.plot(lrs, label='OneCycleLR')
    
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedulers Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_schedulers()