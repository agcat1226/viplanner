# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Optimizer utilities including early stopping scheduler.
"""

import torch.optim as optim


class EarlyStopScheduler:
    """
    Learning rate scheduler with early stopping.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        factor: float = 0.5,
        patience: int = 3,
        min_lr: float = 1e-5,
        verbose: bool = True,
    ):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best_loss = float("inf")
        self.num_bad_epochs = 0
        self.stopped = False

    def step(self, val_loss: float) -> bool:
        """
        Update learning rate based on validation loss.
        
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            # Reduce learning rate
            for param_group in self.optimizer.param_groups:
                old_lr = param_group["lr"]
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group["lr"] = new_lr
                
                if self.verbose:
                    print(f"[INFO] Reducing learning rate: {old_lr:.6f} -> {new_lr:.6f}")
                
                # Stop if minimum learning rate reached
                if new_lr <= self.min_lr:
                    self.stopped = True
                    return True
            
            self.num_bad_epochs = 0

        return False


def count_parameters(model) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
