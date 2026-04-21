# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Logging utilities for training.
"""

import os
from typing import Optional


class WandbLogger:
    """
    Wrapper for Weights & Biases logging.
    """

    def __init__(
        self,
        project: str,
        entity: str,
        name: str,
        config: dict,
        log_dir: str,
        api_key: Optional[str] = None,
    ):
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config
        self.log_dir = log_dir
        
        os.makedirs(log_dir, exist_ok=True)
        
        try:
            import wandb
            if api_key:
                os.environ["WANDB_API_KEY"] = api_key
            os.environ["WANDB_MODE"] = "online"
            
            wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                dir=log_dir,
            )
            self.wandb = wandb
            self.enabled = True
        except Exception as e:
            print(f"[WARN] Wandb not available: {e}")
            self.wandb = None
            self.enabled = False

    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics"""
        if self.enabled:
            try:
                if step is not None:
                    self.wandb.log(metrics, step=step)
                else:
                    self.wandb.log(metrics)
            except Exception as e:
                print(f"[WARN] Logging failed: {e}")

    def watch(self, model):
        """Watch model"""
        if self.enabled:
            try:
                self.wandb.watch(model)
            except Exception as e:
                print(f"[WARN] Model watch failed: {e}")

    def finish(self):
        """Finish logging"""
        if self.enabled:
            try:
                self.wandb.finish()
            except Exception as e:
                print(f"[WARN] Finish failed: {e}")
