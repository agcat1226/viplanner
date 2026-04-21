#!/usr/bin/env python3
# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for configuration modules.
"""

from viplanner.configs import DataConfig, TrainConfig


def test_data_config_defaults():
    """Test DataConfig default values"""
    cfg = DataConfig()
    assert cfg.max_depth == 15.0
    assert cfg.max_goal_distance == 15.0
    assert cfg.ratio == 0.9
    assert cfg.pairs_per_image == 4


def test_data_config_custom():
    """Test DataConfig with custom values"""
    cfg = DataConfig(
        max_depth=20.0,
        max_goal_distance=12.0,
        ratio=0.8,
    )
    assert cfg.max_depth == 20.0
    assert cfg.max_goal_distance == 12.0
    assert cfg.ratio == 0.8


def test_train_config_defaults():
    """Test TrainConfig default values"""
    cfg = TrainConfig()
    assert cfg.epochs == 100
    assert cfg.batch_size == 64
    assert cfg.lr == 2e-3
    assert cfg.optimizer == "sgd"
    assert cfg.sem == True
    assert cfg.rgb == False


def test_train_config_model_save_name():
    """Test model save name generation"""
    cfg = TrainConfig(
        sem=True,
        env_list=["test_env"],
        cost_map_name="cost_map_sem",
        optimizer="sgd",
        file_name="experiment1",
    )
    name = cfg.get_model_save()
    assert "test_env" in name
    assert "DepSem" in name
    assert "Sem" in name
    assert "SGD" in name
    assert "experiment1" in name


def test_train_config_paths():
    """Test path generation"""
    cfg = TrainConfig(file_path="/tmp/test")
    assert "/tmp/test" in cfg.all_model_dir
    assert "/tmp/test" in cfg.data_dir
    assert "/tmp/test" in cfg.log_dir


if __name__ == "__main__":
    print("Running config tests...")
    
    test_data_config_defaults()
    print("✓ DataConfig defaults")
    
    test_data_config_custom()
    print("✓ DataConfig custom values")
    
    test_train_config_defaults()
    print("✓ TrainConfig defaults")
    
    test_train_config_model_save_name()
    print("✓ TrainConfig model save name")
    
    test_train_config_paths()
    print("✓ TrainConfig paths")
    
    print("\nAll tests passed!")
