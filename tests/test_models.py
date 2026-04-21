#!/usr/bin/env python3
# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for model modules.
"""

import torch
import pytest

from viplanner.models import AutoEncoder, DualAutoEncoder, PlannerNet


def test_plannernet_forward():
    """Test PlannerNet forward pass"""
    model = PlannerNet(layers=[2, 2, 2, 2])
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert output.shape[0] == 2
    assert output.shape[1] == 512


def test_autoencoder_forward():
    """Test AutoEncoder forward pass"""
    model = AutoEncoder(encoder_channel=16, k=5)
    depth = torch.randn(2, 1, 360, 640)
    goal = torch.randn(2, 7)
    
    preds, fear = model(depth, goal)
    assert preds.shape == (2, 5, 3)
    assert fear.shape == (2, 1)


def test_dual_autoencoder_forward():
    """Test DualAutoEncoder forward pass"""
    model = DualAutoEncoder(
        in_channel=16,
        knodes=5,
        decoder_small=False,
    )
    depth = torch.randn(2, 1, 360, 640)
    sem = torch.randn(2, 3, 360, 640)
    goal = torch.randn(2, 7)
    
    preds, fear = model(depth, sem, goal)
    assert preds.shape == (2, 5, 3)
    assert fear.shape == (2, 1)


def test_dual_autoencoder_small_decoder():
    """Test DualAutoEncoder with small decoder"""
    model = DualAutoEncoder(
        in_channel=16,
        knodes=5,
        decoder_small=True,
    )
    depth = torch.randn(2, 1, 360, 640)
    sem = torch.randn(2, 3, 360, 640)
    goal = torch.randn(2, 7)
    
    preds, fear = model(depth, sem, goal)
    assert preds.shape == (2, 5, 3)
    assert fear.shape == (2, 1)


if __name__ == "__main__":
    print("Running model tests...")
    test_plannernet_forward()
    print("✓ PlannerNet forward pass")
    
    test_autoencoder_forward()
    print("✓ AutoEncoder forward pass")
    
    test_dual_autoencoder_forward()
    print("✓ DualAutoEncoder forward pass")
    
    test_dual_autoencoder_small_decoder()
    print("✓ DualAutoEncoder with small decoder")
    
    print("\nAll tests passed!")
