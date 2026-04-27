#!/bin/bash
# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

echo "================================"
echo "Running VIPlanner Tests"
echo "================================"

# Check if pytest is available
if command -v pytest &> /dev/null; then
    echo "Using pytest..."
    pytest tests/ -v
else
    echo "pytest not found, running tests directly..."
    echo ""
    echo "Testing models..."
    python tests/test_models.py
    echo ""
    echo "Testing configs..."
    python tests/test_configs.py
fi

echo ""
echo "================================"
echo "Tests completed!"
echo "================================"
