# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cost-map loader (no open3d / pypose dependency).

Reads the pre-built cost map produced by the deprecated cost_builder:
  maps/data/<name>_map.txt     – 2-D cost array  (x_cells × y_cells)
  maps/data/<name>_ground.txt  – 2-D ground-height array
  maps/params/config_<name>.yaml
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Minimal config dataclasses (mirrors deprecated viplanner.config.costmap_cfg)
# ---------------------------------------------------------------------------

@dataclass
class _General:
    resolution: float = 0.1
    root_path: str = ""
    ply_file: str = "cloud.ply"
    clear_dist: float = 1.0
    sigma_smooth: float = 3.0
    x_min: Optional[float] = None
    y_min: Optional[float] = None
    x_max: Optional[float] = None
    y_max: Optional[float] = None


@dataclass
class _SemCostMap:
    negative_reward: float = 0.5
    obstacle_threshold: float = 0.5


@dataclass
class _CostMapCfg:
    general: _General = None
    sem_cost_map: _SemCostMap = None
    semantics: bool = True
    geometry: bool = False
    map_name: str = "cost_map_sem"
    x_start: float = 0.0
    y_start: float = 0.0

    def __post_init__(self):
        if self.general is None:
            self.general = _General()
        if self.sem_cost_map is None:
            self.sem_cost_map = _SemCostMap()


def _load_cfg(yaml_path: str) -> _CostMapCfg:
    """Parse the YAML config written by the deprecated cost_builder.
    
    The file uses !!python/object tags; we register constructors that
    map them to plain dicts so no deprecated viplanner package is needed.
    """
    class _Loader(yaml.SafeLoader):
        pass

    def _make_dict_constructor(tag):
        def _ctor(loader, node):
            return loader.construct_mapping(node, deep=True)
        _Loader.add_constructor(tag, _ctor)

    for tag in [
        "tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.GeneralCostMapConfig",
        "tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.SemCostMapConfig",
        "tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.TsdfCostMapConfig",
        "tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.CostMapConfig",
        "tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.ReconstructionCfg",
    ]:
        _make_dict_constructor(tag)

    with open(yaml_path) as f:
        raw = yaml.load(f, Loader=_Loader)  # noqa: S506

    # The YAML uses python/object tags; yaml.safe_load strips them to plain dicts
    gen_raw = raw.get("general", {})
    sem_raw = raw.get("sem_cost_map", {})

    general = _General(
        resolution=float(gen_raw.get("resolution", 0.1)),
        root_path=str(gen_raw.get("root_path", "")),
        x_min=gen_raw.get("x_min"),
        y_min=gen_raw.get("y_min"),
        x_max=gen_raw.get("x_max"),
        y_max=gen_raw.get("y_max"),
    )
    sem = _SemCostMap(
        negative_reward=float(sem_raw.get("negative_reward", 0.5)),
        obstacle_threshold=float(sem_raw.get("obstacle_threshold", 0.5)),
    )
    cfg = _CostMapCfg(
        general=general,
        sem_cost_map=sem,
        semantics=bool(raw.get("semantics", True)),
        geometry=bool(raw.get("geometry", False)),
        map_name=str(raw.get("map_name", "cost_map_sem")),
        x_start=float(raw.get("x_start", 0.0)),
        y_start=float(raw.get("y_start", 0.0)),
    )
    return cfg


# ---------------------------------------------------------------------------
# CostMap
# ---------------------------------------------------------------------------

class CostMap:
    """
    Lightweight cost-map wrapper.

    Attributes
    ----------
    cost_array   : torch.Tensor  [Nx, Ny]  – obstacle / traversability cost
    ground_array : torch.Tensor  [Nx, Ny]  – ground height
    cfg          : _CostMapCfg
    device       : torch.device
    """

    def __init__(
        self,
        cfg: _CostMapCfg,
        cost_array: np.ndarray,
        ground_array: np.ndarray,
        gpu_id: Optional[int] = 0,
    ):
        self.cfg = cfg
        if torch.cuda.is_available() and gpu_id is not None:
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

        self.cost_array   = torch.tensor(cost_array,   dtype=torch.float32, device=self.device)
        self.ground_array = torch.tensor(ground_array, dtype=torch.float32, device=self.device)
        self.num_x, self.num_y = self.cost_array.shape

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, root_path: str, map_name: str, gpu_id: Optional[int] = 0) -> "CostMap":
        root_path = os.path.expanduser(root_path)
        cfg = _load_cfg(os.path.join(root_path, "maps", "params", f"config_{map_name}.yaml"))
        cost   = np.loadtxt(os.path.join(root_path, "maps", "data", f"{map_name}_map.txt"))
        ground = np.loadtxt(os.path.join(root_path, "maps", "data", f"{map_name}_ground.txt"))
        return cls(cfg, cost, ground, gpu_id)

    # ------------------------------------------------------------------
    def pos2ind_norm(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world-frame xy positions to normalised grid indices in [-1, 1]
        (compatible with F.grid_sample).

        Args:
            points: [..., 3]  world-frame xyz
        Returns:
            norm_inds: [..., 2]  normalised (x, y) in [-1, 1]
        """
        start = torch.tensor(
            [self.cfg.x_start, self.cfg.y_start],
            dtype=torch.float32, device=points.device,
        )
        H = (points[..., :2] - start) / self.cfg.general.resolution   # [..., 2]
        norm = torch.tensor(
            [self.num_x / 2.0, self.num_y / 2.0],
            dtype=torch.float32, device=points.device,
        )
        return (H - norm) / norm   # [..., 2]  in [-1, 1]
