# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cubic-spline trajectory optimizer (pure PyTorch, no external deps)."""

import torch

torch.set_default_dtype(torch.float32)


class _CubicSpline:
    """Batch cubic spline interpolation in PyTorch."""

    def h_poly(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B, S]  -> returns [B, 4, S]
        alpha = torch.arange(4, device=t.device, dtype=t.dtype)
        tt = t[:, None, :] ** alpha[None, :, None]          # [B, 4, S]
        A = torch.tensor(
            [[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]],
            dtype=t.dtype, device=t.device,
        )
        return A @ tt                                        # [B, 4, S]

    def interp(
        self,
        x: torch.Tensor,   # [B, N]   knot x-coords
        y: torch.Tensor,   # [B, N, D] knot values
        xs: torch.Tensor,  # [B, S]   query x-coords
    ) -> torch.Tensor:     # [B, S, D]
        m = (y[:, 1:, :] - y[:, :-1, :]) / (x[:, 1:] - x[:, :-1]).unsqueeze(2)
        m = torch.cat([m[:, :1], (m[:, 1:] + m[:, :-1]) / 2, m[:, -1:]], dim=1)

        idxs = torch.searchsorted(x[0, 1:].contiguous(), xs[0].contiguous())
        dx = (x[:, idxs + 1] - x[:, idxs])                 # [B, S]
        hh = self.h_poly((xs - x[:, idxs]) / dx)           # [B, 4, S]
        hh = hh.permute(0, 2, 1)                            # [B, S, 4]

        out = (hh[:, :, 0:1] * y[:, idxs, :]
               + hh[:, :, 1:2] * m[:, idxs] * dx[:, :, None]
               + hh[:, :, 2:3] * y[:, idxs + 1, :]
               + hh[:, :, 3:4] * m[:, idxs + 1] * dx[:, :, None])
        return out


class TrajOpt:
    """Generate dense waypoints from sparse predicted key-points via cubic spline."""

    def __init__(self):
        self._cs = _CubicSpline()

    def TrajGeneratorFromPFreeRot(
        self, preds: torch.Tensor, step: float = 0.1
    ) -> torch.Tensor:
        """
        Args:
            preds: [B, K, 3]  predicted key-points in camera frame
            step:  interpolation step (fraction of knot spacing)
        Returns:
            waypoints: [B, S, 3]
        """
        B, K, D = preds.shape
        # prepend origin (0,0,0)
        origin = torch.zeros(B, 1, D, device=preds.device, dtype=preds.dtype)
        pts = torch.cat([origin, preds], dim=1)   # [B, K+1, D]
        N = K + 1

        xs = torch.arange(0, N - 1 + step, step, device=preds.device, dtype=preds.dtype)
        xs = xs.unsqueeze(0).expand(B, -1)        # [B, S]
        x  = torch.arange(N, device=preds.device, dtype=preds.dtype).unsqueeze(0).expand(B, -1)

        return self._cs.interp(x, pts, xs)        # [B, S, D]
