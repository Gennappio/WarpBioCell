"""Synthetic tissue shapes as signed-distance functions (negative inside) evaluated on grid nodes.

All lengths in um. The ellipsoid distance is the standard first-order approximation
(``|f| / |grad f|`` of the implicit function), exact on the axes and within a few percent
elsewhere for moderate aspect ratios; the sphere and the union are exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from warpbiocell.fields.scalar_field import GridGeometry


@dataclass(frozen=True)
class Sphere:
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 200.0

    def __post_init__(self):
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")

    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        c = self.center
        return np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) - self.radius

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        c = np.asarray(self.center, dtype=float)
        return c - self.radius, c + self.radius

    @property
    def volume(self) -> float:
        return 4.0 / 3.0 * np.pi * self.radius**3


@dataclass(frozen=True)
class Ellipsoid:
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radii: tuple[float, float, float] = (300.0, 200.0, 150.0)  # semi-axes

    def __post_init__(self):
        if min(self.radii) <= 0.0:
            raise ValueError("semi-axes must be positive")

    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        c, a = self.center, self.radii
        u, v, w = (x - c[0]) / a[0], (y - c[1]) / a[1], (z - c[2]) / a[2]
        k1 = np.sqrt(u**2 + v**2 + w**2)
        k2 = np.sqrt((u / a[0]) ** 2 + (v / a[1]) ** 2 + (w / a[2]) ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(k2 > 0.0, k1 * (k1 - 1.0) / k2, -min(a))
        return d

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        c = np.asarray(self.center, dtype=float)
        a = np.asarray(self.radii, dtype=float)
        return c - a, c + a

    @property
    def volume(self) -> float:
        return 4.0 / 3.0 * np.pi * float(np.prod(self.radii))


@dataclass(frozen=True)
class Union:
    shapes: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if not self.shapes:
            raise ValueError("a union needs at least one shape")

    def sdf(self, x, y, z) -> np.ndarray:
        return np.minimum.reduce([s.sdf(x, y, z) for s in self.shapes])

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lows, highs = zip(*(s.bounds for s in self.shapes))
        return np.minimum.reduce(lows), np.maximum.reduce(highs)

    @property
    def volume(self) -> float:
        return float("nan")  # overlaps unknown; measure it on the grid instead


def evaluate_on_grid(shape, geometry: GridGeometry) -> np.ndarray:
    x, y, z = np.meshgrid(*geometry.node_coordinates(), indexing="ij")
    return shape.sdf(x, y, z).astype(np.float32)
