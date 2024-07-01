from tensorclouds.train.utils import register_pytree
from einops import rearrange

import e3nn_jax as e3nn
import jax.numpy as jnp
import numpy as np

from typing import List
from flax import struct


@struct.dataclass
class TensorCloud:

    irreps_array: e3nn.IrrepsArray
    mask_irreps_array: jnp.ndarray
    coord: jnp.ndarray
    mask_coord: jnp.ndarray

    @classmethod
    def empty(cls, irreps: e3nn.Irreps):
        dim = sum(mul * ir.dim for mul, ir in irreps)
        return cls(
            irreps_array=e3nn.IrrepsArray(irreps=irreps, array=np.zeros((0, dim))),
            mask_irreps_array=np.zeros((0, irreps.num_irreps), dtype=bool),
            coord=np.zeros((0, 3)),
            mask_coord=np.zeros(0, dtype=bool),
        )

    @property
    def irreps(self) -> e3nn.Irreps:
        return self.irreps_array.irreps

    def __len__(self) -> int:
        return self.irreps_array.shape[0]

    @property
    def mask(self) -> jnp.ndarray:
        return self.mask_coord

    def replace(self, **kwargs):
        return TensorCloud(
            irreps_array=kwargs.get("irreps_array", self.irreps_array),
            mask_irreps_array=kwargs.get("mask_irreps_array", self.mask_irreps_array),
            coord=kwargs.get("coord", self.coord),
            mask_coord=kwargs.get("mask_coord", self.mask_coord),
        )

    @classmethod
    def concatenate(cls, clouds: List["TensorCloud"]) -> "TensorCloud":
        if len(clouds) == 0:
            return cls.empty(clouds[0].irreps)
        irreps = clouds[0].irreps
        irreps_array = e3nn.IrrepsArray(
            irreps,
            jnp.concatenate([cloud.irreps_array.array for cloud in clouds], axis=0),
        )
        mask_irreps_array = jnp.concatenate(
            [cloud.mask_irreps_array for cloud in clouds], axis=0
        )
        coord = jnp.concatenate([cloud.coord for cloud in clouds], axis=0)
        mask_coord = jnp.concatenate([cloud.mask_coord for cloud in clouds], axis=0)
        return cls(irreps_array, mask_irreps_array, coord, mask_coord)

    @classmethod
    def zeros(cls, irreps: e3nn.Irreps, shape: tuple) -> "TensorCloud":
        dim = sum(mul * ir.dim for mul, ir in irreps)
        return cls(
            irreps_array=e3nn.IrrepsArray(
                irreps=irreps, array=np.zeros(shape + (dim,))
            ),
            mask_irreps_array=np.ones((shape, irreps.num_irreps), dtype=bool),
            coord=np.zeros(shape + (3,)),
            mask_coord=np.ones(shape, dtype=bool),
        )

    @classmethod
    def stack(cls, clouds: List["TensorCloud"]) -> "TensorCloud":
        if len(clouds) == 0:
            return cls.empty(clouds[0].irreps)
        irreps = clouds[0].irreps
        irreps_array = e3nn.IrrepsArray(
            irreps, jnp.stack([cloud.irreps_array.array for cloud in clouds], axis=0)
        )
        mask_irreps_array = jnp.stack(
            [cloud.mask_irreps_array for cloud in clouds], axis=0
        )
        coord = jnp.stack([cloud.coord for cloud in clouds], axis=0)
        mask_coord = jnp.stack([cloud.mask_coord for cloud in clouds], axis=0)
        return cls(irreps_array, mask_irreps_array, coord, mask_coord)

    def split(self, piece_sizes: List[int]) -> List["TensorCloud"]:
        assert sum(piece_sizes) == len(self)
        tensor_clouds = []
        start = 0
        for piece in piece_sizes:
            tensor_clouds.append(
                TensorCloud(
                    irreps_array=self.irreps_array[start : start + piece],
                    mask_irreps_array=self.mask_irreps_array[start : start + piece],
                    coord=self.coord[start : start + piece],
                    mask_coord=self.mask_coord[start : start + piece],
                )
            )
            start += piece
        return tensor_clouds

    def centralize(self):
        coord, mask = self.coord, self.mask_coord
        center = (coord * mask[..., None]).sum(-2) / mask.sum(-1)
        coord = (coord - center[None, ...]) * mask[..., None]
        coord = jnp.where(jnp.isnan(coord), 0.0, coord)
        return self.replace(coord=coord)

    def plot(self, view, viewer=None, colors=None, radius=0.04, mid=0.95):
        if viewer is None:
            viewer = (0, 0)

        if colors is None:
            colors = ["gray"] * len(self)

        vectors = rearrange(
            self.irreps_array.filter("1e").array, "... (d c) -> ... d c", c=3
        )
        for i, (coord, vecs, mask) in enumerate(zip(self.coord, vectors, self.mask)):
            if mask is False:
                continue

            x, y, z = coord
            x, y, z = float(x), float(y), float(z)
            for dx, dy, dz in vecs:
                view.addArrow(
                    {
                        "start": {"x": x, "y": y, "z": z},
                        "end": {
                            "x": x + float(dx),
                            "y": y + float(dy),
                            "z": z + float(dz),
                        },
                        "radius": radius,
                        "mid": mid,
                        "color": colors[i],
                    },
                    viewer=viewer,
                )
        # view.zoomTo()
        return view

    def __repr__(self) -> str:
        return f"TensorCloud(irreps={self.irreps}, shape={self.irreps_array.shape})"

    def __add__(self, other: "TensorCloud") -> "TensorCloud":
        return TensorCloud(
            irreps_array=self.irreps_array + other.irreps_array,
            mask_irreps_array=self.mask_irreps_array & other.mask_irreps_array,
            coord=self.coord + other.coord,
            mask_coord=self.mask_coord & other.mask_coord,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, scalar: float) -> "TensorCloud":
        return self.__rmul__(scalar)

    def __rmul__(self, scalar: float) -> "TensorCloud":
        if (
            getattr(scalar, "shape", None)
            and len(scalar.shape) > 0
            and scalar.shape[0] == 2
        ):
            scalar_irreps, scalar_coord = scalar
        else:
            scalar_coord = scalar
            scalar_irreps = scalar

        return TensorCloud(
            irreps_array=scalar_irreps * self.irreps_array,
            mask_irreps_array=self.mask_irreps_array,
            coord=scalar_coord * self.coord,
            mask_coord=self.mask_coord,
        )

    def __neg__(self):
        return -1.0 * self

    def __div__(self, scalar: float) -> "TensorCloud":
        return TensorCloud(
            irreps_array=self.irreps_array / scalar,
            mask_irreps_array=self.mask_irreps_array,
            coord=self.coord / scalar,
            mask_coord=self.mask_coord,
        )

    def dot(self, other):
        mask_features = self.mask_irreps_array & other.mask_irreps_array

        features_dot = mask_features * e3nn.IrrepsArray(
            self.irreps_array.irreps, self.irreps_array.array * other.irreps_array.array
        )
        features_dot = features_dot.array.sum(axis=-1)

        mask_coord = self.mask_coord & other.mask_coord
        coord_dot = self.coord * other.coord
        coord_dot = (mask_coord[..., None] * coord_dot).sum(axis=-1)

        return features_dot, coord_dot

    def norm(self):
        mask_features = self.mask_irreps_array
        mask_coord = self.mask_coord
        features_norm = (mask_features * self.irreps_array).array ** 2
        features_norm = features_norm.sum(axis=-1)
        coord_norm = self.coord**2
        coord_norm = (mask_coord[..., None] * coord_norm).sum(axis=-1)
        return features_norm, coord_norm

    def __getitem__(self, index: int) -> "TensorCloud":
        return TensorCloud(
            irreps_array=self.irreps_array[index],
            mask_irreps_array=self.mask_irreps_array[index],
            coord=self.coord[index],
            mask_coord=self.mask_coord[index],
        )


# register_pytree(TensoCloud)
