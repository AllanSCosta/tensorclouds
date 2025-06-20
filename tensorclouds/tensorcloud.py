from typing import List

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import struct
import py3Dmol


@struct.dataclass
class TensorCloud:
    """
    A TensorCloud is a collection of tensors with associated coordinates and masks.
    It is used to represent a cloud of tensors in a 3D space, where each tensor can have
    different irreducible representations (irreps).

    Attributes:
        irreps_array (e3nn.IrrepsArray): The array of tensors with their irreducible representations.
        mask_irreps_array (jax.Array): A boolean mask indicating which irreps are present.
        coord (jax.Array): The coordinates of the tensors in 3D space.
        mask_coord (jax.Array): A boolean mask indicating which coordinates are valid.
        label (jax.Array, optional): An optional label for the TensorCloud, e.g., for classification tasks.

    """

    irreps_array: e3nn.IrrepsArray
    mask_irreps_array: jax.Array
    coord: jax.Array
    mask_coord: jax.Array
    label: jax.Array = None

    @property
    def shape(self):
        return self.irreps_array.shape[:-1]

    @classmethod
    def empty(cls, irreps: e3nn.Irreps):
        irreps = e3nn.Irreps(irreps)
        dim = sum(mul * ir.dim for mul, ir in irreps)
        return cls(
            irreps_array=e3nn.IrrepsArray(irreps=irreps, array=np.zeros((0, dim))),
            mask_irreps_array=np.zeros((0, irreps.num_irreps), dtype=bool),
            coord=np.zeros((0, 3)),
            mask_coord=np.zeros(0, dtype=bool),
        )

    @classmethod
    def ones_like(cls, tc):
        return cls(
            irreps_array=e3nn.ones(tc.irreps_array.irreps),
            mask_irreps_array=np.ones_like(tc.mask_irreps_array),
            coord=np.ones_like(tc.coord),
            mask_coord=np.ones_like(tc.mask_coord),
        )

    @classmethod
    def zeros_like(cls, tc):
        return cls(
            irreps_array=e3nn.zeros(tc.irreps_array.irreps),
            mask_irreps_array=np.zeros_like(tc.mask_irreps_array),
            coord=np.zeros_like(tc.coord),
            mask_coord=np.zeros_like(tc.mask_coord),
        )

    @property
    def irreps(self) -> e3nn.Irreps:
        return self.irreps_array.irreps

    def __len__(self) -> int:
        return self.irreps_array.shape[0]

    @property
    def mask(self) -> jax.Array:
        return self.mask_coord

    def replace(self, **kwargs):
        return TensorCloud(
            irreps_array=kwargs.get("irreps_array", self.irreps_array),
            mask_irreps_array=kwargs.get("mask_irreps_array", self.mask_irreps_array),
            coord=kwargs.get("coord", self.coord),
            mask_coord=kwargs.get("mask_coord", self.mask_coord),
            label=kwargs.get("label", self.label),
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

        label = jnp.concatenate([cloud.label for cloud in clouds], axis=0)

        return cls(irreps_array, mask_irreps_array, coord, mask_coord, label)

    @classmethod
    def zeros(cls, irreps: e3nn.Irreps, shape: tuple) -> "TensorCloud":
        irreps = e3nn.Irreps(irreps)
        dim = sum(mul * ir.dim for mul, ir in irreps)
        return cls(
            irreps_array=e3nn.IrrepsArray(
                irreps=irreps, array=np.zeros(shape + (dim,))
            ),
            mask_irreps_array=np.ones(shape + (irreps.num_irreps,), dtype=bool),
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
                    label=self.label[start : start + piece],
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

    def plot(self, view=None, viewer=None, colors=None, radius=0.04, mid=0.95):
        if view is None:
            view = py3Dmol.view(width=800, height=400)

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
            label=self.label,
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
            label=self.label,
        )

    def __neg__(self):
        return -1.0 * self

    def __div__(self, scalar: float) -> "TensorCloud":
        return TensorCloud(
            irreps_array=self.irreps_array / scalar,
            mask_irreps_array=self.mask_irreps_array,
            coord=self.coord / scalar,
            mask_coord=self.mask_coord,
            label=self.label,
        )

    def __sub__(self, other: "TensorCloud") -> "TensorCloud":
        return TensorCloud(
            irreps_array=self.irreps_array - other.irreps_array,
            mask_irreps_array=self.mask_irreps_array & other.mask_irreps_array,
            coord=self.coord - other.coord,
            mask_coord=self.mask_coord & other.mask_coord,
            label=self.label,
        )

    def __rsub__(self, other: "TensorCloud") -> "TensorCloud":
        return TensorCloud(
            irreps_array=other.irreps_array - self.irreps_array,
            mask_irreps_array=self.mask_irreps_array & other.mask_irreps_array,
            coord=other.coord - self.coord,
            mask_coord=self.mask_coord & other.mask_coord,
            label=self.label,
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
        """
        Compute the norm of the irreps_array and coord of the TensorCloud.
        Returns:
            Tuple[jax.Array, jax.Array]: A tuple containing the norms of the irreps_array and coord.
        """

        mask_features = self.mask_irreps_array
        mask_coord = self.mask_coord
        features_norm = (mask_features * self.irreps_array).array ** 2
        features_norm = features_norm.sum(axis=-1)
        coord_norm = self.coord**2
        coord_norm = (mask_coord[..., None] * coord_norm).sum(axis=-1)
        return features_norm, coord_norm

    def __getitem__(self, index: int) -> "TensorCloud":
        """
        Get a single TensorCloud from the collection by index.
        Args:
            index (int): The index of the TensorCloud to retrieve.
        Returns:
            TensorCloud: A new TensorCloud instance containing the data at the specified index.
        """
        return TensorCloud(
            irreps_array=self.irreps_array[index],
            mask_irreps_array=self.mask_irreps_array[index],
            coord=self.coord[index],
            mask_coord=self.mask_coord[index],
            label=self.label[index] if self.label is not None else None,
        )


from typing import Dict


@struct.dataclass
class TensorClouds:

    irreps: e3nn.Irreps
    _tensorclouds: Dict[str, TensorCloud]

    @classmethod
    def create(cls, **kwargs):
        _tensorclouds = kwargs
        _tensorclouds = {k: v for k, v in _tensorclouds.items() if v is not None}
        _tensorclouds = {k: v for k, v in _tensorclouds.items() if len(v) > 0}

        irreps = None
        for k, v in _tensorclouds.items():
            if not isinstance(v, TensorCloud):
                raise ValueError(f"Expected TensorCloud for {k}, got {type(v)}")
            irreps = v.irreps if irreps is None else irreps
            assert (
                v.irreps == irreps
            ), f"All TensorClouds must have the same irreps, but got {v.irreps} and {irreps}"
        irreps = irreps
        return cls(irreps=irreps, _tensorclouds=_tensorclouds)

    def __len__(self):
        return len(self._tensorclouds)

    def __getitem__(self, key):
        return self._tensorclouds[key]

    def replace(self, **kwargs):
        new_tensorclouds = self._tensorclouds.copy()
        new_tensorclouds.update(kwargs)
        return TensorClouds(**new_tensorclouds)

    def __repr__(self):
        return f"TensorClouds({', '.join(f'{k}: {v.shape}' for k, v in self._tensorclouds.items())})"

    def __rmul__(self, other):
        # if isinstance(other, (int, float)):
        return TensorClouds.create(
            **{k: v * other for k, v in self._tensorclouds.items()}
        )
        # else:
        # raise ValueError(f"Cannot multiply {type(other)} with TensorClouds")

    def __mul__(self, other):
        # if isinstance(other, (int, float)):
        return self.__rmul__(other)
        # else:
        # raise ValueError(f"Cannot multiply {type(other)} with TensorClouds")

    def __add__(self, other):
        if isinstance(other, TensorClouds):
            return TensorClouds.create(
                **{k: v + other[k] for k, v in self._tensorclouds.items()}
            )
        else:
            raise ValueError(f"Cannot add {type(other)} with TensorClouds")

    def __radd__(self, other):
        if isinstance(other, TensorClouds):
            return self.__add__(other)
        else:
            raise ValueError(f"Cannot add {type(other)} with TensorClouds")

    def __sub__(self, other):
        if isinstance(other, TensorClouds):
            return TensorClouds.create(
                **{k: v - other[k] for k, v in self._tensorclouds.items()}
            )
        else:
            raise ValueError(f"Cannot subtract {type(other)} with TensorClouds")

    def __rsub__(self, other):
        if isinstance(other, TensorClouds):
            return TensorClouds.create(
                **{k: other[k] - v for k, v in self._tensorclouds.items()}
            )
        else:
            raise ValueError(f"Cannot subtract {type(other)} with TensorClouds")

    def __div__(self, other):
        if isinstance(other, (int, float)):
            return TensorClouds.create(
                **{k: v / other for k, v in self._tensorclouds.items()}
            )
        else:
            raise ValueError(f"Cannot divide {type(other)} with TensorClouds")

    def __rdiv__(self, other):
        if isinstance(other, (int, float)):
            return TensorClouds.create(
                **{k: other / v for k, v in self._tensorclouds.items()}
            )
        else:
            raise ValueError(f"Cannot divide {type(other)} with TensorClouds")

    @property
    def shapes(self):
        return {k: v.shape for k, v in self._tensorclouds.items()}

    @property
    def mask_coord(self):
        return {k: v.mask_coord for k, v in self._tensorclouds.items()}

    @property
    def mask_irreps_array(self):
        return {k: v.mask_irreps_array for k, v in self._tensorclouds.items()}

    # def centralize(self,):
