from .tensorcloud import TensorCloud
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple
import e3nn_jax as e3nn

def compute_rotation_for_alignment(x: TensorCloud, y: TensorCloud):
    """Computes the rotation matrix that aligns two point clouds."""

    # We are only interested in the coords.
    x = x.coord * x.mask_coord[:, None]
    y = y.coord * y.mask_coord[:, None]

    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    assert x.shape == y.shape

    # Compute the covariance matrix.
    covariance = jnp.matmul(y.T, x)
    ndim = x.shape[-1]
    assert covariance.shape == (ndim, ndim)

    # Compute the SVD of the covariance matrix.
    u, _, v = jnp.linalg.svd(covariance, full_matrices=True)

    # Compute the rotation matrix that aligns the two point clouds.
    rotation = u @ v
    rotation = rotation.at[:, 0].set(
        rotation[:, 0] * jnp.sign(jnp.linalg.det(rotation))
    )
    assert rotation.shape == (ndim, ndim)

    return rotation


def align_with_rotation(
    x0: TensorCloud,
    x1: TensorCloud,
) -> Tuple[TensorCloud, TensorCloud]:
    """Aligns x0 to x1 via a rotation."""
    R = compute_rotation_for_alignment(x0, x1)
    coord = e3nn.IrrepsArray("1o", x0.coord)
    print(f'coord in align {coord}')
    rotated_coord = coord.transform_by_matrix(R).array
    x0 = x0.replace(
        coord=rotated_coord,
        irreps_array=x0.irreps_array.transform_by_matrix(R),
    )
    return x0, x1



def compute_rmsd(x0: TensorCloud, x1: TensorCloud) -> float:
    """Computes the RMSD between two aligned TensorCloud objects."""
    x0_aligned, x1_aligned = align_with_rotation(x0, x1)
    print(f'aligned: {x0_aligned.coord, x1_aligned.coord}')
    #Compute the RMSD
    diff = x0_aligned.coord - x1_aligned.coord
    print(f'diff before mask: {diff.shape, diff}')
    diff = diff * x0_aligned.mask_coord[:, None] 
    print(f'diff after mask: {diff.shape, diff}')
    rmsd = jnp.sqrt((jnp.sum(diff**2))/ jnp.sum(x0_aligned.mask_coord))
    
    return rmsd

