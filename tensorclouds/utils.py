from .tensorcloud import TensorCloud
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple
import e3nn_jax as e3nn
from einops import rearrange, repeat




def compute_rotation_for_alignment(x: TensorCloud, y: TensorCloud):
    """Computes the rotation matrix that aligns two point clouds art to Ca only."""

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


def compute_rotation_for_alignment_all_atoms(x: TensorCloud, y: TensorCloud):
    """Computes the rotation matrix that aligns two point clouds wrt all atoms."""

    #get the coordinates of all atoms in the molecule and reshape to n, 24, 3:
    x_coord = x.irreps_array.filter(keep='1e').array
    x_coord = rearrange(x_coord, "r (a c) -> r a c", a=24)
    
    y_coord = y.irreps_array.filter(keep='1e').array
    y_coord = rearrange(y_coord, "r (a c) -> r a c", a=24)
    
    #apply masks.
    x_all = x_coord * x.mask_irreps_array[..., None]
    y_all = y_coord * y.mask_irreps_array[..., None]
    
    #reshape to n*24 , 3
    x = rearrange(x_all, "r a c -> (r a) c") 
    y = rearrange(y_all, "r a c -> (r a) c") 
    

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
    rotated_coord = coord.transform_by_matrix(R).array
    x0 = x0.replace(
        coord=rotated_coord,
        irreps_array=x0.irreps_array.transform_by_matrix(R),
    )
    return x0, x1


def compute_rmsd(x: TensorCloud, y: TensorCloud) -> float:
    """Compute the RMSD between two point clouds after alignment.
    Expects aligned TCs"""
    # # Align the two point clouds.
    # x_aligned, y_aligned = align_with_rotation(x, y)

    # Compute the squared differences.
    print(f' x coord: {x.coord}')
    diff = x.coord - y.coord
    squared_diff = jnp.square(diff)
    mean_squared_diff = jnp.mean(squared_diff)

    # Compute the RMSD.
    rmsd = jnp.sqrt(mean_squared_diff)
    return rmsd