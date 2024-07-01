from typing import Tuple

import jax
import jax.numpy as jnp
from matscipy.neighbours import neighbour_list
import numpy as np

from ..tensorcloud import TensorCloud


def create_radius_graph(
    positions: jnp.ndarray, cutoff: float, edge_buffer_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return jax.pure_callback(
        _create_radius_graph,
        (
            jnp.empty(edge_buffer_size, dtype=jnp.int32),  # senders
            jnp.empty(edge_buffer_size, dtype=jnp.int32),  # receivers
            jnp.empty(edge_buffer_size, dtype=jnp.int32),  # edge_mask
        ),
        jax.lax.stop_gradient(positions),
        cutoff,
        edge_buffer_size,
    )


def _create_radius_graph(
    positions: np.ndarray, cutoff: float, edge_buffer_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    senders, receivers = neighbour_list(
        "ij", positions=positions, cutoff=float(cutoff), cell=np.eye(3)
    )
    num_edges = senders.shape[0]

    assert num_edges < edge_buffer_size

    edge_mask = np.ones(edge_buffer_size, dtype=np.int32)
    edge_mask[num_edges:] = 0

    senders = np.pad(senders, (0, edge_buffer_size - num_edges), constant_values=0)
    receivers = np.pad(receivers, (0, edge_buffer_size - num_edges), constant_values=0)
    return senders, receivers, edge_mask
