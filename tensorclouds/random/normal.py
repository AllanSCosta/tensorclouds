

class NormalDistribution():
    """A normal distribution for TensorClouds."""

    def __init__(
        self,
        irreps_in: e3nn.Irreps,
        irreps_scale: float = 1.0,
        coords_scale: float = 1.0,
        coords_mean: Optional[chex.Array] = None,
        irreps_mean: Optional[e3nn.IrrepsArray] = None,
    ):
        irreps_in = e3nn.Irreps(irreps_in)
        self.irreps_in = irreps_in
        if irreps_mean is None:
            irreps_mean = e3nn.zeros(irreps_in)
        self.irreps_mean = irreps_mean
        if coords_mean is None:
            coords_mean = jnp.zeros(3)
        self.coords_mean = coords_mean
        self.irreps_scale = irreps_scale
        self.coords_scale = coords_scale

    def sample(
        self, key: chex.PRNGKey, leading_shape: Tuple[int, ...] = (), mask: jnp.ndarray = None
    ) -> TensorCloud:  # TODO(Ilan): add option for custom mask..
        """Sample from the distribution."""
        irreps_key, coords_key = jax.random.split(key)
        irreps = (
            e3nn.normal(self.irreps_in, leading_shape=leading_shape, key=irreps_key)
            * self.irreps_scale
        )
        coords = jax.random.normal(coords_key, (*leading_shape, 3)) * self.coords_scale
        if mask is not None:
            irreps = irreps * mask[..., None]
            coords = coords * mask[..., None]
            
        return TensorCloud(
            irreps_array=irreps,
            mask_irreps_array=jnp.ones(leading_shape, dtype=bool),
            coord=coords,
            mask_coord=jnp.ones(leading_shape, dtype=bool),
        )

    def log_likelihood(self, x: TensorCloud) -> chex.Array:
        """Compute the log probability of the input."""
        irreps_log_likelihood = jax.scipy.stats.norm.logpdf(
            x.irreps_array, loc=self.irreps_mean.array, scale=self.irreps_scale
        ).sum(-1)
        coords_log_likelihood = jax.scipy.stats.norm.logpdf(
            x.coord, loc=self.coords_mean, scale=self.coords_scale
        ).sum(-1)
        return irreps_log_likelihood + coords_log_likelihood
