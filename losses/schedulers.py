import jax
import jax.numpy as jnp


class Scheduler:
    def __call__(self, step: int) -> float:
        raise NotImplementedError


class LinearScheduler(Scheduler):
    def __init__(self, init_value, end_value, transition_steps, transition_begin):
        self.init_value = init_value
        self.end_value = end_value
        self.transition_steps = transition_steps
        self.transition_begin = transition_begin

    def __call__(self, step):
        return jax.optax.schedule.linear(
            init_value=self.init_value,
            end_value=self.end_value,
            transition_steps=self.transition_steps,
            transition_begin=self.transition_begin,
        )(step)


class CyclicAnnealingScheduler(Scheduler):
    def __init__(
        self,
        init_value,
        end_value,
        transition_steps,
        transition_begin,
    ):
        self.init_value = init_value
        self.end_value = end_value
        self.transition_steps = transition_steps
        self.transition_begin = transition_begin

    def __call__(self, step):
        eff_step_raw = step - self.transition_begin
        eff_step = eff_step_raw % (self.transition_steps)
        value = jnp.where(eff_step >= self.transition_steps / 2, self.end_value, 0.0)
        value = jnp.where(
            eff_step < self.transition_steps / 2,
            (self.end_value - self.init_value) * eff_step / (self.transition_steps / 2)
            + self.init_value,
            value,
        )
        value = jnp.where(eff_step_raw < 0, self.init_value, value)
        return value
