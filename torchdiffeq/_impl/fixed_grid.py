import torch

from .solvers import FixedGridODESolver
from . import rk_common


class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return tuple(dt * f_ for f_ in func(t, y))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t, y)))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid))

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return rk_common.rk4_alt_step_func(func, t, dt, y)

    @property
    def order(self):
        return 4

class stoch_RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return rk_common.stoch_rk4_step_func(func, t, dt, y, k=self.k)

    @property
    def order(self):
        return 4

class RK4_noise_after_step(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        next_y = rk_common.rk4_alt_step_func(func, t, dt, y)
        return tuple(y_ + self.k * torch.randn_like(y_) for y_ in next_y)

    @property
    def order(self):
        return 4