import torch
from .fixed_grid import LeapFrog

def hamiltonianint(pot_energy, kin_energy, x0, p0, t, step_size = 1.0):

    solver = LeapFrog((pot_energy, kin_energy), (x0, p0), step_size)
    solution = solver.integrate(t)
    return solution[0]

def hamiltonianint_block(pot_energy, kin_energy, x0, p0, t, n_steps):

    x = HamiltonianInt.apply(pot_energy, kin_energy, x0, p0, t, n_steps)
    return x

class HamiltonianInt(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        potfunc, kinfunc, x, p, time, n_steps = \
            args[-6], args[-5], args[-4], args[-3], args[-2], args[-1]
        #TODO: make adequate time grid with step_size instead of n_steps
        step_size = (time[1] - time[0]) / n_steps
        ctx.potfunc, ctx.kinfunc, ctx.time, ctx.n_steps, ctx.step_size = potfunc, kinfunc, time, n_steps, step_size
        x = x.clone().detach()
        p = p.clone().detach()
        with torch.no_grad():
            for nblock in range(len(potfunc)):
                for t in range(n_steps):
                    x += step_size * kinfunc[nblock](0, p) / 2
                    p += step_size * potfunc[nblock](0, x)
                    x += step_size * kinfunc[nblock](0, p) / 2
        ctx.save_for_backward(x, p)
        return x, p

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, p = ctx.saved_tensors
        x.grad = -1*grad_outputs[0]
        p.grad = -1*grad_outputs[1]
        p.detach_()
        p.requires_grad_(True)
        potfunc, kinfunc, time, n_steps, step_size = ctx.potfunc, ctx.kinfunc, ctx.time, ctx.n_steps, ctx.step_size
        for nblock in range(len(potfunc)):
            for t in range(n_steps):
                x.detach_()
                torch.autograd.set_grad_enabled(True)
                x -= step_size * kinfunc[-nblock - 1](0, p) / 2
                x.backward(-1*x.grad)

                x.detach_()
                p.requires_grad_(False)
                x.requires_grad_(True)
                torch.autograd.set_grad_enabled(True)
                p -= step_size * potfunc[-nblock - 1](0, x)
                p.backward(-1*p.grad)

                p.detach_()
                x.requires_grad_(False)
                p.requires_grad_(True)
                torch.autograd.set_grad_enabled(True)
                x -= step_size * kinfunc[-nblock - 1](0, p) / 2
                x.backward(-1*x.grad)

        for nblock in range(len(potfunc)):
            for param in potfunc[nblock].parameters():
                param.grad *= -1
            for param in kinfunc[nblock].parameters():
                param.grad *= -1

        return None, None, -1*x.grad, -1*p.grad, None, None