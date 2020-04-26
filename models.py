import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint

class norm(nn.Module):
    def __init__(self, planes):
        super(norm, self).__init__()
        self.layer = nn.BatchNorm2d(planes)

    def forward(self, x):
        return self.layer(x)


class DownSamplingBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(DownSamplingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm(in_planes)

    def forward(self, x):
        return self.conv1(F.relu(self.bn1(x)))


class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, bias=True):
        super(ConcatConv2d, self).__init__()
        self._layer = nn.Conv2d(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):
    def __init__(self, planes, args, ksize=3, stride=1, padding=1, bias=True):
        super(ODEfunc, self).__init__()
        self.conv1 = ConcatConv2d(planes, planes,
                                  ksize=ksize, stride=stride, padding=padding, bias=bias)
        self.conv2 = ConcatConv2d(planes, planes,
                                  ksize=ksize, stride=stride, padding=padding, bias=bias)
        self.nfe = 0
        self.norm = args.norm
        if self.norm:
            self.bn1 = norm(planes)
            self.bn2 = norm(planes)


    def forward(self, t, x):
        self.nfe += 1
        if self.norm:
            out = self.bn1(x)
            out = F.relu(out, inplace=False)
        else:
            out = F.relu(x, inplace=False)
        out = self.conv1(t, out)
        if self.norm:
            out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(t, out)
        return out


class ResBlock(nn.Module):
    def __init__(self,  planes, args, ksize=3, stride=1, padding=1, bias=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes,
                               kernel_size=ksize, stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=ksize, stride=stride, padding=padding, bias=bias)
        self.norm = args.norm
        if self.norm:
            self.bn1 = norm(planes)
            self.bn2 = norm(planes)


    def forward(self, x):
        shortcut = x
        if self.norm:
            out = self.bn1(x)
            out = F.relu(out, inplace=False)
        else:
            out = F.relu(x, inplace=False)
        out = self.conv1(out)
        if self.norm:
            out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        return out + shortcut


class ODEBlock(nn.Module):
    def __init__(self, odefunc, time_interval, args):
        super(ODEBlock, self).__init__()
        solver_options = None
        self.odefunc = odefunc
        self.solver_method = args.solver_method
        self.n_steps = args.n_steps
        if solver_options is None:
            solver_options = dict()
        self.solver_options = solver_options
        self.time_interval = torch.tensor(time_interval).float()
        self.tol = args.tol
        self.stoch_type = args.stoch_type
        self.stoch_coeff = args.stoch_coeff
        if self.n_steps:
            if self.solver_method not in ['rk4', 'explicit_adams', 'fixed_adams', 'midpoint', 'euler']:
                raise Exception("argument n_steps is useless if you use {}".format(self.solver_method))
            step_size = (time_interval[1] - time_interval[0]) / self.n_steps
            self.solver_options.update({'step_size': step_size})
        if self.stoch_type == 'sde' and self.n_steps is None:
            raise Exception('Stoch type is sde and n_steps is None, provide n_steps or change stoch_type')

        if args.adjoint:
            self.odeint = odeint_adjoint
        else:
            self.odeint = odeint

    def forward(self, x):
        self.time_interval = self.time_interval.type_as(x)
        if self.stoch_type is None:
            out = self.odeint(self.odefunc, x, self.time_interval, rtol=self.tol, atol=self.tol,
                         method=self.solver_method, options=self.solver_options)
        elif self.stoch_type == 'sde':
            self.solver_options.update({'k': self.stoch_coeff})
            out = self.odeint(self.odefunc, x, self.time_interval, rtol=self.tol, atol=self.tol,
                         method='stoch_rk4', options=self.solver_options)
        elif self.stoch_type == 'after_step':
            self.solver_options.update({'k': self.stoch_coeff})
            out = self.odeint(self.odefunc, x, self.time_interval, rtol=self.tol, atol=self.tol,
                              method='rk4_noise_after_step', options=self.solver_options)
        else:
            raise NotImplementedError
        return out[1]

    def set_n_steps(self, n_steps):
        self.n_steps = n_steps
        time_interval = self.time_interval.numpy()
        step_size = (time_interval[1] - time_interval[0]) / n_steps
        if not self.solver_options:
            self.solver_options = dict()
        self.solver_options.update({'step_size': step_size})

    def set_solver_method(self, method):
        self.solver_method = method

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class ODENet_cifar100(nn.Module):
    def __init__(self, args, num_classes=100):
        super(ODENet_cifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm(64)
        self.ode1 = self._make_ode_blocks(args, 64, n_blocks=2)
        self.down1 = DownSamplingBlock(64, 128)
        self.ode2 = self._make_ode_blocks(args, 128, n_blocks=2)
        self.down2 = DownSamplingBlock(128, 256)
        self.ode3 = self._make_ode_blocks(args, 256, n_blocks=2)
        self.down3 = DownSamplingBlock(256, 512)
        self.ode4 = self._make_ode_blocks(args, 512, n_blocks=2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # out = self.conv1(x)
        out = self.down1(self.ode1(out))
        out = self.down2(self.ode2(out))
        out = self.down3(self.ode3(out))
        out = self.avg(self.ode4(out))
        out = self.fc(self.flatten(out))
        return out

    def _make_ode_blocks(self, args, planes, n_blocks=2):
        blocks = []
        for time_interval in range(n_blocks):
            blocks.append(ODEBlock(
                ODEfunc(planes=planes, args=args), [time_interval, time_interval+1], args))
        return nn.Sequential(*blocks)

    def get_nfe(self):
        nfe = []
        for seq in [self.ode1, self.ode2, self.ode3]:
            for block in seq:
                nfe.append(block.nfe)
        return np.mean(nfe)

    def set_nfe(self, value):
        for seq in [self.ode1, self.ode2, self.ode3]:
            for block in seq:
                block.nfe = value

    def set_stoch_type(self, stoch_type):
        for seq in [self.ode1, self.ode2, self.ode3, self.ode4]:
            for block in seq:
                block.stoch_type = stoch_type


class ResNet_cifar100(nn.Module):
    def __init__(self, args, num_classes=100):
        super(ResNet_cifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm(64)
        self.res1 = self._make_res_blocks(args, 64, n_blocks=2)
        self.down1 = DownSamplingBlock(64, 128)
        self.res2 = self._make_res_blocks(args, 128, n_blocks=2)
        self.down2 = DownSamplingBlock(128, 256)
        self.res3 = self._make_res_blocks(args, 256, n_blocks=2)
        self.down3 = DownSamplingBlock(256, 512)
        self.res4 = self._make_res_blocks(args, 512, n_blocks=2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.down1(self.res1(out))
        out = self.down2(self.res2(out))
        out = self.down3(self.res3(out))
        out = self.avg(self.res4(out))
        out = self.fc(self.flatten(out))
        return out

    def _make_res_blocks(self, args, planes, n_blocks=2):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResBlock(planes=planes, args=args))
        return nn.Sequential(*blocks)


class ResNet_cifar10(nn.Module):
    def __init__(self, args, num_classes=10):
        super(ResNet_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm(16)
        self.res1 = self._make_res_blocks(args, 16, n_blocks=2)
        self.down1 = DownSamplingBlock(16, 32)
        self.res2 = self._make_res_blocks(args, 32, n_blocks=2)
        self.down2 = DownSamplingBlock(32, 64)
        self.res3 = self._make_res_blocks(args, 64, n_blocks=2)
        self.avg = nn.AvgPool2d(8)
        self.flatten = Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.down1(self.res1(out))
        out = self.down2(self.res2(out))
        out = self.avg(self.res3(out))
        out = self.fc(self.flatten(out))
        return out

    def _make_res_blocks(self, args, planes, n_blocks=2):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResBlock(planes=planes, args=args))
        return nn.Sequential(*blocks)

class ODENet_cifar10(nn.Module):
    def __init__(self, args, num_classes=10):
        super(ODENet_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.ode1 = self._make_ode_blocks(args, 16, n_blocks=2)
        self.down1 = DownSamplingBlock(16, 32)
        self.ode2 = self._make_ode_blocks(args, 32, n_blocks=2)
        self.down2 = DownSamplingBlock(32, 64)
        self.ode3 = self._make_ode_blocks(args, 64, n_blocks=2)
        self.avg = nn.AvgPool2d(8)
        self.flatten = Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.down1(self.ode1(out))
        out = self.down2(self.ode2(out))
        out = self.avg(self.ode3(out))
        out = self.fc(self.flatten(out))
        return out

    def _make_ode_blocks(self, args, planes, n_blocks=2):
        blocks = []
        for time_interval in range(n_blocks):
            blocks.append(ODEBlock(
                ODEfunc(planes=planes, args=args), [time_interval, time_interval+1], args))
        return nn.Sequential(*blocks)

    def get_nfe(self):
        nfe = []
        for seq in [self.ode1, self.ode2, self.ode3]:
            for block in seq:
                nfe.append(block.nfe)
        return np.mean(nfe)

    def set_nfe(self, value):
        for seq in [self.ode1, self.ode2, self.ode3]:
            for block in seq:
                block.nfe = value

    def set_stoch_type(self, stoch_type):
        for seq in [self.ode1, self.ode2, self.ode3]:
            for block in seq:
                block.stoch_type = stoch_type

class ResNet_tinyimagenet(nn.Module):
    def __init__(self, args):
        super(ResNet_tinyimagenet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res1 = self._make_res_blocks(args, 64, n_blocks=3)
        self.down1 = DownSamplingBlock(64, 128)
        self.res2 = self._make_res_blocks(args, 128, n_blocks=3)
        self.down2 = DownSamplingBlock(128, 256)
        self.res3 = self._make_res_blocks(args, 256, n_blocks=5)
        self.down3 = DownSamplingBlock(256, 512)
        self.res4 = self._make_res_blocks(args, 512, n_blocks=2)
        self.avg = nn.AvgPool2d(2)
        self.flatten = Flatten()
        self.fc = nn.Linear(512, 200)

    def forward(self, x):
        out = self.pre(x)
        out = self.down1(self.res1(out))
        out = self.down2(self.res2(out))
        out = self.down3(self.res3(out))
        out = self.avg(self.res4(out))
        out = self.fc(self.flatten(out))
        return out

    def _make_res_blocks(self, args, planes, n_blocks=2):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResBlock(planes=planes, args=args))
        return nn.Sequential(*blocks)

class ODENet_tinyimagenet(nn.Module):
    def __init__(self, args):
        super(ODENet_tinyimagenet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ode1 = self._make_ode_blocks(args, 64, n_blocks=3)
        self.down1 = DownSamplingBlock(64, 128)
        self.ode2 = self._make_ode_blocks(args, 128, n_blocks=3)
        self.down2 = DownSamplingBlock(128, 256)
        self.ode3 = self._make_ode_blocks(args, 256, n_blocks=5)
        self.down3 = DownSamplingBlock(256, 512)
        self.ode4 = self._make_ode_blocks(args, 512, n_blocks=2)
        self.avg = nn.AvgPool2d(2)
        self.flatten = Flatten()
        self.fc = nn.Linear(512, 200)

    def forward(self, x):
        out = self.pre(x)
        out = self.down1(self.ode1(out))
        out = self.down2(self.ode2(out))
        out = self.down3(self.ode3(out))
        out = self.avg(self.ode4(out))
        out = self.fc(self.flatten(out))
        return out

    def _make_ode_blocks(self, args, planes, n_blocks=2):
        blocks = []
        for time_interval in range(n_blocks):
            blocks.append(ODEBlock(
                ODEfunc(planes=planes, args=args), [time_interval, time_interval+1], args))
        return nn.Sequential(*blocks)

    def get_nfe(self):
        nfe = []
        for seq in [self.ode1, self.ode2, self.ode3, self.ode4]:
            for block in seq:
                nfe.append(block.nfe)
        return np.mean(nfe)

    def set_nfe(self, value):
        for seq in [self.ode1, self.ode2, self.ode3, self.ode4]:
            for block in seq:
                block.nfe = value

    def set_stoch_type(self, stoch_type):
        for seq in [self.ode1, self.ode2, self.ode3, self.ode4]:
            for block in seq:
                block.stoch_type = stoch_type
