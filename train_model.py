import myexman
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn.functional as F

import utils
import utils_data

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import ResNet_cifar100
from models import ResNet_cifar10
from models import ResNet_tinyimagenet
from models import ODENet_cifar100
from models import ODENet_cifar10
from models import ODENet_tinyimagenet
from logger import Logger
from utils import AverageMeter

parser = myexman.ExParser(file=os.path.basename(__file__))
parser.add_argument('--name', default='')
# Architecture
parser.add_argument('--time', type=eval, default=True)
parser.add_argument('--odenet', type=eval, default=False)
parser.add_argument('--n_blocks', type=int, default=2)
parser.add_argument('--norm', type=eval, default=False)
# Data
parser.add_argument('--data', default='cifar100', type=str)
parser.add_argument('--data_seed', default=30, type=int)
parser.add_argument('--val_size', default=0.2, type=float)
parser.add_argument('--train_bs', default=512, type=int)
parser.add_argument('--test_bs', default=512, type=int)
parser.add_argument('--augmentation', default=True, type=eval)
# Integration
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--solver_method', type=str,
                    choices=['dopri5', 'adams', 'rk4', 'explicit_adams',
                             'fixed_adams', 'midpoint', 'euler'], default='rk4')
parser.add_argument('--n_steps', type=int, default=3)
parser.add_argument('--adjoint', type=bool, default=False)
parser.add_argument('--stoch_type', type=str, default=None, choices=[None, 'sde', 'after_step'])
parser.add_argument('--stoch_coeff', type=float, default=0.)
# Optimization
parser.add_argument('--opt', default='sgd')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--nesterov', dest='nesterov', action='store_true')
parser.set_defaults(nesterov=False)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--pretrained', default='')
parser.add_argument('--weight_decay', default=0., type=float)
parser.add_argument('--lr_decay_threshold', default=0.02, type=float)
parser.add_argument('--lr_decay_patience', default=10, type=int)
parser.add_argument('--lr_decay_factor', default=0.1, type=float)
parser.add_argument('--min_lr', default=1e-5, type=float)
parser.add_argument('--warm', default=1, type=int)
# Verbose
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('--log_each', default=1, type=int)
parser.add_argument('--vis_each', default=1, type=int)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load data
np.random.seed(args.data_seed)
torch.manual_seed(args.data_seed)
torch.cuda.manual_seed_all(args.data_seed)
if args.data == 'cifar100':
    train_loader, val_loader, test_loader = utils_data.get_cifar100_loaders(batch_size=args.train_bs,
                                                                           test_batch_size=args.test_bs,
                                                                           val_size=args.val_size,
                                                                           augmentation=args.augmentation)
if args.data == 'cifar10':
    train_loader, val_loader, test_loader = utils_data.get_cifar10_loaders(batch_size=args.train_bs,
                                                                           test_batch_size=args.test_bs,
                                                                           val_size=args.val_size,
                                                                           augmentation=args.augmentation)
if args.data == 'tinyimagenet':
    train_loader, val_loader, test_loader = utils_data.get_tiny_imagenet_loaders(train_batch_size=args.train_bs,
                                                   test_batch_size=args.test_bs,
                                                   augmentation=args.augmentation,
                                                   val_size=args.val_size)


# Seed for training process
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.odenet:
    if args.data == 'cifar100':
        model = ODENet_cifar100(args).to(device)
    if args.data == 'cifar10':
        model = ODENet_cifar10(args).to(device)
    if args.data == 'tinyimagenet':
        model = ODENet_tinyimagenet(args).to(device)
else:
    if args.data == 'cifar100':
        model = ResNet_cifar100(args).to(device)
    if args.data == 'cifar10':
        model = ResNet_cifar10(args).to(device)
    if args.data == 'tinyimagenet':
        model = ResNet_tinyimagenet(args).to(device)

total_params = sum(p.numel() for p in model.parameters())
print("total_params: {}".format(total_params))

criterion = nn.CrossEntropyLoss().to(device)
torch.save(model.state_dict(), os.path.join(args.root, 'model_init.torch'))

if args.verbose:
    print(model)
    for param in model.named_parameters():
        print(param)

# Training setup

parameters = [p for p in model.parameters() if p.requires_grad]

if args.opt == 'adam':
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
elif args.opt == 'adamax':
    optimizer = torch.optim.Adamax(parameters, lr=args.lr, weight_decay=args.weight_decay)
elif args.opt == 'sgd':
    optimizer = torch.optim.SGD(parameters, lr=args.lr, weight_decay=args.weight_decay,
                                momentum=args.momentum, nesterov=args.nesterov)
else:
    raise NotImplementedError

scheduler = ReduceLROnPlateau(optimizer, 'min', threshold_mode='rel',
                              threshold=args.lr_decay_threshold,
                              patience=args.lr_decay_patience,
                              factor=args.lr_decay_factor,
                              min_lr=args.min_lr)

if args.pretrained != '':
    model.load_state_dict(torch.load(os.path.join(args.pretrained, 'model.torch')))
    optimizer.load_state_dict(torch.load(os.path.join(args.pretrained, 'optimizer.torch')))

if args.odenet:
    fmt = {
        'time': '.3f',
        'train.loss': '.4f',
        'val.loss': '.4f',
        'train.acc': '.4f',
        'val.acc': '.4f',
        'train.forward_nfe': '.1f',
        'train.backward_nfe': '.1f',
        'val.forward_nfe': '.1f',
        'lr': '.5f'
    }
else:
    fmt = {
        'time': '.3f',
        'train.loss': '.4f',
        'val.loss': '.4f',
        'train.acc': '.4f',
        'val.acc': '.4f',
        'lr': '.5f'
    }
logger = Logger('logs', base=args.root, fmt=fmt)

# Train
iter_per_epoch = len(train_loader)
if args.warm>0:
    warmup_scheduler = utils.WarmUpLR(optimizer, iter_per_epoch * args.warm)

if args.odenet:
    model.set_stoch_type(None)
t0 = time.time()
best_val_acc = 0.
for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.
    train_acc = 0.
    if args.odenet:
        train_forward_nfe = AverageMeter()
        train_backward_nfe = AverageMeter()
    for x, y in train_loader:
        if epoch <= args.warm:
            warmup_scheduler.step()
        x = x.to(device)
        y = y.to(device)
        if args.odenet:
            model.set_nfe(0)
        logits = model(x)
        if args.odenet:
            train_forward_nfe.update(model.get_nfe())
            model.set_nfe(0)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.odenet:
            train_backward_nfe.update(model.get_nfe())
            model.set_nfe(0)
        train_loss += loss.item() * x.size(0)
        prediction = F.softmax(logits, dim=1).argmax(dim=1)
        train_acc += np.sum(utils.tonp(prediction) == utils.tonp(y))
    train_loss /= len(train_loader.sampler.indices)
    train_acc /= len(train_loader.sampler.indices)

    if epoch > args.warm:
        scheduler.step(train_loss)

    if epoch % args.log_each == 0 or epoch == 1 or epoch == args.epochs:
        with torch.no_grad():
            model.eval()
            val_loss, val_acc, val_nfe = utils.get_classification_metrics(model, val_loader, device, args.odenet)
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.root, 'model_best.torch'))
            torch.save(optimizer.state_dict(), os.path.join(args.root, 'optimizer_best.torch'))
            torch.save({'save_epoch': epoch}, os.path.join(args.root, "save_epoch.torch"))
        for param_group in optimizer.param_groups:
            logger.add_scalar(epoch, 'lr', param_group['lr'])
        logger.add_scalar(epoch, 'train.loss', train_loss)
        logger.add_scalar(epoch, 'val.loss', val_loss)
        logger.add_scalar(epoch, 'train.acc', train_acc)
        logger.add_scalar(epoch, 'val.acc', val_acc)
        if args.odenet:
            logger.add_scalar(epoch, 'train.forward_nfe', train_forward_nfe.avg)
            logger.add_scalar(epoch, 'train.backward_nfe', train_backward_nfe.avg)
            logger.add_scalar(epoch, 'val.forward_nfe', val_nfe)
        logger.add_scalar(epoch, 'time', time.time() - t0)
        t0 = time.time()
        logger.iter_info()
        logger.save()

    if epoch % args.vis_each == 0 or epoch == 1 or epoch == args.epochs:
        utils.plot_metrics(logger.scalar_metrics, args.odenet)
        plt.savefig(os.path.join(args.root, 'training_curves.png'))
        plt.close()

torch.save(model.state_dict(), os.path.join(args.root, 'model_final.torch'))
torch.save(optimizer.state_dict(), os.path.join(args.root, 'optimizer_final.torch'))

if args.verbose:
    print(model)
    for param in model.named_parameters():
        print(param)

parser.done()
