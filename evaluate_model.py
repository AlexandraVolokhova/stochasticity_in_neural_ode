import myexman
import numpy as np
import os
import time
import torch

import utils
import utils_data

from logger import Logger
from models import ResNet_cifar100
from models import ResNet_cifar10
from models import ODENet_cifar100
from models import ODENet_cifar10

parser = myexman.ExParser(file=os.path.basename(__file__))
parser.add_argument('--name', default='')
# Architecture
parser.add_argument('--time', type=eval, default=True)
parser.add_argument('--odenet', type=eval, default=False)
parser.add_argument('--n_blocks', type=int, default=1)
parser.add_argument('--norm', type=eval, default=False)
# Data
parser.add_argument('--data', default='cifar100', type=str)
parser.add_argument('--data_seed', default=30, type=int)
parser.add_argument('--train_bs', default=256, type=int)
parser.add_argument('--test_bs', default=256, type=int)
parser.add_argument('--augmentation', default=True, type=eval)
# Integration
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--solver_method', type=str,
                    choices=['dopri5', 'adams', 'rk4', 'explicit_adams', 'fixed_adams'], default='rk4')
parser.add_argument('--n_steps', type=int, default=3)
parser.add_argument('--adjoint', type=bool, default=False)
parser.add_argument('--stoch_type', type=str, default=None, choices=[None, 'sde', 'after_step'])
parser.add_argument('--stoch_coeff', type=float, default=0.)
parser.add_argument('--seed', default=0, type=int)
# Model
parser.add_argument('--pretrained', default='')
parser.add_argument('--to_ensamble', default='probs', type=str, choices=['probs', 'logits'])
parser.add_argument('--n_estimators', default=1, type=int)
# Verbose
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)

args = parser.parse_args()

if args.pretrained == '':
    raise Exception("provide path for pretrained model using argument --pretrained")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
np.random.seed(args.data_seed)
torch.manual_seed(args.data_seed)
torch.cuda.manual_seed_all(args.data_seed)
if args.data == 'cifar100':
    train_loader, _, test_loader = utils_data.get_cifar100_loaders(batch_size=args.train_bs,
                                                                           test_batch_size=args.test_bs,
                                                                           val_size=0.,
                                                                           augmentation=args.augmentation)
if args.data == 'cifar10':
    train_loader, _, test_loader = utils_data.get_cifar10_loaders(batch_size=args.train_bs,
                                                                           test_batch_size=args.test_bs,
                                                                           val_size=0.,
                                                                           augmentation=args.augmentation)
# Seed for training process
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.odenet:
    if args.data == 'cifar100':
        model = ODENet_cifar100(args).to(device)
    if args.data == 'cifar10':
        model = ODENet_cifar10(args).to(device)
else:
    if args.data == 'cifar100':
        model = ResNet_cifar100(args).to(device)
    if args.data == 'cifar10':
        model = ResNet_cifar10(args).to(device)

total_params = sum(p.numel() for p in model.parameters())
print("total_params: {}".format(total_params))
parser.dumpd['n_params'] = total_params

if args.verbose:
    print(model)
    for param in model.named_parameters():
        print(param)

if args.pretrained != '':
    path_to_model = os.path.join(args.pretrained, 'model_final.torch')
    if os.path.exists(path_to_model):
        model.load_state_dict(torch.load(path_to_model))
    else:
        path_to_model = os.path.join(args.pretrained, 'model_best.torch')
        model.load_state_dict(torch.load(path_to_model))

fmt = {
    'time': '.3f',
    'train.loss': '.4f',
    'test.loss': '.4f',
    'train.acc': '.4f',
    'test.acc': '.4f',
}

logger = Logger('logs', base=args.root, fmt=fmt)

with torch.no_grad():
    model.eval()
    if args.odenet and args.stoch_type is not None:
        for data_name, data_loader in zip(['test', 'train'], [test_loader, train_loader]):
            ens_acc, _, _ = utils.get_ensamble_metrics(model, data_loader,
                                                                args.n_estimators, device,
                                                                to_ensamble=args.to_ensamble)
            for n in ens_acc.keys():
                logger.add_scalar(n, data_name + "_ens_acc", ens_acc[n])
                logger.iter_info()
                logger.save()

        model.set_stoch_type(None)

        for data_name, data_loader in zip(['test', 'train'], [test_loader, train_loader]):
            acc = utils.get_accuracy(model, data_loader, device)
            logger.add_scalar(0, data_name + "_ens_acc", acc)
            logger.iter_info()
            logger.save()

    else:
        for data_name, data_loader in zip(['test', 'train'], [test_loader, train_loader]):
            _, acc, _ = utils.get_classification_metrics(model, data_loader, device, args.odenet)
            logger.add_scalar(1, data_name + "_ens_acc", acc)
            logger.iter_info()
            logger.save()

parser.done()
