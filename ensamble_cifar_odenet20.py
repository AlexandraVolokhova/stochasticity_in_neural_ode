import myexman
import numpy as np
import os
import time
import torch

import utils
import utils_data

from logger import Logger
from models import ODENet

parser = myexman.ExParser(file=os.path.basename(__file__))
parser.add_argument('--name', default='')
# Architecture
parser.add_argument('--time', type=eval, default=True)
parser.add_argument('--n_blocks', type=int, default=1)
parser.add_argument('--norm', type=eval, default=False)
# Data
parser.add_argument('--data', default='cifar10', choices=['cifar10'])
# parser.add_argument('--train_size', default=0.333, type=float)
parser.add_argument('--data_seed', default=30, type=int)
parser.add_argument('--val_size', default=0.2, type=float)
parser.add_argument('--train_bs', default=256, type=int)
parser.add_argument('--test_bs', default=256, type=int)
parser.add_argument('--limit_dataset', default=None, type=int)
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
parser.add_argument('--final', type=eval, default=False)
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

train_loader, val_loader, test_loader = utils_data.get_cifar100_loaders(batch_size=args.train_bs,
                                                                       test_batch_size=args.test_bs,
                                                                       val_size=args.val_size,
                                                                       limit=args.limit_dataset,
                                                                       augmentation=args.augmentation)
# Seed for training process
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

model = ODENet(args).to(device)

total_params = sum(p.numel() for p in model.parameters())
print("total_params: {}".format(total_params))

if args.verbose:
    print(model)
    for param in model.named_parameters():
        print(param)

if args.pretrained != '':
    if args.final:
        path_to_model = os.path.join(args.pretrained, 'model_final.torch')
        model.load_state_dict(torch.load(path_to_model))
    else:
        path_to_model = os.path.join(args.pretrained, 'model_best.torch')
        model.load_state_dict(torch.load(path_to_model))

fmt = {
    'time': '.3f',
    'train.loss': '.4f',
    'val.loss': '.4f',
    'train.acc': '.4f',
    'val.acc': '.4f',
    'train.forward_nfe': '.1f',
    'train.backward_nfe': '.1f',
    'val.forward_nfe': '.1f',
}

logger = Logger('logs', base=args.root, fmt=fmt)

with torch.no_grad():
    model.eval()
    for data_name, data_loader in zip(['val', 'test', 'train'], [val_loader, test_loader, train_loader]):
        ens_acc, mean_acc, std_acc = utils.get_ensamble_metrics(model, data_loader,
                                                                args.n_estimators, device,
                                                                to_ensamble=args.to_ensamble)
        for n in ens_acc.keys():
            logger.add_scalar(n, data_name + "_ens_acc", ens_acc[n])
            logger.iter_info()
            logger.save()

    model.set_stoch_type(None)

    for data_name, data_loader in zip(['val', 'test', 'train'], [val_loader, test_loader, train_loader]):
        acc = utils.get_accuracy(model, data_loader, device)
        logger.add_scalar(0, data_name + "_ens_acc", acc)
        logger.iter_info()
        logger.save()
    # val_loss, val_acc, val_nfe = utils.get_classification_metrics(model, val_loader, device, True)
    # logger.add_scalar(0, "aaaa_ens_acc", val_acc)
    # logger.iter_info()
    # logger.save()

parser.done()
