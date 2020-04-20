import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

def tonp(x):
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def zero_nfe(model):
    if hasattr(model, 'nfe'):
        model.nfe = 0
    elif hasattr(model, 'modules'):
        for m in model.modules():
            if hasattr(m, 'nfe'):
                m.nfe = 0
    else:
        raise Exception("nfe attr not found")


def get_nfe(model):
    if hasattr(model, 'nfe'):
        return model.nfe
    elif hasattr(model, 'modules'):
        nfe = 0
        for m in model.modules():
            if hasattr(m, 'nfe'):
                nfe += m.nfe
        return nfe
    else:
        raise Exception("nfe attr not found")


def get_n_samples(data_loader):
    if hasattr(data_loader.sampler, 'indices'):
        return len(data_loader.sampler.indices)
    else:
        return len(data_loader.dataset)


def get_classification_metrics(model, data_loader, device, ode_flag):
    loss = 0.
    accuracy = 0.
    nfe = AverageMeter()
    n_samples = get_n_samples(data_loader)
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        if ode_flag:
            model.set_nfe(0)
        logits = model(x)
        loss += F.cross_entropy(logits, y).item() * x.size(0)
        prediction = F.softmax(logits, dim=1).argmax(dim=1)
        accuracy += np.sum(tonp(prediction) == tonp(y))
        if ode_flag:
            nfe.update(model.get_nfe())
    accuracy /= n_samples
    loss /= n_samples
    return loss, accuracy, nfe.avg


def get_accuracy(model, data_loader, device):
    accuracy = 0.
    for x, y in data_loader:
        x = x.to(device)
        zero_nfe(model)
        logits = model(x)
        prediction = F.softmax(logits, dim=1).argmax(dim=1)
        accuracy += np.sum(tonp(prediction) == tonp(y))
    n_samples = get_n_samples(data_loader)
    return accuracy / n_samples

# Visualisations


def visualize_points(x, y, alpha=1., colors=('red', 'green')):
    c = []
    for i in y:
        c.append(colors[i])
    plt.scatter(x[:, 0], x[:, 1], color=c, alpha=alpha)
    plt.grid(True)


def visualize_data(train, val, test, args):
    """train, val, test are TensorDatasets consist of x and y Tensors"""
    plt.figure(figsize=(9*3, 8))
    for i, (data, name) in enumerate(zip([train, val, test], ['train', 'val', 'test'])):
        plt.subplot(1, 3, 1 + i)
        visualize_points(data.tensors[0].numpy(), data.tensors[1].numpy())
        title = "{}, {} samples".format(name, len(data))
        plt.xlabel(title, size=15)

    title = "n_samples: {}\nmu_1: {}, mu_2: {}\nstd_1: {}, std_2: {}".format(args.n_samples,
                                                                             args.mu_1, args.mu_2,
                                                                             args.std_1, args.std_2)
    plt.suptitle(title, size=17)


def get_boundaries(tensors):
    """tensors is list of 2D Tensors"""
    xmax, xmin, ymax, ymin = [], [], [], []
    for tensor in tensors:
        xmax.append(tensor[:, 0].max().item())
        xmin.append(tensor[:, 0].min().item())
        ymax.append(tensor[:, 1].max().item())
        ymin.append(tensor[:, 1].min().item())
    return min(xmin), max(xmax), min(ymin), max(ymax)


def get_endpoints(model, dataloader):
    endpoints = []
    labels = []
    for x, y in dataloader:
        endpoints.append(tonp(model[0](x)))
        labels.append(tonp(y))
    endpoints = np.vstack(endpoints)
    labels = np.hstack(labels)
    return endpoints, labels


def plot_dividing_curve_and_endpoints(model, loaders, device, npts=100, title=None):
    """loaders is dict like {'dataset_name': dataset_loader}"""
    # plot dividing curve in data space
    xmin, xmax, ymin, ymax = get_boundaries([l.dataset.tensors[0] for l in loaders.values()])
    ratio = 1.1
    _x = np.linspace(xmin * ratio, xmax * ratio, npts)
    _y = np.linspace(ymin * ratio, ymax * ratio, npts)
    X, Y = np.meshgrid(_x, _y)
    grid = np.vstack([X.flatten(), Y.flatten()]).T
    logits = model(torch.FloatTensor(grid).to(device))
    p = tonp(F.softmax(logits, dim=1).argmax(dim=1)).reshape((npts, npts))

    n_plots = len(loaders)
    plt.figure(figsize=(8 * n_plots, 12))
    for i, (name, loader) in enumerate(loaders.items()):
        plt.subplot(2, n_plots, i + 1)
        visualize_points(tonp(loader.dataset.tensors[0]),
                              tonp(loader.dataset.tensors[1]), alpha=0.6)
        plt.contourf(X, Y, p, alpha=.15)
        plt.title(name, size=15)

    # plot endpoints of ODE and dividing curve in trasformed space
    endpoints = dict()
    for data_name, loader in loaders.items():
        endpoints.update({
            data_name: get_endpoints(model, loader)
        })
    xmin, xmax, ymin, ymax = get_boundaries([x[0] for x in endpoints.values()])
    _x = np.linspace(xmin * ratio, xmax * ratio, npts)
    _y = np.linspace(ymin * ratio, ymax * ratio, npts)
    X, Y = np.meshgrid(_x, _y)
    grid = np.vstack([X.flatten(), Y.flatten()]).T
    logits = model[-1](torch.FloatTensor(grid).to(device))
    p = tonp(F.softmax(logits, dim=1).argmax(dim=1)).reshape((npts, npts))
    for i, (data_name, points_and_labels) in enumerate(endpoints.items()):
        plt.subplot(2, 3, i + 4)
        visualize_points(*points_and_labels, alpha=0.6)
        plt.contourf(X, Y, p, alpha=.15)
        plt.title(data_name, size=15)
    if title:
        plt.suptitle(title, size=17)


def plot_metrics(logs, nfe_flag):
    def get_epoch_and_log(log):
        arr = np.array(log)
        return arr[:, 0], arr[:, 1]

    def plot(log_name):
        epoch, log = get_epoch_and_log(logs[log_name])
        label = ''
        if log[-1] is not None:
            label = '{0}: {1:.4f}'.format(log_name, log[-1])
        plt.plot(epoch, log, label=label)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plot('train.loss')
    plot('val.loss')
    plt.grid()
    plt.legend()
    plt.ylabel('loss', size=15)
    plt.xlabel('epoch', size=15)
    plt.subplot(1, 3, 2)
    plot('train.acc')
    plot('val.acc')
    plt.grid()
    plt.legend()
    plt.ylabel('accuracy', size=15)
    plt.xlabel('epoch', size=15)
    plt.subplot(1, 3, 3)
    if nfe_flag:
        plot('train.forward_nfe')
        plot('train.backward_nfe')
        plot('val.forward_nfe')
        plt.grid()
        plt.legend()
        plt.ylabel('number of function evaluation', size=15)
        plt.xlabel('epoch', size=15)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.N = 0
        self.sum = 0
        self.avg = 0

    def update(self, val):
        self.N += 1
        self.sum += val
        self.avg = self.sum/self.N


def ensamble_probs(probs, y):
    ens_prediction = probs.mean(dim=2).argmax(dim=1)
    ens_acc = np.sum(tonp(ens_prediction) == y)
    return ens_acc

def ensamble_logits(logits, y):
    ens_logits = logits.mean(dim=2)
    predictions = F.softmax(ens_logits).argmax(dim=1)
    ens_acc = np.sum(tonp(predictions) == y)
    return ens_acc


def get_ensamble_metrics(model, data_loader, n_estimators, device, to_ensamble='probs'):
    n_samples = get_n_samples(data_loader)
    if n_samples == 0:
        return None, None, None
    ens_acc = {i: 0. for i in range(1, n_estimators + 1)}
    all_acc = []
    for x, y in data_loader:
        x = x.to(device)
        y = tonp(y)
        logits = []
        for _ in range(n_estimators):
            logits.append(model(x))
        logits = torch.stack(logits, dim=2)
        probs = F.softmax(logits, dim=1)
        all_acc.append(np.sum(tonp(probs.argmax(dim=1)) == y[:, np.newaxis], axis=0))
        for i in range(1, n_estimators + 1):
            if to_ensamble == 'logits':
                ens_acc[i] += ensamble_logits(logits[:, :, :i], y)
            elif to_ensamble == 'probs':
               ens_acc[i] += ensamble_probs(probs[:, :, :i], y)
            else:
                raise NotImplementedError

    all_acc = np.sum(all_acc, axis=0) / n_samples
    mean_acc = dict()
    std_acc = dict()
    for n_estimators in ens_acc.keys():
        ens_acc[n_estimators] /= n_samples
        mean_acc.update({
            n_estimators: all_acc[:n_estimators].mean()
        })
        std_acc.update({
            n_estimators: all_acc[:n_estimators].std()
        })
    return ens_acc, mean_acc, std_acc


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]