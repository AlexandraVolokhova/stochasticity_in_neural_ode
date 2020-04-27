# Stochasticity in neural ODEs
![img](https://github.com/AlexandraVolokhova/stochasticity_in_neural_ode/raw/master/pictures/stoch_trajectories.png)
This repo contains the code for experiments from the paper ["Stochasticity in Neural ODEs: An Empirical Study"](https://openreview.net/forum?id=C4ydiXrYw), where we experimentally explore regularization properties of stochasticity in the neural ODE

## Run experiments
First of all you should define two enviromental virables ```DATA_ROOT``` as the full path to the directory with datasets and ```EXMAN_PATH``` as the path for saving logs. After that, I can train models running the following scripts.

### CIFAR10
cifar10 SDENet no augmentation:

```python train_model.py --data cifar10 --train_bs 512 --test_bs 512 --n_steps 10 --augmentation False --odenet True --stoch_type sde --stoch_coeff 0.79 --lr 0.05 --warm 0 --wd 5e-4 --val_size 0.0```

cifar10 SDENet with augmentation:

```python train_model.py --data cifar10 --train_bs 512 --test_bs 512 --n_steps 10 --augmentation True  --odenet True --stoch_type sde --stoch_coeff 0.2 --lr 0.05 --warm 0 --wd 5e-4 --val_size 0.0```

cifar10 ODENet no augmentation with batchnorm:

```python train_model.py --data cifar10 --train_bs 512 --test_bs 512 --n_steps 6 --augmentation False --norm True  --odenet True --stoch_type none --lr 0.05 --warm 0 --wd 5e-4 --val_size 0.0```

cifar10 ODENet with augmentation with batchnorm:

```python train_model.py --data cifar10 --train_bs 512 --test_bs 512 --n_steps 6 --augmentation True --norm True  --odenet True --stoch_type none --lr 0.05 --warm 0 --wd 5e-4 --val_size 0.0```

cifar10 ODENet no augmentation no batchnorm:

```python train_model.py --data cifar10 --train_bs 512 --test_bs 512 --n_steps 10 --augmentation False --norm False  --odenet True --stoch_type none --lr 0.1 --warm 0 --wd 5e-4 --val_size 0.0```

cifar10 ODENet with augmentation no batchnorm:

```python train_model.py --data cifar10 --train_bs 512 --test_bs 512 --n_steps 10 --augmentation True --norm False  --odenet True --stoch_type none --lr 0.05 --warm 0 --wd 5e-4 --val_size 0.0```

cifar10 ResNet no augmentation:

```python train_model.py --data cifar10 --train_bs 512 --test_bs 512 --augmentation False --norm True  --odenet False --lr 0.4 --warm 0 --wd 5e-4 --val_size 0.0```

cifar10 ResNet with augmentation:

```python train_model.py --data cifar10 --train_bs 512 --test_bs 512 --augmentation True --norm True  --odenet False --lr 0.1--warm 0 --wd 5e-4 --val_size 0.0```

### CIFAR 100

cifar100 SDENet no augmentation:

```python train_model.py --data cifar100 --train_bs 256 --test_bs 256 --n_steps 3 --augmentation False --odenet True --stoch_type sde --stoch_coeff 0.25 --lr 0.1 --warm 2 --wd 5e-4 --val_size 0.0```

cifar100 SDENet with augmentation:

```python train_model.py --data cifar100 --train_bs 256 --test_bs 256 --n_steps 3 --augmentation True  --odenet True --stoch_type sde --stoch_coeff 0.05 --lr 0.1 --warm 2 --wd 5e-4 --val_size 0.0```

cifar100 ODENet no augmentation with batchnorm:

```python train_model.py --data cifar100 --train_bs 256 --test_bs 256 --n_steps 3 --augmentation False --norm True  --odenet True --stoch_type none --lr 0.1 --warm 2 --wd 5e-4 --val_size 0.0```

cifar100 ODENet with augmentation with batchnorm:

```python train_model.py --data cifar100 --train_bs 256 --test_bs 256 --n_steps 3 --augmentation True --norm True  --odenet True --stoch_type none --lr 0.1 --warm 2--wd 5e-4 --val_size 0.0```

cifar100 ODENet no augmentation no batchnorm:

```python train_model.py --data cifar100 --train_bs 256 --test_bs 256 --n_steps 3 --augmentation False --norm False  --odenet True --stoch_type none --lr 0.1 --warm 2 --wd 5e-4 --val_size 0.0```

cifar100 ODENet with augmentation no batchnorm:

```python train_model.py --data cifar100 --train_bs 256 --test_bs 256 --n_steps 3 --augmentation True --norm False  --odenet True --stoch_type none --lr 0.1 --warm 2 --wd 5e-4 --val_size 0.0```

cifar100 ResNet no augmentation:

```python train_model.py --data cifar100 --train_bs 256 --test_bs 256 --augmentation False --norm True  --odenet False --lr 0.4 --warm 2 --wd 5e-4 --val_size 0.0```

cifar100 ResNet with augmentation:

```python train_model.py --data cifar100 --train_bs 256 --test_bs 256 --augmentation True --norm True  --odenet False --lr 0.2--warm 2 --wd 5e-4 --val_size 0.0```

### TinyImagenet
You may need to download the [dataset](https://tiny-imagenet.herokuapp.com)

tiny imagenet SDENet no augmentation:

```python train_model.py --data tinyimagenet --train_bs 256 --test_bs 256 --n_steps 2 --augmentation False --odenet True --stoch_type sde --stoch_coeff 0.4 --lr 0.05 --warm 0 --wd 1e-5 --val_size 0.0```

tiny imagenet SDENet with augmentation:

```python train_model.py --data tinyimagenet --train_bs 256 --test_bs 256 --n_steps 2 --augmentation True  --odenet True --stoch_type sde --stoch_coeff 0.3 --lr 0.01 --warm 3 --wd 1e-4 --val_size 0.0```

tiny imagenet ODENet no augmentation with batchnorm:

```python train_model.py --data tinyimagenet --train_bs 256 --test_bs 256 --n_steps 2 --augmentation False --norm True  --odenet True --stoch_type none --lr 0.1 --warm 0 --wd 1e-5 --val_size 0.0```

tiny imagenet ODENet with augmentation with batchnorm:

```python train_model.py --data tinyimagenet --train_bs 256 --test_bs 256 --n_steps 2 --augmentation True --norm True  --odenet True --stoch_type none --lr 0.05 --warm 3--wd 1e-4 --val_size 0.0```

tiny imagenet ODENet no augmentation no batchnorm:

```python train_model.py --data tinyimagenet --train_bs 256 --test_bs 256 --n_steps 2 --augmentation False --norm False  --odenet True --stoch_type none --lr 0.05 --warm 0 --wd 1e-5 --val_size 0.0```

tiny imagenet ODENet with augmentation no batchnorm:

```python train_model.py --data tinyimagenet --train_bs 256 --test_bs 256 --n_steps 2 --augmentation True --norm False  --odenet True --stoch_type none --lr 0.01 --warm 3 --wd 1e-4 --val_size 0.0```

tiny imagenet ResNet no augmentation:

```python train_model.py --data tinyimagenet --train_bs 256 --test_bs 256 --augmentation False --norm True  --odenet False --lr 0.05 --warm 3 --wd 1e-4 --val_size 0.0```

tiny imagenet ResNet with augmentation:

```python train_model.py --data tinyimagenet --train_bs 256 --test_bs 256 --augmentation True --norm True  --odenet False --lr 0.1--warm 3 --wd 1e-4 --val_size 0.0```

## External libraries:
We adapted code from the following repositories:
* [torchdiffeq](https://github.com/rtqichen/torchdiffeq) is a library for solving differential equations numerically using PyTorch. We added a numerical solver for stochastic equations there.
* [exman](https://github.com/ferrine/exman) is an experiment manager (a logger and an argument parser), our adopted version is in the ```myexman``` directory
