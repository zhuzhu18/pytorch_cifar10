# PyTorch-CIFAR10

This project borrows ideas from [PyTorch-Template](https://github.com/Lyken17/PyTorch-Template), 
aimming to provide a easy and extensible template for my future pytorch projects.


## Structure
```
├── datasets 
        (wrapper for datasets)
├── models
        (network descriptions)
├── options
        (hyper-parameters parser for command line inputs)
├── trainers
        (define how model's forward / backward and logs)
├── utils
        (toolbox for drawing and scheduling)
├── cifar.py
        (main .py)
├── main.sh
        (setup for the project)
└── work
        (default folder to store logs/models)
```

## Usage
*  Train ResNet-20 (pre-activation) on CIFAR10
    ```bash
    python cifar.py \
       	/home/zzh/dataset/cifar10 \
       	--arch resnet20 \
       	--dataset cifar10
    ```
#### or
* Train ResNet-18 on ImageNet
    ```bash
    sh main.sh
    ```