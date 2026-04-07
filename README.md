# Using paraboloid neurons to train DLA models on CIFAR100 with PyTorch

Paraboloid neuron demonstration of the [GeoND Library](https://geond.tech) for [PyTorch](http://pytorch.org/) on the CIFAR100 dataset. This repository uses Version 1.1 of the GeoND Library. Adapted from [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models).

## Requirements
- Linux only.
- Python 3.9+, use of a virtual environment recommended.
- Install the rest of the requirements by running:
```
pip install -r requirements.txt
```
- (Optional) Download the pre-trained models by running:
```
wget -i models.txt
```

## Models
- ### resnet18
Our baseline ResNet18 model. After creating the model, we make some changes to accomodate the resolution of CIFAR100 images:
```
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
```
#### Evaluation
Download the pretrained model and run:
```
python train.py ./data --dataset torch/cifar100 --dataset-download --num-classes 100 --img-size 32 --epochs 300 --resume resnet18.pth.tar --eval true
```
#### Training from scratch
Run:
```
python train.py ./data --dataset torch/cifar100 --dataset-download --num-classes 100 --img-size 32 --opt sgd --momentum 0.9 --weight-decay 5e-4 --sched cosine --epochs 300 --lr 0.1 --batch-size 128 --min-lr 1e-5 --aa rand-m9-mstd0.5-inc1 --mixup 0.2 --cutmix 1.0 --reprob 0.25 --remode pixel --smoothing 0.1
```




- ### resnet18-paraboloidout
A ResNet18 model (modified for CIFAR100 as above) with a layer of paraboloid neurons as the output layer. After the Version 1.1 update which fixed some issues, it is possible to use paraboloid layers as output layers. Note that models with a paraboloid output layer (or a paraboloid layer before a linear output layer) seem to perform better without momentum. The training script will handle this through a command line argument. This model achieves the best accuracy.

In terms of code, first we import the Library:
```
try:
    import geondpt as gpt
except ImportError:
    import geondptfree as gpt
```

Then we replace the existing output layer:
```
model.fc = gpt.ParaboloidOutput(model.fc.in_features, model.fc.out_features, h_factor = 0.01, lr_factor = 1., wd_factor = 1., grad_factor = 1., input_factor = 0.5, output_factor = 0.1, p_factor=0.0001, init = 'spotlight')
```
Note that ```ParaboloidOutput``` is the same as ```Paraboloid```, it just uses a base configuration more appropriate for output layers.

#### Evaluation
Download the pretrained model and run:
```
python train.py ./data --dataset torch/cifar100 --dataset-download --num-classes 100 --img-size 32 --epochs 300 --paraboloid true --paraboloidout true --resume resnet18-paraboloidout.pth.tar --eval true
```
#### Training from scratch
Run:
```
python train.py ./data --dataset torch/cifar100 --dataset-download --num-classes 100 --img-size 32 --opt sgd --momentum 0.9 --weight-decay 5e-4 --sched cosine --epochs 300 --lr 0.1 --batch-size 128 --min-lr 1e-5 --aa rand-m9-mstd0.5-inc1 --mixup 0.2 --cutmix 1.0 --reprob 0.25 --remode pixel --smoothing 0.1 --paraboloid true --paraboloidout true
```


- ### resnet18-paraboloid

A ResNet18 model (modified for CIFAR100 as above) with a paraboloid neuron layer with 1024 units inserted before the linear output layer. Note that models with a paraboloid output layer (or a paraboloid layer before a linear output layer) seem to perform better without momentum. The training script will handle this through a command line argument. This model achieves better accuracy than the baseline model, though lower than the ```resnet18-paraboloidout``` model. However, this indicates that paraboloid neurons can have applications in transfer learning.

In terms of code, again, we first import the Library:
```
try:
    import geondpt as gpt
except ImportError:
    import geondptfree as gpt
```
Then we replace the existing output layer:
```
model.fc = nn.Sequential(gpt.Paraboloid(model.fc.in_features, 1024, h_factor = 0.01, lr_factor = 1000., wd_factor = 1., grad_factor = 1., input_factor = 0.1, output_factor = 0.1, p_factor=0.0001, init = 'live'), nn.Linear(1024, model.fc.out_features))
```

#### Evaluation
Download the pretrained model and run:
```
python train.py ./data --dataset torch/cifar100 --dataset-download --num-classes 100 --img-size 32 --epochs 300 --paraboloid true --resume resnet18-paraboloid.pth.tar --eval true
```
#### Training from scratch
Run:
```
python train.py ./data --dataset torch/cifar100 --dataset-download --num-classes 100 --img-size 32 --opt sgd --momentum 0.9 --weight-decay 5e-4 --sched cosine --epochs 300 --lr 0.1 --batch-size 128 --min-lr 1e-5 --aa rand-m9-mstd0.5-inc1 --mixup 0.2 --cutmix 1.0 --reprob 0.25 --remode pixel --smoothing 0.1 --paraboloid true
```

## Evaluation of pretrained models
|   Model           | Accuracy |
| ----------------- |-------- |
| **resnet18** - baseline   | 78.89% |
| **resnet18-paraboloidout**         | **79.33%** |
| **resnet18-paraboloid**      | 79.05% |

Note that, due to numerical issues, the results may not always line up exactly.

## Overfitting

While searching for the best parameters, we noticed that there are some sets that overfit the data, even with the augmentations applied. More specifically, for ```resnet18-paraboloidout```, this involves using ```input_factor = 1.0``` instead of ```input_factor = 0.5``` and for ```resnet18-paraboloid``` this involves using ```lr_factor = 10.``` instead of ```lr_factor = 1000.```. We do not yet have any interpretation for this behavior.

To review the overfitting without training models from scratch, we include the log files for the following models in this repository. Studying them will reveal that the overfitting log lists lower values loss for the loss function, yet worse accuracy. Below is a list of the log files:

- ```resnet18``` (baseline): [https://github.com/GeoND-tech/GeoNDv1.1-CIFAR100/blob/main/resnet18-summary.csv](https://github.com/GeoND-tech/GeoNDv1.1-CIFAR100/blob/main/resnet18-summary.csv)
- ```resnet18-paraboloidout```: [https://github.com/GeoND-tech/GeoNDv1.1-CIFAR100/blob/main/resnet18-paraboloidout-summary.csv](https://github.com/GeoND-tech/GeoNDv1.1-CIFAR100/blob/main/resnet18-paraboloidout-summary.csv)
- ```resnet18-paraboloid```: [https://github.com/GeoND-tech/GeoNDv1.1-CIFAR100/blob/main/resnet18-paraboloid-summary.csv](https://github.com/GeoND-tech/GeoNDv1.1-CIFAR100/blob/main/resnet18-paraboloid-summary.csv)
- Overfitting ```resnet18-paraboloidout```: [https://github.com/GeoND-tech/GeoNDv1.1-CIFAR100/blob/main/resnet18-paraboloidout-overfit-summary.csv](https://github.com/GeoND-tech/GeoNDv1.1-CIFAR100/blob/main/resnet18-paraboloidout-overfit-summary.csv)

To overfit ```resnet18-paraboloidout```, run:
```
python train.py ./data --dataset torch/cifar100 --dataset-download --num-classes 100 --img-size 32 --opt sgd --momentum 0.9 --weight-decay 5e-4 --sched cosine --epochs 300 --lr 0.1 --batch-size 128 --min-lr 1e-5 --aa rand-m9-mstd0.5-inc1 --mixup 0.2 --cutmix 1.0 --reprob 0.25 --remode pixel --smoothing 0.1 --paraboloid true --paraboloidout true --overfit true
```

To overfit ```resnet18-paraboloid```, run:
```
python train.py ./data --dataset torch/cifar100 --dataset-download --num-classes 100 --img-size 32 --opt sgd --momentum 0.9 --weight-decay 5e-4 --sched cosine --epochs 300 --lr 0.1 --batch-size 128 --min-lr 1e-5 --aa rand-m9-mstd0.5-inc1 --mixup 0.2 --cutmix 1.0 --reprob 0.25 --remode pixel --smoothing 0.1 --paraboloid true --overfit true
```

## References
- Original repository: [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- GeoND Library: [https://geond.tech/download/](https://geond.tech/download/)
- Paraboloid Neurons: [https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf](https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf)

