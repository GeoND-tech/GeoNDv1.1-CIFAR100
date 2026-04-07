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
wget -i models.txt -P checkpoint
```

## IMPORTANT
Including any layer with paraboloid neurons requires a specialized optimizer:
```
optimizer = gpt.GeoNDSGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.wd, nesterov = args.nesterov)
```

## Models
- ### dla
Our baseline Deep Layer Aggregation model.
#### Evaluation
Download the pretrained model and run:
```
python main.py --model dla --eval dla.pth
```
#### Training from scratch
Run:
```
python main.py --model dla
```

- ### dla_paraboloidout
A DLA model with a layer of paraboloid neurons as the output layer. After the Version 1.1 update which fixed some issues, it is possible to use paraboloid layers as output layers. However, the fixes also cause such models to always overfit, see the Evaluation section for details.

In terms of code, first we import the Library:
```
try:
    import geondpt as gpt
except ImportError:
    import geondptfree as gpt
```

Then we replace the existing output layer:
```
#self.linear = nn.Linear(512, num_classes)
self.paraboloid = gpt.ParaboloidOutput(512, num_classes, h_factor = 0.01, lr_factor = 1., wd_factor = 0.1, grad_factor = 1., input_factor = 0.4, output_factor = 0.1, init='spotlight')
```
Note that ```ParaboloidOutput``` is the same as ```Paraboloid```, it just uses a base configuration more appropriate for output layers. We use a slightly different configuration here.

Remember to update the forward function:
```
out = self.layer6(out)
out = F.avg_pool2d(out, 4)
out = out.view(out.size(0), -1)
out = self.paraboloid(out)
#out = self.linear(out)
```

#### Evaluation
Download the pretrained model and run:
```
python main.py --model dla_paraboloidout --eval dla_paraboloidout.pth
```
#### Training from scratch
Note that models with a paraboloid output layer (or a paraboloid layer before a linear output layer) seem to perform better without momentum. To train the model without momentum, run:
```
python main.py --model dla_paraboloidout --momentum 0.0 --nesterov False
```


- ### DLA_paraconv_quarter
A DLA model with the first convolutional layer replaced with a paraboloid convolutional layer with 4 units instead of the original 16. We do this to avoid overfitting and also reduce the size and execution time of the model. This model can achieve better performance with almost the same speed, especially when using the faster licensed version of the library. See the Evaluation section for details.

In terms of code, again, we first import the Library:
```
try:
    import geondpt as gpt
except ImportError:
    import geondptfree as gpt
```
Then we find the line with the first convolutional layer:
```
nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
```
and replace it with:
```
gpt.ParaConv2d(3, 4, kernel_size=3, stride=1, padding=1, output_factor = 1.0 , input_factor = 1.0, lr_factor = 100., wd_factor = 0.1, skip_input_grad = False, init = 'spotlight', h_factor = 0.01, p_factor = 0.0001, grad_factor = 1.),
nn.BatchNorm2d(4),
```
Note that we also update the batch normalization layer. We also need to update the following layer to accept input from 4 units:
```
self.layer1 = nn.Sequential(
    nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1, bias=False),
```

In this case, we do not need to update the forward function, as we replaced an existing layer.

#### Evaluation
Download the pretrained model and run:
```
python main.py --model dla_paraconv_quarter --eval dla_paraconv_quarter.pth
```
#### Training from scratch
Run:
```
python main.py --model dla_paraconv_quarter
```

## Evaluation of pretrained models
|   Model           | Accuracy |
| ----------------- |-------- |
| **resnet18** - baseline   | 78.89% |
| **resnet18-paraboloidout**         | **79.33%** |
| **resnet18-paraboloid**      | 79.05% |

Note that, due to numerical issues and data augmentation, the results may not always line up exactly.

## References
- Original repository: [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- GeoND Library: [https://geond.tech/download/](https://geond.tech/download/)
- Paraboloid Neurons: [https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf](https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf)

