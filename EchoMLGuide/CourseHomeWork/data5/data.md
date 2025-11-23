You just need run the python:

```python
trainset = torchvision.datasets.CIFAR10(root='./data5', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data5', train=False, download=True, transform=transform)
```