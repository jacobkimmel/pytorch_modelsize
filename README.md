# PyTorch Model Size Estimator

This tool estimates the size of a [PyTorch](https://pytorch.org) model in memory for a given input size.  
Estimating the size of a model in memory is useful when trying to determine an appropriate batch size, or when making architectural decisions.

**Note (1):** `SizeEstimator` is only valid for models where dimensionality changes are exclusively carried out by modules in `model.modules()`.

For example, use of `nn.Functional.max_pool2d` in the `forward()` method of a model prevents `SizeEstimator` from functioning properly. There is no direct means to access dimensionality changes carried out by arbitrary functions in the `forward()` method, such that tracking the size of inputs and gradients to be stored is non-trivial for such models.

**Note (2):** The size estimates provided by this tool are theoretical estimates only, and the total memory used will vary depending on implementation details. PyTorch utilizes a few hundred MB of memory for CUDA initialization, and the use of cuDNN alters memory usage in a manner that is difficult to predict. See [this discussion on the PyTorch Forums](https://discuss.pytorch.org/t/gpu-memory-estimation-given-a-network/1713) for more detail.

[See this blog post](http://jacobkimmel.github.io/pytorch_estimating_model_size/) for an explanation of the size estimation logic.

## Usage

To use the size estimator, simply import the `SizeEstimator` class, then provide a model and an input size for estimation.

```python
# Define a model
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()

        self.conv0 = nn.Conv2d(1, 16, kernel_size=3, padding=5)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3)

    def forward(self, x):
        h = self.conv0(x)
        h = self.conv1(h)
        return h

model = Model()

# Estimate Size
from pytorch_modelsize import SizeEstimator

se = SizeEstimator(model, input_size=(16,1,256,256))
print(se.estimate_size())

# Returns
# (size in megabytes, size in bits)
# (408.2833251953125, 3424928768)

print(se.param_bits) # bits taken up by parameters
print(se.forward_backward_bits) # bits stored for forward and backward
print(se.input_bits) # bits for input
```

## Development

This tool is a product of the [Laboratory of Cell Geometry](https://cellgeometry.ucsf.edu/) at the [University of California, San Francisco](https://ucsf.edu).
