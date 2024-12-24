# 前言

Parameter和Buffer都是实例化的Tensor，Parameter是参与梯度运算的参数，Buffer是不参与梯度计算的参数

`class Parameter(torch.Tensor, metaclass=_ParameterMeta):`

- `Parameter` 是一个特殊的张量，它被用来表示模型的参数,自动将 `Parameter` 

 `class Buffer(torch.Tensor, metaclass=_BufferMeta):`
 
- `Buffer` 也是一个特殊的张量，它用于存储那些在模型中不直接参与梯度计算的数据，但可能在模型的前向或后向传播中使用。
- `Buffer` 对象通常用于存储那些需要在模型中共享或在多个地方使用，但又不需要梯度的张量。例如，批量归一化层（BatchNorm）中的运行均值和方差就是作为缓冲区存储的。

假设你正在创建一个自定义的神经网络层，这个层有一个可学习的参数（例如，一个权重矩阵）。你可以使用 `Parameter` 来定义这个权重矩阵。


```python
import torch
import torch.nn as nn

class CustomLinearLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(CustomLinearLayer, self).__init__()
        # 定义一个可学习的权重参数
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        # 初始化权重
        nn.init.xavier_uniform_(self.weight)
        # 定义偏置，如果需要的话
        self.bias = nn.Parameter(torch.Tensor(output_features))
        nn.init.zeros_(self.bias)

    def forward(self, input):
        return torch.matmul(input, self.weight.t()) + self.bias

# 使用自定义层
layer = CustomLinearLayer(10, 5)
print(layer.weight)  # 打印权重参数
print(layer.bias)    # 打印偏置参数
```

在这个例子中，`self.weight` 和 `self.bias` 都是通过 `nn.Parameter` 创建的，这意味着它们是模型的参数，将在训练过程中被优化。


假设你想要在模型中存储一些不参与梯度计算的额外信息，比如一个用于追踪某些统计信息的运行平均值。

```python
class CustomBatchNorm(nn.Module):
    def __init__(self, num_features):
        super(CustomBatchNorm, self).__init__()
        # 定义运行均值和方差作为缓冲区
        self.running_mean = nn.Buffer(torch.zeros(num_features))
        self.running_var = nn.Buffer(torch.ones(num_features))
        # 定义可学习的参数
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # 这里只是一个示例，实际的批量归一化会更复杂
        mean = self.running_mean
        var = self.running_var
        return (x - mean) / torch.sqrt(var + 1e-5) * self.weight + self.bias

# 使用自定义批量归一化层
bn_layer = CustomBatchNorm(10)
print(bn_layer.running_mean)  # 打印运行均值缓冲区
print(bn_layer.running_var)   # 打印运行方差缓冲区
```

在这个例子中，`self.running_mean` 和 `self.running_var` 是通过 `nn.Buffer` 创建的，这意味着它们是模型的缓冲区，不会在训练过程中被优化，但可以在模型的前向传播中使用。

请注意，这些示例仅用于展示如何定义和使用 `Parameter` 和 `Buffer`，并不是实际可用的层实现。在实际应用中，你应该使用 PyTorch 提供的现成层，如 `nn.Linear` 和 `nn.BatchNorm1d`，因为它们已经经过了优化和测试。


