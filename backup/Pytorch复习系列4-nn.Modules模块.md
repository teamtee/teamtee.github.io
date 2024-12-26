# 前言
nn.Modules下面包含很多nn.Module的实例，nn.Module是Pytorch所有神经网络的父类
参考
[1.PyTorch 源码解读之 nn.Module：核心网络模块接口详解](https://zhuanlan.zhihu.com/p/340453841)

# nn.Module基本属性
在Module的__init__函数中可能观察到下面nn.Modules的核心组件

```python
self.training = True  # 控制 training/testing 状态
self._parameters = OrderedDict()  # 在训练过程中会随着 BP 而更新的参数
self._buffers = OrderedDict()  # 在训练过程中不会随着 BP 而更新的参数
self._non_persistent_buffers_set = set()
self._backward_hooks = OrderedDict()  # Backward 完成后会被调用的 hook
self._forward_hooks = OrderedDict()  # Forward 完成后会被调用的 hook
self._forward_pre_hooks = OrderedDict()  # Forward 前会被调用的 hook
self._state_dict_hooks = OrderedDict()  # 得到 state_dict 以后会被调用的 hook
self._load_state_dict_pre_hooks = OrderedDict()  # load state_dict 前会被调用的 hook
self._modules = OrderedDict()  # 子神经网络模块
```

## 基本属性
下面的函数可以获取这些参数
- named_parameters：返回自身parameters,如果 recurse=True 还会返回子模块中的模型参数
- named_buffers：返回自身parameters,如果 recurse=True 还会返回子模块中的模型 buffer
- named_children：返回自身的Modules
-  named_modules：返回自身和子Modules的Moduels(递归调用)

下面的参数是对上面的调用,默认recurse参数为True
- parameters：
-  buffers：
-  children：
-  modules：
添加参数
- add_module：增加子神经网络模块，更新 self._modules
```
add_module(name,module)
```
-  register_parameter：增加通过 BP 可以更新的 parameters （如 BN 和 Conv 中的 weight 和 bias ），更新 self._parameters
- register_buffer：增加不通过 BP 更新的 buffer（如 BN 中的 running_mean 和 running_var）
- self.xxx = xxx ：该方法不会被登记，不属于Paramets和buffer，进行状态转换的时候会被遗漏
下面的函数可以调整梯度
- train()
- eval()
- requires_grad_()
- zero_gred()

下面的函数可以映射parameters和buffers
- `_apply(fn)`:针对parameters和buffers通过调用所有parameters和buffers的tensor的_apply函数实现

```
1. CPU：将所有 parameters 和 buffer 转移到 CPU 上
2. type：将所有 parameters 和 buffer 转变成另一个类型
3. CUDA：将所有 parameters 和 buffer 转移到 GPU 上
4. float：将所有浮点类型的 parameters 和 buffer 转变成 float32
5. double：将所有浮点类型的 parameters 和 buffer 转变成 double 类型
6. half：将所有浮点类型的 parameters 和 buffer 转变成 float16 类型
8. to：移动模块或/和改变模块的类型
```
- `apply`:针对Moduels，
可以自定义一个 init_weights 函数，通过 `net.apply(init_weights)` 来初始化模型权重。

```python
@torch.no_grad()
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
        print(m.weight)

net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)
```

## Hook钩子

钩子是指在特定阶段会被运行的函数，nn.Module的钩子分为全局的钩子（绑定在nn.Module上）和局部的钩子（绑定在特定的子module上）
- 先调用全局的钩子然后再调用局部的钩子
- 下面是所有钩子的种类
全局注册钩子的函数
```python
register_module_forward_pre_hook
register_module_forward_hook
register_module_full_backward_pre_hook
register_module_backward_hook
register_module_full_backward_hook
register_module_buffer_registration_hook
register_module_module_registration_hook
register_module_parameter_registration_hook
```

本地注册除了包含上面的注册函数，还具有下面的注册钩子的函数

 **`register_full_backward_hook`**：`module`，`grad_input`，`grad_output`
- 想要影响当前模块的参数梯度，你应该修改 `grad_input`。如果你想要影响后续操作的梯度，你可以修改 `grad_output`，但这种修改不会影响当前模块的参数梯度。
**`register_backward_hook`**：`module`，`grad_input`，`grad_output`
- 这个钩子可以用来获取模块输出端的梯度信息，但不提供修改这些梯度的能力。
**`register_full_backward_pre_hook`**：`module`，`grad_input`
- 可以修改梯度输入

**`register_forward_pre_hook`**：`module`，`input`、`output`(None)
- 钩子函数可以返回一个修改后的输入，这个输入将被用于模块的前向传播。
**`register_forward_hook`**：`module`，`input`、`output`(None)
- 钩子函数可以返回一个输出，这个输出可以被用于后续的钩子或者操作，但不会改变模块本身的输出。

- **`register_state_dict_pre_hook`**：
- 这个钩子在调用 `state_dict()` 方法之前执行。
- 它允许你在状态字典被创建之前修改模型的状态。
- 钩子函数的签名为 `hook(module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)`。
- 你可以返回一个修改后的 `state_dict`，这个状态字典将被用于后续的保存操作。
 **`register_state_dict_post_hook`**：
- 这个钩子在调用 `state_dict()` 方法之后执行。
- 它允许你在状态字典被创建之后进行一些操作，比如记录日志或者进行额外的验证。
- 钩子函数的签名为 `hook(module, state_dict)`。
- 你不能修改状态字典，因为此时它已经被创建并准备被保存。

 **`register_load_state_dict_pre_hook`**：
- 这个钩子在调用 `load_state_dict()` 方法之前执行。
- 它允许你在状态字典被加载到模型之前进行一些操作，比如修改状态字典的内容。
- 钩子函数的签名为 `hook(module, state_dict, strict, missing_keys, unexpected_keys, error_msgs)`。
- 你可以返回一个修改后的 `state_dict`，这个状态字典将被用于后续的加载操作。
**`register_load_state_dict_post_hook`**：
    
- 这个钩子在调用 `load_state_dict()` 方法之后执行。
- 它允许你在状态字典被加载到模型之后进行一些操作，比如记录日志或者进行额外的验证。
- 钩子函数的签名为 `hook(module, state_dict)`。
- 你不能修改状态字典，因为此时它已经被加载到模型中。

示例
```python
def state_dict_pre_hook(module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    # 在这里可以修改 state_dict 的内容
    # 例如，我们可以添加一些额外的信息到 local_metadata
    local_metadata['custom_info'] = 'This is a custom info.'

# 注册 state_dict 保存前的钩子
handle = module.register_state_dict_pre_hook(state_dict_pre_hook)

# 保存模型状态字典
state_dict = module.state_dict()

# 移除钩子
handle.remove()
```

# Modules

## nn.Linear

- Bilinear:双线性层，两个输入的线性层
	- `__init__(input1_dim,input2_dim,output_dim)`
- Linear：
	- - `__init__(input_dim,output_dim)`
-  Identity：恒等层，不需要参数的初始化
	- `__init__()`
- LazyLinear:在初始化时不需要指定输入特征的大小（`in_features`），该值会在模块第一次前向传播时自动推断。权重和偏置参数在第一次前向传播时才被初始化，之前它们是未初始化
	- `__init__(output_dim)`
## nn.Conv

    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",

- class _ConvNd(Module):  
```
"stride",
"padding",
"dilation",
"groups",
"padding_mode",可以是 `'zeros'`（默认），表示使用零填充；`'reflect'`，表示使用反射填充；`'replicate'`，表示使用复制边缘值的填充；或者 `'circular'`，表示使用循环填充
"output_padding",
"in_channels",
"out_channels",
"kernel_size",
```

> [!Note]
> groups指定分组卷积数：- 在标准的卷积操作中，每个输入通道与所有输出通道的卷积核进行卷积。这意味着如果输入有 `in_channels` 个通道，输出有 `out_channels` 个通道，那么卷积层将有 `in_channels * out_channels` 个参数。分组卷积改变了这一过程，它将输入通道分成 `groups` 个组，每组包含 `in_channels / groups` 个通道。同样，输出通道也被分成 `groups` 个组。每个组内的输入通道只与对应组内的输出通道进行卷积。


- class ConvTransposeNd
	- 参数类似ConvNd
	- 
## nn.Batchnorm

from .batchnorm import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    
    LazyBatchNorm1d,
    LazyBatchNorm2d,
    LazyBatchNorm3d,
    SyncBatchNorm,
)

$$y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta$$
其中，$\gamma$ 和 $\beta$是可学习的参数向量默认情况下，$\gamma$的元素被设置为 1，$\beta$ 的元素被设置为 0,$E[x]$和$Var[x]$是存储在buffer的不更梯度更新的参数，在trian模式下，$E[x]$和$Var[x]$的大小为当前批次的大小并且会更新，在eval模式下，$E[x]$和$Var[x]$不会更新并且为训练好的值

下面为参数的解释
- `num_features`：输入的特征数或通道数。
- `eps`：为防止分母为零而添加到分母的一个小值，默认为 `1e-5`。
- `momentum`：用于计算运行均值和方差的动量值，默认为 `0.1`。
- `affine`：布尔值，当设置为 `True` 时，模块具有可学习的仿射参数，默认为 `True`。
- `track_running_stats`：布尔值，当设置为 `True` 时，模块会跟踪运行均值和方差，默认为 `True`。

[SyncBatchNorm参考这篇](https://zhuanlan.zhihu.com/p/250471767)

## nn.Dropout
```
AlphaDropout,
Dropout,
Dropout1d,
Dropout2d,
Dropout3d,
FeatureAlphaDropout,
```
`Dropout` 可以替代 `Dropout1d`、`Dropout2d` 和 `Dropout3d`，因为 `Dropout` 是一个通用的 `dropout` 实现

AlphaDropout不讲，有兴趣可以去看论文：Self-Normalizing Neural Networks

## nn.normalize

```
CrossMapLRN2d,
GroupNorm,
LayerNorm,
LocalResponseNorm,
RMSNorm,
```
- LayerNorm:是对一个样本进行正则化，通常是对最后一个维度
	- `__init__(dim...)`
- RMSNorm：是对LayerNorm的变体,减少了均值的计算
	 - [参考](https://blog.csdn.net/yjw123456/article/details/138139970)
- GroupNorm:BatchNorm的变体，将Batch分组，然后在组内Norm
	- `num_groups`：指定分组的数量，每组的大小为 `num_channels / num_groups`。`num_groups` 必须能被 `num_channels` 整除。
	- `num_channels`：输入张量的通道数。
- LocalResponseNorm,
这个函数很少使用，基本上被类似Dropout这样的方法取代.
## nn.instancenorm
```
InstanceNorm1d,
InstanceNorm2d,
InstanceNorm3d,
LazyInstanceNorm1d,
LazyInstanceNorm2d,
LazyInstanceNorm3d,
```
相当于对一个样本进行正则化
## nn.Pool
```
AdaptiveAvgPool1d,
AdaptiveAvgPool2d,
AdaptiveAvgPool3d,
AdaptiveMaxPool1d,
AdaptiveMaxPool2d,
AdaptiveMaxPool3d,
AvgPool1d,
AvgPool2d,
AvgPool3d,
FractionalMaxPool2d,
FractionalMaxPool3d,
LPPool1d,
LPPool2d,
LPPool3d,
MaxPool1d,
MaxPool2d,
MaxPool3d,
MaxUnpool1d,
MaxUnpool2d,
MaxUnpool3d,
```
- MaxPool
- AvgPool
- AdaptiveAvgPool
- AdaptiveMaxPool
它能够自动调整池化窗口的大小，以便输出一个具有预定大小的特征图
- FractionalMaxPool:[有兴趣看论文](https://arxiv.org/abs/1412.6071)，同样能够指定池化输出大小
- `LPPool`：是一种基于 Lp 范数的池化操作，它提供了一种在最大池化和平均池化之间平滑过渡的池化方法。
```python
lp_pool = nn.LPPool2d(norm_type=2, kernel_size=3)
```
- MaxUnpool1d：恢复最大池化前的形状
```python
import torch
import torch.nn as nn

# 创建一个输入张量
x = torch.randn(1, 1, 4, 4)

# 创建一个最大池化层，并返回索引
pool = nn.MaxPool2d(2, return_indices=True)
y, ind = pool(x)

# 创建一个 MaxUnpool2d 层
unpool = nn.MaxUnpool2d(2, 2)

# 使用 MaxUnpool2d 恢复数据
y_unpool = unpool(y, ind)

print("原始数据：", x)
print("池化后数据：", y)
print("恢复后数据：", y_unpool)
```
- 
## nn.Loss

二元和多类别分类损失

- **BCELoss (Binary Cross Entropy Loss)**：用于二元分类问题。
$$ -y \log(p) - (1 - y) \log(1 - p) $$
- **BCEWithLogitsLoss**：结合了 Sigmoid 层和 BCELoss，适用于二元分类。
- **CrossEntropyLoss**：用于多类别分类问题，结合了 LogSoftmax 和 NLLLoss。
$$\text{CE}(\mathbf{x}, y) = -\log\left(\frac{\exp(x_y)}{\sum_{i=1}^{C} \exp(x_i)}\right)$$


- **NLLLoss (Negative Log Likelihood Loss)**：用于多类别分类问题，计算输入的负对数似然。
- **NLLLoss2d**：与 NLLLoss 类似，但适用于二维输入。

 回归损失

- **MSELoss (Mean Squared Error Loss)**：用于回归问题，计算预测值和目标值之间的均方误差。
- **SmoothL1Loss**：用于回归问题，结合了 L1 和 L2 损失，对异常值不敏感。
- **L1Loss**：用于回归问题，计算预测值和目标值之间的绝对误差。

 嵌入和距离损失

- **CosineEmbeddingLoss**：用于测量两个向量的余弦距离，常用于学习相似性。
- **HingeEmbeddingLoss**：用于学习向量之间的边界，常用于排序问题。
- **TripletMarginLoss**：用于学习样本之间的相对距离，常用于度量学习。
- **TripletMarginWithDistanceLoss**：结合了 TripletMarginLoss 和距离计算。

排名和排序损失

- **MarginRankingLoss**：用于学习排序，使得正样本的分数高于负样本。
- **MultiLabelMarginLoss**：用于多标签分类问题，每个标签独立计算。

分布损失

- **KLDivLoss (Kullback-Leibler Divergence Loss)**：用于衡量两个概率分布之间的差异。
- **GaussianNLLLoss**：用于高斯分布的负对数似然损失。
- **PoissonNLLLoss**：用于泊松分布的负对数似然损失。

 其他损失

- **HuberLoss**：对 L1 和 L2 损失的组合，对异常值不敏感。
- **SoftMarginLoss**：用于分类问题，当边界不是硬性的时候。
- **MultiLabelSoftMarginLoss**：用于多标签分类问题，每个标签独立计算，允许软边界。

序列和结构化预测损失

- **CTCLoss (Connectionist Temporal Classification Loss)**：用于序列标注问题，如语音识别和手写识别。
- **MultiMarginLoss**：用于多分类问题，每个类别独立计算。
## nn.activation

- **Sigmoid**：将输入压缩到 0 和 1 之间。
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- **Tanh**：将输入压缩到 -1 和 1 之间。

$$ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
- **ReLU (Rectified Linear Unit)**：线性激活函数，负值置零。
$$ \text{ReLU}(x) = \max(0, x)$$
- **LeakyReLU**：ReLU 的变种，允许负值有小的梯度。

$$\text{LeakyReLU}(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0 
\end{cases}$$
- **PReLU (Parametric ReLU)**：LeakyReLU 的变种，负值的斜率是可学习的。
- **RReLU (Randomized ReLU)**：LeakyReLU 的变种，负值的斜率在一定范围内随机选择。
- **ReLU6**：将输入限制在 -6 和 6 之间，类似于 ReLU，但有上限。
- **SELU (Scaled Exponential Linear Unit)**：自归一化激活函数，具有参数学习和缩放。
- **SiLU (Sigmoid Linear Unit)** 或 **Swish**：由 sigmoid 函数和输入的线性组合构成。
- **CELU (Continuously Differentiable Exponential Linear Unit)**：平滑且连续可微的指数线性单元。
- **ELU (Exponential Linear Unit)**：类似于 ReLU，但负值部分有指数运算。
- **GELU (Gaussian Error Linear Unit)**：基于高斯误差函数的激活函数。
- **GLU (Gated Linear Unit)**：门控线性单元，对输入的一部分进行缩放。
- **Hardshrink**：硬缩放函数，当输入小于某个阈值时输出零。
- **Hardsigmoid**：Hardshrink 的变种，输出经过 sigmoid 压缩的值。
- **Hardswish**：Hardsigmoid 的变种，输出经过 swish 激活的值。
- **Hardtanh**：将输入限制在 -1 和 1 之间，类似于 tanh，但有硬性限制。
- **LogSigmoid**：Sigmoid 激活函数与对数的组合。
- **Mish**：一种新型激活函数，结合了 Sigmoid 和 Softplus。
- **Softplus**：ReLU 的平滑版本，对负值部分使用 log 函数。
- **Softshrink**：Softplus 的变种，当输入小于某个阈值时输出零。
- **Softsign**：将输入压缩到 -1 和 1 之间，类似于 tanh，但使用除法。
- **Softmin**：最小值激活函数，计算输入沿指定维度的最小值。
- **Tanhshrink**：Tanh 的变种，输出输入与双曲正切的差。
- **Threshold**：阈值激活函数，当输入超过阈值时才输出输入值
## nn.Transformer

```
"Transformer"
"TransformerEncoder"
"TransformerDecoder"
"TransformerEncoderLayer"
"TransformerDecoderLayer"
```
使用相对简单，简单示范下TransformerEncoderLayer
```python
>>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
>>> memory = torch.rand(32, 10, 512)
>>> tgt = torch.rand(32, 20, 512)
>>> out = decoder_layer(tgt, memory)
```
```python
>>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
>>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
>>> memory = torch.rand(10, 32, 512)
>>> tgt = torch.rand(20, 32, 512)
>>> out = transformer_decoder(tgt, memory)
```

## nn.Sparse
```
Embedding：
EmbeddingBag:句子级别的嵌入向量
```

```python
import torch
import torch.nn as nn
# 创建一个词嵌入层，词汇表大小为 10，嵌入维度为 3
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)
# 输入为词索引
input_indices = torch.tensor([1, 2, 3, 4])
# 获取嵌入向量
output = embedding(input_indices)
print(output)
```
- `mode`：聚合方式，可以是 `'mean'`、`'sum'` 或 `'max'`。默认是 `'mean'`。
```python
import torch
import torch.nn as nn

# 创建一个嵌入袋层，词汇表大小为 10，嵌入维度为 3
embedding_bag = nn.EmbeddingBag(num_embeddings=10, embedding_dim=3, mode='mean')

# 输入为词索引和对应的 offsets
input_indices = torch.tensor([1, 2, 3, 4])
offsets = torch.tensor([0])  # 句子的起始位置

# 获取嵌入向量并进行聚合
output = embedding_bag(input_indices, offsets)
print(output)
```