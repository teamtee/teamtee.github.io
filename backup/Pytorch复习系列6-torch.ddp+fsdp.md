[分布式深度学习训练中DP,DDP,FSDP这三者之间的区别和联系是什么](https://blog.csdn.net/Flemington7/article/details/139031199)
[Pytorch_DDP](https://pytorch.org/tutorials/beginner/ddp_series_theory.html)
[PYtorch_DDP_start](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
[Pytorch_DP](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
[Pytorch_FSDP](https://pytorch.ac.cn/tutorials/intermediate/FSDP_tutorial.html)
[DDP的好的上手教程](https://www.cnblogs.com/gzyatcnblogs/articles/17946484)
[DeepSpeed官网](https://docs.deepspeed.org.cn/en/latest/activation-checkpointing.html)
[DeepSpeed上手好教程](https://www.tutorialspoint.com/deepspeed/index.htm)
[有关Reduce\Gather\Scatter的概念文章](https://cloud.tencent.com/developer/article/2306663)
[DDP深度解析好文](https://zhuanlan.zhihu.com/p/178402798)
[有关FSDP内存消耗的绝世好文章](https://cloud.tencent.com/developer/article/2314837)
## 原理

有关分布式思想有两个概念：
- DP：数据并行
- MP：模型并行
有关分布式的实践有三个概念
- DP：数据并行(数据并行)
- DDP：分布式数据并行(数据并行)
- FSDP：完全共享式数据并行(数据并行+模型并行)
有关分布式模型并行的论文产生了几个概念：
- Zero0：不分片
- ZeRO1：只把优化器状态进行分片
- ZeRO2：对优化器状态 + 梯度进行分片
- ZeRO3：对优化器状态 + 梯度 + 模型参数进行分片
除此之外还有些概念
- 流水线并行：pipline
- 激活检查点：Activation checkpoint
- 模型卸载：model offload

## DDP

DDP和DP的区别是，DP采用中心服务器来更新模型参数、收集梯度、分发新模型，DDP则完全采用分布式的做法，采用Ring-Reduce来同步梯度。
### 原理
参考:[DDP系列第二篇：实现原理与源代码解析](https://zhuanlan.zhihu.com/p/187610959)
DDP的做法如下：
- 模型同步：建立通信后将模型同步
- 参数分组：将参数分为多个组，每组称为Bucket
- 模型训练：通过sampler使得模型训练的数据不重叠，训练获得梯度，标记对应参数为ready
- 梯度同步：某个bucket所有参数ready后会进行异步的All-reduce，同步参数
### 内存分析
参考：[有关FSDP内存消耗的绝世好文章](https://cloud.tencent.com/developer/article/2314837)
假设模型的参数量为a，按照正常的float32加载和运算，那么模型占有的内存为4a字节（float32）
静态内存：4a（模型参数）+4a（模型梯度）+8a（优化器的一阶优化和二阶优化系数)+4a(bucket 梯度) = 20a
如果按照float16加载运算：
静态内存：4a(模型参数)+2a（float16模型参数副本）+2a(模型梯度)+8a（优化器）+2a(bucket 梯度）=18a
### 实战
DDP上手参考：[DDP的好的上手教程](https://www.cnblogs.com/gzyatcnblogs/articles/17946484)

核心代码如下
```python
dist.init_process_group(backend='nccl')

dataset = SimpleDataset(X, Y)
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

model = DDP(model, device_ids=[local_rank], output_device=local_rank)

```
```python 
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. 基础模块 ### 
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        cnt = torch.tensor(0)
        self.register_buffer('cnt', cnt)

    def forward(self, x):
        self.cnt += 1
        # print("In forward: ", self.cnt, "Rank: ", self.fc.weight.device)
        return torch.sigmoid(self.fc(x))

class SimpleDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
# 2. 初始化我们的模型、数据、各种配置  ####
## DDP：从外部得到local_rank参数。从外面得到local_rank参数，在调用DDP的时候，其会自动给出这个参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

## DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

## 假设我们有一些数据
n_sample = 100
n_dim = 10
batch_size = 25
X = torch.randn(n_sample, n_dim)  # 100个样本，每个样本有10个特征
Y = torch.randint(0, 2, (n_sample, )).float()

dataset = SimpleDataset(X, Y)
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

## 构造模型
model = SimpleModel(n_dim).to(local_rank)
## DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))

## DDP: 构造DDP model —————— 必须在 init_process_group 之后才可以调用 DDP
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

## DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_func = nn.BCELoss().to(local_rank)

# 3. 网络训练  ###
model.train()
num_epoch = 100
iterator = tqdm(range(100))
for epoch in iterator:
    # DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    data_loader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in data_loader:
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label.unsqueeze(1))
        loss.backward()
        iterator.desc = "loss = %0.3f" % loss
        optimizer.step()

    # DDP:
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0 and epoch == num_epoch - 1:
        torch.save(model.module.state_dict(), "%d.ckpt" % epoch)
```


上面的启动方式：python -m torch.distributed.launch,通过参数传递
```python
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 ddp.py

```
local-rank通过环境变量传递：torchrun
```python
CUDA_VISIBLE_DEVICES="0,1" torchrun  --nproc_per_node 2 ddp.py
```



### 拓展
参考：[DDP系列第三篇：实战与技巧](https://zhuanlan.zhihu.com/p/250471767)

## FSDP
### 原理
参考：[FSDP作者本人博客动画讲解的绝世好文章](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

由动画我们可以发现FSDP的原理如下：
（1）每一个显卡储存部分参数分片：分片可以是模型参数、梯度、优化器状态
（2）在计算时，通过通信分发计算需要的分片（比如模型参数）
（3）收集结果到对应的显卡
（4）计算结束后丢弃不存储的分片

### 内存分析：
静态内存：
zero-1:4a+4a+(8a/n),节约一半内存
zero-2:4a+(12a/n),节约3/4内存
zero-3:16a/n,

### 实践
参考：[好的fsdp上手教程](https://intel.github.io/intel-extension-for-pytorch/xpu/2.1.10+xpu/tutorials/features/FSDP.html)
