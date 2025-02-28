[Pytorch分布式文章](https://zhuanlan.zhihu.com/p/178402798)-推荐
# 简介

PyTorch的分布式训练允许在多个GPU或多台机器上并行训练模型，显著提升训练速度和扩展性。其核心是通过多进程协作处理数据、模型或优化任务。

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

# 原理

## DDP

### 原理
DDP是数据并行

参考:[DDP系列第二篇：实现原理与源代码解析](https://zhuanlan.zhihu.com/p/187610959)
DDP的做法如下：
- 模型同步：建立通信后将模型同步
- 参数分组：将参数分为多个组，每组称为Bucket
- 模型训练：通过sampler使得模型训练的数据不重叠，训练获得梯度，标记对应参数为ready
- 梯度同步：某个bucket所有参数ready后会进行异步的All-reduce，同步参数
### 内存分析
参考：[有关FSDP内存消耗的绝世好文章](https://cloud.tencent.com/developer/article/2314837)

**全精度训练**：float32加载和运算
假设模型的参数量为a，那么模型占有的内存为4a字节（float32）
静态内存：4a（模型参数）+4a（模型梯度）+8a（优化器的一阶优化和二阶优化系数)+4a(bucket 梯度) = 20a
**半精度训练**：float16加载和运算
静态内存：4a(模型参数)+2a（float16模型参数副本）+2a(模型梯度)+8a（优化器）+2a(bucket 梯度）=18a
## FSDP
FSDP是模型并行+数据并行
### 原理
参考：[FSDP作者本人博客动画讲解的绝世好文章](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
参考：[讲解文章](http://shiyanjun.cn/archives/2292.html)
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

# 实战

分布式训练通常涉及到下面的问题，
- 采用哪一种**分布式**训练方法：DDP/FSDP：
- 采用哪个**框架**进行分布式训练：Pytorch、Deepseed
- 采用哪种方式进行训练：**单机多卡/多机多卡**：
## Pytorch框架启动
参考：[一文读懂分布式训练启动方式](https://zhuanlan.zhihu.com/p/675464874)
Pytorch主要有三种启动方式，不同的启动方式的区别在于如何传递参数
- 手动使用`torch.multiprocessing.spawn`：参数写死在代码里，不推荐
- 使用`torch.distributed.launch`：参数通过函数参数传递，必须使用`argparse`，而且必须有一个`--local-rank`参数，不推荐
- 使用`torchrun`：参数通过环境变量传递，推荐
分布式训练包含下面的概念
- `node`:节点，即机器数，每个机器下面可以有多个进程
- `world`: 总的进程数，通常一个进程一个GPU，因此可以理解为GPU数目
- `rank`: 进程的唯一标识符id
- `local_rank`:进程在本地（本机器）的唯一标识符
### 训练启动
训练启动就是提供必要的信息给程序

#### torch.distributed.launch
采用这种方式时，主程序必须有下面的代码,从参数中获取local_rank
```
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
args = parser.parse_args()
local_rank = args.local_rank
```

单机多卡启动
```
python -m torch.distributed.launch \
    --nproc_per_node=4 \  # 每台机器的GPU数
    --nnodes=1 \          # 总机器数
    train_script.py
```
多机多卡启动
```
# 主节点（假设IP为192.168.1.1）
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=1234 \
    train_script.py
# 从节点
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=1234 \
    train_script.py
```

#### torchrun
所有信息从环境变量中获得
```
os.environ['RANK'] 可以得到在所有机器所有进程中当前GPU的排序
os.environ['LOCAL_RANK'] 可以得到在当前node中当前GPU的排序
os.environ['WORLD_SIZE'] 可以得到GPU的数量
```

单机多卡启动
```
torchrun \
	--nnodes 1 \
	--nproc_per_node 8 \
	--master_port=29502 \
	train.py
```

多机多卡启动
```
# 主节点（假设IP为192.168.1.1）
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=1234 \
    train_script.py

# 从节点
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=1234 \
    train_script.py

```
#### 常用的训练启动环境变量

- `CUDA_VISIBLE_DEVICES`:`export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` ：设置本机器可见的GPU，默认为全部设备，昇腾为`ASCEND_VISIBLE_DEVICES`
- `MASTER_ADDR`：主节点的IP地址
- `MASTER_PORT`：主节点的通信端口
- `TORCH_DISTRIBUTED_DEBUG`：可以设置为INFO或DETAIL，以输出更多调试信息
####  启动实例
[容器多机多卡训练](https://www.cnblogs.com/syw-home/p/18073062)
### 代码适配
分布式训练需要对原本的代码做三件事情
#### 初始化通信

`torch.distributed.init_process_group`用于初始化分布式通信后端，下面时一些参数
- **`backend`**：指定分布式通信后端，例如`nccl`（适用于GPU）、`gloo`（适用于CPU或GPU）
- **`init_method`**：指定初始化方法，可以是`env://`（默认）、`file://`或`tcp://`,指定为`env://`时会从环境变量中读取,当然没有显示指定下面的参数的情况下默认就是`env://`，而torchrun会设置环境变量

```python
import torch.distributed as dist

def setup(rank,local_rank, world_size):
    dist.init_process_group(
        backend='nccl',    # 使用NCCL后端（GPU场景）
        init_method='env://',  # 从环境变量读取配置
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(local_rank)  # 绑定当前GPU

## pytorch.distribute.launch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int) 
parser.add_argument("--local_rank", type=int) 
parser.add_argument("--world_size", type=int)
args = parser.parse_args()
rank = args.rank
local_rank = args.local_rank
world_size = args.world_size
```

```python
## torchrun
import os 
rank = int(os.environ['RANK'] )
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

def setup(rank,local_rank, world_size):
    dist.init_process_group(
        backend='nccl',    # 使用NCCL后端（GPU场景）
    )
    torch.cuda.set_device(local_rank)  # 绑定当前GPU
setup(rank,world_size)
```
#### 数据集适配
``` python
## 数据集合
## 构造
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
```
#### 模型适配

```python

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

##  必须init_process_group 之后才可以调用 
model.to(device)
model = DDP(model, device_ids=[local_rank])
model = FSDP(model, device_id=local_rank)

## 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
```


##### DDP参数

- `find_unused_parameters`：
	- - 如果设置为`True`，`DDP`会在每次迭代中检查模型中是否有未使用的参数。如果有未使用的参数，`DDP`会重新构建梯度图，以确保所有参数都能参与梯度计算。
	- 这个参数在某些动态图模型（如某些Transformer模型）中非常有用，因为这些模型可能会在不同的迭代中使用不同的参数。
	- **注意**：启用`find_unused_parameters=True`可能会增加额外的计算开销，因此建议仅在需要时启用
- `gradient_as_bucket_view`：
	- - 如果设置为`True`，`DDP`会将梯度视为一个连续的内存块（bucket），而不是分散的张量。这可以减少内存占用，提高通信效率。
	- 从PyTorch 1.9开始支持，建议在支持的环境中启用。
- `broadcast_buffers`：
	- 是否在每次迭代开始时广播模型的缓冲区（如`BatchNorm`的运行均值和方差）。如果模型中包含`BatchNorm`层，建议设置为`True`
##### FSDP 参数 (注意，我在使用过程中一直没有调通)
我们在使用 FSDP 时，需要通过配置 auto_wrap_policy 参数来选择模型分片策略，不然显存优化只能达到 ZeRO-stage1 的水准
- `auto_wrap_policy`
	- 自动包装策略，用于决定哪些子模块需要被FSDP包装。`my_auto_wrapping_policy`是一个自定义的包装策略，通常基于子模块的参数数量或其他条件来决定是否对子模块进行分片
```python
import functools
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
```
- `cpu_offload`
	- 是否将部分参数和梯度卸载到CPU，以进一步减少GPU显存占用。虽然会增加通信开销，但可以显著提高内存效率
```python
from torch.distributed.fsdp import CPUOffload
cpu_offload = CPUOffload(offload_params=True)
```
- `mixed_precision`
	- 混合精度策略，用于控制模型的参数、梯度和优化器状态的精度。`mixed_precision_policy`是一个自定义的混合精度策略

```python
from torch.distributed.fsdp.mixed_precision import MixedPrecision
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.float16,  # 模型参数的精度
    buffer_dtype=torch.float16,  # 模型缓冲区的精度
    reduce_dtype=torch.float16,  # 梯度归约的精度
    backward_dtype=torch.float16,  # 反向传播的精度
    keep_low_precision_grads=True  # 是否保持梯度的低精度
)
```
- `sharding_strategy`
	- 定义参数分片的策略，例如`FULL_SHARD`（完全分片）或`SHARD_GRAD_OP`（仅梯度分片）
		- `ShardingStrategy.FULL_SHARD`:全分片
		- `ShardingStrategy.HYBRID_SHARD`：混合策略，介于全分片和下面的分片
		- `ShardingStrategy.SHARD_GRAD_OP`：仅对梯度和优化器状态进行分片
		- `ShardingStrategy.NO_SHARD：不分片
		- 
```python
from torch.distributed.fsdp import ShardingStrategy
sharding_strategy = ShardingStrategy.FULL_SHARD 
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=sharding_strategy,
    device_id=torch.cuda.current_device(),
)
```

- `limit_all_gathers`:
    - **说明**：是否限制所有`all_gather`操作。设置为`True`可以减少通信开销，但可能会影响某些操作的性能。
-  `sync_module_states`:
    - **说明**：是否在初始化时同步模块状态。在某些情况下，可以减少初始化时的通信开销。
- `param_init_fn`:
    - **说明**：参数初始化函数。在某些情况下，可以用于在初始化时将模型参数移动到特定设备。
```python
param_init_fn = lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
```
#### 模型加载与保存
最好只加载一次
```python
if dist.get_rank() == 0:
    model.load_state_dict(torch.load(ckpt_path))
if dist.get_rank() == 0:
	torch.save(model.module.state_dict(), "%d.ckpt" % epoch)
```
### 分布式技巧
[DDP系列第三篇：实战与技巧](https://zhuanlan.zhihu.com/p/250471767)

#### SyncBN
一句话总结，当前PyTorch SyncBN只在DDP单进程单卡模式中支持
```python
# DDP init
dist.init_process_group(backend='nccl')

# 按照原来的方式定义模型，这里的BN都使用普通BN就行了。
model = MyModel()
# 引入SyncBN，这句代码，会将普通BN替换成SyncBN。
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

# 构造DDP模型
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

#### 梯度累加

```python
model = DDP(model)

for 每次梯度累加循环
    optimizer.zero_grad()
    # 前accumulation_step-1个step，不进行梯度同步，累积梯度。
    for _ in range(K-1)::
        with model.no_sync():
            prediction = model(data)
            loss = loss_fn(prediction, label) / K
            loss.backward()  # 积累梯度，不应用梯度改变
    # 第K个step，进行梯度同步
    prediction = model(data)
    loss = loss_fn(prediction, label) / K
    loss.backward()  # 积累梯度，不应用梯度改变
    optimizer.step()
```

#### 进程同步

```python
code_before()
# 在这一步同步
torch.distributed.barrier()
code_after()
```

**在某个进程中执行A操作，其他进程等待其执行完成后再执行B操作：**

```python
if rank == 0:
    do_A()
    torch.distributed.barrier()
else:
    torch.distributed.barrier()
    do_B()
```

#### 避免冗余输出

```python
import logging

# 给主要进程（rank=0）设置低输出等级，给其他进程设置高输出等级。
logging.basicConfig(level=logging.INFO if rank in [-1, 0] else logging.WARN)
# 普通log，只会打印一次。
logging.info("This is an ordinary log.")
# 危险的warning、error，无论在哪个进程，都会被打印出来，从而方便debug。
logging.error("This is a fatal log!")
```

#### 保证DDP性能：确保数据的一致性

我们需要给不同的进程分配不同的、固定的随机数种子：

```python
def main():
    rank = torch.distributed.get_rank()
    # 问题完美解决！
    init_seeds(1 + rank)
```

设置sampler的随机种子(实际种子为seed+epoch)
```python
for epoch in iterator:
    # DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    data_loader.sampler.set_epoch(epoch)
```

## Deepspeed框架启动
[参考教程](https://github.com/bobo0810/LearnDeepSpeed)
[参考示例](https://github.com/deepspeedai/DeepSpeedExamples/blob/master/training/pipeline_parallelism/train.py)
[绝对入门的好教程](https://www.tutorialspoint.com/deepspeed/deepspeed-optimizer.htm)
[官方简短的教程](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)

#### 启动
[详细讲解](https://blog.csdn.net/weixin_42486623/article/details/132793261)
##### deepspeed命令启动
###### 单机多卡
Deepspeed会同时设置环境变量和传递参数
```
deepspeed  --num_gpus 8 train.py --deepspeed  # 不指定--num_gpus 8则会使用所有的显卡
```
指定GPU
```
deepspeed --include localhost:1
```
###### 单/多机多卡 ssh连接
[参考](https://blog.csdn.net/weixin_42486623/article/details/132793261)
首先在每一台机器上安装必要的库
```
apt install pdsh sshd
```
建立一个hostfile,内容为机器ip+显卡数,如果只有一条，等效于单机多卡
```
x.x.x.x slots=8
x.x.x.x slots=8
```
确保每台机器都能够通过ssh免密连接
```
ssh-keygen 
echo id_rsa.pub > known_hosts
```
在每台机器上启动sshd服务，然后在主机器上运行
```
# 主机器
deepspeed \
	--hostfile $HOST_FILE \
	--ssh_port $SSH_PORT \
	train.py
```
###### 单/多机多卡-无ssh(适配k8s)
在每台机器上都即启动该命令，类是torchrun启动
```
deepspeed --hostfile=myhostfile --no_ssh --node_rank=<n> \
    --master_addr=<addr> --master_port=<port> \
    <client_entry.py> <client args> 
```
##### 指定配置文件

deepspeed支持3种方式指定配置文件
- ~~命令行指定：(似乎已经被丢弃)~~
```
deepspeed   train.py --deepspeed --deepspeed_config ds_config.json
```
- 模型初始化时指定路径：
```python
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    config="path",
```
- 模型初始化时传递字典：
```python
deepspeed_config = {
    "train_batch_size": 8,
    "gradient_accumulation_steps": 4,  # 
    "steps_per_print": 2,            # 新增关键参数
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001
        }
    },
    "pipeline": {
        "activation_checkpoint_interval": 1  # 启用激活检查点
    },
    "fp16": {  # 可选：添加混合精度支持
        "enabled": True
    }
}
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    config=deepspeed_config,
```
#### 代码适配
##### 初始化通信

```
deepspeed.init_distributed(
dist_backend='hccl', # 使用NCCL后端（GPU场景）
)
```
##### 适配模型

```python
    model.to(device)
    parameters = filter(lambda x:x.requires_grad,model.parameters())
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
    config=cfg.train.deepspeed_config,
    model=model,
    model_parameters=parameters
)
```
##### 数据集适配
如果在上一步适配模型处没有指定training_data
``` python
## 数据集合
## 构造
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
```

##### 模型训练
```
loss = model_engine.train(**batch) /如果初始化
model_engine.backward(loss)
model_engine.step()
```
##### 模型保存
```python
client_sd = {"step": step, "epoch": epoch}
model_engine.save_checkpoint(args.save_dir, ckpt_id=step, client_sd=client_sd)

# Load checkpoint
_, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
```
#### 模型配置

[参考](https://zhuanlan.zhihu.com/p/654925843)
```python
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "fp16": {
        "enabled": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0.0,
            "warmup_max_lr": 0.001,
            "warmup_num_steps": 100
        }
    },
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "reduce_scatter": true,
        "allgather_partitions": true
    }
}
```
##### 混合精度
```python
"fp16": {
  "enabled": true,
  "loss_scale": 0,
  "loss_scale_window": 1000,
  "hysteresis": 2,
  "min_loss_scale": 1
}
"bf16": { "enabled": true }
```
##### 性能分析
DeepSpeed 提供了训练过程中不同部分所花费时间的详细分
```
"wall_clock_breakdown": true,
```
当启用激活检查点时，可以在 `deepspeed_config` 文件中启用对每个检查点函数的前向和反向时间的分析。
```
{
  "activation_checkpointing": {
    "profile": true
  }
}
```
DeepSpeed 深度性能分析器测量 PyTorch 模型的耗时、浮点运算次数和参数数量，并显示哪些模块或层是瓶颈。
```
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    }
}
```
DeepSpeed 监视器将实时训练指标记录到一个或多个监控后端，包括 PyTorch 的 TensorBoard、WandB 或直接记录到 CSV 文件
```
{
  "tensorboard": {
    "enabled": true,
    "output_path": "output/ds_logs/",
    "job_name": "train_bert"
  }
  "wandb": {
    "enabled": true,
    "team": "my_team",
    "group": "my_group",
    "project": "my_project"
  }
  "csv_monitor": {
    "enabled": true,
    "output_path": "output/ds_logs/",
    "job_name": "train_bert"
  }
}
```
DeepSpeed 提供了对在 `deepspeed.comm` 中启动的所有通信操作的日志记录
```
{
  "comms_logger": {
    "enabled": true,
    "verbose": false,
    "prof_all": true,
    "debug": false
  }
```
##### 自动调参
[参考](https://www.deepspeed.ai/training/)

DeepSpeed 自动调优器使用模型信息、系统信息和启发式方法来高效调整 Zero 阶段、微批大小和其他 Zero 配置。使用自动调优功能不需要 DeepSpeed 用户进行代码更改。
```
{
  "autotuning": {
    "enabled": true,
    "results_dir": null,
    "exps_dir": null,
    "overwrite": false,
    "metric": "throughput",
    "num_nodes": null,
    "num_gpus": null,
    "start_profile_step": 3,
    "end_profile_step": 5,
    "fast": true,
    "num_tuning_micro_batch_sizes": 3,
    "tuner_type": "model_based",
    "tuner_early_stopping": 5,
    "tuner_num_trials": 50,
    "arg_mappings": null
  }
}
```
#### 示例
##### 数据并行
```python
import torch
import deepspeed

# Define a simple neural network model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Initialize DeepSpeed configuration
deepspeed_config = {
    "train_batch_size": 64,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001
        }
    }
}

# Initialize model
model = SimpleModel()

# Initialize DeepSpeed for distributed data parallelity
model_engine, optimizer, _, _ = deepspeed.initialize(
    config=deepspeed_config,
    model=model
)

# Dummy data
inputs = torch.randn(64, 784)
labels = torch.randint(0, 10, (64,))

# Forward pass
outputs = model_engine(inputs)
loss = torch.nn.functional.cross_entropy(outputs, labels)

# Backward pass and optimization
model_engine.backward(loss)
model_engine.step()
```

##### Pipeline
[参考](https://github.com/bobo0810/LearnDeepSpeed/blob/main/training/pipeline_parallelism/train.py)
下面是一个最小的流水线示例，
```python
import torch
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
import os

# 注意：华为昇腾（Ascend）芯片需使用'hccl'后端，NVIDIA GPU使用'nccl'
deepspeed.init_distributed(dist_backend='hccl')  # 假设使用NVIDIA GPU

# DeepSpeed配置需添加流水线并行参数
deepspeed_config = {
    "train_batch_size": 8,
    "gradient_accumulation_steps": 4,  # 
    "steps_per_print": 2,            # 新增关键参数
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001
        }
    },
    "pipeline": {
        "activation_checkpoint_interval": 1  # 启用激活检查点
    },
    "fp16": {  # 可选：添加混合精度支持
        "enabled": True
    }
}
# 修改数据集以包含标签
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_size=1000, input_dim=784, output_dim=10):
        self.data = torch.randn(data_size, input_dim)
        self.labels = torch.randn(data_size, output_dim)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])  # 返回输入和标签的元组

# 定义模型层
class SimpleLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return torch.relu(self.fc(x))

# 构建流水线模型
layers = [
    LayerSpec(SimpleLayer, 784, 128),
    LayerSpec(SimpleLayer, 128, 10)
]

pipeline_model = PipelineModule(
    layers=layers,
    loss_fn=torch.nn.CrossEntropyLoss(),
    num_stages=2,          # 流水线阶段数需等于GPU数
    partition_method='uniform',  # 均匀划分层到各个阶段
)
dataset = SimpleDataset(data_size=10240)

# 初始化DeepSpeed引擎
model_engine, optimizer, _, _ = deepspeed.initialize(
    config=deepspeed_config,
    model=pipeline_model,
    model_parameters=pipeline_model.parameters(),
    training_data=dataset
)

# 准备数据加载器

for step in range(100): 
    loss = model_engine.train_batch()
```

```bash
deepspeed \
	--num_gpus 2 \
	./src/test.py
```

如果需要自定义`dataloader`
```python
# 初始化DeepSpeed引擎
model_engine, optimizer, _, _ = deepspeed.initialize(
    config=deepspeed_config,
    model=pipeline_model,
    model_parameters=pipeline_model.parameters(),
)

# 准备数据加载器
from deepspeed.utils import RepeatingLoader
datasetloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=deepspeed_config["train_batch_size"])
dataloader = RepeatingLoader(dataloader)  # 转为无限循环的迭代器
data_iter = iter(dataloader)
for step in range(100): 
    loss = model_engine.train_batch(data_iter=dataiter)
```

```
##### sheduler

```python
import torch.nn as nn
import torch.optim as optim

# Model definition
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Initialize model and optimizer
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# DeepSpeed configuration for optimizer and scheduler
ds_config = {
    "train_batch_size": 8,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.01,
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0.001,
            "warmup_max_lr": 0.01,
            "warmup_num_steps": 100
        }
    }
}

# Initialize DeepSpeed with model and optimizer
model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, optimizer=optimizer, config_params=ds_config)

# Sample input and forward pass
inputs = torch.randn(8, 10)
outputs = model_engine(inputs)
loss = outputs.mean()

# Backward pass and step
model_engine.backward(loss)
model_engine.step()
lr_scheduler.step()
```