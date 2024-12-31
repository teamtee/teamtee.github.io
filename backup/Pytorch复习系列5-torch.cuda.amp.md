# 前言

混合精度训练的核心观点：**采用更低精度的类型进行运算会使用更少的内存和更快的速度**
必须采用Tensor core的核心显卡： GPU 中的 Tensor Core 天然支持 FP16 乘积的结果与 FP32 的累加
## 原理
[Mixed Precision Training](https://arxiv.org/abs/1710.03740)

[有关参数的讲解的好文章](https://www.53ai.com/news/finetuning/2024083051493.html)
[有关torch.cuda.amp的好文章](https://zhuanlan.zhihu.com/p/348554267)
[讲解DeepSpeed的好文章](https://basicv8vc.github.io/posts/zero/)
[有关FSDP内存消耗的绝世好文章](https://cloud.tencent.com/developer/article/2314837)
## 参数类型
模型在保存的时候通常有下面四种类型
- fp32
- tf32
- fp16
- bf16
![image](https://github.com/user-attachments/assets/2211ccf2-b62a-4de8-8eac-4fbda5d599cc)
我们需要区分下面的概念，保存类型通常时预训练模型已经指定好的，加载类型我们可以指定，在运算时模型会自动将将运算的类型转换为模型的加载类型

- 保存类型：
- 加载类型：
- 运算类型:
指定加载类型
```python
from transformers import AutoModel
# 加载模型时指定参数类型为float16
model = AutoModel.from_pretrained('bert-base-uncased', torch_dtype=torch.float16)
# 模型运算时，如果使用GPU，会自动使用对应的参数类型进行计算
# 例如，在NVIDIA GPU上，float16运算会使用Tensor Cores加速
```
指定加载类型，并且量化
```python
from transformers import AutoModel
from bitsandbytes as bnb

# 指定量化配置
量化配置 = bnb.QuantizationConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_use_double_quant=False,
)

# 加载并量化模型
model = AutoModel.from_pretrained(
    'bert-base-uncased',
    quantization_config=量化配置,
)
```
混合精度运算的核心思想：采用较高精度的参数类型加载模型，但是运算时将一些运算转化为低精度的参数类型来加快**训练和运算**，具体转化什么算子由pytorch自动决定。


## 梯度缩放
然而低精度的参数类型可能出现溢出的现象，这会导致下面的情况
（1）模型前向为Nan：无法采用低精度,必须用高精度
（2）模型前向正常loss为Nan：无法采用低精度,必须用高精度
（3）模型前向正常loss正常梯度为Nan：采用scaler对模型的loss进行缩放之后在反向传播
也就是说梯度在计算出loss后缩放，计算出梯度后还原。

如何理解溢出现象，查看下面的论文原图，如果太小会导致近0溢出，直接被近似为0，因此缩放的关键是对梯度乘以一个放大系数，再得到梯度之后再缩放回去。

![image](https://github.com/user-attachments/assets/4de8eefb-a14c-4680-b1cf-3ed05046dd7e)

Pytorch的GradScaler的放大系数是自适应的，并且会自动跳过nan值
```python
# 用scaler，scale loss(FP16)，backward得到scaled的梯度(FP16)
        scaler.scale(loss).backward()
        # scaler 更新参数，会先自动unscale梯度
        # 如果有nan或inf，自动跳过
        scaler.step(optimizer)
        # scaler factor更新
        scaler.update()
```
## 内存分析

假设模型的参数量为a，按照正常的float32加载和运算，那么模型占有的内存为4a字节（float32）
静态内存：4a（模型参数）+4a（模型梯度）+8a（优化器的一阶优化和二阶优化系数） = 16a
动态内存：4b（激活检查点）

![image](https://github.com/user-attachments/assets/dfbde05d-644b-4be2-a5b7-bb35b57339f3)
按照上面的混合精度训练：
静态内存：4a(模型参数)+2a（float16模型参数副本）+2a(模型梯度)+8a（优化器）=16a
动态内存：2b（激活检查点）


也就是说，从静态内存来看，使用混合精度训练并不能减少内存，从这篇博文的实验看：[有关FSDP内存消耗的绝世好文章](https://cloud.tencent.com/developer/article/2314837)。因为存在float16到float32的互相转化，模型可能缓存副本，导致内存反而更大,可以关闭缓存
```python
with autocast(device_type='cuda', cache_enabled=False):
```


## 实战

使用混合精度运算非常简单
```python
autocast = torch.cuda.amp.autocast
scaler = torch.cuda.amp.GradScaler()
with autocast(dtype=torch.bfloat16):
# with autocast():
	outputs = model(batch)
	scaler.scale(loss).backward()
	scaler.step(optimizer)
	scaler.update()
```
多进程情况下需要在模型的forward和backward函数下加装饰器，因为autocast是线程本地的
```python
MyModel(nn.Module):
    ...
    @autocast()
    def forward(self, input):
```
累积梯度
```python
# scale 归一的loss 并backward  
scaler.scale(loss).backward()

if (i + 1) % iters_to_accumulate == 0:

	scaler.step(optimizer)
	scaler.update()
	optimizer.zero_grad()
```

## 拓展