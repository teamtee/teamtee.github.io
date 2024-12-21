# 前言

Pytorch计算的基本单位就是Tensor,中文名张量。在数学中，一维的量成为Scale-标量，二维的量称为Vecotr-向量，多维的量称为Tensor-张量，Pytorch沿用了这一概念，将计算的基本单位称为张量。

# Tensor是什么
下面的代码创建了一个张量

```python
import torch
a = torch.tensor((1,2,3)) # tensor([1, 2, 3])
```
Tensor具备下面的属性
- shape:张量的形状
- dtype:张量存储的数据类型
- device:张量所在的设备
- grad:梯度
```python
a.shape  # (torch.Size([3])
a.dtype  # torch.int64
a.device # device(type='cpu')
```
看上去Torch的用法和普通的Python的数组类似
- 支持索引
```python
a[1:2] # tensor([2, 3])
```
那么Tensor和Python的list有和区别
- Tensor创建后**大小无法修改**
- Tensor将本身作为**整体**参与运算
```python
a += 1 # tensor([2, 3, 4])
a == 1 # tensor(False,False,False)
```

## Tensor怎么用

## Tensor的创建
**转换创建**
- torch.tensor:接受可迭代对象或者标量
```python
torch.tensor(4)
torch.tesnor([1,3,4])
```
- torch.from_numpy:从numpy创建,注意创建后的张量和numpy共享内存，如果不希望可以用clone,也可以转化为numpy
```python
tensor = torch.from_numpy(numpy_array) # 共享内存
tensor_clone = torch.from_numpy(numpy_array).clone()
tensor.numpy()
```
**形状创建**

- torch.ones、torch.zeros
创建接受形状参数多样，包括:标量-（3），多个标量-(3,4)，可迭代对象-[3,4,5]等
```python
torch.ones(3) 
troch.ones(3,4)
```
- torch.ones_like,torch.zeors_like

**随机创建**
- torch.rand(均匀分布)
- torch.randn(正态分布)
- torch.rand_like
- torch.randn_like
参数和形状创建的torch.ones类似

```python
torch.rand(3)
```
- torch.randint(low,high,size)
- torch.randint_like
```python
torch.randint(0, 10, (2, 3))
```

可以设置随机种子
```python
torch.random.seed(42)
```
## Tensor的索引

- 切片
- 条件索引
```python
a[a > 5]
```
## Tenosr的运算

**矩阵运算**
Tensor的运算支持广播机制
矩阵乘法
- @
- torch.matmul
点乘运算
- *
- torch.multiply
```
import torch
a = torch.rand(3,4)
b = torch.rand(4,3)
c = torch.rand(3,4)
#
a @ b,torch.matmul(a,b)
#
a * c,torch.multiply(a,c)
```
- torch.max
- torch.min
- torch.sum
- torch.exp
- .T
**就地运算**
- tensor.add_()
```
a.add_(5)
a.t_
```
**逐元素操作**
- apply_:
```python
def my_func(tensor):
	return tensor * 2
x = torch.tensor([1.0, 2.0, 3.0]).apply_(my_func)
```
- torch.clip(min,max)
## Pytorch的形状
- torch.transpose：交换两个向量的维度
- torch.permute：重新排列
- torch.vew:`torch.view` 用于改变张量的形状，但要求新的视图在内存中是连续的
	- torch.contiguous
- torch.cat

```
torch.cat((a,b),dim=0)
```


