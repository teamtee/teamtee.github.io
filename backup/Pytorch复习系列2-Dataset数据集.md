# 前言

Dataset是存储数据的集合，。DataLoader则是让我们以不同的方式从Dataset中加载数据的集合。Sampler指定了冲DataLoader中加载数据的方式。

参考
[1.Github源码](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataset.py)
[2.Pytorch的Doc](https://pytorch.org/docs/stable/data.html)
[3.Blog](https://www.cnblogs.com/marsggbo/p/11308889.html)
# Dataset
通常我们使用的Dataset都是Map形式的dataset，即为我们给定index获得对应的数据，当然也有Iterable的类型。
## Map-style的Dataset

Map-style的定义如下:通过泛型类Generic定义的一个抽象接口类，要求子类必须重写__getitem__方法，即为给定index获取数据
```python
class Dataset(Generic[_T_co]):
    def __getitem__(self, index) -> _T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")
    ...
```

实例化使用很简单
- Dataloader来加载数据就必须重写__len__方法，
```python
from torch.utils.data import Dataset
class MyDataset(Dataset):
	def __init__(self,data):
		super().__init__()
		self.data = data
	def __getitem__(self, index):
		return self.data[index]
	def __len__(self):
		return len(self.data)

data = range(1,10)
testDataset = MyDataset(data)
testDatset[2]
```
## Iterable的Dataset
这样的Dataset必须返回一个可迭代对象，并且多线程和单线程的加载行为并不一致
```python
class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         worker_info = torch.utils.data.get_worker_info()
        ...         if worker_info is None:  # single-process data loading, return the full iterator
        ...             iter_start = self.start
        ...             iter_end = self.end
        ...         else:  # in a worker process
        ...             # split workload
        ...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        ...             worker_id = worker_info.id
        ...             iter_start = self.start + worker_id * per_worker
        ...             iter_end = min(iter_start + per_worker, self.end)
        ...         return iter(range(iter_start, iter_end))
```

## Dataset实例

StackDataset：这个类别在拥有多个数据的Dataset的时候特别有用
- 支持元祖和字典的输入
```python
>>> images = ImageDataset()
>>> texts = TextDataset()
>>> tuple_stack = StackDataset(images, texts)
>>> tuple_stack[0] == (images[0], texts[0])
>>> dict_stack = StackDataset(image=images, text=texts)
>>> dict_stack[0] == {'image': images[0], 'text': texts[0]}
```

- ConcatDataset：可以合并两个数据集合为一个
```python
>>> images1 = ImageDataset()
>>> images2 = ImageDataset()
>>> image = StackDataset(images1, image2)
```


- Subset:可以通过索引来取得Dataset的子集,索引支持任何可迭代对象
```python
>>> images = ImageDataset()
>>> subimages1 = Subset(images,range(1,10))
```
- random_split:将数据集按照长度划分为子集合，为了重复还能够指定generate
```python
 >>> generator1 = torch.Generator().manual_seed(42)
>>> generator2 = torch.Generator().manual_seed(42)
>>> random_split(range(10), [3, 7], generator=generator1)
>>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)
```
- ChainDataset：合并多个Iterable的Dataset
```python
from torch.utils.data import ChainDataset, IterableDataset

# 假设我们有两个 IterableDataset
class DatasetA(IterableDataset):
    def __iter__(self):
        yield from [1, 2, 3]

class DatasetB(IterableDataset):
    def __iter__(self):
        yield from [4, 5, 6]

# 使用 ChainDataset 将两个数据集合并
chain_dataset = ChainDataset([DatasetA(), DatasetB()])

# 迭代 ChainDataset
for data in chain_dataset:
    print(data)
```
-
# DataLoader

DataLoader简单的参数如下
- dataset
- batch_size=1
- shuffle=False
- num_workers=0
- collate_fn=None,
- pin_memory=False
- drop_last=False,
下面还有一些较为复杂的参数
## collate_fn
- collate_fn：collate_fn的作用是打包
```python
def my_collate(batch):
    # batch是一个列表，包含了一个批次的所有样本
    # 这里我们简单地将它们堆叠成一个tensor
    return torch.stack(batch, dim=0)
```
```python
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def collate_fn(batch):
    # batch是一个包含多个样本的列表，每个样本是一个元组(sentence, label)
    sentences, labels = zip(*batch)  # 解压
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)  # 填充
    return sentences, labels

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```
- sampler
sampler=None,
batch_sampler=None
## sampler

在PyTorch中，`sampler`是用于控制`DataLoader`如何从数据集中抽取样本的一个重要组件。以下是一些PyTorch中常用的`sampler`的示例：

- RandomSampler
`RandomSampler`用于从数据集中随机抽取样本，可以是有放回的也可以是无放回的。
```python
from torch.utils.data import RandomSampler

# 假设dataset是你的数据集
sampler = RandomSampler(dataset, replacement=True, num_samples=100)
```
在这里，`replacement=True`表示有放回采样，`num_samples=100`表示你想要采样的数量。如果不指定`num_samples`，则默认为数据集的大小。
- SequentialSampler
`SequentialSampler`用于顺序地从数据集中抽取样本。

```python
from torch.utils.data import SequentialSampler

sampler = SequentialSampler(dataset)
```
它会按照数据集的索引顺序返回样本。

- SubsetRandomSampler

`SubsetRandomSampler`用于从一个数据集的子集随机抽取样本。

```python
from torch.utils.data import SubsetRandomSampler

indices = list(range(100, 200))  # 假设我们想要从第100个到第200个样本中随机抽取
sampler = SubsetRandomSampler(indices)
```

这个`sampler`会从指定的索引范围内随机抽取样本。

- WeightedRandomSampler

`WeightedRandomSampler`用于根据给定的权重从数据集中抽取样本。

```python
from torch.utils.data import WeightedRandomSampler

weights = [0.1, 0.2, 0.7]  # 假设有三个样本，权重不同
sampler = WeightedRandomSampler(weights, num_samples=10, replacement=True)
```

在这里，`weights`是每个样本的权重，`num_samples=10`表示你想要采样的数量，`replacement=True`表示有放回采样。
- BatchSampler
`BatchSampler`用于将其他`sampler`得到的单个索引值合并成一个batch。

```python
from torch.utils.data import BatchSampler

batch_size = 32
sampler = BatchSampler(sampler=RandomSampler(dataset), batch_size=batch_size, drop_last=False)
```

`drop_last=False`表示如果最后一批样本数量小于`batch_size`，也会被保留。

这些`sampler`可以应用于`DataLoader`中，以控制数据加载时的采样策略。通过使用不同的`sampler`，你可以实现不同的数据抽样方法，以适应不同的训练需求。