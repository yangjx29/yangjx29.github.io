---
title: Pytorch-Sampler类学习笔记
abbrlink: 1510da230
date: 2024-11-28 10:36:12
updated: 2024-11-28 10:36:13
categories: ML
tags:
mathjax: true
toc: true
---
<meta name="referrer" content="no-referrer"/>

# Pytorch-Sampler类学习笔记

## 前言

我们在训练神经网络时，如果数据量太大，无法一次性将数据放入到网络中进行训练，所以需要进行分批处理数据读取。这一个问题涉及到如何从数据集中进行读取数据的问题，pytorch框提供了Sampler基类与多个子类实现不同方式的数据采样。子类包括：

>__all__ = [
>
>  "BatchSampler",
>
>  "RandomSampler",
>
>  "Sampler",
>
>  "SequentialSampler",
>
>  "SubsetRandomSampler",
>
>  "WeightedRandomSampler",
>
>]

它决定了在训练过程中如何从数据集（`Dataset`）中选择样本

## *1.基类Sampler*

```python
class Sampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source: Optional[Sized] = None) -> None:
        if data_source is not None:
            import warnings

            warnings.warn("`data_source` argument is not used and will be removed in 2.2.0."
                          "You may still have custom implementation that utilizes it.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError
   

```

* 对于所有的采样器来说，都需要继承采样器类，**必须实现的方法为_iter_()**，也就是定义迭代器行为，返回可
  选代对象。除此之外，采样器类并没有定义任何其它的方法

## *2、顺序采样Sequential Sampler*

```python
class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)
```

* 顺序采样类并没有定义过多的方法，其中初始化方法仅仅需要一个Dataset类对象作为参数。
  对于 len ()只负责返回数据源包含的数据个数；iter()方法负责返回一个可迭代对象，这个可选代对象是
  由range产生的顺序数值序列，也就是说选代是按照顺序进行的。
* **常用于验证集或测试集上**，因为测试过程中我们通常不需要对数据进行打乱，按照顺序采样即可。

前述几种方法都只需要self.data source实现了 len ()方法，因为这几种方法都仅仅使用了
len(self.data source)函数。
所以下面采用同样实现了 len()的list类型来代替Dataset类型做测试:

```python
# 定义数据和对应的采样器
data = list([1, 2, 3, 4, 5])
seq_sampler = sampler.SequentialSampler(data_source=data)
# 迭代获取采样器生成的索引
for index in seq_sampler:
    print("index: {}, data: {}".format(str(index), str(data[index])))
>>>
index: 0, data: 1
index: 1, data: 2
index: 2, data: 3
index: 3, data: 4
index: 4, data: 5


```



## *3、随机采样RandomSampler*

```python
class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples
```

* __iter__()方法，定义了核心的索引生成行为，其中if replacement判断处返回了两种随机值，根据是否在初始化中给出replacement参数决定是否重复采样，区别核心在于randint()函数生成的随机数学列是可能包含重复数值的,而randperm()函数生成的随机数序列是绝对不包含重复数值的
* `RandomSampler` **从数据集中随机选择样本，**且每个样本被选择的概率是相等的。通常用于打乱数据集中的样本顺序，特别是在训练阶段。每个样本的选择都是独立且均匀的。

下面分别测试是否使用replacement作为输入的情况，首先是不使用时:

```python 
ran_sampler = sampler.RandomSampler(data_source=data)
for index in ran_sampler:
    print("index: {}, data: {}".format(str(index), str(data[index])))

index: 3, data: 4
index: 4, data: 5
index: 2, data: 3
index: 1, data: 2
index: 0, data: 1

```

可以看出生成的随机索引是不重复的，下面是采用replacement参数的情况

```python
ran_sampler = sampler.RandomSampler(data_source=data, replacement=True)
for index in ran_sampler:
    print("index: {}, data: {}".format(str(index), str(data[index])))

index: 1, data: 2
index: 2, data: 3
index: 4, data: 5
index: 3, data: 4
index: 1, data: 2

```

此时生成的随机索引是有重复的（1出现两次），也就是说第“1”条数据可能会被重复的采样。

## *4.子集随机采样Subset Random Sampler*

```python 
class SubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(len(self.indices), generator=self.generator):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)
```

* 上述代码中len()的作用与前面几个类的相同,依旧是返回数据集的长度,区别在于iter()返回的并不是
  随机数序列，而是**通过随机数序列作为indices的索引**，进而返回打乱的数据本身。需要注意的仍然是采样是不重复的，也是通过randperm()函数实现的。按照网上可以搜集到的资料，**Subset Random sampler应该用于训练集、测试集和验证集的划分**，下面将data划分为train和val两个部分,再次指出iter()返回的的不是索引,而是索引对应的数据:
* 可以在指定的索引子集中进行随机采样，这样你可以控制哪些数据被用于训练或验证，而不是整个数据集。

```python
print('***********')
sub_sampler_train = sampler.SubsetRandomSampler(indices=data[0:2])
for index in sub_sampler_train:
    print("index: {}".format(str(index)))
print('------------')
sub_sampler_val = sampler.SubsetRandomSampler(indices=data[2:])
for index in sub_sampler_val:
    print("index: {}".format(str(index)))
    
# train：
index: 2
index: 1
# val：
index: 3
index: 4
index: 5

```

## *5.加权随机采样WeightedRandomSampler*

```python
class WeightedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.

    Example:
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """

    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={num_samples}")
        if not isinstance(replacement, bool):
            raise ValueError(f"replacement should be a boolean value, but got replacement={replacement}")

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError("weights should be a 1d sequence but given "
                             f"weights have shape {tuple(weights_tensor.shape)}")

        self.weights = weights_tensor
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples
```

* 对于Weighted Random Sampler类的 init()来说，replacement参数依旧用于控制采样是否是有放回的;
  num sampler用于控制生成的个数;weights参数对应的是“样本”的权重而不是“类别的权重”。其中 iter_()方法返回的数值为随机数序列，只不过生成的随机数序列是按照weights指定的权重确定的
* `WeightedRandomSampler` 按照给定的样本权重随机采样。每个样本的选择概率是与其权重成正比的。它通常用于数据集不平衡的情况，赋予少数类样本更大的权重，以增加其被采样的机会。在处理类别不平衡的数据时，可以通过设置每个样本的权重，使得少数类样本有更高的采样概率，帮助模型学习到更好的分类边界。

```python
# 加权随机采样
data=[1,2,5,78,6,56]
# 位置为[0]权重为0.1，位置为[1] 权重为0.2
weights=[0.1,0.2,0.3,0.4,0.8,0.3,5]
rsampler=sampler.WeightedRandomSampler(weights=weights,num_samples=10,replacement=True)

for index in rsampler:
    print("index: {}".format(str(index)))

index: 5
index: 4
index: 6
index: 6
index: 6

```

## *6、批采样BatchSampler*

```python
class BatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
```

通过将 `Sampler` 和批次大小结合，`BatchSampler` 提供了一种高效的批量采样方式。它的返回值是一个批量样本的索引序列：

```python 
seq_sampler = sampler.SequentialSampler(data_source=data)
batch_sampler = sampler.BatchSampler(seq_sampler, 4, False)
print(list(batch_sampler))

[[0, 1, 2, 3], [4, 5]]


```

