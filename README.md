# NAS-Bench-Graph

Graph neural architecture search (GraphNAS) has recently aroused considerable attention in both academia and industry. However, two key challenges seriously hinder the further research of GraphNAS. First, since there is no consensus for the experimental setting, the empirical results in different research papers are often not comparable and even not reproducible, leading to unfair comparisons. Secondly, GraphNAS often needs extensive computations, which makes it highly inefficient and inaccessible to researchers without access to large-scale computation. To solve these challenges, we propose NAS-Bench-Graph, a tailored benchmark that supports unified, reproducible, and efficient evaluations for GraphNAS. Specifically, we construct a unified, expressive yet compact search space, covering 26,206 unique graph neural network (GNN) architectures and propose a principled evaluation protocol. To avoid unnecessary repetitive training, we have trained and evaluated all of these architectures on nine representative graph datasets, recording detailed metrics including train, validation, and test performance in each epoch, the latency, the number of parameters, etc. Based on our proposed benchmark, the performance of GNN architectures can be directly obtained by a look-up table without any further computation, which enables fair, fully reproducible, and efficient comparisons.

## Usage 

You can use the benchmark by this repository.
At first, read the benchmark of a certain dataset.

```
from readbench import lightread
bench = lightread('cora')
```

The data is stored as a `dict` in the benchmark.
To obtain the data, you declare an architecture by specifying its macro space and opertions.
![arch](https://user-images.githubusercontent.com/17705534/173767528-eda1bc64-f4d8-4da1-a0e9-8470f55ccc6a.png)


```
from hpo import Arch
arch = Arch([0, 1, 2, 1], ['gcn', 'gin', 'fc', 'cheb'])
```

The macro space is described by a list of integers, indicating the input feature map of each layer (0 for the raw input, 1 for the feature map of the 1st layer, etc.)
You can declare the architecture by any topological order. You can also declare the architecture as follows.

```
arch = Arch([0, 1, 1, 2], ['gcn', 'cheb', 'gin', 'fc'])
```

The you can get the data by the look-up table.
In this repository, you can only obatain valid/test performance, latency and number of parameters. Refer to the next part if you want infomation of training process.

```
info = bench[arch.valid_hash()]
# valid performance
info['valid_perf']
# test performance
info['perf']
# latency
info['latency']
# number of parameters
info['para']
```

# Obtain training process information

Please download the data from https://figshare.com/articles/dataset/NAS-bench-Graph/20070371 
Since we run each dataset for 3 times, each dataset is corresponding to 3 files.
Choose one file to read

```
from readbench import read
bench = read('cora0.bench')
```

Get training process information of an architecture at a certain epoch.

```
info = bench[arch.valid_hash()]
epoch = 50
# training performance, validation performance, test performance, training loss, validation loss, test loss, and current best test performance at epoch 50 
print(info['dur'][epoch])
```

# Example usage of NNI and AutoGL
You can refer to `runnni.py` to use methods in NNI.

For usage of `AutoGL`, please refer to the `agnn` branch in `AutoGL`.
