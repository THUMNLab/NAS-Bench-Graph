# NAS-Bench-Graph

This repository provides the official codes and all evaluated architectures for NAS-Bench-Graph, a tailored benchmark for graph neural architecture search.


## Usage 
First, read the benchmark of a certain dataset by specifying the name. The nine supported datasets are: cora, citeseer, pubmed, cs, physics, photo, computers, arxiv, and proteins. For example, for the Cora dataset:
```
from readbench import lightread
bench = lightread('cora')
```
The data is stored as a `dict` in Python.

Then, an architecture needs to be specified by its macro space and opertions.
We consider the macro space as a directed acyclic graph (DAG) and constrain the DAG to have only one input node for each intermediate node. Therefore, the macro space can be specificed by a list of integers, indicating the input node index for each computing node (0 for the raw input, 1 for the first computing node, etc.). Then, the operations can be specified by a list of strings with the same length. For example, we provide the code to specify the architecture in the following figure:
![arch](https://user-images.githubusercontent.com/17705534/173767528-eda1bc64-f4d8-4da1-a0e9-8470f55ccc6a.png)

```
from architecture import Arch
arch = Arch([0, 1, 2, 1], ['gcn', 'gin', 'fc', 'cheb'])
```

Notice that we assume all leaf nodes (i.e., nodes without descendants) are connected to the output, so there is no need to specific the output node. Besides, the list can be specified in any order, e.g., the following code can specific the same architecture:
```
arch = Arch([0, 1, 1, 2], ['gcn', 'cheb', 'gin', 'fc'])
```

The the benchmark data can be obtained by a look-up table. In this repository, we only provide the validation and test performance, the latency, and the number of parameters as follows:

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

For the complete benchmark, please first download the data from https://figshare.com/articles/dataset/NAS-bench-Graph/20070371, which contains the training/validation/testing performance at each epoch. Since we run each dataset with three random seeds, each dataset has 3 files, e.g.,

```
from readbench import read
bench = read('cora0.bench')   # cora1.bench and cora2.bench 
```

The full metric for any epoch can be obtained as follows.
```
info = bench[arch.valid_hash()]
epoch = 50
info['dur'][epoch][0]   # training performance
info['dur'][epoch][1]   # validation performance
info['dur'][epoch][2]   # testing performance
info['dur'][epoch][3]   # training loss
info['dur'][epoch][4]   # validation loss
info['dur'][epoch][5]   # testing loss
info['dur'][epoch][6]   # best performance
```

# Example usage of NNI and AutoGL
NAS-Bench-Graph can be used together with other libraries such AutoGL and NNI.

For the usage of [AutoGL](https://github.com/THUMNLab/AutoGL), please refer to the [agnn branch](https://github.com/THUMNLab/AutoGL/tree/agnn).

You can also refer to `runnni.py` to use the benchmark together with [NNI](https://github.com/microsoft/nni/).


