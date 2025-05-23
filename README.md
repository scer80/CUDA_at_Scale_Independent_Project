# CUDA at Scale Independent Project

## Contents

1. [Docker :package:](#docker)
2. [Data :floppy_disk:](#data)
3. [Train :muscle:](#train)


### Docker :package: <a name="docker"></a>

**Build docker image**
```shell
make build
```
**Open shell in docker container**
```shell
make shell
```

### Data :floppy_disk: <a name="data"></a>

#### Download MNIST data

```shell
make download
```

#### Visualize a sample

**Build `mnist_export`**

```shell
make mnist_export
```

**Execute `mnist_export`**

```shell
./mnist_export --dataset train --index 1 --out out
```

### Train :muscle: <a name="train"></a>

**Build `mnist_train`**

```shell
make mnist_train
```

**Execute `mnist_export`**

```shell
./mnist_train --lr 1e-3 --epochs 10 --batch_size 50
```
