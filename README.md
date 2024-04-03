## Overview

This is the PyTorch implementation of the paper [CSI-Based MIMO Indoor Positioning Using Attention-Aided Deep Learning](https://ieeexplore.ieee.org/document/10325508).
If you feel this repo helpful, please cite our paper:

```
@article{wan2024csibased,
  title={CSI-Based MIMO Indoor Positioning Using Attention-Aided Deep Learning},
  author={Wan, Rongjie and Chen, Yuxing and Song, Suwen and Wang, Zhongfeng},
  journal={IEEE Communications Letters},
  year={2024},
  volume={28},
  number={1},
  pages={53-57},
  publisher={IEEE}
  doi={10.1109/LCOMM.2023.3335408}
}
```


## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.9
- [PyTorch >= 1.12](https://pytorch.org/get-started/locally/)
- [thop](https://github.com/Lyken17/pytorch-OpCounter)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is from [CTW2019](https://data.ieeemlc.org/Ds1Detail).
For the sake of program simplicity, the data has been pre-divided in the code and the data file has been named as random.h5. If you wish to perform dataset partitioning on your own, you can refer to /utils/data.py. Alternatively, you can uncomment lines 67-77 in /dataset/CTW2019.py and comment out lines 78-83 to perform dataset partitioning while running the code.
You can also use the data from [KU Leuven ultra dense indoor MAMIMO CSI dataset](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset).

You can generate your own dataset according to the [DeepMIMO](https://www.deepmimo.net/) as well. The details of data pre-processing can be found in their website.

#### B. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── CLNet  # The cloned CLNet repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── main.py
├── CTW2019  # The data folder
│   ├── random.mat
│   ├── ...
├── Experiments
│   ├── run.sh  # The bash script
...
```

## Train CLNet from Scratch

An example of run.sh is listed below. Simply use it with `sh run.sh`. It starts to train CLNet from scratch. Change scenario using `--scenario`, and change dataset type using '--datatype'. Change '--nc' from 3 to 2 if using "KU Leuven" dataset.

``` bash
python /home/CLNet/main.py \
  --data-dir '/home/CTW2019' \
  --scenario 'random' \
  --epochs 300 \
  --batch-size 32 \
  --workers 8 \
  --nc 3 \
  --scheduler cosine \
  --gpu 0 \
  --datatype 'CTW2019' \
  2>&1 | tee log.out
```


## Acknowledgment

This repository is modified from the [CRNet open source code](https://github.com/Kylin9511/CRNet). Thanks Zhilin for his amazing work.
Thanks Maximilian Arnold for providing the pre-processed CTW2019 dataset.
Thanks SibrenDe Bast and SofiePollin group for providing the pre-processed KU Leuven ultra dense indoor MAMIMO CSI dataset.
 you can find their related work in [CSI-based Positioning in Massive MIMO systems using Convolutional Neural Networks](https://github.com/sibrendebast/MaMIMO-CSI-positioning-using-CNNs) and [MaMIMO CSI-Based Positioning using CNNs: Peeking inside the Black Box](https://github.com/sibrendebast/inside-the-black-box)
