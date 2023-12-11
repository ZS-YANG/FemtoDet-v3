# FemtoDet-v3

[FemtoDet](https://github.com/yh-pengtu/FemtoDet)的非官方实现

paper:[Femtodet: an object detection baseline for energy versus performance tradeoffs](https://arxiv.org/abs/2301.06719)

## 写在前面

原作者代码基于 mmdet2.x，由于mmdet3.x进行了大量更新，原始代码不能直接应用于mmdet3.0的项目中，因此基于mmdet3.2.0对原始代码进行了重构
初始代码基于

## To list

- [ ] 发布在VOC0712数据集上的训练结果
- [ ] 对代码进行进一步重构解耦
- [ ] 更新集教程文档
- [ ] 精度对齐(使用[femtodet_0stage_yolox.py](configs%2FfemtoDet%2Ffemtodet_0stage_yolox.py)
  的训练结果比[femtodet_0stage.py](configs%2FfemtoDet%2Ffemtodet_0stage.py)高很多，需要找到原因)

### Dependencies and Installation

按照[mmdetection](https://github.com/open-mmlab/mmdetection/tree/v3.2.0)的要求安装torch 和 mmcv

之后克隆本项目，进行mmdet的安装

```commandline
git clone https://github.com/ZS-YANG/FemtoDet-v3.git
cd FemtoDet-v3
pip install .
```

### Preparation

1. Download the dataset.

   We mainly train FemtoDet on [Pascal VOC 0712](http://host.robots.ox.ac.uk/pascal/VOC/), you should firstly download
   the datasets. By default, we assume the dataset is stored in ./data/.

2. Dataset preparation.
   使用[pascal_voc.py](tools%2Fdataset_converters%2Fpascal_voc.py)
   转换数据集格式，转换完成后配置[femtoDet](configs%2FfemtoDet)中的数据集路径

3. Download the initialized models.

   We trained our designed backbone on ImageNet 1k, and used it
   for [the inite weights](https://pan.baidu.com/s/1JGsvlvzPkb5nxGBaRSD7ng?pwd=hx8k))(hx8k) of FemtoDet.

### Results (trained on VOC) and Models

原始作者训练精度

| Detector | Params | box AP50 | Config                                |
|----------|--------|----------|---------------------------------------|
|          |        | 37.1     | ./configs/femtoDet/femtodet_0stage.py |
| FemtoDet | 68.77k | 40.4     | ./configs/femtoDet/femtodet_1stage.py |
|          |        | 44.4     | ./configs/femtoDet/femtodet_2stage.py |
|          |        | 46.5     | ./configs/femtoDet/femtodet_3stage.py |

### References

If you find the code useful for your research, please consider citing:

```bib
@InProceedings{Tu_2023_ICCV,
    author    = {Tu, Peng and Xie, Xu and Ai, Guo and Li, Yuexiang and Huang, Yawen and Zheng, Yefeng},
    title     = {FemtoDet: An Object Detection Baseline for Energy Versus Performance Tradeoffs},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {13318-13327}
}
@misc{tu2023femtodet,
      title={FemtoDet: An Object Detection Baseline for Energy Versus Performance Tradeoffs}, 
      author={Peng Tu and Xu Xie and Guo AI and Yuexiang Li and Yawen Huang and Yefeng Zheng},
      year={2023},
      eprint={2301.06719},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```