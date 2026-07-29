# FemtoDet-v3

基于 MMDetection 3.x 重构的 [FemtoDet](https://github.com/yh-pengtu/FemtoDet) 非官方实现。

论文：

* [FemtoDet: An Object Detection Baseline for Energy Versus Performance Tradeoffs](https://arxiv.org/abs/2301.06719)
* ICCV 2023

## 项目简介

FemtoDet 是一个面向极低参数量和低能耗场景设计的目标检测模型，用于研究检测精度、模型规模、计算开销与能耗之间的权衡。

原作者代码基于 MMDetection 2.x 实现。由于 MMDetection 3.x 在配置系统、数据处理流程、模型注册机制及训练接口等方面进行了较大调整，原始代码无法直接运行在 MMDetection 3.x 环境中。

本项目基于 **MMDetection 3.2.0** 对原始 FemtoDet 代码进行了适配和重构，使其能够在 MMDetection 3.x 框架下完成训练、验证和推理。

> 本项目为非官方实现，与 FemtoDet 原作者及 OpenMMLab 官方无直接关联。

## 当前进展

* [ ] 发布 Pascal VOC 2007+2012 数据集上的训练结果和模型权重
* [ ] 进一步重构代码，降低模块间耦合
* [ ] 补充训练、验证及推理教程
* [x] 完成模型精度对齐
* [x] 完成 MMDetection 3.2.0 适配

目前发现使用 [`femtodet_0stage_yolox.py`](configs/femtoDet/femtodet_0stage_yolox.py) 训练得到的精度明显高于 [`femtodet_0stage.py`](configs/femtoDet/femtodet_0stage.py)，相关配置差异及精度变化原因仍在进一步分析中。

## 环境依赖

建议按照 [MMDetection 3.2.0 官方安装文档](https://github.com/open-mmlab/mmdetection/tree/v3.2.0) 安装以下依赖：

* Python
* PyTorch
* MMCV
* MMEngine
* MMDetection 3.2.0

建议使用独立的 Conda 或 Python 虚拟环境安装本项目，以避免依赖冲突。

## 安装方法

克隆本项目并安装：

```bash
git clone https://github.com/ZS-YANG/FemtoDet-v3.git
cd FemtoDet-v3
pip install -v -e .
```

安装完成后，可通过以下命令确认 MMDetection 是否能够正常导入：

```bash
python -c "import mmdet; print(mmdet.__version__)"
```

## 数据准备

### 1. 下载数据集

本项目主要在 Pascal VOC 2007 和 Pascal VOC 2012 数据集上训练和验证。

数据集下载地址：

* [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

默认情况下，项目假设数据集存放在 `data/` 目录下，目录结构示例如下：

```text
FemtoDet-v3/
├── data/
│   └── VOCdevkit/
│       ├── VOC2007/
│       │   ├── Annotations/
│       │   ├── ImageSets/
│       │   ├── JPEGImages/
│       │   └── SegmentationClass/
│       └── VOC2012/
│           ├── Annotations/
│           ├── ImageSets/
│           ├── JPEGImages/
│           └── SegmentationClass/
├── configs/
├── tools/
└── ...
```

### 2. 转换数据集格式

使用以下脚本将 Pascal VOC 数据集转换为 MMDetection 所需的数据格式：

```bash
python tools/dataset_converters/pascal_voc.py \
    data/VOCdevkit \
    --out-dir data/VOCdevkit
```

转换完成后，根据实际数据集路径修改 [`configs/femtoDet`](configs/femtoDet) 目录下配置文件中的数据集路径。

重点检查以下配置项：

```python
data_root = 'data/VOCdevkit/'
```

同时确认训练集、验证集和测试集的标注文件路径配置正确。

### 3. 下载预训练权重

FemtoDet 使用在 ImageNet-1K 数据集上预训练的 Backbone 权重进行初始化。

预训练权重下载地址：

* [百度网盘](https://pan.baidu.com/s/1JGsvlvzPkb5nxGBaRSD7ng?pwd=hx8k)
* 提取码：`hx8k`

下载完成后，将权重文件保存到本地目录，例如：

```text
checkpoints/
└── femtodet_backbone_imagenet1k.pth
```

随后在配置文件中修改预训练权重路径：

```python
load_from = 'checkpoints/femtodet_backbone_imagenet1k.pth'
```

具体配置字段应以对应模型配置文件为准。

## 模型配置

当前提供的主要配置文件如下：

```text
configs/femtoDet/
├── femtodet_0stage.py
├── femtodet_0stage_yolox.py
├── femtodet_1stage.py
├── femtodet_2stage.py
└── femtodet_3stage.py
```

不同 Stage 配置对应不同的模型结构及精度、参数量和计算开销组合。

## 模型训练

以 `femtodet_0stage.py` 为例，启动单卡训练：

```bash
python tools/train.py \
    configs/femtoDet/femtodet_0stage.py
```

指定工作目录：

```bash
python tools/train.py \
    configs/femtoDet/femtodet_0stage.py \
    --work-dir work_dirs/femtodet_0stage
```

使用多卡训练：

```bash
bash tools/dist_train.sh \
    configs/femtoDet/femtodet_0stage.py \
    8
```

其中，`8` 表示参与训练的 GPU 数量。

## 模型验证

使用训练得到的模型权重进行验证：

```bash
python tools/test.py \
    configs/femtoDet/femtodet_0stage.py \
    work_dirs/femtodet_0stage/best_coco_bbox_mAP_epoch_xx.pth
```

实际权重文件名请根据训练输出结果进行替换。

## 训练日志可视化

MMDetection 训练过程中会在工作目录中生成训练日志和可视化文件。

默认输出目录示例：

```text
work_dirs/femtodet_0stage/
```

可根据所使用的可视化后端，通过 TensorBoard 或其他工具查看训练过程。

## VOC 数据集结果

以下结果来自 FemtoDet 原作者在 Pascal VOC 数据集上的训练结果。

| Detector | Stage | Parameters | Box AP50 | Config                                                      |
| -------- | ----: | ---------: | -------: | ----------------------------------------------------------- |
| FemtoDet |     0 |    68.77 K |     37.1 | [`femtodet_0stage.py`](configs/femtoDet/femtodet_0stage.py) |
| FemtoDet |     1 |    68.77 K |     40.4 | [`femtodet_1stage.py`](configs/femtoDet/femtodet_1stage.py) |
| FemtoDet |     2 |    68.77 K |     44.4 | [`femtodet_2stage.py`](configs/femtoDet/femtodet_2stage.py) |
| FemtoDet |     3 |    68.77 K |     46.5 | [`femtodet_3stage.py`](configs/femtoDet/femtodet_3stage.py) |

> 当前表格展示的是原作者公布的参考精度。本项目基于 MMDetection 3.2.0 复现得到的完整训练结果和模型权重将在后续发布。

## 已知问题

* 不同配置文件之间仍存在部分重复代码，需要进一步解耦。
* `femtodet_0stage_yolox.py` 与 `femtodet_0stage.py` 的训练精度存在较大差异，具体原因仍在分析。
* 当前尚未发布基于本项目代码训练得到的完整模型权重。
* 部分训练和推理流程仍需补充更详细的使用文档。

## 致谢

本项目基于以下开源项目进行开发：

* [FemtoDet](https://github.com/yh-pengtu/FemtoDet)
* [MMDetection](https://github.com/open-mmlab/mmdetection)
* [MMEngine](https://github.com/open-mmlab/mmengine)
* [MMCV](https://github.com/open-mmlab/mmcv)

感谢原作者和 OpenMMLab 社区提供的优秀工作。

## 引用

本项目仅对 FemtoDet 进行 MMDetection 3.x 适配和非官方重构。

在研究工作中使用本项目时，请引用 FemtoDet 原论文：

```bibtex
@inproceedings{Tu_2023_ICCV,
    author    = {Tu, Peng and Xie, Xu and Ai, Guo and Li, Yuexiang and Huang, Yawen and Zheng, Yefeng},
    title     = {FemtoDet: An Object Detection Baseline for Energy Versus Performance Tradeoffs},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
    month     = {October},
    year      = {2023},
    pages     = {13318--13327}
}
```

```bibtex
@misc{tu2023femtodet,
    title         = {FemtoDet: An Object Detection Baseline for Energy Versus Performance Tradeoffs},
    author        = {Peng Tu and Xu Xie and Guo Ai and Yuexiang Li and Yawen Huang and Yefeng Zheng},
    year          = {2023},
    eprint        = {2301.06719},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV}
}
```

## License

本项目的许可证请参考仓库中的 `LICENSE` 文件。原始 FemtoDet、MMDetection、MMCV 和 MMEngine 代码分别遵循其对应的开源许可证。
