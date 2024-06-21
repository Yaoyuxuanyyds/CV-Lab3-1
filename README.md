<h1 align = "center">Self-Supervised vs Supervised</h1>



### 项目概述

​	**本项目实现了 SimCLR 自监督学习算法，并分别在 Tiny ImageNet 和 CIFAR 10 数据集上尝试训练ResNet-18，随后在CIFAR-100数据集上使用 Linear Classification Protocol 对其性能进行评测。**

​	**同时在ImageNet数据集上采用监督学习训练得到的 ResNet18 在相同的 Protocol 下进行评测，并比较二者相对于在 CIFAR-100 数据集上从零开始以监督学习方式进行训练所带来的提升。**



### 目录结构
项目包含以下主要文件和脚本：
- `view.py`：包含用于生成图像多个视图的 `ViewGen` 类。
- `utils.py`：包含实用函数，例如保存检查点、保存配置文件和计算准确性。
- `transforms.py`：定义了自定义图像变换，包括高斯模糊。
- `get_dataset.py`：管理数据集加载和变换。
- `simclr.py`：主要的 SimCLR 框架，包括训练逻辑。
- `train.ipynb`：用于训练 SimCLR 模型的 Jupyter notebook。
- `protocol_test.ipynb`：用于测试 SimCLR 框架的 Jupyter notebook。



### Requirement
- Python 3.x
- PyTorch
- torchvision
- numpy
- yaml
- tensorboard

#### 安装
1. 克隆仓库：
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```
2. 安装所需依赖：
   ```sh
   pip install -r requirements.txt
   ```



### 项目目录

- `view.py`：
  - 定义了 `ViewGen` 类，用于生成图像的多个随机裁剪视图。

- `utils.py`：
  - `save_checkpoint`：保存模型检查点。
  - `save_config_file`：保存配置文件。
  - `accuracy`：计算预测结果的准确率。

- `transforms.py`：
  - 定义了 `GaussianBlur` 类，用于对图像应用高斯模糊。

- `get_dataset.py`：
  - `GetTransformedDataset` 类：提供数据集加载和数据变换的功能。
  - `get_cifar100_data_loaders`：获取 CIFAR-100 数据集的数据加载器。

- `simclr.py`：
  - `simclr_framework` 类：包含 SimCLR 模型的训练逻辑。
  - `info_nce_loss` 方法：计算对比学习损失。
  - `train` 方法：训练 SimCLR 模型。



### 使用说明

1. 准备数据集：将数据集下载并放置在指定目录下。
2. 训练模型：运行 `train.ipynb` notebook 文件以开始训练 SimCLR 模型。
3. 测试模型：使用 `protocol_test.ipynb` notebook 文件进行模型测试。



---

​								*本项目为 Fudan University Computer Vision 课程作业，作者：律己zZZ*