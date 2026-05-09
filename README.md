# 图像修复 GAN (Image Inpainting GAN)

本项目是一个基于 PyTorch 实现的生成对抗网络（GAN），用于图像修复任务。模型能够识别并重建图像中被随机生成的正方形遮挡（Mask）覆盖的区域。默认使用 CelebA 人脸数据集进行训练。

## 功能特性

*   **生成器 (Generator)**: 采用基于卷积神经网络的编码器-解码器 (Encoder-Decoder) 架构。包含 4 次下采样（提取特征）和 4 次上采样（恢复图像），最终使用 `Tanh` 激活函数输出修复后的图像。
*   **判别器 (Discriminator)**: 采用类似 PatchGAN 的架构，通过多层卷积对图像的局部区域进行真假判别，输出范围为 0 到 1 的概率值。
*   **损失函数组合**:
    *   **判别器**: 使用二元交叉熵损失 (BCE Loss)。
    *   **生成器**: 结合了对抗损失 (BCE Loss) 和 L1 重建损失。L1 损失的权重设置为 50，且仅针对生成图像与真实图像的遮挡区域进行计算。
*   **质量评估**: 训练过程中集成了 `torchmetrics`，可实时监控 PSNR（峰值信噪比）和 SSIM（结构相似性）指标。

## 环境依赖

在运行代码前，请确保安装了以下依赖库：


pip install torch torchvision torchmetrics numpy pillow matplotlib
注：推荐使用支持 CUDA 的 GPU 环境以加速训练。代码已包含自动检测并调用 GPU 的逻辑。
数据集准备
下载 CelebA 数据集（推荐使用 img_align_celeba 版本）。

解压数据集，确保该目录下包含 .jpg 格式的图像文件。

打开代码文件，找到主程序部分，将 img_dir 变量修改为您本地的数据集绝对路径：

Python
# 示例路径
img_dir = r'/root/autodl-tmp/img_align_celeba' 
训练参数配置
代码中设定的默认超参数如下：

输入图像尺寸: 128 x 128 像素 (预处理时自动缩放与归一化)

遮挡区域尺寸: 32 x 32 像素 (尺寸为图像的 1/4，位置随机)

批大小 (Batch Size): 64

训练轮数 (Epochs): 100

优化器: Adam

生成器学习率: 0.00005, betas: (0.5, 0.999)

判别器学习率: 0.0002, betas: (0.5, 0.999)

运行方法
配置好环境和数据集路径后，直接在终端中运行 Python 脚本即可开始训练：

Bash
python main.py
(假设您的代码文件命名为 main.py)

输出说明
在训练运行期间，代码会自动在根目录下生成以下输出：

res_img/ 目录: 存放训练过程中的可视化验证图像。每隔 10 个 Batch 保存一次。

res_{epoch}_{batch}.png: 使用 torchvision.utils 生成的拼接图像。

vis_epoch{epoch}_batch{batch}.png: 使用 matplotlib 生成的对比图，包含三行内容：真实图像 (Real)、带遮挡的图像 (Masked) 以及修复合成后的图像 (Fixed)。

traning.log 文件: 本地日志文件。记录每个打印周期的 Epoch、Batch、D loss (判别器损失) 和 G loss (生成器损失)。

控制台输出: 实时显示上述日志信息，并同步打印当前批次图像的 PSNR 和 SSIM 评估数值。
