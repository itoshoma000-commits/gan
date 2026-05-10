# 图像修复生成对抗网络 (Image Inpainting GAN)

本项目是一个基于 PyTorch 构建的生成对抗网络 (GAN)，主要用于图像修复（Image Inpainting）任务。模型会随机遮挡输入图像的一部分，并训练生成器（Generator）来预测和填补缺失的像素，同时使用判别器（Discriminator）来提升生成内容的真实度。

## 核心特性

*   **生成器 (Generator)**: 采用经典的编码器-解码器 (Encoder-Decoder) 架构。包含 4 层下采样卷积和 4 层上采样转置卷积，使用 LeakyReLU 和 ReLU 激活函数，输出层使用 Tanh。
*   **判别器 (Discriminator)**: 采用类似 PatchGAN 的架构，包含多层卷积，输出经过 Sigmoid 激活函数，用于评估图像局部的真实性（0 到 1 之间）。
*   **损失函数设计**: 结合了对抗损失 (BCE Loss) 和重建损失 (L1 Loss)。总损失计算方式为：判别真伪的 BCE Loss 加上放大 50 倍的遮挡区域 L1 重建损失，以确保细节的恢复。
*   **实时评估指标**: 训练过程中集成 `torchmetrics`，实时计算 PSNR (峰值信噪比) 和 SSIM (结构相似性)。
*   **可视化与日志**: 自动将训练进度输出到控制台并保存至 `traning.log`，定期保存生成的可视化对比图（包含真实图像、遮挡图像和修复结果）。

## 环境依赖

在运行代码之前，请确保安装了以下 Python 库：


pip install torch torchvision numpy Pillow matplotlib torchmetrics
数据集准备
代码默认使用 img_align_celeba 数据集（例如 CelebA），并期望指定目录下包含 .jpg 格式的图像文件。

准备你的图像数据集（仅需图片文件，无需标签）。

打开主程序文件，将 __main__ 中的 img_dir 变量修改为你本地数据集的绝对路径。
运行训练
直接运行 Python 脚本即可开始训练流程：

Bash
python main.py
训练参数概览
Image Size: 128 x 128

Mask Size: 32 x 32 (随机位置)

Batch Size: 64

Epochs: 100

Optimizer: Adam

Learning Rate: Generator (5e-5), Discriminator (2e-4)

Betas: (0.5, 0.999)

产出与结果记录
运行代码后，会在当前目录下生成以下内容：

traning.log: 记录训练日志，包含 Epoch、Batch、G_loss、D_loss 以及 PSNR 和 SSIM 的指标变化。

res_img/ (文件夹): 存放训练过程中的可视化结果。

res_{epoch}_{batch}.png: 拼接后的修复图像直接输出。

vis_epoch{epoch}_batch{batch}.png: 使用 Matplotlib 绘制的对比图（包含 Real、Masked、Fixed 三种状态），直观展示修复效果。系统每隔 10 个 Batch 保存一次。
