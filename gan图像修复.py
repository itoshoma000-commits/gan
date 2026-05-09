import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms,utils
import numpy as np
import os
from PIL import Image
import logging
import matplotlib.pyplot as plt

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


'''配置日志记录'''
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s',
#保存文件，输出控制台
    handlers=[logging.FileHandler("traning.log"),logging.StreamHandler()]
)

#读取文件
class Imgset(Dataset):
    def __init__(self,root,transform=None):
        self.root = root
        self.transform = transform
        self.image=[ f for f in os.listdir(root) if f.endswith('jpg')]#防止标签读取报错
    def __len__(self):
            return len(self.image)#样本数
    def __getitem__(self,index):
        try:
            imgpath=os.path.join (self.root,self.image[index])
            img = Image.open(imgpath).convert('RGB')#RGB打开
            if self.transform :
                img=self.transform(img)#预处理
                return img
        except  Exception as e:
            logging.error(f"Error loading image {self.image[index]}:{e}")
            return self.__getitem__(0)
    #遮挡生成
def maskcreat(imgsize):
    mask=torch.zeros((1,imgsize,imgsize))
    w,h=imgsize//4,imgsize//4
    x=np.random.randint(0,imgsize-w)
    y=np.random.randint(0,imgsize-h)
    mask[:,y:y+h,x:x+w]=1
    return mask
class Generator(nn.Module):
    def __init__ (self):
        super(Generator,self).__init__()
        #下采样
        def block(inchannel,outchannel,normalize=True):
            layers = [nn.Conv2d(inchannel,outchannel,4,2,1)]
            if normalize:
                layers.append(nn.BatchNorm2d(outchannel))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        #上采样
        def unblock(inchannel,outchannel,):
            return nn.Sequential(
                nn.ConvTranspose2d(inchannel,outchannel,4,2,1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU()
            )
        #编码
        self.down1=block(3,64,False)
        self.down2=block(64,128)
        self.down3 = block(128, 256)
        self.down4 = block(256,  512)
        #解码
        self.up1=unblock(512,256)
        self.up2=unblock(256,128)
        self.up3=unblock(128,64)
        #输出
        self.up4=nn.Sequential(nn.ConvTranspose2d(64,3,4,2,1),nn.Tanh())
    def forward(self,x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        up1 = self.up1(down4)
        up2 = self.up2(up1)
        up3 = self.up3(up2)
        return self.up4(up3)
class Discriminator(nn.Module):
    def __init__ (self):
        super(Discriminator,self).__init__()
        #卷积层
        def layer2(inchannel,outchannel,stride=2):
            return nn.Sequential(nn.Conv2d(inchannel,outchannel,4,stride,1),nn.BatchNorm2d(outchannel),nn.LeakyReLU(0.2))
        #patchGAN
        self.model=nn.Sequential(
            layer2(3,64),
            layer2(64,128),
            layer2(128,256),
            layer2(256,512,1),
            nn.Conv2d(512,1,4,1,1),
            nn.Sigmoid()#0-1
        )
    def forward(self,x):
        return self.model(x)

def visualize(real, masked, fixed, epoch, batch_idx, output_dir, num=4):
    num = min(num, real.size(0))

    #指向CPU 转为numpy
    real = real.detach().cpu()
    masked = masked.detach().cpu()
    fixed = fixed.detach().cpu()

    #反归一
    def denorm(x):
        x = x * 0.5 + 0.5
        x = torch.clamp(x, 0, 1)
        return x.numpy()

    real_np = denorm(real)
    masked_np = denorm(masked)
    fixed_np = denorm(fixed)

    #参数对齐
    real_np = np.transpose(real_np, (0, 2, 3, 1))
    masked_np = np.transpose(masked_np, (0, 2, 3, 1))
    fixed_np = np.transpose(fixed_np, (0, 2, 3, 1))


    fig, axes = plt.subplots(3, num, figsize=(num * 2.5, 7.5))

    # 兼容 num=1 时的维度问题
    if num == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(num):
        #real
        axes[0, i].imshow(real_np[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Real')
        #masked
        axes[1, i].imshow(masked_np[i])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Masked')

        #fixed
        axes[2, i].imshow(fixed_np[i])
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Fixed')

    plt.tight_layout()
    #防止重复读取

    plt.savefig(os.path.join(output_dir, f'vis_epoch{epoch}_batch{batch_idx}.png'))
    plt.close(fig)

if __name__ == "__main__":


    #写入子文件夹
    output_dir = 'res_img'
    os.makedirs(output_dir, exist_ok=True)
    #指向gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #实例化生成器和判别器
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    #定义优化器学习率和动量参考
    optG = optim.Adam(netG.parameters(), lr=0.00005, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    l1_loss = nn.L1Loss()#损失函数

    #图像预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),#缩放128
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])#归一化


    #加载数据集
    img_dir = r'/root/autodl-tmp/img_align_celeba'#指向jpg文件
    dataset =Imgset(root=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    print(f"Dataset Size: {len(dataset)}")


    #循环训练
    for epoch in range(100):
        for i, imgs in enumerate(dataloader):
            real_imgs = imgs.to(device)
            batch_size = imgs.size(0)


            #当前batch生成遮挡
            masks = torch.stack([maskcreat(128) for _ in range(batch_size)]).to(device)
            masked_imgs = real_imgs * (1 - masks)


            #生成器训练部分
            optG.zero_grad()
            gen_imgs = netG(masked_imgs)#修复后图像
            d_out_fake = netD(gen_imgs)#判别器评价

            #二元分类
            valid = torch.ones_like(d_out_fake).to(device)
            fake = torch.zeros_like(d_out_fake).to(device)

            #计算总损失函数
            g_loss = criterion(d_out_fake, valid) + 50 * l1_loss(gen_imgs * masks, real_imgs * masks)
            g_loss.backward()
            optG.step()

            #判别器训练部分
            optD.zero_grad()
            real_loss = criterion(netD(real_imgs), valid)
            fake_loss = criterion(netD(gen_imgs.detach()), fake)#datch避免梯队传回
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optD.step()


            #初始化指标
            psnr_metric = PeakSignalNoiseRatio().to(device)
            ssim_metric = StructuralSimilarityIndexMeasure().to(device)
            #PSNR SSIM
            psnr_val = psnr_metric(gen_imgs, real_imgs)
            ssim_val = ssim_metric(gen_imgs, real_imgs)

            #10批次一轮
            if i % 10 == 0:
                logging.info(f"[Epoch {epoch}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
                comp_imgs = masked_imgs + gen_imgs * masks
                utils.save_image(comp_imgs[:4], os.path.join(output_dir, f"res_{epoch}_{i}.png"), normalize=True)
                visualize(real_imgs, masked_imgs, comp_imgs, epoch, i, output_dir, num=4)
                print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
