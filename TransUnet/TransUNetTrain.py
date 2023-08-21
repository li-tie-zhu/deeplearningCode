# -*- codeing = utf-8 -*-
# @Time : 2023/6/16 11:04
# @Author : 李铁柱
# @File : UNetTrain.py
# @Software: PyCharm

import os
import time

import torch
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.transforms import transforms, Grayscale
from transunet import *
import matplotlib.pyplot as plt
import  numpy as np

os.environ['OPENBLAS_NUM_THREADS'] = '1'


class CreatDataSet(Dataset):
    def __init__(self,rootPath,transform=None,transform_label=None):
        self.rootPath = rootPath
        self.transform = transform
        self.transform_label = transform_label
        self.dataset = self.load_dataset()

    def load_dataset(self):
        data_set = []
        label_mapping = {}
        for idx,subdir in enumerate(sorted(os.listdir(self.rootPath))):
            subdir_path = os.path.join(self.rootPath, subdir)
            if os.path.isdir(subdir_path):
                label_mapping[subdir] = idx
                #templist = []
                transformed_img=None
                transformed_img_label = None
                for file_name in sorted(os.listdir(subdir_path)):
                    file_path = os.path.join(subdir_path, file_name)
                    #print(file_path)
                    if file_name.endswith('.png') and '_mask' not in file_name:
                        image = Image.open(file_path)
                        transformed_img = self.transform(image)
                        image_name=file_name
                        #templist.append(transformed_img)
                    elif file_name.endswith('_mask.png'):
                        image = Image.open(file_path)
                        image_lable = file_name
                        transformed_img_label = self.transform_label(image)  # 对标签只需要resize成一样大小，不需要做和输入图片一样的处理
                        #templist.append(transformed_img_label)
                        #data_set.append(templist)
                        #templist = []
                    if transformed_img != None and transformed_img_label != None:
                        data_set.append((transformed_img,transformed_img_label))
                        transformed_img = None
                        transformed_img_label = None
                    # else:
                    #     print(f"Skipping an image and its label: {file_path}")
        return data_set

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image,label= self.dataset[item]
        return image, label


data_tranform =transforms.Compose([
    transforms.Resize((128,128)),
    Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图像
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])

data_tranform_label =transforms.Compose([
    transforms.Resize((128,128)),
    Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图像
    transforms.ToTensor(),
])

rootPath = '/home/postgraduate/huchao/deeplearning2/CNN/Dataset_BUSI_with_GT'

data_set = CreatDataSet(rootPath,data_tranform,data_tranform_label)

train_ratio = 0.8
train_size = int(train_ratio*len(data_set))
test_size = len(data_set)-train_size
print(len(data_set))
print(train_size,test_size)

train_dataset,test_dataset = random_split(data_set,[train_size,test_size])
train_dataloader =DataLoader(train_dataset,batch_size=32,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False)

print(len(train_dataset))
print(len(test_dataset))
image,target = data_set[208]
print(image.shape)
transposed_image = np.transpose(image, (1,2,0))
image = image.squeeze()
target = target.squeeze()
plt.figure()
plt.imshow(image,cmap='gray')
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(target,cmap='gray')
plt.axis('off')
plt.show()
#plt.imshow(target)

class DiceLossSoftmax(nn.Module):
    def __init__(self, num_classes=2):
        super(DiceLossSoftmax, self).__init__()
        self.epsilon = 1e-5
        self.num_classes = num_classes

    def forward(self, predict, target):
        with torch.no_grad():
            target = F.one_hot(target.long(), self.num_classes)
            target = torch.Tensor.permute(target, [0, 4, 2, 3, 1])
            target = target[:, :, :, :, 0]
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        #         pre = torch.sigmoid(predict).view(num, -1)
        pre = predict.view(num, predict.size(1), -1)
        tar = target.view(num, target.size(1), -1)

        intersection = (pre * tar).sum(-1)
        union = (pre + tar).sum(-1)

        score = 1 - (2 * (intersection).sum(-1) / (union + self.epsilon).sum(-1))
        return score.mean()

def loss_function(pre, mask, DICELossFunction):
    loss = DICELossFunction(pre, mask)
    return loss


class trainer():
    def __init__(self, model, loader, dicelossfunction, optims, metrics_function=None):
        super(trainer, self).__init__()
        self.model = model
        self.loader = loader
        self.loss_f = loss_function
        self.dicelossfunction = dicelossfunction
        self.optims = optims
        self.metrics_function = metrics_function
        self.epoch_now = 0

    def train_one_step(self, data, mask):
        pre = self.model(data)
        pre = F.softmax(pre, dim=1)
        loss = self.loss_f(pre, mask, self.dicelossfunction)

        randintger = np.random.randint(0, 10000)
        # Utils.SaveImageGray([torch.argmax(pre, dim=1, keepdim=True).detach().cpu().numpy(),
        #                      mask.detach().cpu().numpy()
        #                      ],
        #                     ['pre' + str(randintger), 'mask' + str(randintger)],
        #                     './debug'
        #                     )
        self.optims.zero_grad()
        loss.backward()
        self.optims.step()

        # if self.metrics_function != None:
        #     metrics = self.metrics_function(pre, mask)
        # else:
        #     metrics = {'None': 0}
        # metrics = self.dict_util.ExtendDicts(metrics, {'loss': loss.data.item() * -1})
        return loss.data.item()#, metrics

    def train_one_epoch(self, visual_step=25):
        self.model.train()
        total_loss = []
        total_metrics = []
        for i, batch_datas in enumerate(self.loader):
            datas, mask = batch_datas
            datas = datas.to(device)  # 将 datas 移动到 GPU
            mask = mask.to(device)
            #data = torch.cat(datas, 1)
            step_loss = self.train_one_step(datas, mask)
            total_loss.append(step_loss)
            #total_metrics.append(step_metrics)
            if i % visual_step == 0 and i != 0:
                print('training step:  ', i, '  avg_loss: ', np.mean(total_loss))
                # if self.metrics_function != None:
                #     self.dict_util.PrintDict(self.dict_util.MeanDictList(total_metrics[-visual_step:]), 5)
        self.epoch_now += 1
        re_loss = np.mean(total_loss)
        #re_metrics = self.dict_util.MeanDictList(total_metrics)
        print('training epoch:  ', self.epoch_now, '  avg_loss: ', re_loss)
        # if self.metrics_function != None:
        #     self.dict_util.PrintDict(re_metrics, 5)
        return re_loss#, re_metrics

class valider():
    def __init__(self, model, loader, dicelossfunction, metrics_function = None):
        super(valider,self).__init__()
        self.loader = loader
        self.model = model
        self.loss_f = loss_function
        self.dicelossfunction = dicelossfunction
        self.metrics_function = metrics_function
        self.epoch_now = 0

    def valid_one_step(self, data, mask):
        pre = self.model(data)
        pre = F.softmax(pre, dim = 1)
        loss = self.loss_f(pre, mask, self.dicelossfunction)
        # if self.metrics_function != None:
        #     metrics = self.metrics_function(pre, mask)
        # else:
        #     metrics = {'None': 0}
        # self.dict_util.ExtendDicts(metrics, {'loss': loss.data.item()* -1})
        return loss.data.item()#, metrics

    def valid_one_epoch(self):
        self.model.eval()
        total_loss = []
        total_metrics = []
        for i, batch_datas in enumerate(self.loader):
            datas, mask = batch_datas
            datas = datas.to(device)  # 将 datas 移动到 GPU
            mask = mask.to(device)
            #data = torch.cat(datas, 1)
            step_loss = self.valid_one_step(datas, mask)
            total_loss.append(step_loss)
            #total_metrics.append(step_metrics)
        self.epoch_now += 1
        re_loss = np.mean(total_loss)
        #re_metrics = self.dict_util.MeanDictList(total_metrics)
        print('validing epoch:  ', self.epoch_now, '  avg_loss: ', re_loss)
        # if self.metrics_function != None:
        #     self.dict_util.PrintDict(re_metrics, 5)
        return re_loss

model = TransUNet(img_dim=128,
                          in_channels=1,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=2)
device = torch.device('cuda:4')
model = model.to(device)
print('model biuld')
dicefuc = DiceLossSoftmax(2)
print('loss initialized')
optim = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False)
trainer = trainer(model, train_dataloader, dicefuc, optim)
valider = valider(model, test_dataloader, dicefuc)
print('trainer valider saver initialized')

print("train start")
epoch = 40
for e in range(epoch):
    tr_loss = trainer.train_one_epoch(visual_step=1)
    print('----------+----------+----------+----------+----------+----------')
    va_loss = valider.valid_one_epoch()
    print('----------*----------*----------*----------*----------*----------')



# 假设test_dataloader是测试集的数据加载器
for i, batch_datas in enumerate(test_dataloader):
    datas, mask = batch_datas
    datas = datas.to(device)  # 将 datas 移动到 GPU
    mask = mask.to(device)
    # 假设你的分割模型是segmentation_model
    with torch.no_grad():
        # 将数据传入分割模型，得到分割结果
        segmentation_result = model(datas)

    # 将张量转换成numpy数组以便显示
    datas_np = datas[0].cpu().numpy()  # 假设批次大小为1，这里取第一张图片
    mask_np = mask[0].cpu().numpy()
    segmentation_result_np = segmentation_result[0].cpu().numpy()

    # 可以使用matplotlib显示图像
    plt.figure(figsize=(10, 5))

    # 显示原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(np.transpose(datas_np, (1, 2, 0)))  # 假设数据的shape为(C, H, W)，转换成(H, W, C)以适合matplotlib
    plt.title('Original Image')
    plt.axis('off')

    # 显示真实标签（如果有的话）
    plt.subplot(2, 2, 2)
    plt.imshow(np.squeeze(mask_np), cmap='gray')  # 假设掩膜是单通道图像
    plt.title('True Mask')
    plt.axis('off')

    # 显示背景信息
    plt.subplot(2, 2, 3)
    plt.imshow(segmentation_result_np[0], cmap='gray')  # 假设分割结果是单通道图像
    plt.title('background Result')
    plt.axis('off')

    # 显示分割结果
    plt.subplot(2, 2, 4)
    plt.imshow(segmentation_result_np[1], cmap='gray')  # 假设分割结果是单通道图像
    plt.title('Segmentation Result')
    plt.axis('off')
    plt.show()


