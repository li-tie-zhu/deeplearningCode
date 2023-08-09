# -*- codeing = utf-8 -*-
# @Time : 2023/6/13 20:16
# @Author : 李铁柱
# @File : skinTrain.py
# @Software: PyCharm

import os
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
import torch
from torchvision.transforms import Grayscale, RandomHorizontalFlip, RandomVerticalFlip, RandomErasing
import time
from VIT import *
from ResNet2 import *
print("aaa")
class CreatDataSet(Dataset):
    def __init__(self,rootPath,transform=None):
        self.rootPath = rootPath
        self.transform = transform
        self.dataset = self.load_dataset()

    def load_dataset(self):
        data_set = []
        label_mapping = {}
        for idx,subdir in enumerate(os.listdir(self.rootPath)):
            subdir_path = os.path.join(self.rootPath, subdir)
            if os.path.isdir(subdir_path):
                label_mapping[subdir] = idx
                for file_name in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file_name)
                    if file_name.endswith('.jpg'):
                        image = Image.open(file_path)
                        transformed_img = data_tranform(image)
                        numeriv_label = label_mapping[subdir]
                        data_set.append([transformed_img, numeriv_label])
        return data_set

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label


data_tranform =transforms.Compose([
    transforms.Resize((32,32)),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
])

rootPath_train = '../CNN/skinLesions/skinlesions/train'
rootPath_test = '../CNN/skinLesions/skinlesions/test'

data_set_train = CreatDataSet(rootPath_train,data_tranform)
data_set_test = CreatDataSet(rootPath_test,data_tranform)

print(len(data_set_train))
print(len(data_set_test))

train_dataloader =DataLoader(data_set_train,batch_size=32,shuffle=True)
test_dataloader = DataLoader(data_set_test,batch_size=32,shuffle=False)



def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train() # 改回训练模式
            n += y.shape[0]
    return acc_sum / n
def train(net, train_iter, test_iter, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f%%, test acc %.3f%%, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, 100*train_acc_sum / n, 100*test_acc, time.time() - start))

# model_vit = ViT(
#         image_size = 256,
#         patch_size = 32,
#         num_classes = 3,
#         dim = 1024,
#         depth = 6,
#         heads = 16,
#         mlp_dim = 2048,
#         dropout = 0.1,
#         emb_dropout = 0.1
#     )
model  = resnet18(pretrained=False,num_classes=3)
num_epochs = 20
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam(model_vit.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(Model.parameters(), lr=learning_rate,momentum=0.5)
train(model_vit, train_dataloader, test_dataloader, optimizer, device, num_epochs)


