import os
import copy
import argparse
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils import data
import torchvision
from torchvision import datasets, models, transforms
from deeplab import Res50_Deeplab
from visualize import Visualizer



K = 1
INPUT_SIZE = (256, 256)
H, W = 100, 100
MODEL = 'resnet50'

N_CLASSES = 2
CLASS_NAMES = ['cow', 'horse']
#CLASS_NAMES = ['cat', 'cow', 'dog', 'horse', 'sheep']
DATASET = 'VOC2012'
DATA_DIR = '/tmp4/hank/VOCdevkit/VOC2012/'

GPU = 2
N_WORKERS = 8
BATCH_SIZE = 8
N_EPOCHS = 300

LR = 1e-4
LR_B = 1e-1
MOMENTUM = 0.9
MOMENTUM_B = 0.9
DECAY = 5e-4
DECAY_B = 5e-4

EXP_NAME = MODEL+'_2class_cam_2maps'
LAMBDA_CLS = 1e4
LAMBDA_GEO = 0#1e0
LAMBDA_SEM = 0#1e4
LAMBDA_ORT = 0
LAMBDA_EQV = 0#1e2



global feat_target
          
def get_feat_target(self, input, output):
    global feat_target
    feat_target.append(nn.functional.interpolate(output, size=(H,W),
                                                 mode='bilinear', align_corners=True))    
    #l2_norm = torch.norm(output, p=2, dim=1).unsqueeze(1)
    #feat_target.append(torch.div(output, l2_norm.expand_as(output)))
    


class PartBasis(nn.Module):
    def __init__(self, dim_feat, k):
        super(PartBasis, self).__init__()
        self.w = nn.Parameter(torch.abs(torch.FloatTensor(dim_feat, k).normal_()))
        
    def forward(self, x=None):
        out = nn.ReLU()(self.w)
        return out



def preprosess_img(inputs, phase, flip, angle, translate, scale, shear):
    inputs = transforms.functional.resize(inputs, INPUT_SIZE)
    if phase == 'train':
        if flip:
            inputs = transforms.functional.hflip(inputs)
        inputs = transforms.functional.affine(inputs, angle, translate, scale, shear,
                                              resample=Image.BILINEAR, fillcolor=(128,128,128))
        inputs = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2)(inputs)
    inputs = transforms.ToTensor()(inputs)
    inputs = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(inputs)
    return inputs


def preprosess_sal(inputs, phase, flip, angle, translate, scale, shear):
    inputs = transforms.functional.resize(inputs, (H,W))
    if phase == 'train':
        if flip:
            inputs = transforms.functional.hflip(inputs)
        inputs = transforms.functional.affine(inputs, angle, translate, scale, shear,
                                              resample=Image.BILINEAR, fillcolor=0)
    inputs = transforms.ToTensor()(inputs)
    return inputs



class VOCDataset(data.Dataset):
    def __init__(self, phase):
        self.phase = phase
        self.size = []
        self.img_list = []
        #if phase == 'train':
        #    list_file = DATA_DIR+'ImageSets/Main/{}_{}_large.txt'
        #else:
        list_file = DATA_DIR+'ImageSets/Main/{}_{}.txt'
        for cls_idx, cls in enumerate(CLASS_NAMES):
            with open(list_file.format(cls, phase), 'r') as f:
                #if phase == 'train':
                #    img_ids = [(s, cls_idx) for s in f.read().splitlines()]
                #else:
                img_ids = [(s.split(' ')[0], cls_idx) for s in f.read().splitlines() if s[-2:] == ' 1']
                self.size.append(len(img_ids))
                self.img_list += img_ids
        print(self.size)

    
    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, i):
        img = Image.open(DATA_DIR+'JPEGImages/'+self.img_list[i][0]+'.jpg')
        sal = Image.open(DATA_DIR+'Saliency/'+self.img_list[i][0]+'_RBD.png')
        label = self.img_list[i][1]
        
        flip = np.random.randint(2)
        angle = np.random.randn()*10
        translate = (np.random.random(2)*0.2).tolist()
        scale = np.random.random()*0.5+1
        shear = np.random.randn()*5
        
        img = preprosess_img(img, self.phase, flip, angle, translate, scale, shear)
        sal = preprosess_sal(sal, self.phase, flip, angle, translate, scale, shear)

        return img, sal, label



def transforms_img(inputs, flip, angle, translate, scale, shear):
    inputs = inputs - torch.min(inputs)
    inputs = inputs / torch.max(inputs)
    inputs = transforms.ToPILImage()(inputs)
    if flip:
        inputs = transforms.functional.hflip(inputs)
    inputs_tf = transforms.functional.affine(inputs, angle, translate, scale, shear,
                                             resample=Image.BILINEAR, fillcolor=(128,128,128))
    inputs_tf = transforms.ToTensor()(inputs_tf)
    inputs_tf = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(inputs_tf)
    return inputs_tf



def transforms_pam(inputs, flip, angle, translate, scale, shear):
    inputs_min = torch.min(inputs)
    inputs = inputs - inputs_min
    inputs_max = torch.max(inputs)
    inputs = inputs / inputs_max
    inputs = transforms.ToPILImage()(inputs)
    if flip:
        inputs = transforms.functional.hflip(inputs)
    inputs_tf = transforms.functional.affine(inputs, angle, translate, scale, shear,
                                             resample=Image.BILINEAR, fillcolor=0)
    inputs_tf = transforms.ToTensor()(inputs_tf)
    inputs_tf = (inputs_tf * inputs_max) + inputs_min
    return inputs_tf




def train():

    device = torch.device("cuda:"+str(GPU) if torch.cuda.is_available() else "cpu")
    viz = Visualizer(EXP_NAME)    

    # ================= Prepare training data ==================
    img_datasets = {x: VOCDataset(x) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(img_datasets[x], batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=N_WORKERS) for x in ['train', 'val']}
    dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}

    class_weight = []
    for i in range(N_CLASSES):
        pos = 1.*img_datasets['train'].size[i] / dataset_sizes['train']
        class_weight.append((1-pos) / pos)
    class_weight = torch.Tensor(class_weight).to(device)
    
    
    # ===================== Base model =====================
    model = models.resnet50(pretrained=True).to(device)
    #model.fc = nn.Linear(model.fc.in_features, N_CLASSES).to(device)
    model.avgpool = nn.Sequential(
        nn.Conv2d(2048, 2, kernel_size=(1, 1), stride=(1, 1), bias=True).to(device),
        nn.AdaptiveAvgPool2d(output_size=(1, 1))
    )
    model.fc = nn.Linear(2, N_CLASSES).to(device)    
    print(model)

    # ================= training parameters ================
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    
    # ==================== Start training ======================
    one = torch.Tensor([1]).to(device)

    for epoch in range(N_EPOCHS):
        print('-' * 20)
        print('Epoch {}/{}'.format(epoch, N_EPOCHS-1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss1 = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, saliency, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                saliency = saliency.to(device).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)
                    
                    # Classification loss
                    _, preds = torch.max(outputs, 1)
                    loss_cls = nn.CrossEntropyLoss()(outputs, labels)
                    loss = LAMBDA_CLS * loss_cls


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()

                # statistics
                running_loss1 += loss_cls.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                

            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss1 = LAMBDA_CLS * running_loss1 / dataset_sizes[phase]
            epoch_loss = epoch_loss1

            print('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        if epoch % 50 == 49:
            model_saved = '../models/model_{}_epoch_{}.pt'.format(EXP_NAME, str(epoch))
            torch.save(model.state_dict(), model_saved)
            
    print('Training complete.')
    return model



model = train()
