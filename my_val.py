import os
import copy
import numpy as np
import cv2
from PIL import Image


import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from deeplab import Res50_Deeplab



K = 1
N_CLASSES = 2
INPUT_SIZE = (256, 256)
H, W = 100, 100

GPU = 2
device = torch.device("cuda:"+str(GPU) if torch.cuda.is_available() else "cpu")



class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

label_colors = [(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]


classes = [9, 12]
#classes = [7, 9, 11, 12, 16]
class_mask = [label_colors[c] for c in classes]



model = Res50_Deeplab(N_CLASSES * K + 1).to(device)

saved_state_dict = torch.load('../models/model_resnet50_avgpool_noWeights_2class_40002_epoch_299.pt', map_location='cpu')
model.load_state_dict(saved_state_dict)
model.eval()



preprocess = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]) 



def pams_to_mask(pams):
    pams = pams - pams.min()
    pams = pams / pams.max()
    
    pams_norm = pams.clone().detach()
    pams_norm -= pams_norm.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
    pams_norm = pams_norm.div(pams_norm.max(2, keepdim=True)[0].max(3, keepdim=True)[0])
    
    pams_norm = pams_norm[:,:-1,:,:] - pams_norm[:,-1:,:,:]
    pams_norm -= pams_norm.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
    pams_norm = pams_norm.div(pams_norm.max(2, keepdim=True)[0].max(3, keepdim=True)[0])
    
    pams = pams[0,:-1,:,:].permute(1,2,0).detach().cpu().numpy()
    pams_norm = pams_norm[0,:,:,:].permute(1,2,0).detach().cpu().numpy()

    idx = np.argmax(pams, axis=2)
    for i in range(pams.shape[2]):
        idx[(idx == i) & (pams_norm[:,:,i] < 0.5)] = N_CLASSES * K

    return idx




with open('/tmp4/hank/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt', 'r') as f:
    img_list = f.read().splitlines()
    
    count = 0
    tp = np.zeros(N_CLASSES)
    pos_gt = np.zeros(N_CLASSES)
    pos_pred = np.zeros(N_CLASSES)
    
    
    for im_id in img_list:
        img = Image.open('/tmp4/hank/VOCdevkit/VOC2012/JPEGImages/'+im_id+'.jpg').convert('RGB')
        img_np = cv2.resize(np.array(img), (H,W))
        
        mask_file = '/tmp4/hank/VOCdevkit/VOC2012/SegmentationClass/'+im_id+'.png'
        if os.path.isfile(mask_file):
            mask = Image.open(mask_file).convert('RGB')
            mask_np = cv2.resize(np.array(mask), (H,W), interpolation=cv2.INTER_NEAREST)
        else:
            continue
        
        masks_gt = [(mask_np[:,:,0] == label_colors[c][0]) & (mask_np[:,:,1] == label_colors[c][1]) & 
                    (mask_np[:,:,2] == label_colors[c][2]) for c in classes]
        
        if not np.any(np.array(masks_gt)):
            continue

        img_variable = preprocess(img).to(device)
        outputs = model(img_variable.unsqueeze(0))
        outputs = nn.functional.interpolate(outputs, size=(H,W), 
                                                mode='bilinear', align_corners=True)

        pams = nn.Softmax(dim=1)(outputs)
        mask_pred = pams_to_mask(outputs) #/ K
        #probs = nn.AdaptiveMaxPool2d(output_size=(1,1))(pams[:,:-1,:,:])
        #probs = torch.mean(probs.view(num_classes, K), 1)
        #cls = 0 if probs[0] > probs[1] else 1
            
        for i in range(N_CLASSES):
            mask_i = mask_pred == i
            tp[i] += np.sum(mask_i & masks_gt[i])
            pos_gt[i] += np.sum(masks_gt[i])
            pos_pred[i] += np.sum(mask_i)
            
        count += 1
        if count % 10 == 1:
            print(count)    
        
    iou = 1. * tp / (pos_gt + pos_pred - tp)
    print(iou, count)
