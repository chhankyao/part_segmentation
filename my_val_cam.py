import os
import copy
import numpy as np
import cv2
from PIL import Image


import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms



K = 1
N_CLASSES = 2
INPUT_SIZE = (256, 256)
H, W = 100, 100

GPU = 5
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


model = models.resnet50(pretrained=True).to(device)
model.fc = nn.Linear(model.fc.in_features, N_CLASSES).to(device)

saved_state_dict = torch.load('../models/model_resnet50_2class_cam_epoch_299.pt')
model.load_state_dict(saved_state_dict)
model.eval()


global feature
def get_feat(self, input, output):
    global feature
    feature = output

model.layer4.register_forward_hook(get_feat)



preprocess = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]) 



def img2cam(img, model):
    logit = model(img)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    feat = feature.data.cpu().numpy()[0,:,:,:]
    cam = weight_softmax.dot(feat.reshape((2048, 8*8)))
    cam = cam.reshape(N_CLASSES,8,8).transpose(1,2,0)
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam



def cams_to_mask(cams):
    cams = cv2.resize(cams, (W,H), interpolation=cv2.INTER_LINEAR)
    act = np.max(cams, axis=2)
    idx = np.argmax(cams, axis=2)
    idx[act < 0.8] = N_CLASSES
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

        img_variable = preprocess(img).unsqueeze(0).to(device)
        cams = img2cam(img_variable, model)
        mask_pred = cams_to_mask(cams)
            
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
