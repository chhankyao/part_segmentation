import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms


def preprocess_img(img, img_size, phase, flip, angle, translate, scale, shear):
    inputs = transforms.functional.resize(img, (img_size, img_size))
    if phase == 'train':
        if flip:
            inputs = transforms.functional.hflip(inputs)
        inputs = transforms.functional.affine(inputs, angle, translate, scale, shear,
                                              resample=Image.BILINEAR, fillcolor=(128, 128, 128))
        inputs = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2)(inputs)
    inputs = transforms.ToTensor()(inputs)
    inputs = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(inputs)
    return inputs


def preprocess_sal(img, img_size, phase, flip, angle, translate, scale, shear):
    inputs = transforms.functional.resize(img, (img_size, img_size))
    if phase == 'train':
        if flip:
            inputs = transforms.functional.hflip(inputs)
        inputs = transforms.functional.affine(inputs, angle, translate, scale, shear,
                                              resample=Image.BILINEAR, fillcolor=0)
    inputs = transforms.ToTensor()(inputs)
    return inputs


class VOCDataset(data.Dataset):
    def __init__(self, data_dir, phase, class_names, img_size, feat_size):
        self.data_dir = data_dir
        self.phase = phase
        self.img_size = img_size
        self.feat_size = feat_size

        self.size = []
        self.img_list = []
        if phase == 'train':
            list_file = data_dir + 'ImageSets/Main/{}_{}_large.txt'
        else:
            list_file = data_dir + 'ImageSets/Main/{}_{}.txt'
        for cls_idx, cls in enumerate(class_names):
            with open(list_file.format(cls, phase), 'r') as f:
                if phase == 'train':
                    img_ids = [(s, cls_idx) for s in f.read().splitlines()]
                else:
                    img_ids = [(s.split(' ')[0], cls_idx) for s in f.read().splitlines() if s[-2:] == ' 1']
                self.size.append(len(img_ids))
                self.img_list += img_ids

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        img = Image.open(self.data_dir + 'JPEGImages/' + self.img_list[i][0] + '.jpg')
        sal = Image.open(self.data_dir + 'Saliency/' + self.img_list[i][0] + '_RBD.png')
        label = self.img_list[i][1]

        flip = np.random.randint(2)
        angle = np.random.randn() * 10
        translate = (np.random.random(2) * 0.2).tolist()
        scale = np.random.random() * 0.5 + 1
        shear = np.random.randn() * 5

        img = preprocess_img(img, self.img_size, self.phase, flip, angle, translate, scale, shear)
        sal = preprocess_sal(sal, self.feat_size, self.phase, flip, angle, translate, scale, shear)

        return img, sal, label
