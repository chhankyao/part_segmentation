from PIL import Image
import torch
from torchvision import transforms


def transforms_img(inputs, flip, angle, translate, scale, shear):
    inputs = inputs - torch.min(inputs)
    inputs = inputs / torch.max(inputs)
    inputs = transforms.ToPILImage()(inputs)
    if flip:
        inputs = transforms.functional.hflip(inputs)
    inputs_tf = transforms.functional.affine(inputs, angle, translate, scale, shear,
                                             resample=Image.BILINEAR, fillcolor=(128, 128, 128))
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
