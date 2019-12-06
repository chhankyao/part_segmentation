import torch
import torch.nn.funtional as F
from PIL import Image
from torchvision import transforms
import numpy as np


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


def transform_img(inputs, flip, angle, translate, scale, shear):
    theta = torch.zeros(1, 2, 3)
    theta[:, :, :2] = torch.tensor([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
    theta[:, :, 2] = torch.tensor(translate)

    nb, nc, h, w = [int(s) for s in inputs.size()]
    h2, w2 = int(round(h*scale)), int(round(w*scale))
    size_o = torch.tensor([nb, nc, h2, w2])
    grid = F.affine_grid(theta, size_o, align_corners=True)
    inputs_tf = F.grid_sample(inputs, grid, padding_mode="border")

    x1 = int(round((w2 - w) / 2.))
    y1 = int(round((h2 - h) / 2.))
    input_tf = inputs_tf[:, :, y1:y1+h, x1:x1+w]

    if flip:
        inputs_tf = inputs_tf.flip([3])

    return inputs_tf
