import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

from utils import *
from datasets import *
from models import *
from visualize import Visualizer
from deeplab import Res50_Deeplab


global feat_target
def get_feat_target(self, input, output):
    feat_target.append(output)
    # feat_norm = torch.norm(output, p=2, dim=1, keepdim=True)
    # feat_target.append(torch.div(output, feat_norm))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='cls1', help="experiment name")
    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads for batch generation")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--feat_size", type=int, default=100, help="size of each feat dimension")
    parser.add_argument("--k", type=int, default=2, help="number of parts per class")
    parser.add_argument("--n_classes", type=int, default=1, help="number of classes")
    parser.add_argument("--w_cls", type=float, default=0, help="classification loss weight")
    parser.add_argument("--w_sem", type=float, default=1e4, help="classification loss weight")
    parser.add_argument("--w_ort", type=float, default=0, help="classification loss weight")
    parser.add_argument("--w_geo", type=float, default=1e2, help="classification loss weight")
    parser.add_argument("--w_eqv", type=float, default=1e3, help="classification loss weight")
    parser.add_argument("--data_dir", type=str, default='/tmp4/hank/VOCdevkit/VOC2012/', help="path to data files")
    parser.add_argument("--weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on validation set")
    opt = parser.parse_args()
    print(opt)

    k = opt.k
    nc = opt.n_classes
    h, w = opt.feat_size, opt.feat_size

    viz = Visualizer(opt.exp_name)
    device = torch.device("cuda:"+str(opt.gpu) if torch.cuda.is_available() else "cpu")


    # ================= Prepare training data ==================
    if nc == 1:
        class_names = ['horse']
    elif nc == 2:
        class_names = ['cow', 'horse']
    elif nc == 5:
        class_names = ['cat', 'cow', 'dog', 'horse', 'sheep']

    datasets = {x: VOCDataset(opt.data_dir, x, class_names, opt.img_size, h) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=opt.batch_size, shuffle=True,
                                                  num_workers=opt.n_cpu) for x in ['train', 'val']}

    class_weights = []
    for i in range(nc):
        pos = 1. * datasets['train'].size[i] / len(datasets['train'])
        class_weights.append((1-pos) / pos)
    class_weights = torch.Tensor(class_weights).to(device)


    # ===================== Base model =====================
    model = Res50_Deeplab(nc * k + 1).to(device)
    params = model.state_dict()

    resnet = models.resnet50(pretrained=True).to(device)
    state_dict = resnet.state_dict().copy()
    for name, param in params.items():
        if name in state_dict and param.size() == state_dict[name].size():
            params[name].copy_(state_dict[name])
    model.load_state_dict(params)


    # ==================== Part basis ======================
    part_basis = PartBasis(1024, nc * k).to(device)


    # =================== Reference feature =================
    vgg19 = models.vgg19(pretrained=True).to(device)
    vgg19.eval()

    global feat_target
    vgg19.features[31].register_forward_hook(get_feat_target)
    vgg19.features[35].register_forward_hook(get_feat_target)


    # ================= training settings ================
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    optimizer_b = optim.SGD(part_basis.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    exp_lr_scheduler_b = lr_scheduler.StepLR(optimizer_b, step_size=100, gamma=0.5)


    # ==================== Start training ======================
    coord = np.tile(1. * np.arange(w) / (w-1), (h, 1))
    u = torch.from_numpy(coord).to(device).float()
    v = torch.from_numpy(coord.transpose()).to(device).float()

    for epoch in range(opt.n_epochs):
        print('Epoch {}/{}'.format(epoch, opt.n_epochs-1))

        if epoch % opt.evaluation_interval == 0:
            phases = ['train', 'val']
        else:
            phases = ['train']

        for phase in phases:
            if phase == 'train':
                model.train()
                part_basis.train()
            else:
                model.eval()
                part_basis.eval()

            running_loss_cls = 0.0
            running_loss_geo = 0.0
            running_loss_sem = 0.0
            running_loss_ort = 0.0
            running_loss_eqv = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, saliency, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                saliency = saliency.to(device).float()

                nb = inputs.size(0)
                label_mask = torch.zeros((nb, nc), dtype=torch.bool).to(device)
                label_mask = label_mask.scatter_(1, labels.view(-1, 1), 1).repeat_interleave(k, dim=1)

                # Zero the parameter gradients
                optimizer.zero_grad()
                optimizer_b.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    outputs = nn.functional.interpolate(outputs, size=(h, w),
                                                        mode='bilinear', align_corners=True)

                    pams = nn.Softmax(dim=1)(outputs)[:, :-1, :, :]
                    pams_masked = pams.masked_select(label_mask.view(-1, nc*k, 1, 1)).view(-1, k, h, w)


                    # Classification loss
                    probs = nn.AdaptiveMaxPool2d(output_size=(1, 1))(pams).squeeze()
                    _, preds = torch.max(torch.sum(probs.view(-1, nc, k), 2), 1)
                    sample_weights = class_weights.index_select(0, labels).view(-1, 1)
                    loss_cls = nn.BCELoss(weight=sample_weights)(probs, label_mask.float())

                    #pams = pams.view(-1, n_classes, k, h, w)
                    #select_mask = torch.zeros((inputs.size(0), n_classes), dtype=torch.uint8).to(device)
                    #select_mask = select_mask.scatter_(1, labels.view(-1, 1), 1).view(-1, n_classes, 1, 1, 1)
                    #pams_selected = torch.masked_select(pams, select_mask).view(-1, k, h, w)

                    #cams = torch.sum(outputs[:, :-1, :, :].view(-1, n_classes, k, h, w), 2)
                    #probs = nn.AdaptiveAvgPool2d(output_size=(1, 1))(cams).view(-1, n_classes)
                    #_, preds = torch.max(probs, 1)
                    #loss_cls = nn.CrossEntropyLoss(weight=class_weights)(probs, labels)


                    # Semantic loss
                    basis = part_basis()
                    basis_masked = basis.view(1024, nc, k).index_select(1, labels).transpose(0, 1)

                    feat_target = []
                    _ = vgg19(inputs)
                    feat_t = nn.functional.interpolate(torch.cat(feat_target, dim=1), size=(h, w),
                                                       mode='bilinear', align_corners=True)

                    feat_r = torch.bmm(basis_masked, pams_masked.view(-1, k, h*w)).view(-1, 1024, h, w)
                    #feat_r = torch.zeros(feat_t.size()).to(device)
                    #for i in range(nb):
                    #    feat_r[i, :, :, :] = torch.mm(basis_masked[:, i, :],
                    #                                  pams_masked[i, :, :, :].view(k, -1)).view(-1, h, w)
                    loss_sem = nn.MSELoss()(feat_r, feat_t * saliency)


                    # Orthonormal loss
                    wn = torch.norm(basis_masked, p=2, dim=1, keepdim=True)
                    ww = basis_masked.div(wn.expand_as(ww) + 1e-9).view(-1, k)
                    wwt = torch.matmul(ww.transpose(0, 1), ww)
                    loss_ort = nn.MSELoss()(wwt, torch.eye(k).to(device))


                    # Geometric concentration loss
                    pams_sum = pams_masked.sum(2, keepdim=True).sum(3, keepdim=True)
                    pams_geo = pams_masked.div(pams_sum + 1e-9)
                    center_u = (pams_geo * u).sum(2, keepdim=True).sum(3, keepdim=True)
                    center_v = (pams_geo * v).sum(2, keepdim=True).sum(3, keepdim=True)
                    dist = torch.sqrt((u - center_u)**2 + (v - center_v)**2)
                    loss_geo = torch.sum(pams_geo * dist)


                    # Equivariance loss
                    inputs = inputs.detach()#.cpu()
                    outputs = outputs.detach()#.cpu()
                    #inputs_tf = torch.zeros(inputs.size())
                    #feat_tf = torch.zeros(outputs.size())

                    flip = np.random.randint(2)
                    angle = np.random.randn()*10
                    translate = (np.random.random(2)*0.2).tolist()
                    scale = np.random.random()*0.5+1
                    shear = np.random.randn()*5
                    params_tf = [device, flip, angle, translate, scale]

                    inputs_tf = spatial_transforms(inputs, *params_tf)
                    pams_tf = spatial_transforms(pams, *params_tf)

                    #for i in range(nb):
                    #    inputs_tf[i, :, :, :] = transforms_img(inputs[i, :, :, :], *params_tf)
                    #    for j in range(outputs.size(1)):
                    #        feat_tf[i, j, :, :] = transforms_pam(outputs[i, j, :, :], *params_tf)
                    #pams_tf = nn.Softmax(dim=1)(feat_tf.to(device))

                    outputs_tf = model(inputs_tf.to(device))
                    outputs_tf = nn.functional.interpolate(outputs_tf, size=(h, w),
                                                           mode='bilinear', align_corners=True)
                    outputs_tf = nn.LogSoftmax(dim=1)(outputs_tf)[:, :-1, :, :]
                    loss_eqv = nn.KLDivLoss()(outputs_tf, pams_tf)


                    loss = opt.w_cls * loss_cls + \
                           opt.w_geo * loss_geo + \
                           opt.w_sem * loss_sem + \
                           opt.w_ort * loss_ort + \
                           opt.w_eqv * loss_eqv


                    # Visualization
                    with torch.no_grad():
                        if epoch % 5 == 0:
                            pams_viz = pams.view(-1, nc * k, h, w)
                            pams_tf_viz = pams_tf.view(-1, nc * k, h, w)
                            pams_viz = pams_viz / pams_viz.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
                            pams_tf_viz = pams_tf_viz / pams_tf_viz.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
                            viz.vis_inputs(epoch, inputs, prefix='')
                            viz.vis_inputs(epoch, inputs_tf, prefix='_tf')
                            viz.vis_inputs(epoch, saliency, prefix='saliency')
                            viz.vis_DFF_heatmaps(epoch, pams_viz, threshold=0.1, prefix='pams')
                            viz.vis_DFF_heatmaps(epoch, pams_tf_viz, threshold=0.1, prefix='pams_tf')

                        if phase == 'train':
                            viz.vis_losses(epoch, [loss, loss_cls, loss_geo, loss_sem, loss_ort, loss_eqv],
                                                  ['loss_train', 'loss_cls_train', 'loss_geo_train', 'loss_sem_train',
                                                   'loss_ort_train', 'loss_eqv_train'])
                        else:
                            viz.vis_losses(epoch, [loss, loss_cls, loss_geo, loss_sem, loss_ort, loss_eqv],
                                                  ['loss_val', 'loss_cls_val', 'loss_geo_val', 'loss_sem_val',
                                                   'loss_ort_val', 'loss_eqv_val'])

                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5)
                        nn.utils.clip_grad_norm_(part_basis.parameters(), 5)
                        optimizer.step()
                        optimizer_b.step()
                        exp_lr_scheduler.step()
                        exp_lr_scheduler_b.step()

                # Statistics
                running_loss_cls += loss_cls.item() * nb
                running_loss_geo += loss_geo.item() * nb
                running_loss_sem += loss_sem.item() * nb
                running_loss_ort += loss_ort.item() * nb
                running_loss_eqv += loss_eqv.item() * nb
                running_corrects += torch.sum(preds == labels.data)

            epoch_acc = running_corrects.float() / len(datasets[phase])
            epoch_loss_cls = opt.w_cls * running_loss_cls / len(datasets[phase])
            epoch_loss_geo = opt.w_geo * running_loss_geo / len(datasets[phase])
            epoch_loss_sem = opt.w_sem * running_loss_sem / len(datasets[phase])
            epoch_loss_ort = opt.w_ort * running_loss_ort / len(datasets[phase])
            epoch_loss_eqv = opt.w_eqv * running_loss_eqv / len(datasets[phase])
            epoch_loss = epoch_loss_cls + epoch_loss_geo + epoch_loss_sem + epoch_loss_ort + epoch_loss_eqv

            print('{} Loss: {:.3f}, Acc: {:.3f}'.format(phase, epoch_loss, epoch_acc))
            print('cls loss:{:.3f}, geo loss:{:.3f}, sem loss:{:.3f}, ort loss:{:.3f}, eqv loss:{:.3f}'.format(
                  epoch_loss_cls, epoch_loss_geo, epoch_loss_sem, epoch_loss_ort, epoch_loss_eqv))

        if (epoch+1) % opt.checkpoint_interval == 0:
            model_saved = 'checkpoints/model_{}_epoch_{}.pt'.format(opt.exp_name, str(epoch))
            basis_saved = 'checkpoints/basis_{}_epoch_{}.pt'.format(opt.exp_name, str(epoch))
            torch.save(model.state_dict(), model_saved)
            torch.save(part_basis.state_dict(), basis_saved)

    print('Training complete.')
