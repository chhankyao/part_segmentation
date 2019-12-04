import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import visdom
from tensorboardX import SummaryWriter
import json
import cv2
import os.path as osp



class SobelFilter(nn.Module):
    def __init__(self, in_plane=1):
        super(SobelFilter, self).__init__()

        self.filter_x = nn.Conv2d(in_plane, in_plane, kernel_size=3,
                         padding=1, bias=False)
        self.filter_y = nn.Conv2d(in_plane, in_plane, kernel_size=3,
                         padding=1, bias=False)

        a=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
        self.filter_x.weight=nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
        b=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
        self.filter_y.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))

        # freeze weight
        self.filter_x.weight.requires_grad = False
        self.filter_y.weight.requires_grad = False


    def forward(self, x):
        G_x=self.filter_x(x)
        G_y=self.filter_y(x)

        G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))

        return G


# sobel = SobelFilter().cuda()


# from loss import get_center, softmax

class BatchColorize(object):
    def __init__(self, n=40):
        self.cmap = color_map(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((size[0], 3, size[1], size[2]), dtype=np.float32)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[:,0][mask] = self.cmap[label][0]
            color_image[:,1][mask] = self.cmap[label][1]
            color_image[:,2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[:,0][mask] = color_image[:,1][mask] = color_image[:,2][mask] = 255

        return color_image


def color_map(N=256, normalized=True):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def Batch_Draw_GT_Landmarks(imgs, pred, lms):

    B,_,H,W = imgs.shape
    C = lms.shape[1]
    cmap = color_map(40,normalized=False)
    imgs_cv2 = imgs.detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8)

    centers = np.zeros((B,C,2))

    for b in range(B):
        for c in range(C):
            x_c = int(lms[b][c][0])
            y_c = int(lms[b][c][1])

            img = imgs_cv2[b].copy()
            cv2.drawMarker(img, (x_c,y_c), (int(cmap[c+1][0]), int(cmap[c+1][1]), int(cmap[c+1][2])), 
                           markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=2)
            imgs_cv2[b] = img

    return imgs_cv2.transpose(0,3,1,2)


def Batch_Draw_Bboxes(imgs, bboxes):

    B,C,H,W = imgs.shape
    imgs_cv2 = imgs.detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8)
    for b in range(B):
        x,y,w,h = bboxes[b]
        img = imgs_cv2[b].copy()
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        imgs_cv2[b] = img

    return imgs_cv2.transpose(0,3,1,2)


def Batch_Get_Centers(pred, sm=True):
    B,C,H,W = pred.shape
    if sm:
        pred_softmax = softmax(pred)
    else:
        pred_softmax = pred
    centers = np.zeros((B,C-1,2))

    for b in range(B):
        for c in range(1,C):
            # normalize part map as spatial pdf
            part_map = pred_softmax[b,c,:,:]
            k = float(part_map.sum())
            part_map_pdf = part_map/k
            x_c, y_c = get_center(part_map_pdf)
            x_c = (x_c+1.0)/2*W # [-1,1] -> [0,W]
            y_c = (y_c+1.0)/2*H
            centers[b,c-1,:] = [x_c,y_c]
    return centers


def Batch_Draw_Landmarks(imgs, pred, sm=True):

    B,C,H,W = pred.shape
    cmap = color_map(40,normalized=False)

    if sm:
        pred_softmax = softmax(pred)
    else:
        pred_softmax = pred

    imgs_cv2 = imgs.detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8)

    centers = np.zeros((B,C-1,2))

    part_response = np.zeros((B,C-1,H,W,3)).astype(np.uint8)
    part_response_gradient =np.zeros((B,C-1,H,W,3)).astype(np.uint8)

    for b in range(B):
        for c in range(1,C):

            # normalize part map as spatial pdf
            part_map = pred_softmax[b,c,:,:]
            k = float(part_map.sum())
            part_map_pdf = part_map/k

            response_map = part_map_pdf.detach().cpu().numpy()
            response_map = response_map/response_map.max()

            '''
            response_map_gradient = sobel(part_map_pdf.unsqueeze(0).unsqueeze(0))
            response_map_gradient = response_map_gradient.detach().squeeze().cpu().numpy()
            response_map_gradient = response_map_gradient/response_map_gradient.max()
            '''

            response_map = cv2.applyColorMap((response_map*255.0).astype(np.uint8), cv2.COLORMAP_HOT)[:,:,::-1] # BGR->RGB
            #response_map = np.tile(response_map, (3,1,1)).transpose((1,2,0))*128*128
            #response_map_gradient = cv2.applyColorMap((response_map_gradient*255.0).astype(np.uint8), cv2.COLORMAP_HOT)[:,:,::-1]

            part_response[b,c-1,:,:,:] = response_map.astype(np.uint8)
            #part_response_gradient[b,c-1,:,:,:] = response_map_gradient.astype(np.uint8)

            x_c, y_c = get_center(part_map_pdf)


            centers[b,c-1,:] = [x_c/2,y_c/2]

            x_c = (x_c+1.0)/2*W
            y_c = (y_c+1.0)/2*H

            img = imgs_cv2[b].copy()
            cv2.drawMarker(img, (x_c,y_c), (int(cmap[c][0]), int(cmap[c][1]), int(cmap[c][2])), 
                           markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=2)
            imgs_cv2[b] = img

    return imgs_cv2.transpose(0,3,1,2), centers, part_response.transpose(0,1,4,2,3), part_response_gradient.transpose(0,1,4,2,3)

'''
def denseCRF(img, pred):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    N,H,W = pred.shape

    d = dcrf.DenseCRF2D(W, H, N)  # width, height, nlabels
    U = unary_from_softmax(pred.transpose(0,2,1))
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.array(Q).reshape((N,W,H)).transpose(2,1,0)

    return Q
'''

class Visualizer(object):
    #def __init__(self, args, viz=None):
    def __init__(self, exp, viz=None):

        self.exp_name = exp#args.exp_name
        self.tb_writer = SummaryWriter(log_dir=osp.join('/tmp4/hank/runs', self.exp_name))

        self.vis_interval = 5#args.vis_interval
        self.colorize = BatchColorize(7)#args.num_classes)

        #self.args = args

        # dump args to tensorboard
        #args_str = '{}'.format(json.dumps(vars(args), sort_keys=False, indent=4))
        #self.tb_writer.add_text('Exp_args', args_str, 0)

    def vis_recon_images(self, i_iter, recon_imgs):
        if i_iter % (self.vis_interval*1) == 0 :
            recon_imgs = vutils.make_grid(recon_imgs, normalize=False, scale_each=False)
            self.tb_writer.add_image('Recon', recon_imgs, i_iter)

    def vis_recon_images_v2(self, i_iter, recon_imgs, masked_imgs):
        if i_iter % self.vis_interval == 0 :
            for c in range(len(recon_imgs)):
                viz_img = torch.cat((masked_imgs[c], recon_imgs[c]), dim=0)
                viz_img = vutils.make_grid(viz_img, normalize=False, scale_each=False)
                self.tb_writer.add_image('Recon {}'.format(c), viz_img, i_iter)

    def vis_align_images(self, i_iter, align_imgs, mean):
        if i_iter % self.vis_interval == 0 :
            i_shape = align_imgs.shape
            mean_tensor = torch.tensor(mean).float().expand(i_shape[0], i_shape[3], i_shape[2], 3).transpose(1,3)
            imgs_align_viz = torch.clamp(align_imgs+mean_tensor, 0.0, 255.0)
            imgs_align_viz = vutils.make_grid(imgs_align_viz/255.0, normalize=False, scale_each=False)
            self.tb_writer.add_image('Align', imgs_align_viz, i_iter)


    def vis_images(self, i_iter, imgs, tps_imgs, saliency_imgs, edge_imgs, mean, pred):
        if i_iter % self.vis_interval == 0 :
            i_shape = imgs.shape
            mean_tensor = torch.tensor(mean).float().expand(i_shape[0], i_shape[3], i_shape[2], 3).transpose(1,3)
            imgs_viz = torch.clamp(imgs+mean_tensor, 0.0, 255.0)
            self.imgs_viz = imgs_viz
            imgs_viz_grid = vutils.make_grid(imgs_viz/255.0, normalize=False, scale_each=False)
            self.imgs_viz_grid = imgs_viz_grid
            self.tb_writer.add_image('Input', imgs_viz_grid, i_iter)

            tps_imgs_viz = torch.clamp(tps_imgs+mean_tensor, 0.0, 255.0)
            tps_imgs_viz = vutils.make_grid(tps_imgs_viz/255.0, normalize=False, scale_each=False)
            self.tb_writer.add_image('Transformed', tps_imgs_viz, i_iter)

            # saliency
            if saliency_imgs is not None:
                sal_viz = torch.clamp(saliency_imgs.float().unsqueeze(dim=1)*255.0, 0.0, 255.0)
                sal_viz = vutils.make_grid(sal_viz/255.0, normalize=False, scale_each=False)
                self.tb_writer.add_image('Saliency', sal_viz, i_iter)

            # edges
            if edge_imgs is not None:
                edge_viz = torch.clamp(edge_imgs.float().unsqueeze(dim=1)*255.0, 0.0, 255.0)
                edge_viz = vutils.make_grid(edge_viz/255.0, normalize=False, scale_each=False)
                self.tb_writer.add_image('Edge', edge_viz, i_iter)

            # landmarks
            lm_viz, _, part_pdf_viz, part_pdf_grad_viz = Batch_Draw_Landmarks(imgs_viz, pred)
            lm_viz = torch.tensor(lm_viz.astype(np.float32))
            lm_viz = vutils.make_grid(lm_viz/255.0, normalize=False, scale_each=False)
            self.tb_writer.add_image('Landmark', lm_viz, i_iter)

            # part segmentaiton
            #pred = pred.transpose(0,2,3,1)
            '''
            pred_crf = np.zeros((pred.shape[0], pred.shape[2], pred.shape[3]))

            for i in range(i_shape[0]):
                output_dcrf_i = denseCRF(imgs_viz[i].numpy().squeeze().transpose(2,1,0).astype(np.uint8).copy(), 
                                         softmax(pred[i]).cpu().numpy())
                pred_crf[i] = np.asarray(np.argmax(output_dcrf_i, axis=2), dtype=np.int)

            pred_crf = self.colorize(pred_crf)
            pred_crf = vutils.make_grid(torch.tensor(pred_crf), normalize=False, scale_each=False)
            pred_crf = (self.imgs_viz_grid + pred_crf)/2
            self.tb_writer.add_image('Part Map (CRF)', pred_crf, i_iter)
            '''

            pred = pred.detach().cpu().float().numpy()
            pred = np.asarray(np.argmax(pred, axis=1), dtype=np.int)
            pred = self.colorize(pred)
            pred = vutils.make_grid(torch.tensor(pred), normalize=False, scale_each=False)
            pred = (self.imgs_viz_grid + pred)/2
            self.tb_writer.add_image('Part Map', pred, i_iter)


    def vis_DFF_heatmaps(self, i_iter, response_maps, threshold=0.5, prefix=''):
        if i_iter % self.vis_interval == 0:
            B,K,H,W = response_maps.shape
            part_response = np.zeros((B,K,H,W,3)).astype(np.uint8)

            for b in range(B):
                for k in range(K):
                    response_map = response_maps[b,k,...].cpu().numpy()
                    response_map = cv2.applyColorMap((response_map*255.0).astype(np.uint8), cv2.COLORMAP_HOT)[:,:,::-1] # BGR->RGB
                    part_response[b,k,:,:,:] = response_map.astype(np.uint8)

            part_response = part_response.transpose(0,1,4,2,3)
            part_response = torch.tensor(part_response.astype(np.float32))
            for k in range(K):
                map_viz_single = vutils.make_grid(part_response[:,k,:,:,:].squeeze()/255.0, normalize=False, scale_each=False)
                self.tb_writer.add_image('{} PART_{}'.format(prefix, k), map_viz_single, i_iter)

            # color segmentation
            response_maps_np = response_maps.cpu().numpy()
            response_maps_np = np.concatenate((np.ones((B,1,H,W))*threshold, response_maps_np), axis=1)
            response_maps_np = np.asarray(np.argmax(response_maps_np, axis=1), dtype=np.int)
            response_maps_np = self.colorize(response_maps_np)
            response_maps_np = vutils.make_grid(torch.tensor(response_maps_np), normalize=False, scale_each=False)
            #response_maps_np = (self.imgs_viz_grid + response_maps_np)/2
            self.tb_writer.add_image('{} Map'.format(prefix), response_maps_np, i_iter)


    def vis_inputs(self, i_iter, inputs, prefix=''):
        if i_iter % self.vis_interval == 0:
            #B,K,H,W = inputs.shape
            #img = np.zeros((B,K,H,W,3)).astype(np.uint8)
            inputs_np = inputs.cpu().numpy()
            inputs_np -= inputs_np.min()
            inputs_np /= inputs_np.max()
            #response_maps_np = np.concatenate((np.ones((B,1,H,W))*threshold, response_maps_np), axis=1)
            #response_maps_np = np.asarray(np.argmax(response_maps_np, axis=1), dtype=np.int)
            #response_maps_np = self.colorize(response_maps_np)
            inputs_np = vutils.make_grid(torch.tensor(inputs_np), normalize=False, scale_each=False)
            self.tb_writer.add_image('inputs_{}'.format(prefix), inputs_np, i_iter)


    def vis_landmarks(self, i_iter, imgs, mean, pred, lms):
        if i_iter % self.vis_interval == 0 :
            i_shape = imgs.shape
            mean_tensor = torch.tensor(mean).float().expand(i_shape[0], i_shape[3], i_shape[2], 3).transpose(1,3)
            imgs_viz = torch.clamp(imgs+mean_tensor, 0.0, 255.0)
            self.imgs_viz = imgs_viz

            lm_viz = Batch_Draw_GT_Landmarks(imgs_viz, pred, lms)
            lm_viz = torch.tensor(lm_viz.astype(np.float32))

            lm_viz = vutils.make_grid(lm_viz/255.0, normalize=False, scale_each=False)
            self.tb_writer.add_image('Landmark_GT', lm_viz, i_iter)


    def vis_bboxes(self, i_iter, bboxes):
        if i_iter % self.vis_interval == 0 :
            bbox_viz = Batch_Draw_Bboxes(self.imgs_viz, bboxes)
            bbox_viz = torch.tensor(bbox_viz.astype(np.float32))

            bbox_viz = vutils.make_grid(bbox_viz/255.0, normalize=False, scale_each=False)
            self.tb_writer.add_image('BBOX_GT', bbox_viz, i_iter)


    def vis_losses(self, i_iter, losses, names):
        for i, loss in enumerate(losses):
            self.tb_writer.add_scalar('data/'+ names[i], loss, i_iter)


    def vis_embeddings(self, i_iter, part_feat_list_all):
        # check visualization interval
        if i_iter % (self.vis_interval*10) != 0:
            return

        feat_list = []
        img_list = []
        label_list = []

        for i in range(len(part_feat_list_all)):
            # i: img index
            for j in range(len(part_feat_list_all[i])):
                # j : part index
                if part_feat_list_all[i][j].shape[0] != 0 :
                    label_list.append(j)
                    img_list.append(self.imgs_viz[i:i+1,...])
                    feat_list.append(part_feat_list_all[i][j].detach().cpu())

        label_tensor = torch.tensor(label_list)
        img_tensor = torch.cat(img_list, dim=0)
        feat_tensor = torch.cat(feat_list, dim=0)
        print('show embedding iter {}'.format(i_iter))
        self.tb_writer.add_embedding(feat_tensor,
                                     tag='part_feature',
                                     metadata=label_tensor,
                                     label_img=img_tensor,
                                     global_step=i_iter)
