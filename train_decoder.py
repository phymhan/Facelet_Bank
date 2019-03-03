import os
import argparse
import random
import functools
import math
import numpy as np
import scipy
from scipy import stats
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.utils
from network import base_network, facelet_net
from network import networks
from tqdm import tqdm
import time
from util.util import upsample2d
import itertools
import cv2
from util import alignface
from network.decoder import vgg_decoder


###############################################################################
# Options | Argument Parser
###############################################################################
class Options():
    def initialize(self, parser):
        parser.add_argument('--mode', type=str, default='train', help='train | test | embedding')
        parser.add_argument('--name', type=str, default='exp', help='experiment name')
        parser.add_argument('--dataroot', required=True, default='datasets/UTKFace', help='path to images')
        parser.add_argument('--datafile', type=str, default='', help='text file listing images')
        parser.add_argument('--dataroot_val', type=str, default='')
        parser.add_argument('--datafile_val', type=str, default='')
        parser.add_argument('--pretrained_model_path', type=str, default='pretrained_models/resnet18-5c106cde.pth', help='path to pretrained models')
        parser.add_argument('--pretrained_model_path_IP', type=str, default='pretrained_models/alexnet-owt-4df8aa71.pth', help='pretrained model path to IP net')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints')
        parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=100, help='batch size')
        parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load')
        parser.add_argument('--which_model', type=str, default='resnet18', help='which model')
        parser.add_argument('--n_layers', type=int, default=3, help='only used if which_model==n_layers')
        parser.add_argument('--nf', type=int, default=64, help='# of filters in first conv layer')
        parser.add_argument('--loadSize', type=int, default=240, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--weight', nargs='+', type=float, default=[], help='weights for FM loss')
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout p')
        parser.add_argument('--lambda_FM', type=float, default=0.0, help='weight for feature matching loss')
        parser.add_argument('--print_freq', type=int, default=10, help='print loss every print_freq iterations')
        parser.add_argument('--display_id', type=int, default=1, help='visdom window id, to disable visdom set id = -1.')
        parser.add_argument('--display_port', type=int, default=8097)
        parser.add_argument('--transforms', type=str, default='resize_affine_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--affineScale', nargs='+', type=float, default=[0.95, 1.05], help='scale tuple in transforms.RandomAffine')
        parser.add_argument('--affineDegrees', type=float, default=5, help='range of degrees in transforms.RandomAffine')
        parser.add_argument('--use_color_jitter', action='store_true', help='if specified, add color jitter in transforms')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--epoch_count', type=int, default=1, help='starting epoch')
        parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of saving the latest results')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--norm_layer', type=str, default='none')
        parser.add_argument('--scale_feat', action='store_true')
        parser.add_argument('--scale_delta', action='store_true')
        parser.add_argument('--norm_feat', type=str, default='group')
        parser.add_argument('--norm_delta', type=str, default='group')
        parser.add_argument('--criterion_rec', type=str, default='mse')
        return parser

    def get_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)
        self.opt = self.parser.parse_args()
        self.opt.use_gpu = len(self.opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.opt.isTrain = self.opt.mode == 'train'
        if self.opt.mode == 'train':
            self.print_options(self.opt)
        return self.opt
    
    def print_options(self, opt):
        message = ''
        message += '--------------- Options -----------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


###############################################################################
# Dataset and Dataloader
###############################################################################
class SiameseNetworkDataset(Dataset):
    def __init__(self, rootdir, source_file, transform=None, warp=True):
        self.rootdir = rootdir
        self.landmarkdir = rootdir.rstrip('/').rstrip('\\') + '_landmark'
        self.source_file = source_file
        self.transform = transform
        self.warp = warp
        self.precompute_landmark = os.path.exists(self.landmarkdir)
        with open(self.source_file, 'r') as f:
            self.source_file = f.readlines()
        if self.warp:
            if not self.precompute_landmark:
                face_d, face_p = alignface.load_face_detector('models/shape_predictor_68_face_landmarks.dat')
                self.detector = face_d
                self.predictor = face_p
            self.M_pool = [None for _ in range(self.__len__())]

    def __getitem__(self, index):
        # s = self.source_file[index].split()
        # imgA = Image.open(os.path.join(self.rootdir, s[0])).convert('RGB')
        # imgB = Image.open(os.path.join(self.rootdir, s[1])).convert('RGB')
        # label = int(s[2])
        # if self.transform != None:
        #     imgA = self.transform(imgA)
        #     imgB = self.transform(imgB)
        # return imgA, imgB, torch.LongTensor(1).fill_(label).squeeze()
        s = self.source_file[index].split()
        imgA = cv2.imread(os.path.join(self.rootdir, s[0]))
        imgB = cv2.imread(os.path.join(self.rootdir, s[1]))
        if self.warp:
            if self.precompute_landmark:
                lmA = self.load_landmark(os.path.join(self.landmarkdir, s[0] + '.landmark'))
                lmB = self.load_landmark(os.path.join(self.landmarkdir, s[1] + '.landmark'))
            else:
                lmA = alignface.detect_landmarks_from_image(imgA, self.detector, self.predictor)
                lmB = alignface.detect_landmarks_from_image(imgB, self.detector, self.predictor)
            if lmA is not None and lmB is not None:
                if self.M_pool[index] is None:
                    M, _ = alignface.fit_face_landmarks(lmB, lmA, landmarks=list(range(68)))
                    self.M_pool[index] = M
                else:
                    M = self.M_pool[index]
                imgB = alignface.warp_to_template(imgB, M, imgA.shape[:2])
        label = int(s[2])
        imgA = Image.fromarray(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB))
        imgB = Image.fromarray(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB))
        if self.transform != None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB, torch.LongTensor(1).fill_(label).squeeze()

    def __len__(self):
        # # shuffle source file
        # random.shuffle(self.source_file)
        return len(self.source_file)

    def load_landmark(self, p):
        with open(p, 'r') as f:
            l = f.readlines()
        return np.array([[float(l_.split()[0]), float(l_.split()[1].rstrip('\n'))] for l_ in l])


class SingleImageDataset(Dataset):
    def __init__(self, rootdir, source_file, transform=None):
        self.rootdir = rootdir
        self.transform = transform
        if source_file:
            with open(source_file, 'r') as f:
                self.source_file = f.readlines()
        else:
            self.source_file = os.listdir(rootdir)

    def __getitem__(self, index):
        imgA = Image.open(os.path.join(self.rootdir, self.source_file[index].rstrip('\n'))).convert('RGB')
        if self.transform != None:
            imgA = self.transform(imgA)
        return imgA, self.source_file[index]

    def __len__(self):
        return len(self.source_file)


class PairImageDataset(Dataset):
    def __init__(self, rootdir, source_file, transform=None):
        self.rootdir = rootdir
        self.source_file = source_file
        self.transform = transform
        with open(self.source_file, 'r') as f:
            self.source_file = [line.rstrip('\n') for line in f.readlines()]

    def __getitem__(self, index):
        s = self.source_file[index].split()
        imgA = Image.open(os.path.join(self.rootdir, s[0])).convert('RGB')
        imgB = Image.open(os.path.join(self.rootdir, s[1])).convert('RGB')
        if self.transform != None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB

    def __len__(self):
        return len(self.source_file)


class ImageEmbeddingDataset(Dataset):
    def __init__(self, rootdir, source_file, transform=None):
        self.rootdir = rootdir
        self.source_file = source_file
        self.transform = transform
        with open(self.source_file, 'r') as f:
            self.source_file = [line.rstrip('\n') for line in f.readlines()]

    def __getitem__(self, index):
        s = self.source_file[index].split()
        imgA = Image.open(os.path.join(self.rootdir, s[0])).convert('RGB')
        embB = float(s[1])
        if self.transform != None:
            imgA = self.transform(imgA)
        return imgA, torch.FloatTensor(1).fill_(embB).squeeze()

    def __len__(self):
        return len(self.source_file)


###############################################################################
# Loss Functions
###############################################################################
# import from models.networks
def total_variation_loss(mat):
    # return torch.mean(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
    #        torch.mean(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))
    return torch.mean(torch.pow(mat[:, :, :, :-1] - mat[:, :, :, 1:], 2)) + \
           torch.mean(torch.pow(mat[:, :, :-1, :] - mat[:, :, 1:, :], 2))


class TVLoss(nn.Module):
    def __init__(self, eps=1e-3, beta=2):
        super(TVLoss, self).__init__()
        self.eps = eps
        self.beta = beta

    def forward(self, input):
        x_diff = input[:, :, :-1, :-1] - input[:, :, :-1, 1:]
        y_diff = input[:, :, :-1, :-1] - input[:, :, 1:, :-1]

        sq_diff = torch.clamp(x_diff * x_diff + y_diff * y_diff, self.eps, 10000000)
        return torch.norm(sq_diff, self.beta / 2.0) ** (self.beta / 2.0)

class FMLoss(nn.Module):
    def __init__(self, weight=None):
        super(FMLoss, self).__init__()
        self.weight = weight if weight != None else [1.0, 1.0, 1.0]
        self.criterion = torch.nn.MSELoss()

    def forward(self, fz, fx):
        loss = 0.0
        for fz_, fx_, w_ in zip(fz, fx, self.weight):
            fx_ = fx_.detach()
            fx_.requires_grad = False
            loss += self.criterion(fz_, fx_) * w_
        return loss


###############################################################################
# Networks and Models
###############################################################################
def _scale(feat):
    numel = 0.0
    for f in feat:
        numel += f.size(1)
    return [f/numel for f in feat]


def _normalize(feat, norm):
    if norm == 'group':
        # group normalize
        feat = [f/torch.sum(f.pow(2).view(f.size(0), -1), dim=1).sqrt().view(f.size(0), 1, 1, 1) for f in feat]
    elif norm == 'concat':
        # concat normalize
        norm = torch.sum(feat[0].pow(2).view(feat[0].size(0), -1), dim=1) + \
               torch.sum(feat[1].pow(2).view(feat[1].size(0), -1), dim=1) + \
               torch.sum(feat[2].pow(2).view(feat[2].size(0), -1), dim=1)
        norm = norm.sqrt().view(norm.size(0), 1, 1, 1)
        feat = [f/norm for f in feat]
    return feat


def _minus(feat1, feat2):
    return [f1 - f2 for f1, f2 in zip(feat1, feat2)]


def _inner_prod(feat1, feat2):
    a = 0.0
    n = feat1[0].size(0)
    for f1, f2 in zip(feat1, feat2):
        a += torch.sum(f1.view(n, -1) * f2.view(n, -1), dim=1)
    return a.view(n, 1, 1, 1)


class SiameseNetwork(nn.Module):
    def __init__(self, base=None, facelet=None, scale_feat=True, scale_delta=True, norm_feat='group', norm_delta='group'):
        super(SiameseNetwork, self).__init__()
        # base
        self.base = base
        self.facelet = facelet
        self.norm_feat = norm_feat
        self.norm_delta = norm_delta
        self.scale_feat = scale_feat
        self.scale_delta = scale_delta

    def _process_feature(self, feat, scale=True, norm='group'):
        if scale:
            feat = _scale(feat)
        feat = _normalize(feat, norm)
        return feat

    def get_feature(self, x):
        feat = self.base.forward(x)
        return feat

    def forward(self, x1, x2):
        feat1 = self.get_feature(x1)
        feat2 = self.get_feature(x2)

        delta1 = self.get_delta(feat1)
        feat_diff = _minus(feat1, feat2)

        feat_diff = self._process_feature(feat_diff, self.scale_feat, self.norm_feat)
        delta1 = self._process_feature(delta1, self.scale_delta, self.norm_delta)

        a = _inner_prod(feat_diff, delta1)
        # print(a)

        return feat1, feat2, a

    def get_delta(self, feat):
        delta = self.facelet.forward(feat)
        return delta

    def get_inner_prod(self, x):
        feat = self.get_feature(x)
        delta = self.get_delta(feat)

        feat = self._process_feature(feat, self.scale_feat, self.norm_feat)
        delta = self._process_feature(delta, self.scale_delta, self.norm_delta)

        a = self.inner_prod(feat, delta)
        return a

    def get_heat_map(self, x):
        feat = self.base.forward(x)
        # feature_ = torch.cat([feat[0]*0, upsample2d(feat[1], 56)*1, upsample2d(feat[2], 56)*0], 1)
        feature_ = feat[0]
        heat_map = torch.sum(feature_.pow(2), 1, keepdim=True)
        return heat_map

    def get_heat_map_fc(self, x):
        feat = self.base.forward(x)

        norms = [torch.sum(f.pow(2).view(f.size(0), -1), dim=1).sqrt() for f in feat]
        print(norms)

        delta = self.get_delta(feat)

        norms = [torch.sum(f.pow(2).view(f.size(0), -1), dim=1).sqrt() for f in delta]
        print(norms)

        feature_ = torch.cat([delta[0]*1, upsample2d(delta[1], 56)*1, upsample2d(delta[2], 56)*1], 1)
        # feature_ = delta[2]
        heat_map = torch.sum(feature_.pow(2), 1, keepdim=True)
        return heat_map


###############################################################################
# Helper Functions | Utilities
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_attr_value(fname):
    return float(fname.split('_')[0])


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_grad(params):
    """Sets gradients of all model parameters to zero."""
    for p in params():
        if p.grad is not None:
            p.grad.data.zero_()


# just modify the width and height to be multiple of 4
def __adjust(img):
    ow, oh = img.size

    # the size needs to be a multiple of this number, 
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4 
    if ow % mult == 0 and oh % mult == 0:
        return img
    w = (ow - 1) // mult
    w = (w + 1) * mult
    h = (oh - 1) // mult
    h = (h + 1) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)
        
    return img.resize((w, h), Image.BICUBIC)


def __scale_width(img, target_width):
    ow, oh = img.size
    
    # the size needs to be a multiple of this number, 
    # because going through generator network may change img size
    # and eventually cause size mismatch error    
    mult = 4
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if (ow == target_width and oh % mult == 0):
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)
    
    return img.resize((w, h), Image.BICUBIC)


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


def tensor2image(image_tensor):
    image = image_tensor[0].cpu().numpy().transpose((1, 2, 0))
    image = (image * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])) * 255
    return image.transpose((2, 0, 1))


def feature2image(image_tensor):
    image = image_tensor[0].cpu().numpy().transpose((1, 2, 0))
    image = (image - image.min()) / (image.max() - image.min()) * 255
    return image.transpose((2, 0, 1))


###############################################################################
# Main Routines
###############################################################################
def get_model(opt):
    base = base_network.VGG(pretrained=True)
    opt.pretrained = False
    facelet = facelet_net.Facelet(opt)
    net = SiameseNetwork(base=base, facelet=facelet,
                         scale_feat=opt.scale_feat, scale_delta=opt.scale_delta,
                         norm_feat=opt.norm_feat, norm_delta=opt.norm_delta)
    if opt.mode == 'train' and not opt.continue_train:
        print('>> weight not initialized')
        # net.facelet.apply(weights_init)
    else:
        # HACK: strict=False
        net.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, opt.name, '{}_net.pth'.format(opt.which_epoch))), strict=True)

    if opt.mode != 'train':
        set_requires_grad(net, False)
        net.eval()

    if opt.use_gpu:
        net.cuda()
    return net


def get_decoder(opt):
    decoder = vgg_decoder(False)
    if opt.use_gpu:
        decoder.cuda()
    return decoder

def get_vgg(opt):
    vgg = base_network.VGG(pretrained=True)
    set_requires_grad(vgg, False)
    if opt.use_gpu:
        vgg.cuda()
    return vgg


def get_transform(opt):
    transform_list = []
    if opt.transforms == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.transforms == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.transforms == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.transforms == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.transforms == 'none':
        transform_list.append(transforms.Lambda(
            lambda img: __adjust(img)))
    elif opt.transforms == 'resize_affine_crop':
        transform_list.append(transforms.Resize([opt.loadSize, opt.loadSize], Image.BICUBIC))
        transform_list.append(transforms.RandomAffine(degrees=opt.affineDegrees, scale=tuple(opt.affineScale),
                                                      resample=Image.BICUBIC, fillcolor=127))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.transforms == 'resize_affine_center':
        transform_list.append(transforms.Resize([opt.loadSize, opt.loadSize], Image.BICUBIC))
        transform_list.append(transforms.RandomAffine(degrees=opt.affineDegrees, scale=tuple(opt.affineScale),
                                                      resample=Image.BICUBIC, fillcolor=127))
        transform_list.append(transforms.CenterCrop(opt.fineSize))
    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.transforms)

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if opt.isTrain and opt.use_color_jitter:
        transform_list.append(transforms.ColorJitter())  # TODO

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    return transforms.Compose(transform_list)


def forward(image, vgg, decoder):
    vgg_feat = vgg.forward(image)
    return decoder.forward(vgg_feat, image)


# Routines for training
def train(opt, vgg, decoder, dataloader, dataloader_val=None):
    opt.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    if opt.display_id >= 0 and not os.path.exists(os.path.join(opt.save_dir, 'images')):
        os.makedirs(os.path.join(opt.save_dir, 'images'))
    
    # criterion
    if opt.criterion_rec == 'mse':
        criterionRec = torch.nn.MSELoss()
    elif opt.criterion_rec == 'l1':
        criterionRec = torch.nn.L1Loss()
    criterionFM = FMLoss(opt.weight)

    # optimizer
    optimizer = optim.Adam(decoder.parameters(), lr=opt.lr)

    dataset_size, dataset_size_val = opt.dataset_size, opt.dataset_size_val
    loss_history = []
    total_iter = 0
    num_iter_per_epoch = math.ceil(dataset_size / opt.batch_size)
    opt.display_val_acc = not not dataloader_val
    loss_legend = ['Rec']
    if opt.lambda_FM > 0:
        loss_legend.append('FM')
    if opt.display_id >= 0:
        import visdom
        vis = visdom.Visdom(server='http://localhost', port=opt.display_port)
        plot_loss = {'X': [], 'Y': [], 'leg': loss_legend}

    torch.save(decoder.cpu().state_dict(), os.path.join(opt.save_dir, 'init_net.pth'))
    if opt.use_gpu:
        decoder.cuda()

    # start training
    for epoch in range(opt.epoch_count, opt.num_epochs+opt.epoch_count):
        epoch_iter = 0

        for i, data in enumerate(dataloader, 0):
            img0, path0 = data
            if opt.use_gpu:
                img0 = img0.cuda()
            epoch_iter += 1
            total_iter += 1

            optimizer.zero_grad()

            # net forward
            recon = forward(img0, vgg, decoder)

            losses = {}
            # Reconstruction loss
            loss = criterionRec(recon, img0)
            losses['Rec'] = loss.item()

            # feature matching
            if opt.lambda_FM > 0.0:
                feat_image = vgg.forward(img0)
                feat_recon = vgg.forward(recon)
                this_loss = criterionFM(feat_image, feat_recon) * opt.lambda_FM
                loss += this_loss
                losses['FM'] = this_loss.item()

            loss.backward()
            optimizer.step()

            if total_iter % opt.print_freq == 0:
                print("epoch %02d, iter %06d, loss: %.4f" % (epoch, total_iter, loss.item()))
                if opt.display_id >= 0:
                    plot_loss['X'].append(epoch-1+epoch_iter/num_iter_per_epoch)
                    plot_loss['Y'].append([losses[k] for k in plot_loss['leg']])
                    vis.line(
                        X=np.stack([np.array(plot_loss['X'])] * len(plot_loss['leg']), 1),
                        Y=np.array(plot_loss['Y']),
                        opts={'title': 'loss', 'legend': plot_loss['leg'], 'xlabel': 'epoch', 'ylabel': 'loss'},
                        win=opt.display_id
                    )

                if opt.display_id >= 0:
                    images = [tensor2image(img0.detach()), tensor2image(recon.detach())]
                    vis.images(images, win=opt.display_id + 10)

                    # save concatenated images
                    images_pad = []
                    for image in images:
                        images_pad.append(image.transpose((1, 2, 0)))
                    scipy.misc.imsave(os.path.join(opt.save_dir, 'images', 'ep%02d_it%06d.png' % (epoch, total_iter)),
                                      np.concatenate(images_pad, axis=1))


                loss_history.append(loss.item())
            
            if total_iter % opt.save_latest_freq == 0:
                torch.save(decoder.cpu().state_dict(), os.path.join(opt.save_dir, 'latest_net.pth'))
                if opt.use_gpu:
                    decoder.cuda()
                if epoch % opt.save_epoch_freq == 0:
                    torch.save(decoder.cpu().state_dict(), os.path.join(opt.save_dir, '{}_net.pth'.format(epoch)))
                    if opt.use_gpu:
                        decoder.cuda()

        torch.save(decoder.cpu().state_dict(), os.path.join(opt.save_dir, 'latest_net.pth'))
        if opt.use_gpu:
            decoder.cuda()
        if epoch % opt.save_epoch_freq == 0:
            torch.save(decoder.cpu().state_dict(), os.path.join(opt.save_dir, '{}_net.pth'.format(epoch)))
            if opt.use_gpu:
                decoder.cuda()

    with open(os.path.join(opt.save_dir, 'loss.txt'), 'w') as f:
        for loss in loss_history:
            f.write(str(loss)+'\n')


def F_decode(opt, net, decoder, dataloader):
    import visdom
    vis = visdom.Visdom(server='http://localhost', port=opt.display_port)

    delta = -0.003

    for i, data in enumerate(dataloader, 0):
        img0, path0 = data
        if opt.use_gpu:
            img0 = img0.cuda()
        emb0 = net.forward_once(img0)
        W1, W2, W3 = net.get_delta(emb0)
        emb1 = [emb0[0] + W1 * delta, emb0[1] + W2 * delta * 1, emb0[2] + W3 * delta * 1]

        img1 = decoder.forward(emb1, img0)

        images = []
        images += [tensor2image(img0.detach())]
        images += [tensor2image(img1.detach())]
        vis.images(images, win=opt.display_id + 10)

        # save_image(images[0], 'results/image_%d_A.png' % i)
        # save_image(images[1], 'results/image_%d_B.png' % i)


# Routines for visualization
def save_image(npy, path):
    scipy.misc.imsave(path, npy.transpose((1,2,0)))


###############################################################################
# main()
###############################################################################
# TODO: set random seed

if __name__=='__main__':
    opt = Options().get_options()

    # get model
    vgg = get_vgg(opt)
    decoder = get_decoder(opt)

    if opt.mode == 'train':
        # get dataloader
        dataset = SingleImageDataset(opt.dataroot, opt.datafile, get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=not opt.serial_batches, num_workers=opt.num_workers, batch_size=opt.batch_size)
        opt.dataset_size = len(dataset)
        # val dataset
        if opt.dataroot_val:
            dataset_val = SingleImageDataset(opt.dataroot_val, opt.datafile_val, get_transform(opt))
            dataloader_val = DataLoader(dataset_val, shuffle=False, num_workers=0, batch_size=1)
            opt.dataset_size_val = len(dataset_val)
        else:
            dataloader_val = None
            opt.dataset_size_val = 0
        print('dataset size = %d' % len(dataset))
        # train
        train(opt, vgg, decoder, dataloader, dataloader_val)
    else:
        raise NotImplementedError('Mode [%s] is not implemented.' % opt.mode)
