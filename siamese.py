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
from network import base_network
from tqdm import tqdm
import time


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
        parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load')
        parser.add_argument('--which_model', type=str, default='resnet18', help='which model')
        parser.add_argument('--n_layers', type=int, default=3, help='only used if which_model==n_layers')
        parser.add_argument('--nf', type=int, default=64, help='# of filters in first conv layer')
        parser.add_argument('--pooling', type=str, default='avg', help='empty: no pooling layer, max: MaxPool, avg: AvgPool')
        parser.add_argument('--loadSize', type=int, default=240, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--weight', nargs='+', type=float, default=[], help='weights for CE')
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout p')
        parser.add_argument('--finetune_fc_only', action='store_true', help='fix feature extraction weights and finetune fc layers only, if True')
        parser.add_argument('--fc_dim', type=int, nargs='*', default=[], help='dimension of fc')
        parser.add_argument('--fc_relu_slope', type=float, default=0.3)
        parser.add_argument('--fc_residual', action='store_true', help='use residual fc')
        parser.add_argument('--cnn_dim', type=int, nargs='+', default=[64, 1], help='cnn kernel dims for feature dimension reduction')
        parser.add_argument('--cnn_pad', type=int, default=1, help='padding of cnn layers defined by cnn_dim')
        parser.add_argument('--cnn_relu_slope', type=float, default=0.7)
        parser.add_argument('--no_cxn', action='store_true', help='if true, do **not** add batchNorm and ReLU between cnn and fc')
        parser.add_argument('--lambda_regularization', type=float, default=0.0, help='weight for feature regularization loss')
        parser.add_argument('--lambda_contrastive', type=float, default=0.0, help='weight for contrastive loss')
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
        parser.add_argument('--draw_prob_thresh', type=float, default=0.16)
        parser.add_argument('--which_model_stn', type=str, default='None')
        parser.add_argument('--fineSize_ST', type=int, default=128, help='fineSize for STN')
        parser.add_argument('--theta_loss', type=str, default='mse')
        parser.add_argument('--lambda_theta', type=float, default=0.0, help='regularization on theta in STN')

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
    def __init__(self, rootdir, source_file, transform=None):
        self.rootdir = rootdir
        self.source_file = source_file
        self.transform = transform
        with open(self.source_file, 'r') as f:
            self.source_file = f.readlines()

    def __getitem__(self, index):
        s = self.source_file[index].split()
        imgA = Image.open(os.path.join(self.rootdir, s[0])).convert('RGB')
        imgB = Image.open(os.path.join(self.rootdir, s[1])).convert('RGB')
        label = int(s[2])
        if self.transform != None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB, torch.LongTensor(1).fill_(label).squeeze()

    def __len__(self):
        # # shuffle source file
        # random.shuffle(self.source_file)
        return len(self.source_file)


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


###############################################################################
# Networks and Models
###############################################################################
# moved to models.networks
class SiameseNetwork(nn.Module):
    def __init__(self, base=None):
        super(SiameseNetwork, self).__init__()
        # base
        self.base = base
        self.feature_num = 0
        self.feature_dim = None
        self.fc = []

    def forward_once(self, x):
        theta = None
        if self.stn:
            x, theta = self.stn(x)
        output = self.base.forward(x)
        if self.cnn:
            output = self.cnn(output)
        if self.cxn:
            output = self.cxn(output)
        if self.pooling == 'avg':
            output = nn.AvgPool2d(output.size(2))(output)
        elif self.pooling == 'max':
            output = nn.MaxPool2d(output.size(2))(output)
        return output

    def forward(self, input1, input2):
        feature1 = self.forward_once(input1)
        feature2 = self.forward_once(input2)


        if self.fc and self.residual:
            output = torch.cat((feature1, feature2), dim=1)
            output = self.fc(output) + (feature1-feature2)
        elif self.fc and not self.residual:
            output = torch.cat((feature1-feature2, feature1, feature2), dim=1)
            output = self.fc(output)
        else:
            output = feature1-feature2

        return feature1, feature2, output, theta1, theta2

    def load_pretrained(self, state_dict):
        # used when loading pretrained base model
        # warning: self.cnn and self.fc won't be initialized
        self.base.load_pretrained(state_dict)


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


# def get_prediction(score):
#     batch_size = score.size(0)
#     score_cpu = score.detach().cpu().numpy()
#     pred = stats.mode(score_cpu.argmax(axis=1).reshape(batch_size, -1), axis=1)
#     return pred[0].reshape(batch_size)
def get_prediction(score, draw_thresh=0.1):
    batch_size = score.size(0)
    prob = torch.sigmoid(score)
    idx1 = torch.abs(0.5-prob) < draw_thresh
    idx0 = (prob > 0.5) & (1-idx1)
    idx2 = (prob <= 0.5) & (1-idx1)
    pred = idx0*0 + idx1*1 + idx2*2
    pred_cpu = pred.detach().cpu().numpy()
    pred = stats.mode(pred_cpu.reshape(batch_size, -1), axis=1)
    return pred[0].reshape(batch_size)


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
def my_get_model(opt):
    base = base_network.VGG()



def get_model(opt):
    # define base model
    base = None
    if opt.which_model == 'alexnet':
        base = networks.AlexNetFeature(input_nc=3, pooling='')
    elif 'resnet' in opt.which_model:
        base = networks.ResNetFeature(input_nc=3, which_model=opt.which_model)
    else:
        raise NotImplementedError('Model [%s] is not implemented.' % opt.which_model)

    # define Siamese Network
    # FIXME: SiameseNetwork or SiameseFeature according to opt.mode
    if opt.mode == 'train' or opt.mode == 'attention':
        stn = networks.define_STN(input_nc=3, which_model_stn=opt.which_model_stn, size_in=opt.fineSize_ST,
                                  size_out=opt.fineSize, output_theta=True, init_type=opt.init_type,
                                  gpu_ids=opt.gpu_ids)
        net = networks.SiameseNetwork(base, pooling=opt.pooling, cnn_dim=opt.cnn_dim, cnn_pad=opt.cnn_pad,
                                      cnn_relu_slope=opt.cnn_relu_slope, fc_dim=opt.fc_dim,
                                      fc_relu_slope=opt.fc_relu_slope, fc_residual=opt.fc_residual,
                                      dropout=opt.dropout, no_cxn=opt.no_cxn, stn=stn)
    else:  # 'embedding' or 'optimize'
        stn = networks.define_STN(input_nc=3, which_model_stn=opt.which_model_stn, size_in=opt.fineSize_ST,
                                  size_out=opt.fineSize, output_theta=False, init_type=opt.init_type,
                                  gpu_ids=opt.gpu_ids)
        net = networks.SiameseFeature(base, pooling=opt.pooling, cnn_dim=opt.cnn_dim, cnn_pad=opt.cnn_pad,
                                      cnn_relu_slope=opt.cnn_relu_slope, stn=stn)

    # initialize | load weights
    if opt.mode == 'train' and not opt.continue_train:
        net.apply(weights_init)
        if opt.pretrained_model_path:
            if isinstance(net, torch.nn.DataParallel):
                net.module.load_pretrained(opt.pretrained_model_path)
            else:
                net.load_pretrained(opt.pretrained_model_path)
        if stn is not None:
            if isinstance(net, torch.nn.DataParallel):
                net.module.stn.init_identity()
            else:
                net.stn.init_identity()
    else:
        # HACK: strict=False
        net.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, opt.name, '{}_net.pth'.format(opt.which_epoch))), strict=False)
    
    if opt.mode != 'train':
        set_requires_grad(net, False)
        net.eval()
    
    if opt.use_gpu:
        net.cuda()
    return net


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


# Routines for training
def train(opt, net, dataloader, dataloader_val=None):
    if opt.which_model_stn.lower() == 'none':
        opt.lambda_theta = 0
    if opt.lambda_contrastive > 0:
        criterion_constrastive = networks.ContrastiveLoss()
    opt.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    # criterion
    criterion = networks.BinaryCrossEntropyLoss()
    criterion_theta = networks.AffineIdentityLoss(opt.theta_loss, torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor)

    # optimizer
    if opt.finetune_fc_only:
        if isinstance(net, nn.DataParallel):
            param = net.module.get_finetune_parameters()
        else:
            param = net.get_finetune_parameters()
    else:
        param = net.parameters()
    optimizer = optim.Adam(param, lr=opt.lr)

    dataset_size, dataset_size_val = opt.dataset_size, opt.dataset_size_val
    loss_history = []
    total_iter = 0
    num_iter_per_epoch = math.ceil(dataset_size / opt.batch_size)
    opt.display_val_acc = not not dataloader_val
    loss_legend = ['classification']
    if opt.lambda_theta > 0:
        loss_legend.append('theta')
    if opt.lambda_contrastive > 0:
        loss_legend.append('contrastive')
    if opt.lambda_regularization > 0:
        loss_legend.append('regularization')
    if opt.display_id >= 0:
        import visdom
        vis = visdom.Visdom(server='http://localhost', port=opt.display_port)
        # plot_data = {'X': [], 'Y': [], 'leg': ['loss']}
        plot_loss = {'X': [], 'Y': [], 'leg': loss_legend}
        plot_acc = {'X': [], 'Y': [], 'leg': ['train', 'val'] if opt.display_val_acc else ['train']}

    torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, 'init_net.pth'))
    if opt.use_gpu:
        net.cuda()
    
    # start training
    for epoch in range(opt.epoch_count, opt.num_epochs+opt.epoch_count):
        epoch_iter = 0
        pred_train = []
        target_train = []

        for i, data in enumerate(dataloader, 0):
            img0, img1, label = data
            if opt.use_gpu:
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            epoch_iter += 1
            total_iter += 1

            optimizer.zero_grad()

            # net forward
            feat1, feat2, score, th1, th2 = net(img0, img1)

            losses = {}
            # classification loss
            loss = criterion(score, label)
            losses['classification'] = loss.item()

            # regularization on theta
            if opt.lambda_theta > 0:
                this_loss = (criterion_theta(th1)+criterion_theta(th2))/2 * opt.lambda_theta
                loss += this_loss
                losses['theta'] = this_loss.item()

            # contrastive loss
            if opt.lambda_contrastive > 0:
                # new label: 0 similar (1), 1 dissimilar (0, 2)
                idx = label == 1
                label_new = label.clone()
                label_new[idx] = 0
                label_new[1-idx] = 1
                this_loss = criterion_constrastive(
                    feat1.view(img0.size(0), -1), feat2.view(img0.size(0), -1), label_new.float()
                    ) * opt.lambda_contrastive
                loss += this_loss
                losses['contrastive'] = this_loss.item()
            
            # regularization
            if opt.lambda_regularization > 0:
                reg1 = feat1.pow(2).mean()
                reg2 = feat2.pow(2).mean()
                this_loss = (reg1 + reg2) * opt.lambda_regularization
                loss += this_loss
                losses['regularization'] = this_loss.item()
            
            # loss = loss_classification + loss_contrastive + loss_regularization

            # get predictions
            pred_train.append(get_prediction(score, opt.draw_prob_thresh))
            target_train.append(label.cpu().numpy())

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

                    if opt.which_model_stn.lower() != 'none':
                        image_tensor = img0[0:1, ...]
                        image = image_tensor[0].cpu().numpy().transpose((1, 2, 0))
                        image = (image * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])) * 255
                        image_tensor_transformed, _ = net.stn(image_tensor)
                        image_transformed = image_tensor_transformed.detach()[0].cpu().numpy().transpose((1, 2, 0))
                        image_transformed = (image_transformed * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])) * 255
                        vis.images([image.transpose((2, 0, 1)), image_transformed.transpose((2, 0, 1))], win=opt.display_id + 10)

                loss_history.append(loss.item())
            
            if total_iter % opt.save_latest_freq == 0:
                torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, 'latest_net.pth'))
                if opt.use_gpu:
                    net.cuda()
                if epoch % opt.save_epoch_freq == 0:
                    torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, '{}_net.pth'.format(epoch)))
                    if opt.use_gpu:
                        net.cuda()
        
        curr_acc = {}
        # evaluate training
        err_train = np.count_nonzero(np.concatenate(pred_train) - np.concatenate(target_train)) / dataset_size
        curr_acc['train'] = 1 - err_train

        # evaluate val
        if opt.display_val_acc:
            pred_val = []
            target_val = []
            for i, data in enumerate(dataloader_val, 0):
                img0, img1, label = data
                if opt.use_gpu:
                    img0, img1 = img0.cuda(), img1.cuda()
                _, _, output, _, _ = net.forward(img0, img1)
                pred_val.append(get_prediction(output, opt.draw_prob_thresh))
                target_val.append(label.cpu().numpy())
            err_val = np.count_nonzero(np.concatenate(pred_val) - np.concatenate(target_val)) / dataset_size_val
            curr_acc['val'] = 1 - err_val

        # plot accs
        if opt.display_id >= 0:
            plot_acc['X'].append(epoch)
            plot_acc['Y'].append([curr_acc[k] for k in plot_acc['leg']])
            vis.line(
                X=np.stack([np.array(plot_acc['X'])] * len(plot_acc['leg']), 1),
                Y=np.array(plot_acc['Y']),
                opts={'title': 'accuracy', 'legend': plot_acc['leg'], 'xlabel': 'epoch', 'ylabel': 'accuracy'},
                win=opt.display_id+1
            )
            sio.savemat(os.path.join(opt.save_dir, 'mat_loss'), plot_loss)
            sio.savemat(os.path.join(opt.save_dir, 'mat_acc'), plot_acc)

        torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, 'latest_net.pth'))
        if opt.use_gpu:
            net.cuda()
        if epoch % opt.save_epoch_freq == 0:
            torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, '{}_net.pth'.format(epoch)))
            if opt.use_gpu:
                net.cuda()

    with open(os.path.join(opt.save_dir, 'loss.txt'), 'w') as f:
        for loss in loss_history:
            f.write(str(loss)+'\n')


# Routines for testing
def test(opt, net, dataloader):
    dataset_size_val = opt.dataset_size_val
    pred_val = []
    target_val = []
    for i, data in enumerate(dataloader, 0):
        img0, img1, label = data
        if opt.use_gpu:
            img0, img1 = img0.cuda(), img1.cuda()
        _, _, output, _, _ = net.forward(img0, img1)

        pred_val.append(get_prediction(output, opt.draw_prob_thresh).squeeze())
        target_val.append(label.cpu().numpy().squeeze())
        print('--> batch #%d' % (i+1))

    err = np.count_nonzero(np.stack(pred_val) - np.stack(target_val)) / dataset_size_val
    print('================================================================================')
    print('accuracy: %.6f' % (100. * (1-err)))


# Routines for extracting embedding
def embedding(opt, net, dataloader):
    features = []
    labels = []
    for _, data in enumerate(dataloader, 0):
        img0, path0 = data
        if opt.use_gpu:
            img0 = img0.cuda()
        feature = net.forward(img0)
        feature = feature.cpu().detach().numpy()
        features.append(feature.reshape([1, net.feature_dim]))
        labels.append(get_attr_value(path0[0]))
        print('--> %s' % path0[0])

    X = np.concatenate(features, axis=0)
    labels = np.array(labels)
    np.save(os.path.join(opt.checkpoint_dir, opt.name, 'features_%s.npy' % opt.which_epoch), X)
    np.save(os.path.join(opt.checkpoint_dir, opt.name, 'labels_%s.npy' % opt.which_epoch), labels)


# Routines for visualization
def optimize(opt, net, dataloader):
    import visdom
    vis = visdom.Visdom(server='http://localhost', port=opt.display_port)

    netIP = networks.AlexNetFeature(input_nc=3, pooling='None')
    netIP.cuda()
    if isinstance(netIP, torch.nn.DataParallel):
        netIP.module.load_pretrained(opt.pretrained_model_path_IP)
    else:
        netIP.load_pretrained(opt.pretrained_model_path_IP)

    for i, data in enumerate(dataloader, 0):
        img0, emb1 = data
        if opt.use_gpu:
            img0, emb1 = img0.cuda(), emb1.cuda()
        img1 = optimize_image(img0, emb1, net, netIP, opt)

        images = []
        images += [tensor2image(img0.detach())]
        images += [tensor2image(img1.detach())]
        vis.images(images, win=opt.display_id + 10)

        # hack
        save_image(images[0], 'samples_vis/optimize/iter%d_o.png' % i)
        save_image(images[1], 'samples_vis/optimize/iter%d_t.png' % i)


def optimize_image(img, emb, netE, netIP, opt, n_iter=100, lr=0.1):
    img_orig = img
    img = img_orig.clone()
    img.requires_grad = True
    optim_input = optim.LBFGS([img], lr=lr)
    emb = emb.view(1, 1, 1, 1)
    # tv_loss = TVLoss()

    def closure():
        optim_input.zero_grad()
        img_emb = netE.forward(img)

        loss = 0.5 * torch.nn.MSELoss()(img_emb, emb) + total_variation_loss(img) * 0.1
        loss.backward()
        return loss

    for _ in tqdm(range(100)):
        optim_input.step(closure)

    return img


# def optimize_image(img, emb, netE, netIP, opt, n_iter=100, lr=0.1):
#     img_orig = img
#     img = img_orig.clone()
#     img.zero_()
#     img.requires_grad = True
#     optim_input = optim.LBFGS([img], lr=lr)
#     emb = emb.view(1, 1, 1, 1)
#
#     def closure():
#         optim_input.zero_grad()
#         img_emb = netE.forward(img)
#
#         feature_orig = netIP(img_orig).detach()
#         feature_orig.requires_grad = False
#         loss_IP = torch.nn.MSELoss()(netIP(img), feature_orig) * 1
#         loss = 0.5 * torch.nn.MSELoss()(img_emb, emb) + 0.001 * total_variation_loss(img) + 0.1 * loss_IP
#         loss.backward()
#         return loss
#
#     for _ in tqdm(range(n_iter)):
#         optim_input.step(closure)
#
#     return img

def save_image(npy, path):
    scipy.misc.imsave(path, npy.transpose((1,2,0)))


def attention(opt, net, dataloader):
    import visdom
    vis = visdom.Visdom(server='http://localhost', port=opt.display_port)

    update_relus(net)

    for i, data in enumerate(dataloader, 0):
        img0, img1, label = data
        if opt.use_gpu:
            img0, img1 = img0.cuda(), img1.cuda()
        att0, att1 = get_attention(img0, img1, label, net, opt)
        # att0, att1 = att0.abs(), att1.abs()
        alpha = 0.01
        images = []
        image = tensor2image(img0.detach())
        image_ = image
        images += [image]
        image = feature2image(att0.detach())
        image = image * (1-alpha) + image_ * alpha
        images += [image]
        image = tensor2image(img1.detach())
        image_ = image
        images += [image]
        image = feature2image(att1.detach())
        image = image * (1 - alpha) + image_ * alpha
        images += [image]
        vis.images(images, win=opt.display_id + 10)

        # hack
        save_image(images[0], 'samples_vis/attention/iter%d_A_x.png' % i)
        save_image(images[1], 'samples_vis/attention/iter%d_A_a.png' % i)
        save_image(images[2], 'samples_vis/attention/iter%d_B_x.png' % i)
        save_image(images[3], 'samples_vis/attention/iter%d_B_a.png' % i)
        print('--> iter#%d' % i)
        # time.sleep(1)


def get_attention(img0, img1, label, net, opt):
    """
        backprop / deconv
    """
    img0.requires_grad = True
    img1.requires_grad = True
    feat1, feat2, score, _, _ = net(img0, img1)
    net.zero_grad()
    # prob = torch.nn.functional.sigmoid(score)
    grad_tensor = torch.FloatTensor([1]).view(1, 1, 1, 1).cuda()
    torch.autograd.backward(score, grad_tensor)
    return img0.grad, img1.grad


# def get_attention(img0, img1, label, net, opt):
#     """
#         backprop
#     """
#     img0.requires_grad = True
#     img1.requires_grad = True
#     feat1, feat2, score, _, _ = net(img0, img1)
#     loss = networks.BinaryCrossEntropyLoss()(score, label.cuda())
#     loss.backward()
#     return img0.grad, img1.grad


def update_relus(model):
    """
        Updates relu activation functions so that it only returns positive gradients
    """
    from torch.nn import ReLU

    def relu_hook_function(module, grad_in, grad_out):
        """
        If there is a negative gradient, changes it to zero
        """
        if isinstance(module, ReLU):
            return (torch.clamp(grad_in[0], min=0.0),)

    def set_relu_hook(m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('ReLU') != -1:
            m.register_backward_hook(relu_hook_function)

    net.apply(set_relu_hook)


###############################################################################
# main()
###############################################################################
# TODO: set random seed

if __name__=='__main__':
    opt = Options().get_options()

    # get model
    net = get_model(opt)

    if opt.mode == 'train':
        # get dataloader
        dataset = SiameseNetworkDataset(opt.dataroot, opt.datafile, get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=not opt.serial_batches, num_workers=opt.num_workers, batch_size=opt.batch_size)
        opt.dataset_size = len(dataset)
        # val dataset
        if opt.dataroot_val:
            dataset_val = SiameseNetworkDataset(opt.dataroot_val, opt.datafile_val, get_transform(opt))
            dataloader_val = DataLoader(dataset_val, shuffle=False, num_workers=0, batch_size=10)
            opt.dataset_size_val = len(dataset_val)
        else:
            dataloader_val = None
            opt.dataset_size_val = 0
        print('dataset size = %d' % len(dataset))
        # train
        train(opt, net, dataloader, dataloader_val)
    elif opt.mode == 'test':
        # get dataloader
        dataset = SiameseNetworkDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=opt.num_workers, batch_size=opt.batch_size)
        opt.dataset_size_val = len(dataset)
        # test
        test(opt, net, dataloader)
    elif opt.mode == 'embedding':
        # get dataloader
        dataset = SingleImageDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)
        # get embedding
        embedding(opt, net, dataloader)
    elif opt.mode == 'optimize':
        # get dataloader
        dataset = ImageEmbeddingDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)
        # get embedding
        optimize(opt, net, dataloader)
    elif opt.mode == 'attention':
        # get dataloader
        dataset = SiameseNetworkDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)
        # get embedding
        attention(opt, net, dataloader)
    else:
        raise NotImplementedError('Mode [%s] is not implemented.' % opt.mode)
