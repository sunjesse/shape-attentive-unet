# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import torch.utils.data as data
# Our libs
from data.augmentations import Compose, PaddingCenterCrop
from data.ac17_dataloader import AC17Data as AC17, AC17_2DLoad as load2D
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm
from loss import ACLoss
import random
from PIL import Image
#colors = loadmat('data/color150.mat')['colors']

def get_heatmap(LRP):
    print(LRP.shape)
    heatmap = LRP[0].permute(1, 2, 0).data.cpu().numpy()
    heatmap = np.absolute(heatmap)
    heatmap = 255 * (heatmap-heatmap.min()) / (heatmap.max()-heatmap.min()+1e-10)
    return heatmap.astype(np.uint8)

def visualize_result(data, pred, edge, att, args):
    (img, seg, info) = data

    # segmentation
    #seg_color = colorEncode(seg.astype(np.uint8), [args.num_class,3])

    # prediction
    #pred_color = colorEncode(pred.astype(np.uint8), [args.num_class,3])
    
    # aggregate images and save
    #im_vis = np.concatenate((img, seg_color, pred_color),
    #                       axis=1).astype(np.uint8)
    #normalize image to [0, 1] first.
    img = img[0].cpu().numpy() #3 channels are the same, just take one of them since grayscale image.
    img = (img - img.min())/(img.max()-img.min())
    img = (img * 255).astype(np.uint8) #Then scale it up to [0, 255] to get the final image.
    img = np.average(img, axis=0)
    pred_img = (pred * 85).astype(np.uint8)
    seg_img = (seg * 85).astype(np.uint8)
    edge = edge[0][0].cpu().numpy()
    edge = (edge* 255).astype(np.uint8)
    att = att[0][0].cpu().numpy()
    att = (((att>0.6)*att)*255).astype(np.uint8)
    #heat = get_heatmap(LRP)
    im_vis = np.concatenate((img, seg_img, pred_img, edge, att), axis=1).astype(np.uint8)
    img_name = info.split('/')[-1] + '.png'
    cv2.imwrite(os.path.join(args.result,
                img_name), im_vis)


def eval(loader_val, segmentation_module, args, crit):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    loss_meter = AverageMeter()

    segmentation_module.eval()
    for batch_data in loader_val:
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data["mask"][0])
        torch.cuda.synchronize()
        batch_data["image"] = batch_data["image"].unsqueeze(0).cuda()
        #batch_data["mask"][0] = batch_data["mask"][0].cuda()
        #batch_data["mask"][1] = batch_data["mask"][1].cuda()

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, args.gpu)
            feed_dict = batch_data.copy()
            #print(torch.max(feed_dict['image']))   

            # forward pass
            scores_tmp, loss = segmentation_module(feed_dict, epoch=0, segSize=segSize)
            scores = scores + scores_tmp  / len(args.imgSize)
            loss_meter.update(loss)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
                #print(np.amax(pred))
        torch.cuda.synchronize()
        # calculate accuracy
        intersection, union = intersectionAndUnion(pred, seg_label, args.num_class)
        intersection_meter.update(intersection)
        union_meter.update(union)


    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        if i >= 1:
            print('class [{}], IoU: {:.4f}'.format(i, _iou))
    print('loss: {:.4f}'.format(loss_meter.average()))


def evaluate(segmentation_module, loader_val, args):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()
    pbar = tqdm(total=len(loader_val))
    for batch_data in loader_val:
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data["mask"][0])
        torch.cuda.synchronize()
        batch_data["image"] = batch_data["image"].unsqueeze(0).cuda()
        #batch_data["mask"][0] = batch_data["mask"][0].cuda()
        #batch_data["mask"][1] = batch_data["mask"][1].cuda()

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, args.gpu)
            feed_dict = batch_data.copy()
            #print(torch.max(feed_dict['image']))   

            # forward pass
            scores, edge, att, loss = segmentation_module(feed_dict, epoch=0, segSize=segSize)
            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
        torch.cuda.synchronize()
        
        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, args.num_class)
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_meter.update(acc)
            # visualization
        if True:# args.visualize
            visualize_result(
                (batch_data['image'], seg_label, batch_data["name"]),
                pred, edge, att, args)
        
        #Free up memroy
        #del sal
        
        pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))


def main(args):
    torch.cuda.set_device(args.gpu)

    # Network Builders
    builder = ModelBuilder()
    
    net_encoder = None
    net_decoder = None
    unet = None

    if args.unet == False:
        net_encoder = builder.build_encoder(
            arch=args.arch_encoder,
            fc_dim=args.fc_dim,
            weights=args.weights_encoder)
        net_decoder = builder.build_decoder(
            arch=args.arch_decoder,
            fc_dim=args.fc_dim,
            num_class=args.num_class,
            weights=args.weights_decoder,
            use_softmax=True)
    else:
        unet = builder.build_unet(num_class=args.num_class,
            arch=args.arch_unet,
            weights=args.weights_unet)

    #crit = nn.NLLLoss()
    crit = ACLoss()

    if args.unet == False:
        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    else:
        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit,
                                                is_unet=args.unet, unet=unet)
    '''
    # Dataset and Loader
    dataset_val = dl.loadVal()

    loader_val = torchdata.DataLoader(
        dataset_val,
        batch_size=5,
        shuffle=False,
        num_workers=1,
        drop_last=True)
    '''
    test_augs = Compose([PaddingCenterCrop(256)])
    
    dataset_val = AC17(
            root=args.data_root,
            split='val',
            k_split=args.k_split,
            augmentations=test_augs,
            img_norm=args.img_norm)
    ac17_val = load2D(dataset_val, split='val', deform=False)
    loader_val = data.DataLoader(
        ac17_val,
        batch_size=1,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, args)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'
    
    DATA_ROOT = os.getenv('DATA_ROOT', '/home/rexma/Desktop/MRI_Images/AC17')

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', required=True,
                        help="a name for identifying the model to load")
    #parser.add_argument('--suffix', default='_epoch_20.pth',
    #                    help="which snapshot to load")
    parser.add_argument('--arch_encoder', default='resnet50dilated',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')
    parser.add_argument('--unet', default=True,
                        help='Use a UNet?')
    parser.add_argument('--arch_unet', default='albunet',
                        help='UNet architecture?')

    # Path related arguments
    
    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=4, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--checkpoint', type=str, required=True, help="checkpoint path")
    parser.add_argument('--test-split', type=str, default='val')
    parser.add_argument('--data-root', type=str, default=DATA_ROOT)
    parser.add_argument('--img-norm', default=True, action='store_true', help="normalize img value to [0, 1]")
    parser.add_argument('--contour_type', default='i')
    parser.add_argument('--imgSize', default=[128], nargs='+', type=int)
    parser.add_argument('--imgMaxSize', default=128, type=int)
    parser.add_argument('--k_split', default=1) 
    # Misc argument
    parser.add_argument('--ckpt', default='/home/rexma/Desktop/JesseSun/ac17_seg/ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--visualize', action='store_true',
                        help='output visualization?')
    parser.add_argument('--result', default='/home/rexma/Desktop/JesseSun/ac17_seg/result',
                        help='folder to output visualization results')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu id for evaluation')
    parser.add_argument('--show_SRmap', default=True, type=bool,
                        help='Show the saliency relevance mapping')

    args = parser.parse_args()
    args.arch_encoder = args.arch_encoder.lower()
    args.arch_decoder = args.arch_decoder.lower()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # absolute paths of model weights
    if args.unet == False:
        args.weights_encoder = os.path.join(args.ckpt, args.id,
                                            'encoder' + args.suffix)
        args.weights_decoder = os.path.join(args.ckpt, args.id,
                                            'decoder' + args.suffix)
    
        assert os.path.exists(args.weights_encoder) and \
            os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'
    
    else:
        args.weights_unet = args.checkpoint
        assert os.path.exists(args.weights_unet), 'checkpoint does not exist!'

    args.result = os.path.join(args.result, args.id)
    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
