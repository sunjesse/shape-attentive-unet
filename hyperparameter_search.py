from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
import numpy as np
import time
import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn import metrics
import random
import csv
import pickle

# import math
from distutils.version import LooseVersion
import math
# Numerical libs
import torch.utils.data as data
from data.augmentations import Compose, RandomSized, AdjustContrast, AdjustBrightness, RandomVerticallyFlip, RandomHorizontallyFlip, RandomRotate, PaddingCenterCrop
# Our libs
from data.ac17_dataloader import AC17Data as AC17, AC17_2DLoad as load2D
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, parse_devices, accuracy, intersectionAndUnion
from lib.nn import UserScatteredDataParallel, async_copy_to,  user_scattered_collate, patch_replication_callback
import lib.utils.data as torchdata
from lib.utils import as_numpy
from loss import ACLoss
from radam import RAdam

save_path = '/home/rexma/Destkop/JeseSun/ac17_seg/hypersearch'

def eval(loader_val, segmentation_module, crit):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    loss_meter = AverageMeter()

    segmentation_module.eval()
    for batch_data in loader_val:
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data["mask"])
        torch.cuda.synchronize()
        batch_data["image"] = batch_data["image"].unsqueeze(0).cuda()
        batch_data["mask"] = batch_data["mask"].cuda()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, 4, segSize[0], segSize[1])
            feed_dict = batch_data.copy()
            #print(torch.max(feed_dict['image']))   

            # forward pass
            scores, loss = segmentation_module(feed_dict, epoch=0, segSize=segSize)
            loss_meter.update(loss)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
        torch.cuda.synchronize()
        # calculate accuracy
        intersection, union = intersectionAndUnion(pred, seg_label, 4)
        intersection_meter.update(intersection)
        union_meter.update(union)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        if i >= 1:
            print('class [{}], IoU: {:.4f}'.format(i, _iou))
    print('loss: {:.4f}'.format(loss_meter.average()))
    return loss_meter.average()

# train one epoch
def train(segmentation_module, loader_train, optimizers, epoch, space):
    adjust_learning_rate(optimizers, epoch, space['lr'])

    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_j1 = AverageMeter()
    ave_j2 = AverageMeter()
    ave_j3 = AverageMeter()

    segmentation_module.train()

    # main loop
    tic = time.time()
    iter_count = 0

    for batch_data in loader_train:
        data_time.update(time.time() - tic)
        batch_data["image"] = batch_data["image"].cuda()
        batch_data["mask"] = batch_data["mask"].cuda()
        segmentation_module.zero_grad()
        # forward pass
        loss, acc = segmentation_module(batch_data, epoch)
        loss = loss.mean()

        jaccard = acc[1]
        for j in jaccard:
            j = j.float().mean()
        acc = acc[0].float().mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        iter_count += 4

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        ave_j1.update(jaccard[0].data.item()*100)
        ave_j2.update(jaccard[1].data.item()*100)
        ave_j3.update(jaccard[2].data.item()*100)

        if iter_count % 40 == 0:
            # calculate accuracy, and display
            print('Epoch: [{}/{}], Iter: [{}], Time: {:.2f}, Data: {:.2f},'
                  'Accuracy: {:4.2f}, Jaccard: [{:4.2f},{:4.2f},{:4.2f}], '
                  'Loss: {:.6f}'
                  .format(epoch, 30, iter_count,
                        batch_time.average(), data_time.average(),
                        ave_acc.average(),
                        ave_j1.average(), ave_j2.average(),
                        ave_j3.average(), ave_total_loss.average()))

    #Average jaccard across classes.
    j_avg = (ave_j1.average() + ave_j2.average() + ave_j3.average())/3

    return ave_total_loss.average()

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    #ssert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, space):
    (unet, crit) = nets
    '''
    optimizer_unet = torch.optim.Adam(
        group_weight(unet),
        lr = space['lr'],
        betas=(space['b1'], space['b2']))
    '''
    optimizer_unet = torch.optim.SGD(
            group_weight(unet),
            lr=space['lr'],
            momentum=0.9,
            weight_decay=1e-4)
    '''
    optimizer_unet = RAdam(
            group_weight(unet),
            lr=space['lr'],
            betas=(space['b1'], space['b2']))
    '''
    return [optimizer_unet]


def adjust_learning_rate(optimizers, cur_iter, lr):
    scale_running_lr = 0.5*(1+math.cos(3.14159*(cur_iter-10)/30))
    running_lr_encoder = lr * scale_running_lr

    optimizer_unet = optimizers[0]
    for param_group in optimizer_unet.param_groups:
        param_group['lr'] = running_lr_encoder

def run_model(space):
     # Network Builders
    builder = ModelBuilder()
    net_encoder=None
    net_decoder=None
    unet=None

    unet = builder.build_unet(num_class=4,
        arch='AlbuNet',
        weights='')

    crit = ACLoss()

    segmentation_module = SegmentationModule(
        net_encoder, net_decoder,  crit, is_unet=True, unet=unet)

    train_augs = Compose([PaddingCenterCrop(224), RandomHorizontallyFlip(), RandomVerticallyFlip(), RandomRotate(180)])
    test_augs = Compose([PaddingCenterCrop(224)])
    # Dataset and Loader
    dataset_train = AC17(
            root=os.getenv('DATA_ROOT', '/home/rexma/Desktop/MRI_Images/AC17'),
            split='train',
            k_split=1,
            augmentations=train_augs,
            img_norm=True)
    ac17_train = load2D(dataset_train, split='train', deform=False)

    loader_train = data.DataLoader(
        ac17_train,
        batch_size=4,  # we have modified data_parallel
        shuffle=True,
        num_workers=5,
        drop_last=True,
        pin_memory=True)

    dataset_val = AC17(
            root=os.getenv('DATA_ROOT', '/home/rexma/Desktop/MRI_Images/AC17'),
            split='val',
            k_split=1,
            augmentations=test_augs,
            img_norm=True)
    ac17_val = load2D(dataset_val, split='val', deform=False)
    loader_val = data.DataLoader(
        ac17_val,
        batch_size=1,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Set up optimizers
    nets = (unet, crit)
    optimizers = create_optimizers(nets, space)
    
    val_losses = []
    train_losses = []
    status = STATUS_OK
    print("Searching " + "lr: " + str(space['lr']))#+ " b1: " + str(space['b1']) + "b2: " + str(space['b2']))
    # Main loop
    for epoch in range(1, 31):
        t_iou = train(segmentation_module, loader_train, optimizers, epoch, space)
        v_iou = eval(loader_val, segmentation_module, crit)
        train_losses.append(t_iou)
        val_losses.append(v_iou)
        if epoch == 3 and v_iou >= 1.0:
            status = STATUS_FAIL
            break
    
    #values to be returned
    opt_name = 'lr' + str(space['lr'])# + "_b1" + str(space['b1']) + "_b2" + str(space['b2'])
    model_dict = {'loss':min(val_losses),
                  'status':status,
                  'name': opt_name}
    return model_dict
    
# Hyperparameter search space
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--evals', default=30)
    args = parser.parse_args()
     
    # Hyperparameter search space
    space = {'lr': hp.loguniform('l2',np.log(0.001), np.log(1.0))}
             #'b1': hp.loguniform('b1',np.log(0.5), np.log(0.9)),
             #'b2': hp.loguniform('b2', np.log(0.5), np.log(0.999))}

    pickle_loc = save_path + '/trials.p'
    csv_loc = save_path + '/search.csv'
    try: # load pickle file to continue search if already started
        trials = pickle.load(open(pickle_loc, 'rb'))
    except:
        trials = Trials()
        print('new trial started')
    evals = args.evals
    evals_inc = min(evals, 3)
    while evals_inc <= evals:
        best = fmin(
            fn=run_model,  # "Loss" function to minimize
            space=space,  # Hyperparameter space
            algo=tpe.suggest,  # Tree-structured Parzen Estimator (TPE)
            max_evals=evals_inc,  # num trials
            trials = trials
        )
        results = []
        for trial in trials.trials:
            results.append(trial['result'])
        # save trials
        pickle.dump(trials, open(pickle_loc, "wb"))
        # save results to csv
        keys = results[0].keys()
        with open(csv_loc, 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        if evals_inc == evals:
            break
        evals_inc = min(evals, evals_inc+3)
