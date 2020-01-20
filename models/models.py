import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from . import resnet, resnext, mobilenet, dpn, drn
from lib.nn import SynchronizedBatchNorm2d
from . import convcrf as cc
from smoothgrad import generate_smooth_grad
from guided_backprop import GuidedBackprop
from vanilla_backprop import VanillaBackprop
from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)
import math
from .refine_blocks import RefineBlock
from .resnet import BasicBlock as ResBlock
from . import GSConv as gsc
import cv2
from .norm import Norm2d

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()
    
    def intersectionAndUnion(self, imPred, imLab, numClass):
        imPred = np.asarray(imPred.cpu()).copy()
        imLab = np.asarray(imLab.cpu()).copy()

        imPred += 1
        imLab += 1
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        imPred = imPred * (imLab > 0)

        # Compute area intersection:
        intersection = imPred * (imPred == imLab)
        (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

        # Compute area union:
        (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
        (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
        area_union = area_pred + area_lab - area_intersection
        
        jaccard = area_intersection/area_union
        #print("I: " + str(area_intersection))
        #print("U: " + str(area_union))
        jaccard = (jaccard[1]+jaccard[2])/2
        return jaccard if jaccard <= 1 else 0

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 1).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)

        acc = acc_sum.float() / (pixel_sum.float() + 1e-10) #When you +falsePos, acc == Jaccard.
        
        #class 1
        v1 = (label == 1).long()
        pred1 = (preds == 1).long()
        anb1 = torch.sum(v1 * pred1)
        try:
            j1 = anb1.float() / (torch.sum(v1).float() + torch.sum(pred1).float() - anb1.float() + 1e-10)
        except:
            j1 = 0
        #class 2 
        v2 = (label == 2).long()
        pred2 = (preds == 2).long()
        anb2 = torch.sum(v2 * pred2)
        try: 
            j2 = anb2.float() / (torch.sum(v2).float() + torch.sum(pred2).float() - anb2.float() + 1e-10)
        except:
            j2 = 0

        #print(str(j1) + " " + str(j2))
        if j1 > 1:
            j1 = 0

        if j2 > 1:
            j2 = 0
        
        #class 3
        v3 = (label == 3).long()
        pred3 = (preds == 3).long()
        anb3 = torch.sum(v3 * pred3)
        try:
            j3 = anb3.float() / (torch.sum(v3).float() + torch.sum(pred3).float() - anb3.float() + 1e-10)
        except:
            j3 = 0

        #print(str(j1) + " " + str(j2))
        #print(str(j1.item()) + " " + str(j2.item()))
        j1 = j1 if j1 <= 1 else 0
        j2 = j2 if j2 <= 1 else 0
        j3 = j3 if j3 <= 1 else 0
        
        jaccard = [j1, j2, j3] #acc_sum.float() / (pixel_sum.float() + falsePos + 1e-10)
        
        #jaccard = torch.from_numpy(np.array(self.intersectionAndUnion(pred, label, 3)))
        return acc, jaccard

    def jaccard(self, pred, label): #ACCURACY THAT TAKES INTO ACCOUNT BOTH TP AND FP.
        AnB = torch.sum(pred.long() & label) #TAKE THE AND
        return AnB/(pred.view(-1).sum().float() + label.view(-1).sum().float() - AnB)
    
class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, is_unet=False, unet=None, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.is_unet = is_unet
        self.unet = unet
        self.deep_sup_scale = deep_sup_scale
     
    def forward(self, feed_dict, epoch, *, segSize=None):
        #training
        if segSize is None:
            p = self.unet(feed_dict['image'])
            loss = self.crit(pred, feed_dict['mask'], epoch=epoch)
            acc = self.pixel_acc(torch.round(nn.functional.softmax(pred[0], dim=1)).long(), feed_dict['mask'][0].long().cuda())
            return loss, acc
            
        #test
        if segSize == True:
            p = self.unet(feed_dict['image'])[0]
            pred = nn.functional.softmax(p, dim=1)
            return pred

        #inference
        else:
            p = self.unet(feed_dict['image'])
            loss = self.crit((p[0], p[1]), (feed_dict['mask'][0].long().unsqueeze(0), feed_dict['mask'][1]))
            pred = nn.functional.softmax(p[0], dim=1)
            return pred, p[1], p[2], loss

    def SRP(self, model, pred, seg):
        output = pred #actual output
        _, pred = torch.max(nn.functional.softmax(pred, dim=1), dim=1) #class index prediction
        
        tmp = []
        for i in range(output.shape[1]):
            tmp.append((pred == i).long().unsqueeze(1))
        T_model = torch.cat(tmp, dim=1).float()

        LRP = self.layer_relevance_prop(model, output * T_model)

        return LRP
    
    def layer_relevance_prop(self, model, R):
        for l in range(len(model.layers), 0, -1):
            print(model.layers[l-1])
            R  = model.layers[l-1].relprop(R)
        return R


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )

def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)
    
    def build_unet(self, num_class=1, arch='albunet', weights=''):
        arch = arch.lower()

        if arch == 'albunet':
            unet = AlbuNet(num_classes=num_class)
        else:
            raise Exception('Architecture undefined!')
        
        if len(weights) > 0:
            unet.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
            print("Loaded pretrained UNet weights.")
        print('Loaded weights for unet')
        return unet

    def build_encoder(self, arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        print('Loaded weights for net_encoder')
        return net_encoder

    def build_decoder(self, arch='ppm_deepsup',
                      fc_dim=512, num_class=1,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            checkpoint = torch.load(weights, map_location=lambda storage, loc: storage) #Cast dict() around when iterating through it to change values
            for key in checkpoint:
                if len(checkpoint[key].shape) > 0 and checkpoint[key].shape != net_decoder.state_dict()[key].shape:
                    if checkpoint[key].shape[0] == 2 and net_decoder.state_dict()[key].shape[0] == 1:
                        checkpoint[key] = checkpoint[key][0].unsqueeze(0)
            net_decoder.load_state_dict(checkpoint, strict=False)
            print('Loaded weights for net_decoder')
        return net_decoder


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                #nn.LeakyReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            #nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x


### ---- UNets ---- ###

##Building Blocks
#def conv3x3(in_, out):
#    return nn.Conv2d(in_, out, 3, padding=1)

#class ConvRelu(nn.Module):
#    def __init__(self, in_, out):
#        super(ConvRelu, self).__init__()
#        self.conv = conv3x3(in_, out)
#       self.activation = nn.ReLU(inplace=True)

#    def forward(self, x):
#        x = self.conv(x)
#        x = self.activation(x)
#        return x
class CenterBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(CenterBlock, self).__init__()
        self.in_channels = in_channels
        self.is_deconv=is_deconv

        if self.is_deconv: #Dense Block
            self.conv1 = conv3x3_bn_relu(in_channels, middle_channels)
            self.conv2 = conv3x3_bn_relu(in_channels+middle_channels, middle_channels)
            self.convUp = nn.Sequential(
                                nn.ConvTranspose2d(in_channels+2*middle_channels, out_channels,
                                                kernel_size=4, stride=2, padding=1),
                                nn.ReLU(inplace=True)
            )
        
        else:
            self.convUp = nn.Unsample(scale_factor=2, mode='bilinear', align_corners=True),
            self.conv1 = conv3x3_bn_relu(in_channels, middle_channels),
            self.conv2 = conv3x3_bn_relu(in_channels+middle_channels, out_channels)  
            
    def forward(self, x):
        tmp = []
        if self.is_deconv == False:
            convUp = self.convUp(x); tmp.append(convUp)
            conv1 = self.conv1(convUp); tmp.append(conv1)
            conv2 = self.conv2(torch.cat(tmp, 1))
            return conv2

        else: 
            tmp.append(x)
            conv1 = self.conv1(x); tmp.append(conv1)
            conv2 = self.conv2(torch.cat(tmp, 1)); tmp.append(conv2)
            convUp = self.convUp(torch.cat(tmp, 1))
            return convUp

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        
        if is_deconv:
            self.block = nn.Sequential(
                conv3x3_bn_relu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                conv3x3_bn_relu(in_channels, middle_channels),
                conv3x3_bn_relu(middle_channels, out_channels),
            )

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x):
        return self.block(x)
    
class SkipConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        
        ## initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x):
        return self.block(x)

#ResNet + UNet
class AlbuNet1(nn.Module):
    def __init__(self, num_classes=2, num_filters=32, pretrained=True, is_deconv=True):
        super(AlbuNet1, self).__init__()
        
        self.num_classes = num_classes
        
        self.pool = nn.MaxPool2d(2,2)
               
        #self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        #self.encoder = se_resnet.se_resnet50()
        self.encoder = drn.drn_d_105(pretrained=True)
        #self.encoder = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        #for n, p in self.encoder.named_parameters():
        #    print(n)

        self.relu = nn.ReLU(inplace=True)
        
        #self.conv1 = nn.Sequential(self.encoder.conv1,
        #                self.encoder.bn1,
        #                self.encoder.relu)
        #self.conv1 = self.encoder.layer0
        self.conv1 = nn.Sequential(
                      self.encoder.layer0,
                      self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = nn.Sequential(
                        self.encoder.layer4,
                        self.encoder.layer5,
                        self.encoder.layer6,
                        self.encoder.layer7,
                        self.encoder.layer8
        )

        
        '''
        for p in self.conv1.parameters():
            p.requires_grad = False
        for p in self.conv2.parameters():
            p.requires_grad = False
        for p in self.conv3.parameters():
            p.requires_grad = False
        for p in self.conv4.parameters():
            p.requires_grad = False
        for p in self.conv5.parameters():
            p.requires_grad = False
        '''
        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(32 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(16 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = conv3x3_bn_relu(num_filters, num_filters)
        self.final = nn.Sequential(nn.BatchNorm2d(num_filters),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(num_filters, self.num_classes, kernel_size=1)
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        #print("Conv1: " + str(x.shape) + " => " + str(conv1.shape))
        conv2 = self.conv2(conv1)
        #print("Conv2: " + str(conv1.shape) + " => " + str(conv2.shape))
        conv3 = self.conv3(conv2)
        #print("Conv3: " + str(conv2.shape) + " => " + str(conv3.shape))
        conv4 = self.conv4(conv3)
        #print("Conv4: " + str(conv3.shape) + " => " + str(conv4.shape))
        conv5 = self.conv5(conv4)
        #print("Conv5: " + str(conv4.shape) + " => " + str(conv5.shape))

        center = self.center(self.pool(conv5))
        
        dec5 = self.dec5(torch.cat([center, self.pad(conv5, center)], 1)) #dim=1 because NCHW, channels is dim 1.
        #print("Deconv5: " + str(center.shape) + " => " + str(dec5.shape))
        dec4 = self.dec4(torch.cat([dec5, self.pad(conv4, dec5)], 1))
        #print("Deconv4: " + str(dec5.shape) + " => " + str(dec4.shape))
        dec3 = self.dec3(torch.cat([dec4, self.pad(conv3, dec4)], 1))
        #print("Deconv3: " + str(dec4.shape) + " => " + str(dec3.shape))
        dec2 = self.dec2(torch.cat([dec3, self.pad(conv2, dec3)], 1))
        #print("Deconv2: " + str(dec3.shape) + " => " + str(dec2.shape))
        dec1 = self.dec1(dec2)
        #print("Deconv1: " + str(dec2.shape) + " => " + str(dec1.shape))
        dec0 = self.pad(self.dec0(dec1), x)
        #print("Deconv0: " + str(dec1.shape) + " => " + str(dec0.shape))

        '''
        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec(0), dim=1))
        else:
            x_out = self.final(dec0)
        '''
        
        #x_out = nn.functional.log_softmax(self.final(dec0), dim=1) #DIM = 1, perform channel-wise log softmax.
        x_out = self.final(dec0)
        return x_out

    def pad(self, x, y):
        diffX = y.shape[3] - x.shape[3]
        diffY = y.shape[2] - x.shape[2]

        return nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                     diffY // 2, diffY - diffY //2))

    def relprop(self, R):
        for l in range(len(self.layers), 0, -1):
            R  = self.layers[l-1].relprop(R)
        return R


class AlbuNet6(nn.Module): #DPN Unet
    
    def __init__(self, num_classes=2, num_filters=32, pretrained=True, is_deconv=True, use_crf=False):#False):
        super(AlbuNet6, self).__init__()
        self.num_classes = num_classes
        self.use_crf = use_crf

        self.pool = nn.MaxPool2d(2, 2)

        #self.encoder = torchvision.models.densenet121(pretrained=pretrained)
        #del self.encoder.classifier #just to save my sanity.
        
        self.encoder = dpn.dpn92(pretrained=True, test_time_pool=False)
        del self.encoder.classifier
        
        #for n, p in self.encoder.named_parameters():
            #print(n)
        self.conv1 = self.encoder.features.conv1_1
        self.conv2 = nn.Sequential(self.encoder.features.conv2_1,
                                   self.encoder.features.conv2_2,
                                   self.encoder.features.conv2_3)
        self.conv3 = nn.Sequential(self.encoder.features.conv3_1,
                                   self.encoder.features.conv3_2,
                                   self.encoder.features.conv3_3,
                                   self.encoder.features.conv3_4)
        self.conv4 = nn.Sequential(self.encoder.features.conv4_1,
                                   self.encoder.features.conv4_2,
                                   self.encoder.features.conv4_3,
                                   self.encoder.features.conv4_4,
                                   self.encoder.features.conv4_5,
                                   self.encoder.features.conv4_6,
                                   self.encoder.features.conv4_7,
                                   self.encoder.features.conv4_8,
                                   self.encoder.features.conv4_9,
                                   self.encoder.features.conv4_10,
                                   self.encoder.features.conv4_11,
                                   self.encoder.features.conv4_12,
                                   self.encoder.features.conv4_13,
                                   self.encoder.features.conv4_14,
                                   self.encoder.features.conv4_15,
                                   self.encoder.features.conv4_16,
                                   self.encoder.features.conv4_17,
                                   self.encoder.features.conv4_18,
                                   self.encoder.features.conv4_19,
                                   self.encoder.features.conv4_20)
        self.conv5 = nn.Sequential(self.encoder.features.conv5_1,
                                   self.encoder.features.conv5_2,
                                   self.encoder.features.conv5_3,
                                   self.encoder.features.conv5_bn_ac)
        
        self.center = DecoderBlock(2688, num_filters * 8 * 2, num_filters * 8, is_deconv)
        
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(256+ num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(128+ num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(128+ num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = conv3x3_bn_relu(num_filters, num_filters)
        #conv_final = nn.Conv2d(num_filters, self.num_classes, kernel_size=1)
        #self.final = Conv2d(conv_final)

        self.skip_conv5 = nn.Sequential(nn.Conv2d(2688, 512, kernel_size=1, padding=0, stride=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True))
        self.skip_conv4 = nn.Sequential(nn.Conv2d(1552, 256, kernel_size=1, padding=0, stride=1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True))
        self.skip_conv3 = nn.Sequential(nn.Conv2d(704, 128, kernel_size=1, padding=0, stride=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True))
        self.skip_conv2 = nn.Sequential(nn.Conv2d(336, 128, kernel_size=1, padding=0, stride=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True))
        
        self.final = nn.Conv2d(num_filters, self.num_classes, kernel_size=1)

        #self.layers = nn.Sequential():q:q
    
    def update_graph(self):
        self.layers = [mod for mod in self.modules() if type(mod) not in [nn.Sequential, SynchronizedBatchNorm2d, 
                                                                            nn.BatchNorm2d]]

    def forward(self, x):
        conv1 = self.conv1(x)
        #print("Conv1: " + str(x.shape) + " => " + str(conv1.shape))
        conv2 = self.conv2(conv1)
        #print("Conv2: " + str(conv1[0].shape) + " => " + str(conv2[0].shape))
        conv3 = self.conv3(conv2)
        #print("Conv3: " + str(conv2[0].shape) + " => " + str(conv3[0].shape))
        conv4 = self.conv4(conv3)
        #print("Conv4: " + str(conv3[0].shape) + " => " + str(conv4[0].shape))
        conv5 = self.conv5(conv4)
        #print("Conv5: " + str(conv4[0].shape) + " => " + str(conv5.shape))

        center = self.center(self.pool(conv5))
        
        dec5 = self.dec5(torch.cat([center, self.skip_conv5(self.pad(conv5, center))], 1))
        #print("Deconv5: " + str(center.shape) + " => " + str(dec5.shape))
        dec4 = self.dec4(torch.cat([dec5, self.skip_conv4(self.pad(conv4, dec5))], 1))
        #print("Deconv4: " + str(dec5.shape) + " => " + str(dec4.shape))
        dec3 = self.dec3(torch.cat([dec4, self.skip_conv3(self.pad(conv3, dec4))], 1))
        #print("Deconv3: " + str(dec4.shape) + " => " + str(dec3.shape))
        dec2 = self.dec2(torch.cat([dec3, self.skip_conv2(self.pad(conv2, dec3))], 1))
        #print("Deconv2: " + str(dec3.shape) + " => " + str(dec2.shape))
        dec1 = self.dec1(dec2)
        #print("Deconv1: " + str(dec2.shape) + " => " + str(dec1.shape))
        dec0 = self.pad(self.dec0(dec1), x)
        #print("Deconv0: " + str(dec1.shape) + " => " + str(dec0.shape))

        '''
        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec(0), dim=1))
        else:
            x_out = self.final(dec0)
        '''
        
        #x_out = nn.functional.log_softmax(self.final(dec0), dim=1) #DIM = 1, perform channel-wise log softmax.
    
        x_out = self.final(dec0)
    
        #self.update_graph()
        return x_out

    def pad(self, x, y):
        if len(x) == 1:
            diffX = y.shape[3] - x.shape[3]
            diffY = y.shape[2] - x.shape[2]

            return nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                         diffY // 2, diffY - diffY //2))
        else: #len == 2
            diffX = y.shape[3] - x[0].shape[3]
            diffY = y.shape[2] - x[0].shape[2]
            
            d1 = nn.functional.pad(x[0], (diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2))
            d2 = nn.functional.pad(x[1], (diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2))

            return torch.cat([d1, d2], 1)


class AlbuNetj(nn.Module): #densenet
    def __init__(self, num_classes=2, num_filters=32, pretrained=True, is_deconv=True):
        super(AlbuNetj, self).__init__()

        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2,2)
        self.avgpool2 = nn.AvgPool2d(2, stride=2)
        self.avgpool4 = nn.AvgPool2d(4, stride=4)
        self.avgpool8 = nn.AvgPool2d(8, stride=8)
        self.encoder = torchvision.models.densenet121(pretrained=pretrained)
        #self.encoder = se_resnet.se_resnet50()
        #self.encoder = gen_efficientnet.efficientnet_b2(pretrained=True)
       
        #for n, p in self.encoder.named_parameters():
        #    print(n)
        print("DENSE U-NET")

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                   self.encoder.features.norm0)
        self.conv2 = nn.Sequential(self.encoder.features.denseblock1,
                                   self.encoder.features.transition1)
        self.conv3 = nn.Sequential(self.encoder.features.denseblock2,
                                   self.encoder.features.transition2)
        self.conv4 = nn.Sequential(self.encoder.features.denseblock3,
                                   self.encoder.features.transition3)
        self.conv5 = nn.Sequential(self.encoder.features.denseblock4,
                                   self.encoder.features.norm5)

        self.center = DecoderBlock(1024, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = conv3x3_bn_relu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, self.num_classes, kernel_size=1)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        #print("Conv1: " + str(x.shape) + " => " + str(conv1.shape))
        conv2 = self.conv2(conv1)
        #print("Conv2: " + str(conv1.shape) + " => " + str(conv2.shape))
        conv3 = self.conv3(conv2)
        #print("Conv3: " + str(conv2.shape) + " => " + str(conv3.shape))
        conv4 = self.conv4(conv3)
        #print("Conv4: " + str(conv3.shape) + " => " + str(conv4.shape))
        conv5 = self.conv5(conv4)
        #print("Conv5: " + str(conv4.shape) + " => " + str(conv5.shape))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, 
                                    self.pad(conv5, center)],1))
        #print("Deconv5: " + str(center.shape) + " => " + str(dec5.shape))
        dec4 = self.dec4(torch.cat([dec5, 
                                    self.pad(conv4, dec5)],1))
        #print("Deconv4: " + str(dec5.shape) + " => " + str(dec4.shape))
        dec3 = self.dec3(torch.cat([dec4, 
                                    self.pad(conv3, dec4)],1))
        #print("Deconv3: " + str(dec4.shape) + " => " + str(dec3.shape))
        dec2 = self.dec2(torch.cat([dec3, 
                                    self.pad(conv2, dec3)], 1))
        #print("Deconv2: " + str(dec3.shape) + " => " + str(dec2.shape))
        dec1 = self.dec1(dec2)
        #print("Deconv1: " + str(dec2.shape) + " => " + str(dec1.shape))
        dec0 = self.pad(self.dec0(dec1), x)
        #print("Deconv0: " + str(dec1.shape) + " => " + str(dec0.shape))

       
        #x_out = nn.functional.log_softmax(self.final(dec0), dim=1) #DIM = 1, perform channel-wise log softmax.
        x_out = self.final(dec0)
        return x_out

    def pad(self, x, y):
        diffX = y.shape[3] - x.shape[3]
        diffY = y.shape[2] - x.shape[2]

        return nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY //2))
class AlbuNetz(nn.Module): #densenet
    def __init__(self, num_classes=4, num_filters=64, pretrained=True, is_deconv=True):
        super(AlbuNetz, self).__init__()

        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2,2)
        self.avgpool2 = nn.AvgPool2d(2, stride=2)
        self.avgpool4 = nn.AvgPool2d(4, stride=4)
        self.avgpool8 = nn.AvgPool2d(8, stride=8)
        self.encoder = torchvision.models.densenet161(pretrained=pretrained)
        #self.encoder = se_resnet.se_resnet50()
        #self.encoder = gen_efficientnet.efficientnet_b2(pretrained=True)

        #for n, p in self.encoder.named_parameters():
        #    print(n)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                   self.encoder.features.norm0)
        self.conv2 = self.encoder.features.denseblock1
        self.conv2t = self.encoder.features.transition1
        self.conv3 = self.encoder.features.denseblock2
        self.conv3t = self.encoder.features.transition2
        self.conv4 = self.encoder.features.denseblock3
        self.conv4t = self.encoder.features.transition3
        self.conv5 = nn.Sequential(self.encoder.features.denseblock4,
                                   self.encoder.features.norm5)

        #self.center = DecoderBlock(1024, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.center = conv3x3_bn_relu(2208, num_filters * 8 * 2)
        self.dec5 = RefineBlock(inchannels=[num_filters * 8 * 2, 2208], outchannels=1024, img_dim=[1/32, 1/16])
        self.dec4 = RefineBlock(inchannels=[2048, 2112], outchannels=512, img_dim=[1/16, 1/8])
        self.dec3 = RefineBlock(inchannels=[1024, 768], outchannels=512, img_dim=[1/8, 1/4])
        self.dec2 = RefineBlock(inchannels=[1024,  384], outchannels=256, img_dim=[1/4, 1/2])
        self.dec1 = DecoderBlock(512, 256, num_filters, is_deconv) 
        self.dec0 = conv3x3_bn_relu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, self.num_classes, kernel_size=1)
        

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(self.conv2t(conv2))
        conv4 = self.conv4(self.conv3t(conv3))
        conv5 = self.conv5(self.conv4t(conv4))
        
        center = self.center(self.pool(conv5))
        
        dec5 = self.dec5([center, conv5])
        dec4 = self.dec4([dec5, conv4])
        dec3 = self.dec3([dec4, conv3])
        dec2 = self.dec2([dec3, conv2])
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        #x_out = nn.functional.log_softmax(self.final(dec0), dim=1) #DIM = 1, perform channel-wise log softmax.
        x_out = self.final(dec0)
        return x_out

    def pad(self, x, y):
        diffX = y.shape[3] - x.shape[3]
        diffY = y.shape[2] - x.shape[2]

        return nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                        diffY // 2, diffY - diffY //2))

class AlbuNetl(nn.Module): #haunet
    def __init__(self, num_classes=4, num_filters=32, pretrained=True, is_deconv=True):
        super(AlbuNetl, self).__init__()

        self.num_classes = num_classes
        print("HAU-Net")
        self.pool = nn.MaxPool2d(2,2)
        self.encoder = torchvision.models.densenet121(pretrained=pretrained)
        #self.encoder = se_resnet.se_resnet50()
        #self.encoder = gen_efficientnet.efficientnet_b2(pretrained=True)

        #for n, p in self.encoder.named_parameters():
        #    print(n)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                   self.encoder.features.norm0)
        self.conv2 = self.encoder.features.denseblock1
        self.conv2t = self.encoder.features.transition1
        self.conv3 = self.encoder.features.denseblock2
        self.conv3t = self.encoder.features.transition2
        self.conv4 = self.encoder.features.denseblock3
        self.conv4t = self.encoder.features.transition3
        self.conv5 = nn.Sequential(self.encoder.features.denseblock4,
                                   self.encoder.features.norm5)

        #self.center = DecoderBlock(1024, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.center = conv3x3_bn_relu(1024, num_filters * 8 * 2)
        self.dec5 = RefineBlock(inchannels=[512, 1024, 1024, 512, 256], outchannels=512, img_dim=[1/32, 1/16, 1/8, 1/4, 1/2])
        self.dec4 = RefineBlock(inchannels=[512, 1024, 512, 256], outchannels=256, img_dim=[1/16, 1/8, 1/4, 1/2])
        self.dec3 = RefineBlock(inchannels=[256, 512, 256], outchannels=128, img_dim=[1/8, 1/4, 1/2])
        self.dec2 = RefineBlock(inchannels=[128,  256], outchannels=64, img_dim=[1/4, 1/2])
        self.dec1 = DecoderBlock(64, 48, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)#conv3x3_bn_relu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, self.num_classes, kernel_size=1)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(self.conv2t(conv2))
        conv4 = self.conv4(self.conv3t(conv3))
        conv5 = self.conv5(self.conv4t(conv4))

        center = self.center(self.pool(conv5))
    
        dec5 = self.dec5([center, conv5, conv4, conv3, conv2])
        dec4 = self.dec4([dec5, conv4, conv3, conv2])
        dec3 = self.dec3([dec4, conv3, conv2])
        dec2 = self.dec2([dec3, conv2])
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        #x_out = nn.functional.log_softmax(self.final(dec0), dim=1) #DIM = 1, perform channel-wise log softmax.
        x_out = self.final(dec0)
       
        return x_out

    def pad(self, x, y):
        diffX = y.shape[3] - x.shape[3]
        diffY = y.shape[2] - x.shape[2]

        return nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                        diffY // 2, diffY - diffY //2))

class AlbuNeta(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super(AlbuNeta, self).__init__()
        self.num_classes = num_classes
        print("UNet")
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        # if self.num_classes > 1:
            # x_out = F.log_softmax(self.final(dec1), dim=1)
        # else:
            # x_out = self.final(dec1)
        x_out = self.final(dec1)

        return x_out

class AlbuNet(nn.Module): #haunet
    def __init__(self, num_classes=4, num_filters=32, pretrained=True, is_deconv=True):
        super(AlbuNet, self).__init__()

        self.num_classes = num_classes
        print("HAUNet w/ Shape Stream")
        self.pool = nn.MaxPool2d(2,2)
        self.encoder = torchvision.models.densenet121(pretrained=pretrained)

        #for n, p in self.encoder.named_parameters():
        #    print(n)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
         
        #Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(512, 1, kernel_size=1)
        self.c5 = nn.Conv2d(1024, 1, kernel_size=1)
        
        self.d0 = nn.Conv2d(128, 64, kernel_size=1)
        self.res1 = ResBlock(64, 64)
        self.d1 = nn.Conv2d(64, 32, kernel_size=1)
        self.res2 = ResBlock(32, 32)
        self.d2 = nn.Conv2d(32, 16, kernel_size=1)
        self.res3 = ResBlock(16, 16)
        self.d3 = nn.Conv2d(16, 8, kernel_size=1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)
        
        self.expand = nn.Sequential(nn.Conv2d(1, num_filters, kernel_size=1),
                                    Norm2d(num_filters),
                                    nn.ReLU(inplace=True))
        
        #Encoder
        self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                   self.encoder.features.norm0)
        self.conv2 = self.encoder.features.denseblock1
        self.conv2t = self.encoder.features.transition1
        self.conv3 = self.encoder.features.denseblock2
        self.conv3t = self.encoder.features.transition2
        self.conv4 = self.encoder.features.denseblock3
        self.conv4t = self.encoder.features.transition3
        self.conv5 = nn.Sequential(self.encoder.features.denseblock4,
                                   self.encoder.features.norm5)

        #Decoder
        self.center = conv3x3_bn_relu(1024, num_filters * 8 * 2)
        self.dec5 = RefineBlock(inchannels=[512, 1024], outchannels=512)
        self.dec4 = RefineBlock(inchannels=[512, 512], outchannels=256)
        self.dec3 = RefineBlock(inchannels=[256, 256], outchannels=128)
        self.dec2 = RefineBlock(inchannels=[128,  128], outchannels=64)
        self.dec1 = DecoderBlock(64, 48, num_filters, is_deconv)
        #self.dec0 = RefineBlock(inchannels=[num_filters*2], outchannels=num_filters)
        #self.dec0 = ConvRelu(num_filters*2, num_filters)
        self.dec0 = conv3x3_bn_relu(num_filters*2, num_filters)
        #self.dec0 = nn.Sequential(conv3x3_bn_relu(num_filters*2, num_filters),
        #                          conv3x3_bn_relu(num_filters, num_filters))
        '''
        self.center = DecoderBlock(1024, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = conv3x3_bn_relu(num_filters * 2 * 2, num_filters)
        self.dec0 = conv3x3_bn_relu(num_filters*2, num_filters)
        '''
        self.final = nn.Conv2d(num_filters, self.num_classes, kernel_size=1)

    def forward(self, x):
        x_size = x.size()

        #Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2t(self.conv2(conv1))
        conv3 = self.conv3t(self.conv3(conv2))
        conv4 = self.conv4t(self.conv4(conv3))
        conv5 = self.conv5(conv4)
       
        #Shape Stream
        ss = F.interpolate(self.d0(conv2), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.res1(ss)
        c3 = F.interpolate(self.c3(conv3), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.d1(ss)
        ss = self.gate1(ss, c3)
        ss = self.res2(ss)
        ss = self.d2(ss)
        c4 = F.interpolate(self.c4(conv4), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.gate2(ss, c4)
        ss = self.res3(ss)
        ss = self.d3(ss)
        c5 = F.interpolate(self.c5(conv5), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.gate3(ss, c5)
        ss = self.fuse(ss)
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(ss)

        ### Canny Edge
        im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()
        ### End Canny Edge

        cat = torch.cat([edge_out, canny], dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
        edge = self.expand(acts)
        
        #Decoder
        conv2 = F.interpolate(conv2, scale_factor=2, mode='bilinear', align_corners=True)
        conv3 = F.interpolate(conv3, scale_factor=2, mode='bilinear', align_corners=True)
        conv4 = F.interpolate(conv4, scale_factor=2, mode='bilinear', align_corners=True)
        
        center = self.center(self.pool(conv5))
        dec5, _ = self.dec5([center, conv5])
        dec4, _ = self.dec4([dec5, conv4])
        dec3, att = self.dec3([dec4, conv3])
        dec2, _ = self.dec2([dec3, conv2])
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(torch.cat([dec1, edge], dim=1))
        '''
        dec5 = self.dec5(torch.cat([center, conv5], dim=1))
        dec4 = self.dec4(torch.cat([dec5, conv4], dim=1))
        dec3 = self.dec3(torch.cat([dec4, conv3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, conv2], dim=1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(torch.cat([dec1, edge], dim=1))
        '''
        x_out = self.final(dec0)
        
        att = F.interpolate(att, scale_factor=4, mode='bilinear', align_corners=True)

        return x_out, edge_out, att

    def pad(self, x, y):
        diffX = y.shape[3] - x.shape[3]
        diffY = y.shape[2] - x.shape[2]

        return nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                        diffY // 2, diffY - diffY //2))


