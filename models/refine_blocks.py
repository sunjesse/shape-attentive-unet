"""RefineNet-CRP-RCU-blocks in PyTorch
RefineNet-PyTorch for non-commercial purposes
Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


def batchnorm(in_planes):
    "batch norm 2d"
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_planes),
                             nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_planes))

class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        self.top1 = nn.Sequential(conv1x1(in_planes,
                                          out_planes),
                                  nn.ReLU(inplace=True))
        self.top2 = nn.Sequential(conv1x1(in_planes,
                                          out_planes),
                                  nn.ReLU(inplace=True))
        self.top3 = nn.Sequential(conv1x1(in_planes,
                                          out_planes),
                                  nn.ReLU(inplace=True))
        '''
        for i in range(n_stages):
            self.tops.append(nn.Sequential(conv1x1(in_planes,
                                                   out_planes),
                                          nn.ReLU(inplace=True)))
        '''

        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        
        ### initialize
        for t in self.modules():
            if isinstance(t, nn.Sequential):
                for m in t:
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                        m = m.cuda()

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            if i == 0:
                top = self.top1(top)
            elif i==1:
                top = self.top2(top)
            elif i==2:
                top = self.top3(top)
            x = top + x
        return x
    
stages_suffixes = {0 : '_conv',
                   1 : '_conv_relu_varout_dimred'}
    
class RCUBlock(nn.Module):
    
    def __init__(self, in_planes, out_planes, n_blocks, n_stages):
        super(RCUBlock, self).__init__()
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}{}'.format(i + 1, stages_suffixes[j]),
                        conv3x3(in_planes,
                                in_planes,
                                stride=1,
                                bias=(j == 0)))
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        
        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()  
            elif isinstance(m, nn.Sequential):
                for i in m:
                    if isinstance(i, nn.Conv2d):
                        n = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                        i.weight.data.normal_(0, math.sqrt(2. / n))
            m = m.cuda()

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = F.relu(x)
                x = getattr(self, '{}{}'.format(i + 1, stages_suffixes[j]))(x)
            x += residual
        return x


class MRFBlock(nn.Module):
    def __init__(self, img_dim=[1/2, 1/4], in_planes=[512], channels=256, bn=False):
        super(MRFBlock, self).__init__()
        self.paths = []
        self.channels = channels
        self.in_planes = in_planes
        self.target_res = max(img_dim)
        self.path1 = nn.Sequential(nn.ConvTranspose2d(self.in_planes[0],
                                                      self.channels,
                                                      kernel_size=4,
                                                      stride=2,
                                                      padding=1),
                                   nn.BatchNorm2d(self.channels),
                                   nn.ReLU(inplace=True))
        self.path2 = nn.Sequential(nn.Conv2d(self.in_planes[1],
                                             self.channels,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             bias=False),
                                   nn.BatchNorm2d(self.channels),
                                   nn.ReLU(inplace=True))
        #self.path3 = None
        '''
        for i in range(len(img_dim)):
            if img_dim[i] == self.target_res:
                self.paths.append(nn.Sequential(nn.Conv2d(self.in_planes[i],
                                                          self.channels,
                                                          kernel_size=1,
                                                          stride=1,
                                                          padding=0,
                                                          bias=False),
                                               nn.BatchNorm2d(self.channels),
                                               nn.ReLU(inplace=True)))
                                               #nn.ReLU(inplace=True)))
            elif img_dim[i]/self.target_res == 0.5: # we upsample by 2
                self.paths.append(nn.Sequential(nn.ConvTranspose2d(self.in_planes[i],
                                                                   self.channels,
                                                                   kernel_size=4,
                                                                   stride=2,
                                                                   padding=1),
                                                nn.BatchNorm2d(self.channels),
                                                nn.ReLU(inplace=True)))
                                                #nn.ReLU(inplace=True)))
        '''
        for p in self.modules():
            if isinstance(p, nn.Sequential):
                for m in p:
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
                    m = m.cuda()

    def forward(self, x): # x is a tuple of incoming feature maps at different resolutions.
        out = []
        for i in range(len(x)):
            if i == 0:
                out.append(self.path1(x[i]))
            elif i == 1:
                out.append(self.path2(x[i]))
        return torch.cat(out, dim=1) #NCHW
        
        #utput = out[0]
        #for i in range(1, len(out)):
        #    output += out[i]


class RCUModule(nn.Module):
    def __init__(self, inchannels=[128, 256, 256], outchannels=256, n_blocks=2, n_stages=2):
        super(RCUModule, self).__init__()
        self.path_count = len(inchannels)
        self.paths = []
        self.out_chs = outchannels

        for i in range(self.path_count):
            self.paths.append(RCUBlock(in_planes=inchannels[i],
                                       out_planes=self.out_chs,
                                       n_blocks=n_blocks,
                                       n_stages=n_stages))

    def forward(self, x): # x is a tuple of incoming feature maps
        assert len(x) == len(self.paths)

        out = []
        for i in range(len(x)):
            out.append(self.paths[i](x[i]))
        return out

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out

class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride



class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


BOTTLENECK_WIDTH = 128
COEFF = 12.0

def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x*weight, kernel, stride, padding)/F.avg_pool2d(weight, kernel, stride, padding)

class SoftGate(nn.Module):
    def __init__(self):
        super(SoftGate, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x) * COEFF


class BottleneckShared(nn.Module):
    def __init__(self, channels):
        super(BottleneckShared, self).__init__()
        rp = BOTTLENECK_WIDTH

        self.logit = nn.Sequential(
            OrderedDict((
                ('conv1', conv1x1(channels, rp)),
                ('bn1', nn.InstanceNorm2d(rp, affine=True)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', conv3x3(rp, rp)),
                ('bn2', nn.InstanceNorm2d(rp, affine=True)),
                ('relu2', nn.ReLU(inplace=True)),
            ))
        )

    def init_layer(self):
        pass

    def forward(self, x):
        return self.logit(x)

class BottleneckLIP(nn.Module):
    def __init__(self, channels):
        super(BottleneckLIP, self).__init__()
        rp = BOTTLENECK_WIDTH

        self.postprocessing = nn.Sequential(
            OrderedDict((
                ('conv', conv1x1(rp, channels)),
                ('bn', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate()),
            ))
        )

    def init_layer(self):
        self.postprocessing[0].weight.data.fill_(0.0)
        pass

    def forward_with_shared(self, x, shared):
        frac = lip2d(x, self.postprocessing(shared))
        return frac


class ProjectionLIP(nn.Module):
    def __init__(self, channels):
        super(ProjectionLIP, self).__init__()

        rp = channels

        self.logit = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                   nn.InstanceNorm2d(channels, affine=True),
                                   SoftGate())

    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x))
        return frac



class GridAttentionBlock(nn.Module):
    def __init__(self, in_features, attn_features, up_factor, normalize_attn=False):
        super(GridAttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.down = nn.Conv2d(in_features, attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(attn_features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        c = self.down(x) # batch_sizexattn_featuersxWxH
        c = self.phi(self.relu(self.bn(c)))
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c) # not a seperate attention map for each channel, its universal. Makes sense, if not, we have so many variables. The general heated area of attention should not change between channels
        # re-weight the local feature
        return a
        
        '''
        f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
       
        if self.normalize_attn:
            output = f.view(N,C,-1).sum(dim=2) # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1,1)).view(N,C)
    
        return a, f#c.view(N,1,W,H), output
        '''
'''
class _MRF(nn.Module):
    def __init__(self, inchannels):
        super(_MRF, self).__init__()
        #self.avgpool2 = nn.AvgPool2d(2, stride=2)
        #self.avgpool4 = nn.AvgPool2d(4, stride=4)
        #self.avgpool8 = nn.AvgPool2d(8, stride=8)
        self.lip_1 = None
        self.lip_2 = None
        self.lip_3 = None

        self.reduce_4 = None
        self.reduce_3 = None
        self.reduce_2 = None
        self.bn_4 = None
        self.bn_3 = None
        self.bn_2 = None
        self.skip = nn.Sequential(nn.Conv2d(inchannels[1],
                                            128,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True))
        
        self.up = nn.Sequential(nn.ConvTranspose2d(inchannels[0],
                                                   inchannels[0],
                                                   kernel_size=4,
                                                   stride=2,
                                                   padding=1),
                                nn.BatchNorm2d(inchannels[0]),
                                nn.ReLU(inplace=True))
        if len(inchannels) == 5:
            self.reduce_4 = nn.Sequential(nn.Conv2d(inchannels[-3], 128, kernel_size=1, padding=0, stride=1),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True))
            self.lip_1 = ProjectionLIP(inchannels[-3])
            self.lip_2 = nn.Sequential(ProjectionLIP(inchannels[-2]), 
                                       ProjectionLIP(inchannels[-2]))
            self.lip_3 = nn.Sequential(ProjectionLIP(inchannels[-1]),
                                       ProjectionLIP(inchannels[-1]),
                                       ProjectionLIP(inchannels[-1]))
        if len(inchannels) >= 4:
            self.reduce_3 = nn.Sequential(nn.Conv2d(inchannels[-2], 128, kernel_size=1, padding=0, stride=1),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True))
            if len(inchannels) == 4:
                self.lip_1 = ProjectionLIP(inchannels[-2])
                self.lip_2 = nn.Sequential(ProjectionLIP(inchannels[-1]),
                                           ProjectionLIP(inchannels[-1]))

        if len(inchannels) >= 3:
            self.reduce_2 = nn.Sequential(nn.Conv2d(inchannels[-1], 128, kernel_size=1, padding=0, stride=1),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True))
            if len(inchannels) == 3:
                self.lip_1 = ProjectionLIP(inchannels[-1])

        for p in self.modules():
            if isinstance(p, nn.Sequential):
                for m in p:
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
                    elif isinstance(m, nn.Linear):
                        m.bias.data.zero_()

    def forward(self, channels):
        out = []
        
        out.append(self.up(channels[0]))
        out.append(self.skip(F.relu(channels[1])))
        
        if len(channels) == 5:
            out.append(self.reduce_4(self.lip_1(F.relu(channels[2]))))
            out.append(self.reduce_3(self.lip_2(F.relu(channels[3]))))
            out.append(self.reduce_2(self.lip_3(F.relu(channels[4]))))
        
        elif len(channels) == 4:
            out.append(self.reduce_3(self.lip_1(F.relu(channels[2]))))
            out.append(self.reduce_2(self.lip_2(F.relu(channels[3]))))
        
        elif len(channels) == 3:
            out.append(self.reduce_2(self.lip_1(F.relu(channels[2]))))
    
        return torch.cat(out, dim=1)
'''

class _MRF(nn.Module):
    def __init__(self, inchannels):
        super(_MRF, self).__init__()
        #self.avgpool2 = nn.AvgPool2d(2, stride=2)
        #self.avgpool4 = nn.AvgPool2d(4, stride=4)
        #self.avgpool8 = nn.AvgPool2d(8, stride=8)

        self.up = nn.Sequential(nn.ConvTranspose2d(inchannels[0],
                                                   inchannels[0],
                                                   kernel_size=4,
                                                   stride=2,
                                                   padding=1),
                                nn.BatchNorm2d(inchannels[0]),
                                nn.ReLU(inplace=True))

        for p in self.modules():
            if isinstance(p, nn.Sequential):
                for m in p:
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
                    elif isinstance(m, nn.Linear):
                        m.bias.data.zero_()

    def forward(self, channels):
        if len(channels) == 2:
            out = []
            out.append(channels[1])
            out.append(self.up(channels[0]))
            return torch.cat(out, dim=1)
        elif len(channels) == 1:
            return channels[0]

class RefineBlock(nn.Module):
    def __init__(self, inchannels=[128, 256], outchannels=256):
        super(RefineBlock, self).__init__()
        inchs = sum(inchannels) 
        self.mrf = _MRF(inchannels)
        self.spatialAttn = GridAttentionBlock(outchannels, int(outchannels/4), 2)
        self.channelAttn = SEModule(outchannels, 16)
        self.c3x3rb = nn.Sequential(nn.Conv2d(inchs, 
                                              outchannels, 
                                              kernel_size=3,
                                              padding=1),
                                    nn.BatchNorm2d(outchannels),
                                    nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x): # x is a tuple of incoming feature maps at different resolutions
        fused = self.mrf(x)
        fused = self.c3x3rb(fused)
        spatial = self.spatialAttn(fused)
        channel = self.channelAttn(fused)
        out = torch.mul(spatial.expand_as(channel) + 1, channel) # F(X) = C(X)*(1+S(X))
        #out = F.relu(self.reduce(out))
        return fused, spatial

