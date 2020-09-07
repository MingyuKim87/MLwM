import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_featnet(architecture, inputW=80, inputH=80):
    # if cifar dataset, the last 2 blocks of WRN should be without stride
    isCifar = (inputW == 32) or (inputH == 32)
    if architecture == 'WRN_28_10':
        net = WideResNet(28, 10, isCifar=isCifar)
        return net, net.nChannels
# Resnet
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.droprate = dropRate
        if self.droprate > 0:
            self.dropoutLayer = nn.Dropout(p=self.droprate)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        out = out if self.equalInOut else x
        out = self.conv1(out)
        if self.droprate > 0:
            out = self.dropoutLayer(out)
            #out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(self.relu2(self.bn2(out)))

        if not self.equalInOut:
            return self.convShortcut(x) + out
        else:
            return x + out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            in_plances_arg = i == 0 and in_planes or out_planes
            stride_arg = i == 0 and stride or 1
            layers.append(block(in_plances_arg, out_planes, stride_arg, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropRate=0.0, userelu=True, isCifar=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate) if isCifar \
                else  NetworkBlock(n, nChannels[0], nChannels[1], block, 2, dropRate)
        # 2nd block

        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True) if userelu else None
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)

        if self.relu is not None:
            out = self.relu(out)

        out = F.avg_pool2d(out, out.size(3))
        out = out.view(-1, self.nChannels)

        return out

# Classifier for trainer (Perceptron)
class ClassifierTrain(nn.Module):

    def __init__(self, nCls, nFeat=640, scaleCls = 10.):
        super(ClassifierTrain, self).__init__()

        self.scaleCls =  scaleCls
        self.nFeat =  nFeat
        self.nCls =  nCls

        # weights of base categories 
        self.weight = torch.FloatTensor(nFeat, nCls).normal_(0.0, np.sqrt(2.0/nFeat)) # Dimension nFeat * nCls
        self.weight = nn.Parameter(self.weight, requires_grad=True)

        # bias
        self.bias = nn.Parameter(torch.FloatTensor(1, nCls).fill_(0), requires_grad=True) # Dimension 1 * nCls

        # Scale of cls (Heat Parameter)
        self.scaleCls = nn.Parameter(torch.FloatTensor(1).fill_(scaleCls), requires_grad=True)

        # Method
        self.applyWeight = self.applyWeightCosine

    def getWeight(self):
        return self.weight, self.bias, self.scaleCls

    def applyWeightCosine(self, feature, weight, bias, scaleCls):
        batchSize, nFeat =feature.size()

        feature = F.normalize(feature, p=2, dim=1, eps=1e-12) ## Attention: normalized along 2nd dimension!!!
        weight = F.normalize(weight, p=2, dim=0, eps=1e-12)## Attention: normalized along 1st dimension!!!

        clsScore = scaleCls * (torch.mm(feature, weight) )#+ bias)
        return clsScore

    def forward(self, feature):
        weight, bias, scaleCls = self.getWeight()
        clsScore = self.applyWeight(feature, weight, bias, scaleCls)
        return clsScore

# Classifer for validation (cosine similarity)
class ClassifierEval(nn.Module):
    '''
    There is nothing to be learned in this classifier
    it is only used to evaluate netFeat episodically
    '''
    def __init__(self, nKnovel, nFeat):
        super(ClassifierEval, self).__init__()

        self.nKnovel = nKnovel
        self.nFeat = nFeat

        # bias & scale of classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=False)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=False)

    def apply_classification_weights(self, features, cls_weights):
        '''
        (B x n x nFeat, B x nKnovel x nFeat) -> B x n x nKnovel
        (B x n x nFeat, B x nKnovel*nExamplar x nFeat) -> B x n x nKnovel*nExamplar if init_type is nn
        '''
        features = F.normalize(features, p=2, dim=features.dim()-1, eps=1e-12)
        cls_weights = F.normalize(cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)
        cls_scores = self.scale_cls * torch.baddbmm(1.0, self.bias.view(1, 1, 1), 1.0, features, cls_weights.transpose(1,2))
        return cls_scores

    def forward(self, features_supp, features_query):
        '''
        features_supp: (B, nKnovel * nExamplar, nFeat)
        features_query: (B, nKnovel * nTest, nFeat)
        '''
        B = features_supp.size(0)

        weight = features_supp.view(B, self.nKnovel, -1, self.nFeat).mean(2)
        cls_scores = self.apply_classification_weights(features_query, weight)

        return cls_scores
