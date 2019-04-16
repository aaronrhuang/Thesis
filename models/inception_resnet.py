import torch
import torch.nn as nn
from models.se_module import SELayer


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    '''
    input 300*300*3
    output 35*35*128
    '''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class StemBlock(nn.Module):
    '''
    input 299*299*3
    output 35*35*384
    '''
    def __init__(self):
        super(StemBlock, self).__init__()
        self.model_a = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1)
        )
        self.branch_a0 = nn.MaxPool2d(3, stride=2)
        self.branch_a1 = BasicConv2d(64, 96, kernel_size=3, stride=2)
        self.branch_b0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )
        self.branch_b1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )
        self.branch_c0 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.branch_c1 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x = self.model_a(x)
        x_0 = self.branch_a0(x)
        x_1 = self.branch_a1(x)
        x = torch.cat((x_0, x_1), 1)
        x_0 = self.branch_b0(x)
        x_1 = self.branch_b1(x)
        x = torch.cat((x_0, x_1), 1)
        x_0 = self.branch_c0(x)
        x_1 = self.branch_c1(x)
        x = torch.cat((x_0, x_1), 1)
        return x


class InceptionResA(nn.Module):
    '''
    input 35*35*384
    output 35*35*384
    '''

    def __init__(self, planes = 384, scale=1.0, reduction=16):
        super(InceptionResA, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.scale = scale
        self.branch_0 = BasicConv2d(384, 32, kernel_size=1, stride=1)
        self.branch_1 = nn.Sequential(
            BasicConv2d(384, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(384, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )
        self.branch_all = BasicConv2d(128, 384, kernel_size=1, stride=1)
        self.se = SELayer(384, reduction)

    def forward(self, x):
        x = self.relu(x)
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_new = torch.cat((x_0, x_1, x_2), 1)
        x_new = self.branch_all(x_new)
        out = self.se(x_new)
        x = x + x_new * self.scale
        return x


class ReductionA(nn.Module):
    '''
    input 35*35*384
    output 17*17*1152
    '''
    def __init__(self):
        super(ReductionA, self).__init__()
        self.branch_0 = nn.MaxPool2d(3, stride=2)
        self.branch_1 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch_2 = nn.Sequential(
            BasicConv2d(384, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        return torch.cat((x_0, x_1, x_2), 1)


class InceptionResB(nn.Module):
    '''
    input 17*17*1152
    output 17*17*1152
    '''
    def __init__(self, scale=1.0, reduction=16):
        super(InceptionResB, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.scale = scale
        self.branch_0 = BasicConv2d(1152, 192, kernel_size=1, stride=1)
        self.branch_1 = nn.Sequential(
            BasicConv2d(1152, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        self.branch_all = BasicConv2d(384, 1152, kernel_size=1, stride=1)
        self.se = SELayer(1152, reduction)

    def forward(self, x):
        x = self.relu(x)
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_new = torch.cat((x_0, x_1), 1)
        x_new = self.branch_all(x_new)
        out = self.se(x_new)
        x = x + x_new * self.scale
        return x


class ReductionB(nn.Module):
    '''
    input 17*17*1152
    ouput 8*8*2144
    '''
    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch_0 = nn.MaxPool2d(3, stride=2)
        self.branch_1 = nn.Sequential(
            BasicConv2d(1152, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(1152, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )
        self.branch_3 = nn.Sequential(
            BasicConv2d(1152, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        return torch.cat((x_0, x_1, x_2, x_3), 1)


class InceptionResC(nn.Module):
    '''
    input 8*8*2144
    output 8*8*2144
    '''
    def __init__(self, scale=1.0, reduction=16):
        super(InceptionResC, self).__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=False)
        self.branch_0 = BasicConv2d(2144, 192, kernel_size=1, stride=1)
        self.branch_1 = nn.Sequential(
            BasicConv2d(2144, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.branch_all = BasicConv2d(448, 2144, kernel_size=1, stride=1)
        self.se = SELayer(2144, reduction)

    def forward(self, x):
        x = self.relu(x)
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_new = torch.cat((x_0, x_1), 1)
        x_new = self.branch_all(x_new)
        out = self.se(x_new)
        x = x + x_new * self.scale
        return x


class InceptionResV2(nn.Module):

    def __init__(self, num_classes=120, layers = [2,2,2]):
        super(InceptionResV2, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        downsample = nn.Sequential(
            nn.Conv2d(64, 384,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(384),
        )
        self.stem = nn.Sequential(self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            SEBasicBlock(64,384,downsample=downsample)
        )
        resA,resB,resC = [],[],[]
        for i in range(layers[0]):
            resA.append(InceptionResA())
        for i in range(layers[1]):
            resB.append(InceptionResB())
        for i in range(layers[2]):
            resC.append(InceptionResC())

        self.inception_resA5 = nn.Sequential(*resA)
        self.reductionA = ReductionA()
        self.inception_resB10 = nn.Sequential(*resB)
        self.reductionB = ReductionB()
        self.inception_resC5 = nn.Sequential(*resC)

        self.avg_pool = nn.AvgPool2d(8, count_include_pad=False)
        self.dropout = nn.Dropout2d(p=0.8)
        self.fc = nn.Linear(2144, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_resA5(x)
        x = self.reductionA(x)
        x = self.inception_resB10(x)
        x = self.reductionB(x)
        x = self.inception_resC5(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
