# ------------------------------------------------------------------------
# Modified from https://github.com/sniklaus/pytorch-liteflownet
# ------------------------------------------------------------------------

import paddle.nn.functional as F
import paddle.fluid.layers as L
import paddle.nn as nn
import paddle
import time
def conv2d(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Conv2D(in_channels,
                     out_channels, 
                     kernel_size, 
                     stride=stride, 
                     padding=padding)


def deconv(in_channels, out_channels, groups):
    return nn.Conv2DTranspose(in_channels,
                              out_channels, 
                              4,
                              padding=1, 
                              stride=2, 
                              groups=groups,
                              bias_attr=False)


def conv1x1(in_channels, out_channels, stride=1):
    return conv2d(in_channels, out_channels, 1, stride, 0)


def conv3x3(in_channels, out_channels, stride=1):
    return conv2d(in_channels, out_channels, 3, stride, 1)


class LeakyReLU(nn.Layer):

    def __init__(self, negative_slope=0.1):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
    

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)


def grid_sample(x, grid):
    h, w = x.shape[2:]
    grid_x = grid[:, :, :, 0:1] * (w / (w - 1))
    grid_y = grid[:, :, :, 1:2] * (h / (h - 1))
    grid = paddle.concat([grid_x, grid_y], 3)
    return L.grid_sampler(x, grid)


backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
    start = time.time()
    h, w = tenFlow.shape[2:]
    if str(tenFlow.shape) not in backwarp_tenGrid:
        paddle.linspace(0, 1, 128, dtype="float32")
        tenHor = paddle.linspace(-1.0 + (1.0 / w), 1.0 - (1.0 / w),
                             w, dtype="float32")  # (w, )
        tenVer = paddle.linspace(-1.0 + (1.0 / h), 1.0 - (1.0 / h),
                             h, dtype="float32")  # (h, )
        tenHor = paddle.reshape(tenHor, (1, 1, 1, -1)) # (1, 1, 1, w)
        tenHor = paddle.expand(tenHor, (1, 1, h, w))   # (1, 1, h, w)
        tenVer = paddle.reshape(tenVer, (1, 1, -1, 1)) # (1, 1, h, 1)
        tenVer = paddle.expand(tenVer, (1, 1, h, w))   # (1, 1, h, w)

        backwarp_tenGrid[str(tenFlow.shape)] = paddle.concat([tenHor, tenVer], 1) # (1, 2, h, w)
    
    tenFlow = paddle.concat([tenFlow[:, 0:1, :, :] / ((w - 1.0) / 2.0),
                        tenFlow[:, 1:2, :, :] / ((h - 1.0) / 2.0),
                       ], 1) # (1, 2, h, w)
    grid = backwarp_tenGrid[str(tenFlow.shape)] + tenFlow # (1, 2, h, w)
    grid = paddle.transpose(grid, (0, 2, 3, 1))
    res = grid_sample(tenInput, grid, )
    end = time.time()
    #print('warp',end - start)
    return res


class Features(nn.Layer):

    def __init__(self):
        super(Features, self).__init__()

        self.moduleOne = nn.Sequential(
            ('0', conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3)),
            ('1', LeakyReLU())
        )

        self.moduleTwo = nn.Sequential(
            ('0', conv3x3(in_channels=32, out_channels=32, stride=2)),
            ('1', LeakyReLU()),
            ('2', conv3x3(in_channels=32, out_channels=32)),
            ('3', LeakyReLU()),
            ('4', conv3x3(in_channels=32, out_channels=32)),
            ('5', LeakyReLU())
        )

        self.moduleThr = nn.Sequential(
            ('0', conv3x3(in_channels=32, out_channels=64, stride=2)),
            ('1', LeakyReLU()),
            ('2', conv3x3(in_channels=64, out_channels=64)),
            ('3', LeakyReLU())
        )

        self.moduleFou = nn.Sequential(
            ('0', conv3x3(in_channels=64, out_channels=96, stride=2)),
            ('1', LeakyReLU()),
            ('2', conv3x3(in_channels=96, out_channels=96)),
            ('3', LeakyReLU())
        )

        self.moduleFiv = nn.Sequential(
            ('0', conv3x3(in_channels=96, out_channels=128, stride=2)),
            ('1', LeakyReLU())
        )

        self.moduleSix = nn.Sequential(
            ('0', conv3x3(in_channels=128, out_channels=192, stride=2)),
            ('1', LeakyReLU())
        )
    

    def forward(self, tenInput):
        tenOne = self.moduleOne(tenInput)
        tenTwo = self.moduleTwo(tenOne)
        tenThr = self.moduleThr(tenTwo)
        tenFou = self.moduleFou(tenThr)
        tenFiv = self.moduleFiv(tenFou)
        tenSix = self.moduleSix(tenFiv)

        return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]


class Matching(nn.Layer):

    def __init__(self, intLevel):
        super(Matching, self).__init__()

        self.fltBackwarp = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

        if intLevel != 2:
            self.moduleFeat = nn.Sequential()
        elif intLevel == 2:
            self.moduleFeat = nn.Sequential(
                ('0', conv1x1(in_channels=32, out_channels=64)),
                ('1', LeakyReLU())
            )
        
        if intLevel == 6:
            self.moduleUpflow = None
        elif intLevel != 6:
            self.moduleUpflow = deconv(in_channels=2, out_channels=2, groups=2)

        if intLevel >= 4:
            self.moduleUpcorr = None
        elif intLevel < 4:
            self.moduleUpcorr = deconv(in_channels=49, out_channels=49, groups=49)
        
        self.moduleMain = nn.Sequential(
            ('0', conv3x3(in_channels=49, out_channels=128)),
            ('1', LeakyReLU()),
            ('2', conv3x3(in_channels=128, out_channels=64)),
            ('3', LeakyReLU()),
            ('4', conv3x3(in_channels=64, out_channels=32)),
            ('5', LeakyReLU()),
            ('6', conv2d(in_channels=32, out_channels=2, 
                kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1, 
                padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))
        )


    def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
        tenFeaturesFirst = self.moduleFeat(tenFeaturesFirst)
        tenFeaturesSecond = self.moduleFeat(tenFeaturesSecond)

        if tenFlow is not None:
            tenFlow = self.moduleUpflow(tenFlow)
        
        if tenFlow is not None:
            tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond, 
                                         tenFlow=tenFlow * self.fltBackwarp)

        if self.moduleUpcorr is None:
            correlation = paddle.fluid.contrib.layers.nn.correlation(tenFeaturesFirst, tenFeaturesSecond, 
                                         pad_size=3,
                                         kernel_size=1,
                                         max_displacement=3,
                                         stride1=1,
                                         stride2=1,)
            tenCorrelation = F.leaky_relu(correlation, 0.1)
        elif self.moduleUpcorr is not None:
            start = time.time()
            correlation = paddle.fluid.contrib.layers.nn.correlation(tenFeaturesFirst, tenFeaturesSecond, 
                                         pad_size=6,
                                         kernel_size=1,
                                         max_displacement=6,
                                         stride1=2,
                                         stride2=2,)
            tenCorrelation = F.leaky_relu(correlation, 0.1)
            end = time.time()
            #print('cor',end - start)
            tenCorrelation = self.moduleUpcorr(tenCorrelation)
            
        return (tenFlow if tenFlow is not None else 0.0) + self.moduleMain(tenCorrelation)


class Subpixel(nn.Layer):

    def __init__(self, intLevel):
        super(Subpixel, self).__init__()

        self.fltBackward = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]

        if intLevel != 2:
            self.moduleFeat = nn.Sequential()
        elif intLevel == 2:
            self.moduleFeat = nn.Sequential(
                ('0', conv1x1(in_channels=32, out_channels=64)),
                ('1', LeakyReLU())
            )
        
        self.moduleMain = nn.Sequential(
            ('0', conv3x3(in_channels=[0, 0, 130, 130, 194, 258, 386][intLevel], out_channels=128)),
            ('1', LeakyReLU()),
            ('2', conv3x3(in_channels=128, out_channels=64)),
            ('3', LeakyReLU()),
            ('4', conv3x3(in_channels=64, out_channels=32)),
            ('5', LeakyReLU()),
            ('6', conv2d(in_channels=32, out_channels=2, 
                         kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1,
                         padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))
        )


    def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
        tenFeaturesFirst = self.moduleFeat(tenFeaturesFirst)
        tenFeaturesSecond = self.moduleFeat(tenFeaturesSecond)

        if tenFlow is not None:
            tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond, tenFlow=tenFlow * self.fltBackward)

        tenFeatures = paddle.concat([tenFeaturesFirst, tenFeaturesSecond, tenFlow], 1)
        return (tenFlow if tenFlow is not None else 0.0) + self.moduleMain(tenFeatures)


class Regularization(nn.Layer):

    def __init__(self, intLevel):
        super(Regularization, self).__init__()

        self.fltBackward = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625][intLevel]
        self.intUnfold = [0, 0, 7, 5, 5, 3, 3][intLevel]

        if intLevel >= 5:
            self.moduleFeat = nn.Sequential()
        elif intLevel < 5:
            self.moduleFeat = nn.Sequential(
                ('0', conv1x1(in_channels=[0, 0, 32, 64, 96, 128, 192][intLevel], out_channels=128)),
                ('1', LeakyReLU())
            )
        
        self.moduleMain = nn.Sequential(
            ('0', conv3x3(in_channels=[0, 0, 131, 131, 131, 131, 195][intLevel], out_channels=128)),
            ('1', LeakyReLU()),
            ('2', conv3x3(in_channels=128, out_channels=128)),
            ('3', LeakyReLU()),
            ('4', conv3x3(in_channels=128, out_channels=64)),
            ('5', LeakyReLU()),
            ('6', conv3x3(in_channels=64, out_channels=64)),
            ('7', LeakyReLU()),
            ('8', conv3x3(in_channels=64, out_channels=32)),
            ('9', LeakyReLU()),
            ('10', conv3x3(in_channels=32, out_channels=32)),
            ('11', LeakyReLU())
        )

        if intLevel >= 5:
            self.moduleDist = nn.Sequential(
                ('0', conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                       kernel_size=[0, 0, 7, 5, 5, 3, 3][intLevel], stride=1,
                       padding=[0, 0, 3, 2, 2, 1, 1][intLevel]))
            )
        elif intLevel < 5:
            self.moduleDist = nn.Sequential(
                ('0', conv2d(in_channels=32, out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel],
                       kernel_size=([0, 0, 7, 5, 5, 3, 3][intLevel], 1), stride=1,
                       padding=([0, 0, 3, 2, 2, 1, 1][intLevel], 0))),
                ('1', conv2d(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], 
                             out_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], 
                             kernel_size=(1, [0, 0, 7, 5, 5, 3, 3][intLevel]),
                             stride=1, padding=(0, [0, 0, 3, 2, 2, 1, 1][intLevel])))
            )
        
        self.moduleScaleX = conv1x1(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1)
        self.moduleScaleY = conv1x1(in_channels=[0, 0, 49, 25, 25, 9, 9][intLevel], out_channels=1)
    

    def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
        b, _, h, w = tenFlow.shape
        tenDifference = tenFirst - backwarp(tenInput=tenSecond, tenFlow=tenFlow * self.fltBackward)
        tenDifference = paddle.pow(tenDifference, 2)
        tenDifference = L.reduce_sum(tenDifference, 1, True) # [b, 1, h, w]
        tenDifference = paddle.sqrt(tenDifference).detach()

        tenFeaturesFirst = self.moduleFeat(tenFeaturesFirst)

        tenMean = paddle.reshape(tenFlow, (b, 2, -1))    # [b, 2, h * w]
        tenMean = L.reduce_mean(tenMean, 2, True)   # [b, 2, 1]
        tenMean = paddle.reshape(tenMean, (b, 2, 1, 1))  # [b, 2, 1, 1]
        tenMean = paddle.expand(tenMean, (b,2, h, w))   # [b, 2, h, w]
        delta = tenFlow - tenMean

        diff = paddle.concat([tenDifference, delta, tenFeaturesFirst], 1)
        tenDist = self.moduleDist(self.moduleMain(diff))
        tenDist = paddle.pow(tenDist, 2.0) * -1.0
        tenDist = tenDist - L.reduce_max(tenDist, 1, True)
        tenDist = paddle.exp(tenDist)

        tenDivisor = L.reduce_sum(tenDist, 1, True)
        tenDivisor = paddle.reciprocal(tenDivisor)

        tenScaleX = F.unfold(x=tenFlow[:, 0:1, :, :], 
                             kernel_sizes=self.intUnfold, 
                             strides=1, 
                             paddings=int((self.intUnfold - 1) / 2)) # [b, c, h * w]
        tenScaleX = paddle.reshape(tenScaleX, (b, -1, h, w))          # [b, c, h, w]
        tenScaleX = self.moduleScaleX(tenDist * tenScaleX) * tenDivisor

        tenScaleY = F.unfold(x=tenFlow[:, 1:2, :, :], 
                             kernel_sizes=self.intUnfold, 
                             strides=1, 
                             paddings=int((self.intUnfold - 1) / 2)) # [b, c, h * w]
        tenScaleY = paddle.reshape(tenScaleY, (b, -1, h, w))          # [b, c, h, w]
        tenScaleY = self.moduleScaleY(tenDist * tenScaleY) * tenDivisor

        return paddle.concat([tenScaleX, tenScaleY], 1)


class Network(nn.Layer):

    def __init__(self, ):
        super(Network, self).__init__()

        self.moduleFeatures = Features()
        levels = [2, 3, 4, 5, 6]
        self.moduleMatching = nn.LayerList([Matching(intLevel) for intLevel in levels])
        self.moduleSubpixel = nn.LayerList([Subpixel(intLevel) for intLevel in levels])
        self.moduleRegularization = nn.LayerList([Regularization(intLevel) for intLevel in levels])
    

    def forward(self, tenFirst, tenSecond):
        ori_h, ori_w = tenFirst.shape[2:]
        tenFeaturesFirst = self.moduleFeatures(tenFirst)
        tenFeaturesSecond = self.moduleFeatures(tenSecond)
        tenFirst = [tenFirst]
        tenSecond = [tenSecond]

        for intLevel in [1, 2, 3, 4, 5]:
            h, w = tenFeaturesFirst[intLevel].shape[2:]
            tenFirst.append(F.interpolate(tenFirst[-1],(h, w), align_corners=False))
            tenSecond.append(F.interpolate(tenSecond[-1], (h, w), align_corners=False))
        
        tenFlow = None

        for intLevel in [-1, -2, -3, -4, -5]:
            start = time.time()
            tenFlow = self.moduleMatching[intLevel + 5](tenFirst[intLevel], tenSecond[intLevel],
                                                    tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow)
            end = time.time()
            #print(intLevel,end - start)
            start = time.time()
            tenFlow = self.moduleSubpixel[intLevel + 5](tenFirst[intLevel], tenSecond[intLevel],
                                                    tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow)
            end = time.time()
            #print(intLevel,end - start)
            start = time.time()
            tenFlow = self.moduleRegularization[intLevel + 5](tenFirst[intLevel], tenSecond[intLevel],
                                                          tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow)
            end = time.time()
            #print(intLevel,end - start)
        cur_h, cur_w = tenFlow.shape[2:]
        tenFlow = F.interpolate(tenFlow,(ori_h, ori_w))
        tenFlow[:, 0, :, :] *= float(ori_w) / float(cur_w)
        tenFlow[:, 1, :, :] *= float(ori_h) / float(cur_h)
        return tenFlow * 20.0

