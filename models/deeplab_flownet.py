from .deeplabv3 import DeepLabV3P
from .resnet_vd import ResNet50_vd
from .LiteFlowNet import Network
from .fpn import fpn
import paddle.nn as nn
import paddle
import time
import sys
sys.path.append('work/utils')
from utils.warp_optical_flow import opflow_warp
class Deep_op(nn.Layer):
    
    def __init__(self,pretrained=None):
        super().__init__()

        self.backbone = ResNet50_vd()
        self.deeplab = DeepLabV3P(19, self.backbone)
        self.lite_flow_net = Network()
        '''
        self.conv = nn.Conv2D(
            in_channels=2,
            out_channels=1,
            kernel_size=8,
            stride=4, 
            padding=2)
        self.sigmoid = nn.Sigmoid()
        '''
        self.fpn = fpn()

        if (pretrained != None):
            deeplab_dict_path = pretrained['DeepLabV3P']
            load = paddle.load(deeplab_dict_path)
            self.deeplab.set_state_dict(load)
            LFN_dict_path = pretrained['LiteFlowNet']
            load = paddle.load(LFN_dict_path)
            self.lite_flow_net.set_state_dict(load)


    def forward(self, x, clr):
        #op_flow = self.lite_flow_net(x, last_img)
        #print(paddle.max(op_flow))
        #opflow_warp(last_img, op_flow, True)
        #mag = paddle.sqrt(paddle.square(op_flow[:,0,:,:])+paddle.square(op_flow[:,1,:,:]))
        #mag = mag/paddle.max(mag)
        #ang = paddle.atan(op_flow[:,0,:,:]/op_flow[:,0,:,:])*2/3.1415
        #op_flow[:,0,:,:] = mag
        #op_flow[:,1,:,:] = ang
        #op_attention = self.conv(op_flow)
        #op_attention = self.sigmoid(op_attention)
        #op_attention = self.fpn(op_flow)
        #print(op_attention)
        #print(op_attention)
        #print(paddle.min(op_attention))
        mask = self.deeplab(x, clr)
        return mask


