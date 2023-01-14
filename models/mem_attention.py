import paddle.nn as nn
import paddle.nn.functional as F
from . import layers
import paddle
import time
import sys
import cv2

class ConvBNLayer(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            is_vd_mode=False,
            act=None,
    ):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 if dilation == 1 else 0,
            dilation=dilation,
            groups=groups,
            bias_attr=False)

        self._batch_norm = layers.SyncBatchNorm(out_channels)
        self._act_op = layers.Activation(act=act)

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        y = self._act_op(y)

        return y

class mem_attention(nn.Layer):
    def __init__(self, mem_num = 3, channels = 256, ratio = 0.25):
        super(mem_attention, self).__init__()
        self.mem_num = mem_num
        self.channels = channels
        self.ratio = ratio
        self.mem = {'key':[], 'value':[]}
        '''
        nn.Conv2D(int(channels/8),int(channels/8),2,2,0),
            nn.ReLU(),
        '''
        self.key_enc = nn.Sequential(
            ConvBNLayer(int(channels*ratio), int(channels*ratio/2),3,1,act = 'relu')
            #nn.Conv2D(int(channels/8),int(channels/8),3,1,1),
            #nn.ReLU()
        )
        '''
        nn.Conv2D(int(channels/4),int(channels/4),2,2,0),
            nn.ReLU(),
        '''
        
        self.value_enc = nn.Sequential(
            #nn.Conv2D(int(channels/2),int(channels/2),3,1,1),
            #ConvBNLayer(int(channels*ratio), int(channels*ratio),3,1,act = 'relu')
            nn.ReLU()
        )
        self.bn_relu = nn.Sequential(
            layers.SyncBatchNorm(int(channels*ratio)),
            nn.ReLU(),

        )
        self.softmax = nn.Softmax(axis = 1)
        self.act = nn.Sigmoid()
    
    def flatten(self, x, dim = 5):
        if(dim==5):
            B,C,K,H,W = x.shape
            res = paddle.reshape(x, (B,C,K*H*W))
        elif(dim==4):
            B,C,H,W = x.shape
            res = paddle.reshape(x, (B,C,H*W))
        return res
    
    def forward(self, x, clr, renew):
        if(clr):
            self.mem['key'] = []
            self.mem['value'] = []
            return None
        B,C,H,W = x.shape
        H = int(H/2)
        W = int(W/2)
        feat_cur = x[:,int(C*(1-self.ratio)):,:,:]
        '''
        print(paddle.max(feat_cur))
        print(paddle.min(feat_cur))
        print(paddle.mean(feat_cur))
        '''
        cur_key = self.key_enc(feat_cur)
        '''
        print(paddle.max(cur_key))
        print(paddle.min(cur_key))
        print(paddle.mean(cur_key))
        '''
        cur_key = F.interpolate(cur_key,[H,W])
        cur_value = self.value_enc(feat_cur)
        cur_value = F.interpolate(cur_value,[H,W])
        if(len(self.mem['key'])<self.mem_num):
            self.mem['key'].append(paddle.reshape(cur_key,(B,int(C*self.ratio/2),1,H,W)).detach())
            self.mem['value'].append(paddle.reshape(cur_value,(B,int(C*self.ratio),1,H,W)).detach())
            return [x,x]
        mem_key = self.mem['key'][0]
        for i in range(1,len(self.mem['key'])):
            mem_key = paddle.concat((mem_key, self.mem['key'][i]), 2)
        key_ten = self.flatten(mem_key, 5)
        '''
        if(C ==256):
            for i in range(128):
                    a = self.mem['value'][0].numpy()*255
                    a = a[0][i][0][:][:]
                    cv2.imwrite('feat1/'+str(i)+'.png', a)
        '''
        cur_key_ten = self.flatten(cur_key, 4)
        key_ten_norm = self.softmax(key_ten)*10
        cur_key_ten_norm = self.softmax(cur_key_ten)*10
        attention = paddle.matmul(key_ten_norm, cur_key_ten_norm, True, False) #B KHW HW B C KHW * B C HW
        #attention = self.act(attention)
        attention = self.softmax(attention)
        mem_value = self.mem['value'][0]
        for i in range(1,len(self.mem['value'])):
            mem_value = paddle.concat((mem_value, self.mem['value'][i]), 2)
        value_ten = self.flatten(mem_value, 5)
        emb_ten = paddle.matmul(value_ten, attention, False, False)
        emb_ten = paddle.reshape(emb_ten, (B,int(C*self.ratio),H,W))
        emb_ten = F.interpolate(emb_ten,[H*2,W*2],mode='BILINEAR')
        emb_ten = self.bn_relu(emb_ten)
        #print(paddle.max(emb_ten))
        #print(paddle.min(emb_ten))
        #print(paddle.mean(emb_ten))
        emb_ten = emb_ten*(paddle.mean(feat_cur)/paddle.mean(emb_ten))+feat_cur
        x = x[:,:int(C*(1-self.ratio)),:,:]
        x = paddle.concat((x, emb_ten), 1)
        if(renew):
            for i in range(self.mem_num-1):
                self.mem['key'][i] = self.mem['key'][i+1]
                self.mem['value'][i] = self.mem['value'][i+1]
            self.mem['key'][self.mem_num-1] = paddle.reshape(cur_key,(B,int(C*self.ratio/2),1,H,W)).detach()
            self.mem['value'][self.mem_num-1] = paddle.reshape(cur_value,(B,int(C*self.ratio),1,H,W)).detach()
        return x
