import sys
sys.path.append('work/models')
import os
import math
import numpy as np
from PIL import Image
import paddle
from LiteFlowNet import Network
import paddle as F
import paddle.fluid.dygraph as dg
import cv2
from warp_optical_flow import opflow_warp

def prepare_data(first_image_path, second_image_path):
    tenFirst = np.array(Image.open(first_image_path).convert("RGB")).astype("float32")
    tenSecond = np.array(Image.open(second_image_path).convert("RGB")).astype("float32")
    mean = np.array([0.411618, 0.434631, 0.454253]).astype("float32")

    tenFirst = tenFirst / 255. - mean
    tenSecond = tenSecond / 255. - mean

    tenFirst = tenFirst.transpose((2, 0, 1))
    tenSecond = tenSecond.transpose((2, 0, 1))

    h, w = tenFirst.shape[1:]

    tenFirst = dg.to_variable(tenFirst)
    tenSecond = dg.to_variable(tenSecond)

    tenFirst = F.reshape(tenFirst, (1, 3, h, w))
    tenSecond = F.reshape(tenSecond, (1, 3, h, w))

    r_h, r_w = int(math.floor(math.ceil(h / 32.0) * 32.0)), int(math.floor(math.ceil(w / 32.0) * 32.0))
    tenFirst = F.nn.functional.interpolate(tenFirst, (r_h, r_w))
    tenSecond = F.nn.functional.interpolate(tenSecond, (r_h, r_w))
    return tenFirst, tenSecond, (h, w), (r_h, r_w)


def get_model():
    model = Network()
    state_dict = paddle.load('liteflownet_model.pdparams')
    model.set_state_dict(state_dict)
    return model


if __name__ == '__main__':
    model = get_model()
    first, second, original_size, resized_size = prepare_data('data/camvid/images/01TP/0001TP_006720.png', 'data/camvid/images/01TP/0001TP_006690.png')
    flow = model(first, second)
    h, w = original_size
    r_h, r_w = resized_size
    flow = F.nn.functional.interpolate(flow, (h, w))

    flow[:, 0, :, :] *= float(w) / float(r_w)
    flow[:, 1, :, :] *= float(h) / float(r_h)
    #flow = F.transpose(flow[0], (1, 2, 0)).numpy() # [h, w, 2]
    img1 = cv2.imread('data/camvid/images/01TP/0001TP_006690.png')
    img2 = cv2.imread('data/camvid/images/01TP/0001TP_006720.png')
    img1= img1.transpose((2, 0, 1))
    img2= img2.transpose((2, 0, 1))
    tenFirst = dg.to_variable(img1)
    tenSecond = dg.to_variable(img2)
    flow = dg.to_variable(flow)
    tenFirst = F.reshape(tenFirst, (1, 3, h, w))
    tenSecond = F.reshape(tenSecond, (1, 3, h, w))
    opflow_warp(tenFirst, flow, True)

