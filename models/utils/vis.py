import numpy as np
import paddle
import paddle.nn.functional as F
import cv2

def vis(pred, label, ID, clip, num_classes):
    color = [[64,128,64],[192,0,128],[0,128,192],[0,128,64],[128,0,0],[64,0,128],
    [64,0,192],[192,128,64],[192,192,128],[64,64,128],[128,0,192],[192,0,64],[128,128,64],
    [192,0,192],[128,64,64],[64,192,128],[64,64,0],[128,64,128],[128,128,192]]
    pred = paddle.squeeze(pred, axis=1)
    label = paddle.squeeze(label, axis=1)
    mask = label != 255
    pred = pred + 1
    label = label + 1
    pred = pred * mask
    label = label * mask
    pred = F.one_hot(pred, num_classes + 1)
    pred = pred[:, :, :, 1:]
    batchsize, h, w = label.shape
    for batch in range(batchsize):
        vis_res = np.zeros((h,w,3))      
        for i in range(num_classes):
            pred_i = pred[batch, :, :, i].numpy()
            vis_res[:,:,0] = vis_res[:,:,0] + pred_i*color[i%19][0]
            vis_res[:,:,1] = vis_res[:,:,1] + pred_i*color[i%19][1]
            vis_res[:,:,2] = vis_res[:,:,2] + pred_i*color[i%19][2]
        cv2.imwrite('vis/'+clip+'_'+str(ID)+'_'+str(batch)+'.png', vis_res)
    