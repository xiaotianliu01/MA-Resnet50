import numpy as np
import paddle
import paddle.nn.functional as F

def cal_miou(pred, label, num_classes, cls_count, bs):
    pred = paddle.squeeze(pred, axis=1)
    label = paddle.squeeze(label, axis=1)
    mask = label != 255
    pred = pred + 1
    label = label + 1
    pred = pred * mask
    label = label * mask
    pred = F.one_hot(pred, num_classes + 1)
    label = F.one_hot(label, num_classes + 1)
    pred = pred[:, :, :, 1:]
    label = label[:, :, :, 1:]
    iouc_sum = np.zeros(num_classes)
    batchsize = label.shape[0]
    for batch in range(batchsize):
        iouc=[]      
        for i in range(num_classes):
            pred_i = pred[batch, :, :, i]
            label_i = label[batch, :, :, i]
            pred_area_i = paddle.sum(pred_i)
            label_area_i = paddle.sum(label_i)
            intersect_area_i = paddle.sum(pred_i * label_i)
            if (label_area_i!=0):
                iou = (intersect_area_i/(pred_area_i+label_area_i-intersect_area_i)).numpy()[0]
                cls_count[i] = cls_count[i] + 1
            else:
                iou = 0
            iouc.append(iou)
        iouc_sum = iouc_sum + iouc
    return iouc_sum, cls_count
    