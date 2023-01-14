from dataset.cityscapes import Cityscapes
from models.pspnet import PSPNet
from models.deeplabv3 import DeepLabV3P
from models.resnet_vd import ResNet50_vd
from models.sfnet import SFNet
from utils.cal_miou import cal_miou
from utils.vis import vis
import time
import paddle
import yaml
import numpy as np
import random

def generate_dataset(mode, camvid_clips, yml_path, batch_size, shuffle):
    f = open(yml_path)
    config = yaml.load(f,Loader=yaml.FullLoader)
    dataloader = []
    for clip in camvid_clips:
        data = Cityscapes(config[mode+'_dataset']['transforms'],'data/camvid',mode, 'camvid',clip)
        loader = paddle.io.DataLoader(data,
                    use_shared_memory=False,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=True,
                    num_workers=2,
                    return_list=True)
        dataloader.append(loader)
    return dataloader


if __name__ == '__main__':
    model_name = 'SFNet'
    yml_path="work/dataset/cityscapes.yml"
    camvid_clips = ['16E5','05VD','06R0','01TP']
    camvid_identify = {50:'01TP',70:'05VD',40:'06R0',120:'16E5'}
    #camvid_identify = {100:'01TP',140:'05VD',80:'06R0',240:'16E5'}
    val_loader = generate_dataset('val', camvid_clips, yml_path, 2, False)
    backbone = ResNet50_vd()
    #backnone.set_state_dict(paddle.load('resnet50_vd_ssld_v2_imagenet/model.pdparams'))
    if(model_name == 'PSPNet'):
        model = PSPNet(19,backbone)
    elif(model_name == 'DeepLabv3P'):
        model = DeepLabV3P(19,backbone)
    elif(model_name == 'SFNet'):
        model = SFNet(19,backbone)
    #model.set_state_dict(paddle.load('model/home/aistudio/PSPNet_model/best_PSPNet_camvid_19cls_mem.pdparams'))
    with paddle.no_grad():
        print('start val...')
        mean_ap = 0
        data_sum = 0
        time_sum = []
        for clip_id, loader_val in enumerate(val_loader):
            miou = np.zeros(19)
            r_h = 720
            r_w = 960
            h = 720
            w = 960
            for step_id, data in enumerate(loader_val):
                im = data[0]
                if(step_id == 0):
                    out_test = model(im,True)
                im = data[0]
                im = paddle.nn.functional.interpolate(im, (r_h, r_w))
                label = data[1].astype('int64')
                if(step_id >= len(loader_val)-1):
                    out = model(im,True)
                    continue
                else:
                    s = time.time()
                    out = model(im,False)
                    e = time.time()
                    time_sum.append(e-s)
                if out is None:
                    continue
                '''
                if(step_id == 2):
                    break
                '''
                out[0] = paddle.nn.functional.interpolate(out[0], (h, w))
                #label = paddle.nn.functional.interpolate(label, (r_h, r_w))
                pred = paddle.argmax(out[0], axis=1, keepdim=True, dtype='int32')
                vis(label, label, step_id, camvid_clips[clip_id], 19)
                iou = cal_miou(pred, label, 19)/2
                miou = miou + iou
            data_sum = data_sum + len(loader_val)-1
            mean_ap = mean_ap + miou
            print('miou classes for '+camvid_clips[clip_id]+' : ',miou/(len(loader_val)-1))
            print('mean_iou for '+camvid_clips[clip_id]+' : ',np.mean(miou)/(len(loader_val)-1))
        print('current_miou_cls: ',mean_ap/data_sum)
        print('current_miou: ',np.mean(mean_ap)/data_sum)
        print('fps: ',1/np.mean(time_sum)*2)


