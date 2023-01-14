from dataset.cityscapes import Cityscapes
from models.deeplabv3 import DeepLabV3P
from models.pspnet import PSPNet
from models.sfnet import SFNet
from models.resnet_vd import ResNet50_vd
from utils.cal_miou import cal_miou
from utils.vis import vis
import paddle
import yaml
import numpy as np
import time

def val_prog(model, loader_val, model_name, best_ap):
    cont = 0
    time_cal = []
    with paddle.no_grad():
        print('start val...')
        s = time.time()
        mean_ap = 0
        cls_count = np.zeros(2)
        cls_count = cls_count + 0.00000001
        miou = np.zeros(2)
        r_h = 640
        r_w = 640
        for step_id, data in enumerate(loader_val):
            im = data[0]
            if len(im.shape)<4:
                #out = model(im,True)
                cont = cont + 1
                continue
            else:
                im = paddle.nn.functional.interpolate(im, (r_h, r_w))
                v = time.time()
                out = model(im,False)
                w = time.time()
                time_cal.append(w-v)
            if out is None:
                cont = cont + 1
                continue
            label = data[1].astype('int64')
            label = paddle.nn.functional.interpolate(label, (r_h, r_w))
            pred = paddle.argmax(out[0], axis=1, keepdim=True, dtype='int32')
            iou, cls_count = cal_miou(pred, label, 2, cls_count, 2)
            #print(cls_count)
            #vis(pred, label, step_id, 'clip', 2)
            miou = miou + iou
            if(int(time.time()-s)%100 == 0):
                print(step_id/len(loader_val)*100,'%: miou:', np.mean(miou/cls_count))
        print('miou_cls : ',miou/cls_count)
        mean_ap = np.mean(miou/cls_count)
        print('current_miou: ',mean_ap,'best_miou: ',best_ap)
        print('fps: ',1/np.mean(time_cal)*2)
        return True, mean_ap

if __name__ == '__main__':
    model_name = 'PSPNet'
    yml_path="work/dataset/cityscapes.yml"
    f = open(yml_path)
    config = yaml.load(f,Loader=yaml.FullLoader)
    dataset_val = Cityscapes(config['val_dataset']['transforms'],'data/camvid','val', 'VSPW')
    loader_val = paddle.io.DataLoader(dataset_val,
                    use_shared_memory=False,
                    batch_size=2,
                    shuffle=False,
                    drop_last=False,
                    num_workers=2,
                    return_list=True)
    print(len(loader_val))
    backbone = ResNet50_vd()
    if(model_name == 'PSPNet'):
        model = PSPNet(2,backbone)
    elif(model_name == 'DeepLabv3P'):
        model = DeepLabV3P(2,backbone)
    elif(model_name == 'SFNet'):
        model = SFNet(2,backbone)
    model.set_state_dict(paddle.load('model/pre_50_psp.pdparams'))
    flag, ap = val_prog(model, loader_val, model_name, 0)
