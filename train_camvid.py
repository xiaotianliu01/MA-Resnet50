from dataset.cityscapes import Cityscapes
from models.pspnet import PSPNet
from models.deeplabv3 import DeepLabV3P
from models.sfnet import SFNet
from models.resnet_vd import ResNet50_vd
from utils.cal_miou import cal_miou
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
    camvid_clips = ['01TP','05VD','06R0','16E5']
    camvid_identify = {50:'01TP',70:'05VD',40:'06R0',120:'16E5'}
    #camvid_identify = {100:'01TP',140:'05VD',80:'06R0',240:'16E5'}
    train_loader = generate_dataset('train', camvid_clips, yml_path, 2, False)
    val_loader = generate_dataset('val', camvid_clips, yml_path, 2, False)
    ce_loss = paddle.nn.CrossEntropyLoss(ignore_index= 255, reduction='mean', axis=1)
    backbone = ResNet50_vd()
    if(model_name == 'PSPNet'):
        model = PSPNet(19,backbone)
    elif(model_name == 'DeepLabv3P'):
        model = DeepLabV3P(19,backbone)
    elif(model_name == 'SFNet'):
        model = SFNet(19,backbone)
    model.set_state_dict(paddle.load('SFnet_model/best_SFNet_camvid_19cls.pdparams'))
    epoch = 50
    loss_sum = 0
    cal = 10
    best_ap = 0
    counter = 0
    lr = [0.1,0.05,0.01,0.004]
    lr_cnt = 0
    #sche = paddle.optimizer.lr.PolynomialDecay(0.0005, 10, 0, 0.9)
    #sche = paddle.optimizer.lr.PiecewiseDecay(boundaries=[1,2,3], values=[0.1,0.01,0.002,0.0004,0.0001], verbose=False)
    for i in range(epoch):
        random.shuffle(train_loader)
        model.train()
        optimizer = paddle.optimizer.SGD(learning_rate=lr[lr_cnt], parameters=model.parameters(), weight_decay=4.0e-5)
        print('start train...')
        for clip_id, loader in enumerate(train_loader):
            for step_id, data in enumerate(loader):
                im = data[0]
                label = data[1].astype('int64')
                if(step_id == len(loader)-1):
                    out = model(im,True)
                    continue
                else:
                    out = model(im,False)
                if out is None:
                    continue
                loss = ce_loss(out[0], label)
                loss_sum = loss_sum + loss.item()
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                if((step_id+1)%cal == 0):
                    print("epoch[{}],clip:{},[{}/{}],loss:{}".format(i,camvid_identify[int(len(loader))], step_id+1,int(len(loader)),loss_sum/cal))
                    loss_sum = 0
        static_model=model.state_dict()
        paddle.save(static_model,'./SFnet_model/'+model_name+'_camvid_19cls_mem_0.125.pdparams')
        with paddle.no_grad():
            print('start val...')
            mean_ap = 0
            data_sum = 0
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
                        out = model(im,False)
                    if out is None:
                        continue
                    out[0] = paddle.nn.functional.interpolate(out[0], (h, w))
                    pred = paddle.argmax(out[0], axis=1, keepdim=True, dtype='int32')
                    iou = cal_miou(pred, label, 19)/2
                    miou = miou + iou
                data_sum = data_sum + len(loader_val)-2
                mean_ap = mean_ap + miou
                print('miou classes for '+camvid_clips[clip_id]+' : ',miou/(len(loader_val)-2))
                print('mean_iou for '+camvid_clips[clip_id]+' : ',np.mean(miou)/(len(loader_val)-2))
            if np.mean(mean_ap)/data_sum > best_ap:
                counter = 0
                static_model=model.state_dict()
                paddle.save(static_model,'./SFnet_model/best_'+model_name+'_camvid_19cls_mem_0.125.pdparams')
                best_ap = np.mean(mean_ap)/data_sum
            else:
                counter = counter+1
                if(counter>=10):
                    print('lr_step')
                    #model.set_state_dict(paddle.load('SFnet_model/best_'+model_name+'_camvid_19cls_mem.pdparams'))
                    counter = 0
                    lr_cnt = lr_cnt+1
                    if(lr_cnt == 4):
                        print('stop')
                        break
            print('current_miou_cls: ',mean_ap/data_sum,'best_miou: ',best_ap)
            print('current_miou: ',np.mean(mean_ap)/data_sum,'best_miou: ',best_ap)


