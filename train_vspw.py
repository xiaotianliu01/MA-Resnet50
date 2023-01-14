from dataset.cityscapes import Cityscapes
from models.deeplabv3 import DeepLabV3P
from models.pspnet import PSPNet
from models.sfnet import SFNet
from models.resnet_vd import ResNet50_vd
from utils.cal_miou import cal_miou
import paddle
import yaml
import numpy as np
import time
def val_prog(model, loader_val, model_name, best_ap):
    cnt = 0
    with paddle.no_grad():
        print('start val...')
        s = time.time()
        mean_ap = 0
        miou = np.zeros(2)
        cls_count = np.zeros(2)
        r_h = 640
        r_w = 640
        for step_id, data in enumerate(loader_val):
            im = data[0]
            if len(im.shape)<4:
                out = model(im,True)
                cnt = cnt + 1
                continue
            else:
                im = paddle.nn.functional.interpolate(im, (r_h, r_w))
                out = model(im,False)
            if out is None:
                continue
            label = data[1].astype('int64')
            label = paddle.nn.functional.interpolate(label, (r_h, r_w))
            pred = paddle.argmax(out[0], axis=1, keepdim=True, dtype='int32')
            iou, cls_count = cal_miou(pred, label, 2, cls_count, 2)
            miou = miou + iou
            if(int(time.time()-s)%100 == 0):
                print(step_id/len(loader_val)*100,'%: miou:', np.mean(miou/cls_count))
        print('miou_cls : ',miou/cls_count)
        mean_ap = np.mean(miou/cls_count)
        if mean_ap > best_ap:
            static_model=model.state_dict()
            paddle.save(static_model,'./model/best_'+model_name+'_vspw_125cls_mem.pdparams')
            print('current_miou: ',mean_ap,'best_miou: ',best_ap)
            return True, mean_ap
        else:
            print('current_miou: ',mean_ap,'best_miou: ',best_ap)
            return False, best_ap

if __name__ == '__main__':
    model_name = 'PSPNet'
    yml_path="work/dataset/cityscapes.yml"
    f = open(yml_path)
    config = yaml.load(f,Loader=yaml.FullLoader)
    dataset = Cityscapes(config['train_dataset']['transforms'],'data/camvid','train', 'VSPW')
    loader = paddle.io.DataLoader(dataset,
                    use_shared_memory=False,
                    batch_size=2,
                    shuffle=False,
                    drop_last=False,
                    num_workers=2,
                    return_list=True)
    dataset_val = Cityscapes(config['val_dataset']['transforms'],'data/camvid','val', 'VSPW')
    loader_val = paddle.io.DataLoader(dataset_val,
                    use_shared_memory=False,
                    batch_size=2,
                    shuffle=False,
                    drop_last=False,
                    num_workers=2,
                    return_list=True)
    ce_loss = paddle.nn.CrossEntropyLoss(ignore_index= 255, reduction='mean', axis=1)
    backbone = ResNet50_vd()
    if(model_name == 'PSPNet'):
        model = PSPNet(2,backbone)
    elif(model_name == 'DeepLabv3P'):
        model = DeepLabV3P(2,backbone)
    elif(model_name == 'SFNet'):
        model = SFNet(2,backbone)
    model.set_state_dict(paddle.load('model/PSPNet_vspw_2cls_mem.pdparams'))
    epoch = 100
    loss_sum = 0
    cal = 40
    iter_num = 54026
    lr = [0.1,0.05,0.01]
    lr_cnt = 0
    counter = 0
    best_ap = 0.7720332977507468
    #sche = paddle.optimizer.lr.PolynomialDecay(0.0005, 10, 0, 0.9)
    #sche = paddle.optimizer.lr.PiecewiseDecay(boundaries=[5,7,9], values=[0.1,0.02,0.005, 0.001], verbose=False)
    optimizer = paddle.optimizer.SGD(learning_rate=lr[lr_cnt], parameters=model.parameters(), weight_decay=4.0e-5)
    for i in range(epoch):
        model.train()
        print('start train...')
        clip_num = 0
        clip_id = 0
        for step_id, data in enumerate(loader):
            if(step_id<17760):
                continue
            im = data[0]
            label = data[1].astype('int64')
            if len(im.shape)<4:
                clip_num = 0
                clip_id = clip_id + 1
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
            iter_num = iter_num + 1
            optimizer.clear_grad()
            clip_num = clip_num + 1
            if((step_id+1)%cal == 0):
                print("epoch[{}],iter_num:[{}],clip_id:[{}/{}],[{}/{}],loss:{}".format(i,iter_num,clip_id,2805,step_id+1,int(len(loader)),loss_sum/cal))
                loss_sum = 0
            if((iter_num+1)%18000 == 0):
                static_model=model.state_dict()
                paddle.save(static_model,'./model/'+model_name+'_vspw_2cls_mem.pdparams')
                flag, ap = val_prog(model,loader_val,model_name, best_ap)
                if(flag):
                    counter = 0
                    best_ap = ap
                else:
                    counter = counter+1
                    if(counter == 3):
                        break
                        print('lr_step')
                        lr_cnt = lr_cnt+1
                        if(lr_cnt == 3):
                            break
                        else:
                            optimizer = paddle.optimizer.SGD(learning_rate=lr[lr_cnt], parameters=model.parameters(), weight_decay=4.0e-5)
                            counter = 0
        if(counter == 3):
            break
