import cv2
import numpy as np
import paddle
import paddle.fluid.dygraph as dg
def opflow_warp(img_last,op_flow,test):

    B, C, H, W = img_last.shape
    xx = paddle.arange(0, W)
    yy = paddle.arange(0, H)
    xx, yy = paddle.meshgrid(yy,xx)
    xx = paddle.reshape(xx, [1, 1, H, W])
    yy = paddle.reshape(yy, [1, 1, H, W])
    grid = paddle.concat((yy,xx),1)
    grid = paddle.tile(grid,[B,1,1,1]).astype('float')
    vgrid = grid - op_flow
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0
    vgrid = paddle.transpose(vgrid, [0,2,3,1])  
    output = paddle.nn.functional.grid_sample(img_last.astype('float'), vgrid)
    mask = paddle.ones(img_last.shape).astype('float')
    mask = paddle.nn.functional.grid_sample(mask, vgrid)
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    if(test):
        res = paddle.transpose(output,[0,2,3,1])
        res = res.numpy().reshape((H,W,3))
        cv2.imwrite('work/utils/res.png',res)
        op_flow1 = op_flow.numpy()
        mag, ang = cv2.cartToPolar(op_flow1[0][0][:][:], op_flow1[0][1][:][:])
        hsv = np.zeros((mag.shape[0],mag.shape[1],3))
        hsv[...,1] = 255
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        cv2.imwrite('work/utils/rgb.png',rgb)
    '''
    res = paddle.transpose(output,[0,2,3,1])
    res = res.numpy().reshape((H,W,3))
    mag, ang = cv2.cartToPolar(op_flow[...,0], op_flow[...,1])
    hsv = np.zeros_like(img_cur)
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imwrite('work/utils/res.png',res)
    cv2.imwrite('work/utils/rgb.png',res)
    '''
    return output*mask
'''
img1 = cv2.imread('camvid/images/01TP/0001TP_006690.png')
img2 = cv2.imread('camvid/images/01TP/0001TP_006720.png')
mask = cv2.imread('camvid/labels/01TP/0001TP_006690_P.png')
transmask_opflow(img1, img2, mask)
'''

'''
    res = np.zeros((op_flow.shape[0],op_flow.shape[1]))
    for i in range(op_flow.shape[0]):
        for j in range(op_flow.shape[1]):
            new_x = int(i+op_flow[i][j][1])
            new_y = int(j+op_flow[i][j][0])
            if(new_x>=op_flow.shape[0]):
                new_x = op_flow.shape[0]-1
            if(new_y>=op_flow.shape[1]):
                new_y = op_flow.shape[1]-1
            if(new_x<0):
                new_x = 0
            if(new_y<0):
                new_y = 0
            res[new_x][new_y] = img_last[i][j]
'''
