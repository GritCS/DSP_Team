# -*-encodng: utf-8-*-
'''
@File: Network.py.py
@Contact: 2257925767@qq.com
@Author:wangyu
@Version:

@Desciption:
    env: 

@DateTime: 2021/5/7上午8:37 
'''
import torch
import numpy as np
    

from models import spinal_net
import decoder
import pre_proc
from util_tool import cobb_angle_calc

'''
@:name
    cobb_angle_calc
@:func 
    跟据关键点来计算cobb
@:param
    pts：预测的关键点
@:return
    line1
    line2 
    cobb_angle1: the value of cobb ,type: float
'''

class Network(object):
    def __init__(self):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'hm': 1,
                 'reg': 2,
                 'wh': 2*4,}
        self.model_path ="model_last.pth"
        self.model = spinal_net.SpineNet(heads=heads,
                                         basename='resnet34',
                                         pretrained=True,
                                         down_ratio=4,
                                         final_kernel=1,
                                         head_conv=256)
        self.down_ratio = 4
        self.model_state = False
        self.decoder = decoder.DecDecoder(K=100, conf_thresh=0.2)

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        self.model_state = True 
        return model
        

    def getLandmark(self,cv2_image,input_shape=(512,1024),):  #cv2_image(w,h,3) input_shape=(w,h)
        if self.model_state == False: #如果没有加载模型参数
                self.model = self.load_model(self.model,self.model_path)
                self.model.eval()
        #数据预处理
        images = pre_proc.processing_test(image=cv2_image, input_h=input_shape[1], input_w=input_shape[0])
        #模型预测
        with torch.no_grad():
            output = self.model(images)
            hm = output['hm']
            wh = output['wh']
            reg = output['reg']
        #计算关键点检测
        pts2 = self.decoder.ctdet_decode(hm, wh, reg)  # 17, 11
        pts0 = pts2.copy()
        pts0[:, :10] *= self.down_ratio # cpu().numpy()  shape: 17, 11
        landmark=[]
        for i in range(pts0.shape[0]):
            one_bone = []
            for j in range(2,10):
                one_bone.append(pts0[i][j])
            landmark.append(one_bone)
        return landmark,pts0 

    def getCobb(self,pts0,h,w,input_shape=(512,1024)):  #  h,w,c = ori_image.shape
        x_index = range(0,10,2)   #0,2,4,8
        y_index = range(1,10,2)   #1,3,5,7
        pts0[:, x_index] = pts0[:, x_index]/input_shape[0]*w    #放缩比例，调整到 和 原图大小相同
        pts0[:, y_index] = pts0[:, y_index]/input_shape[1]*h
        sort_ind = np.argsort(pts0[:,1])
        pts0 = pts0[sort_ind]
        pr_landmarks = []
        for i, pt in enumerate(pts0): #一共17组
            pr_landmarks.append(pt[2:4])
            pr_landmarks.append(pt[4:6])
            pr_landmarks.append(pt[6:8])
            pr_landmarks.append(pt[8:10])
        pr_landmarks = np.asarray(pr_landmarks, np.float32)   #[68, 2]
        line1,line2 ,angle = cobb_angle_calc(pr_landmarks)

        return line1,line2,angle




