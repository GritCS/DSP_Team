# -*-encodng: utf-8-*-
'''
@File: main.py
@Contact: 2257925767@qq.com
@Author:wangyu
@Version:

@Desciption:
    env: 

@DateTime: 2021/5/7下午2:26 
'''
from util_tool import more_points_to_json, two_points_to_json, cobb_to_json, conform
import cv2
import numpy
import Network
import time  # 引入time模块

colors = [[0.76590096, 0.0266074, 0.9806378],
          [0.54197179, 0.81682527, 0.95081629],
          [0.0799733, 0.79737015, 0.15173816],
          [0.93240442, 0.8993321, 0.09901344],
          [0.73130136, 0.05366301, 0.98405681],
          [0.01664966, 0.16387004, 0.94158259],
          [0.54197179, 0.81682527, 0.45081629],
          # [0.92074915, 0.09919099 ,0.97590748],
          [0.83445145, 0.97921679, 0.12250426],
          [0.7300924, 0.23253621, 0.29764521],
          [0.3856775, 0.94859286, 0.9910683],  # 10
          [0.45762137, 0.03766411, 0.98755338],
          [0.99496697, 0.09113071, 0.83322314],
          [0.96478873, 0.0233309, 0.13149931],
          [0.33240442, 0.9993321, 0.59901344],
          # [0.77690519,0.81783954,0.56220024],
          # [0.93240442, 0.8993321, 0.09901344],
          [0.95815068, 0.88436046, 0.55782268],
          [0.03728425, 0.0618827, 0.88641827],
          [0.05281129, 0.89572238, 0.08913828],
          ]

'''
接口函数
@:param:
    cv2_image: 要预测图的图像 type：cv2 object
    lm : 为True 结果需要返回脊柱上所有的关键点
    bone:为True 结果返回组成cobb角的两个脊柱的坐标
    cobb:为True 结果返回cobb角预测的值
@:return
    json 串
'''
def auto_cal_cobb(is_object,cv2_image, lm= True , bone = True , cobb=True):
    #检查model是否已经加载了
    assert is_object!=None
    ans_status = "fail"
    #检查图片是否正确加载
    assert isinstance(cv2_image,numpy.ndarray)
    h,w,c = cv2_image.shape
    landmark, pts0 = is_object.getLandmark(cv2_image)
    ans1 = None
    ans2 = None
    ans3 = None
    if lm:
        ans_status = "success"
        ans1 = more_points_to_json(landmark)
    if bone or cobb :
        ans_status = "success"
        line1, line2, cobb_angle = is_object.getCobb(pts0, h, w)
        if bone:
            ans2 = two_points_to_json(line1,line2)
        if cobb:
            ans3 = cobb_to_json(cobb_angle)
    ans = conform(ans_status,ans1,ans2,ans3)
    return ans

if __name__ == '__main__':
    #1. load the model
    model = Network.Network()

    #2.load the test image
    cv2_image = cv2.imread("sunhl-1th-01-Mar-2017-311 E AP.jpg")

    # ==================landmark detection=============
    landmark, pts0 = model.getLandmark(cv2_image)
    h, w, c = cv2_image.shape

    # ===================calculate the cob angle=====
    line1, line2, cobb_angle = model.getCobb(pts0, h, w)


    for i, pt in enumerate(pts0):
        color = colors[i]
        color_255 = (255 * color[0], 255 * color[1], 255 * color[2])

        cv2.circle(cv2_image, (int(pt[2]), int(pt[3])), 10, color_255, -1, 1)
        cv2.circle(cv2_image, (int(pt[4]), int(pt[5])), 10, color_255, -1, 1)
        cv2.circle(cv2_image, (int(pt[6]), int(pt[7])), 10, color_255, -1, 1)
        cv2.circle(cv2_image, (int(pt[8]), int(pt[9])), 10, color_255, -1, 1)
    cv2_bone_image = cv2_image.copy()
    cv2.line(cv2_bone_image,  #
             (int(line1[0]), int(line1[1])),
             (int(line1[2]), int(line1[3])),
             color=(0, 255, 0), thickness=5, lineType=2)
    cv2.line(cv2_bone_image,  #
             (int(line2[0]), int(line2[1])),
             (int(line2[2]), int(line2[3])),
             color=(0, 255, 0), thickness=5, lineType=2)
    #save_the_result_of_cobb
    cv2.imwrite("landmark_test.jpg", cv2_image)
    #save_the_result_of_bone
    cv2.imwrite("bone_test.jpg", cv2_bone_image)

