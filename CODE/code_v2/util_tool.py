# -*-encodng: utf-8-*-
'''
@File: util_tool.py
@Contact: 2257925767@qq.com
@Author:wangyu
@Version:

@Desciption:
    env: 

@DateTime: 2021/5/7下午4:18 
'''
import torch
import numpy as np

def cobb_angle_calc(pts):
    pts = np.asarray(pts, np.float32)  # 68 x 2
    num_pts = pts.shape[0]  # number of points, 68
    mid_p = []
    for i in range(0, num_pts, 4):
        pt1 = (pts[i, :] + pts[i + 2, :]) / 2  # 左中点
        pt2 = (pts[i + 1, :] + pts[i + 3, :]) / 2  # 右中点
        mid_p.append(pt1)  #
        mid_p.append(pt2)
    mid_p = np.asarray(mid_p, np.float32)  # 34 x 2
    vec_m = mid_p[1::2, :] - mid_p[0::2, :]  # 17 x 2 右中点-左中点,任意一个关键点的中线的方向向量
    dot_v = np.matmul(vec_m, np.transpose(vec_m))  # 17 x 17   计算任意两个向量的乘积
    mod_v = np.sqrt(np.sum(vec_m ** 2, axis=1))[:, np.newaxis]  # (17,1) mod_v[i]表示第i个方向向量的模长
    mod_v = np.matmul(mod_v, np.transpose(mod_v))  # 17 x 17   #
    cosine_angles = np.clip(dot_v / mod_v, a_min=0., a_max=1.)  # 计算夹角余弦值 (a*b)/|a||b|
    angles = np.arccos(cosine_angles)  # 17 x 17  #利用反余弦，计算角度
    pos1 = np.argmax(angles, axis=1)  # 返回每行最大值的列 index,i行j列表示, 向量i与向量j的乘积,所以这里返回的是第二个向量的index
    maxt = np.amax(angles, axis=1)  # 返回每行的最大值
    pos2 = np.argmax(maxt)  # 返回整个angles的最大值所在的行index
    cobb_angle1 = np.amax(maxt)  # 返回整个angles的最大值

    cobb_angle1 = cobb_angle1 / np.pi * 180  # 转换为弧度制
    line1 = [mid_p[int(pos2 * 2)][0],mid_p[int(pos2 * 2)][1],mid_p[int(pos2*2+1)][0],mid_p[int(pos2*2+1)][1]]
    line2 = [mid_p[int(pos1[pos2]*2)][0],mid_p[int(pos1[pos2]*2)][1],mid_p[int(pos1[pos2]*2+1)][0],mid_p[int(pos1[pos2]*2+1)][1]]

    return line1,line2, cobb_angle1

def two_points_to_json(t1,t2):
    tjson1 = "{"+"\"x1\":"+"\"{:.2f}\",".format(t1[0])+"\"y1\":"+"\"{:.2f}\",".format(t1[1])+"\"x2\":"+"\"{:.2f}\",".format(t1[2])+"\"y2\":"+"\"{:.2f}\"".format(t1[3])+"}"
    tjson2 = "{"+"\"x1\":"+"\"{:.2f}\",".format(t2[0])+"\"y1\":"+"\"{:.2f}\",".format(t2[1])+"\"x2\":"+"\"{:.2f}\",".format(t2[2])+"\"y2\":"+"\"{:.2f}\"".format(t2[3])+"}"
    tjson = tjson1 + "," + tjson2
    res = "\"two_points\":["  + tjson +  "]"
    return res

def more_points_to_json(t):
    res = ""
    for tjson in t:
        res = res + "{"+"\"x1\":"+"\"{:.2f}\",".format(tjson[0])+"\"y1\":"+"\"{:.2f}\",".format(tjson[1])+"\"x2\":"+"\"{:.2f}\",".format(tjson[2])+"\"y2\":"+"\"{:.2f}\"".format(tjson[3])+","+"\"x3\":"+"\"{:.2f}\",".format(tjson[4])+"\"y3\":"+"\"{:.2f}\",".format(tjson[5])+"\"x4\":"+"\"{:.2f}\",".format(tjson[6])+"\"y4\":"+"\"{:.2f}\"".format(tjson[7])+"},"
    res = res.rstrip(',')
    res = "\"points\":["+res+"]"
    return res

def cobb_to_json(c):
    res = "\"cobb_value\":"  +   "\"{:.4f}\"".format(c)
    return res

def conform(ans_status,f1=None,f2=None,f3=None): #f1 means 17 bones, f2 means 2 bones, f3 means cobb_value
    res = "{\"status\":\""+ans_status+"\","
    if f1!=None:
        res = res + f1 +","
    if f2!=None:
        res = res + f2 +","
    if f3!=None:
        res = res + f3 +","
    res = res.rstrip(',')
    return res + "}"


import numpy as np


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

