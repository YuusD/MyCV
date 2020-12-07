'''
主要的机器视觉模块，实现：识别、跟踪、立体重建、世界坐标输出
'''
from __future__ import print_function

from numba import jit

import os
import cv2
from apriltag import Apriltag
import numpy as np
import time
import queue
import threading
import pickle
import tagUtils as tud
import cameraconfig
import stereomoulds
from sort import *

ql=queue.Queue()
qr=queue.Queue()

def Receive():
    '''
    读取视频图像
    '''
    print("Start Reveive")
    #cap = cv2.VideoCapture("rtsp://admin:admin_123@172.0.0.0") #网络摄像头(海康、大华等)读取
    capl = cv2.VideoCapture(0) #本地默认摄像头
    capr = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("D:/") #读取视频文件
    retl, framel = capl.read()
    retr, framer = capr.read()
    ql.put(framel)
    qr.put(framer)
    while retl:
        retl, framel = calp.read()
        retr, framer = capr.read()
        ql.put(framel)
        qr.put(framer)
 
 
def DisplayandSolve():
     print("Start Displaying and Solving")
     #创建识别器
     detector = Apriltag()
     detector.create_detector(debug=False)
     #创建跟踪器
     mot_tracker = Sort()
     #读取相机内外参
     camerasconfig=cameraconfig.stereoCamera()

     while True:
         if q.empty() !=True:
            framel=ql.get()
            framer=qr.get()
            #读取照片参数
            height, width = framel.shape[0:2]
            #以下开始处理图片

            #立体校正
            map1x, map1y, map2x, map2y, Q= stereomoulds.getRectifyTransform(height, width, camerasconfig)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
            iml_rectified, imr_rectified = stereomoulds.rectifyImage(framel, framer, map1x, map1y, map2x, map2y)

            detections=detector.detect(framel)
            objtracks=list()
            show=None
            if (len(detections) == 0):
                show = frame
            else:
                show = frame
                edges = np.array([[0, 1],
                                  [1, 2],
                                  [2, 3],
                                 [3, 0]])
                for detection in detections:
                    point = tud.get_pose_point(detection.homography)
                    dis = tud.get_distance(detection.homography,122274)
                    obj=[point[0][0],point[0][1],point[2][0],point[2][1],detections.index(detection)]
                    objtracks.append(obj)
                    for j in range(4):
                         cv2.line(show,tuple(point[edges[j,0]]),tuple(point[edges[j,1]]),(0,0,255),2)
                    #print ('dis:' , dis)
            
            cv2.imshow("frame1", show)
            #跟踪更新box
            track_bbs_ids = mot_tracker.update(objtracks)

           
            #立体校正
            map1x, map1y, map2x, map2y, Q= stereomoulds.getRectifyTransform(height, width, camerasconfig)  # 获取畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
            iml_rectified, imr_rectified = stereomoulds.rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
            #立体匹配
            iml_, imr_ = stereomoulds.preprocess(iml_rectified, imr_rectified)
            disp, _ = stereomoulds.disparity_SGBM(iml_, imr_)
            disp = np.divide(disp.astype(np.float32), 16.)  # 除以16得到真实视差（因为SGBM算法得到的视差是×16的）
            # 计算像素点的3D坐标（左相机坐标系下）
            points_3d = cv2.reprojectImageTo3D(disp, Q)
           

            #得到目标中心的3D坐标（左相机坐标系下）
            points_3d=stereomoulds.stereoto3d(track_bbs_idsl, track_bbs_idsr, Q)

            #记录坐标点轨迹到trajectory.txt文件
            file=open("trajectory.txt","a")
            #trajt=file.write(point_3d)
            pickle.dump(point_3d,file)
            file.close()
         if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
     
if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=DisplayandSolve)
    p1.start()
    p2.start()