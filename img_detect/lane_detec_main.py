import os
import cv2
import time
import numpy as np
import biaozhi_detect as bd
import lane_detec0803 as ldm
import lane_detect_road as ldr
import matplotlib.pyplot as plt

img_path = '5.jpg'
img = cv2.imread(img_path)
print(img_path)

t1 = time.time()
shift, shiline = ldm.line_det_shift(img, 'auto')
t2 = time.time()
img_lane, shiline = ldm.line_det_img(img, 'auto')
print('检测依据车道线：', shiline)
print('检测耗时：', (t2 - t1), 's')
plt.imshow(img_lane)
plt.show()
print('车辆正前方十米处与两侧车道线中间位置的偏移距离为：', shift, '米')

left = np.array([400, 400])
right = np.array([800, 400])
img_odom_person = ldm.odom_person(left, right)  # left和right为原图中深度学习算法检测出来矩形框底边线段两端点的坐标
print('交通标志的位置相对于车辆的位置为：', img_odom_person, ' 单位：米')

inference_pb = "sorted_inference_graph.pb"
graph_txt = "graph.pbtxt"
net = cv2.dnn.readNetFromTensorflow(inference_pb, graph_txt)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
image, series, maxscore = bd.bd_img(img, net)
if maxscore > 0:
    dis, series, maxscore = bd.bd_dis(img, net)
    dis[0] = round(dis[0], 3)
    dis[1] = round(dis[1], 3)
    if series == 1:
        biaozhi = 'right turn'
    if series == 2:
        biaozhi = 'limit'
    txt = "The " + biaozhi + " sign is about " + str(dis) + " meters."
    print(txt)
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.putText(image, txt, (10, 50), font, 0.7, (0, 255, 0), 2)
    plt.imshow(image)
    plt.show()

img_path = '3.jpg'
img = cv2.imread(img_path)
print(img_path)

t1 = time.time()
dis, series = ldr.ldr_shift(img)
t2 = time.time()
print('检测耗时：', (t2 - t1), 's')
if series == 0:
    print("前方没有检测到斑马线或直角弯")
else:
    if series == 1:
        biaozhi = 'zebra crossing'
    if series == 2:
        biaozhi = 'corner'
    txt = "About " + str(round(dis, 5)) + " meters from the front " + biaozhi
    print(txt)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, txt, (10, 50), font, 0.7, (0, 255, 0), 2)
    img_road, series = ldr.ldr_img(img)
    plt.imshow(img_road)
    plt.show()



