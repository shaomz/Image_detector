import cv2 as cv
import numpy as np
import lane_detec0803 as ldm

def bd_pre(image, net):
    h, w = image.shape[:2]
    im_tensor = cv.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
    net.setInput(im_tensor)
    cvOut = net.forward()
    maxscore = 0
    left = 0
    top = 0
    right = 0
    bottom = 0
    series = 0
    for detect in cvOut[0, 0, :, :]:
        score = detect[2]
        if score > 0.2:
            # print(score)
            if maxscore < score:
                maxscore = score
                left = detect[3]*w
                top = detect[4]*h
                right = detect[5]*w
                bottom = detect[6]*h
                series = detect[1]
    return left, top, right, bottom, maxscore, series


def bd_img(image, net):
    left, top, right, bottom, maxscore, series = bd_pre(image, net)
    cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 255), 2)
    return image, series, maxscore


def bd_dis(image, net):
    left, top, right, bottom, maxscore, series = bd_pre(image, net)
    zuo = np.array([left, bottom])
    you = np.array([right, bottom])
    dis = ldm.odom_person(zuo, you)
    return dis, series, maxscore

