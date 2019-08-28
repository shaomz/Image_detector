import lane_detec0803 as ldm
import cv2
import numpy as np


def ldr_pre(img):
    img_nkt = ldm.nkt(img)
    img_t = cv2.cvtColor(img_nkt, cv2.COLOR_BGR2HLS)
    img_t = img_t[:, :, 1]
    mask = (img_t[:, 100:300] < 90) & (img_t[:, 100:300] > 0)
    img_t[:, 100:300][mask] = 1
    img_t[:, 100:300][~mask] = 0
    h, w = img_t.shape[:2]
    biaozhi = 0
    bottom = 0
    top = 0
    series = 0
    for i in range(h):
        ii = h - i - 1
        he = np.sum(img_t[ii, 100:300])
        # print(he)
        if (biaozhi == 0) & (he > 47):
            bottom = ii
            biaozhi = 1
            series = 1
        if (biaozhi == 0 | biaozhi == 1) & (he > 85):
            biaozhi = 2
            series = 2
        if (biaozhi == 1) & (he < 47):
            top = ii
            break
        if (biaozhi == 2) & (he < 85):
            top = ii
            break

    if bottom < 270:
        biaozhi = 0

    return h, w, top, bottom, series, biaozhi


def ldr_img(img):
    h, w, top, bottom, series, biaozhi = ldr_pre(img)
    if biaozhi == 1 or biaozhi == 2:
        left = w/3
        right = 2/3*w
        tu = np.array([[left, top], [left, bottom], [right, bottom], [right, top]])
        img_result_plt = ldm.result_plt(tu, img)
        return img_result_plt, series
    return img, 0


def ldr_shift(img):
    h, w, top, bottom, series, biaozhi = ldr_pre(img)
    if biaozhi == 1 or biaozhi == 2:
        cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter, img_size, UNWARPED_SIZE, LANE_WIDTH, mid_point = ldm.l_d()
        dis = (h - bottom) / pixels_per_meter[1]
        return dis, series
    return 0, 0

