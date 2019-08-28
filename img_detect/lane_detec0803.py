# -*- coding: utf-8 -*-
import cv2
import numpy as np


def l_d():
    cam_matrix = np.matrix([[290.24832514, 0., 313.01588114],
                            [0., 291.42743857, 200.99920963],
                            [0., 0., 1.]])
    dist_coeffs = np.matrix([[-0.35157169, 0.16879852, -0.00122305, 0.00073571, -0.0448552]])
    perspective_transform = np.matrix([[ 2.83688558e+00, 3.63121352e+00, -6.15968829e+02],
                                       [-5.00531839e-16, 1.87323135e+01, -3.65133995e+03],
                                       [ 8.88178372e-20, 1.81560690e-02,  1.00000000e+00]])
    pixels_per_meter = [714.4128671629536, 3045.7058138029743]
    img_size = [480, 640]
    UNWARPED_SIZE = [500, 400]
    LANE_WIDTH = 0.225
    mid_point = 160.74289511166455
    return cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter, img_size, UNWARPED_SIZE, LANE_WIDTH, mid_point


class LaneLine:
    def __init__(self, fit_poly, warped_size):
        self.fit_poly = fit_poly
        self.warped_size = warped_size
        self.ploty = np.array(warped_size[1], dtype=np.uint8)
        self.fitx = np.array(self.warped_size[0], dtype=np.uint8)
        self.line_points = np.ones((warped_size[0], warped_size[1]), dtype=np.uint8)

    def get_line_points(self, img_t):
        self.ploty = np.linspace(0, img_t.shape[0] - 1, img_t.shape[0])
        self.fitx = self.fit_poly[0] * self.ploty ** 2 + self.fit_poly[1] * self.ploty + self.fit_poly[2]
        self.line_points = np.stack((self.fitx, self.ploty), axis=-1)
        return self.line_points


class LaneDetection:
    def __init__(self, cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter, img_size, warped_size, LANE_WIDTH, midpoint, auto):
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.img_size = img_size
        self.warped_size = warped_size
        self.M = perspective_transform
        self.left_line = LaneLine([0.0, 0.0, 0.0], warped_size)
        self.right_line = LaneLine([0.0, 0.0, 0.0], warped_size)
        self.pixels_per_meter = pixels_per_meter
        self.LANE_WIDTH = LANE_WIDTH
        self.pixels_per_meter = pixels_per_meter
        self.shiline = 'left'
        self.midpoint = midpoint
        self.auto = auto

    def undistort(self, img):
        return cv2.undistort(img, self.cam_matrix, self.dist_coeffs)

    def warp(self, img):
        return cv2.warpPerspective(img, self.M, (self.warped_size[1], self.warped_size[0]))

    def unwarp(self, img):
        imgsize = (self.img_size[1], self.img_size[0])
        return cv2.warpPerspective(img, self.M, imgsize,
                                   flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

    def extract_lanes_pixels(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        midpoint = histogram.shape[0] // 2
        left_point = histogram.shape[0] // 100
        leftx_base = np.argmax(histogram[left_point:midpoint]) + left_point
        right_point = histogram.shape[0] - histogram.shape[0] // 100
        rightx_base = np.argmax(histogram[midpoint:right_point]) + midpoint
        nwindows = 20  # 框最大限制数
        window_height = binary_warped.shape[0] // nwindows
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 30  # 框宽度的一半
        minpix = 50  # 是否转移框的位置
        left_lane_inds = []
        right_lane_inds = []
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        if sum(left_lane_inds) > sum(right_lane_inds):
            shiline = 'left'
        else:
            shiline = 'right'
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        return leftx, lefty, rightx, righty, shiline

    def get_center_shift(self):
        return -(self.left_line.fitx[499] + self.right_line.fitx[499] - self.warped_size[1]) / 2 / \
               self.pixels_per_meter[0]

    def find_lane(self, img):
        img = self.undistort(img)
        img_t = self.warp(img)
        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2HLS)
        img_t = img_t[:, :, 1]
        mask = (img_t < 90) & (img_t > 0)
        img_t[mask] = 255
        img_t[~mask] = 0
        leftx, lefty, rightx, righty, self.shiline = self.extract_lanes_pixels(img_t)
        if self.auto == 'left':
            self.shiline = 'left'
        if self.auto == 'right':
            self.shiline = 'right'
        if self.shiline == 'left':
            self.left_line.fit_poly = np.polyfit(lefty, leftx, 2)
            self.left_line.get_line_points(img_t)
            self.right_line.fitx = self.left_line.fitx + self.midpoint
            self.right_line.ploty = self.left_line.ploty
            self.right_line.line_points = np.stack((self.right_line.fitx, self.right_line.ploty), axis=-1)
        else:
            self.right_line.fit_poly = np.polyfit(righty, rightx, 2)
            self.right_line.get_line_points(img_t)
            self.left_line.fitx = self.right_line.fitx - self.midpoint
            self.left_line.ploty = self.right_line.ploty
            self.left_line.line_points = np.stack((self.left_line.fitx, self.left_line.ploty), axis=-1)

    def draw_lane_weighted(self, img, alpha=1, beta=1, gamma=0):
        both_lines = np.concatenate((self.left_line.line_points, np.flipud(self.right_line.line_points)), axis=0)
        lanes = np.zeros((self.warped_size[0], self.warped_size[1], 3), dtype=np.uint8)
        cv2.polylines(lanes, [self.left_line.line_points.astype(np.int32)], False, (0, 0, 255), thickness=5)
        cv2.polylines(lanes, [self.right_line.line_points.astype(np.int32)], False, (0, 0, 255), thickness=5)
        cv2.fillPoly(lanes, [both_lines.astype(np.int32)], (0, 255, 0))
        img = self.undistort(img)
        lanes_unwarped = self.unwarp(lanes)
        return cv2.addWeighted(img, alpha, lanes_unwarped, beta, gamma)

    def process_shift(self, img):
        self.find_lane(img)
        shift = self.get_center_shift()
        return shift

    def process_img(self, img):
        self.find_lane(img)
        lane_img = self.draw_lane_weighted(img)
        img_lane = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)
        return img_lane


def line_det_img(img, auto):
    cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter, img_size, UNWARPED_SIZE, LANE_WIDTH, midpoint =l_d()
    ld = LaneDetection(cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter, img_size, UNWARPED_SIZE, LANE_WIDTH, midpoint, auto)
    img_lane = ld.process_img(img)
    shiline = ld.shiline
    return img_lane, shiline


def line_det_shift(img, auto):
    cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter, img_size, UNWARPED_SIZE, LANE_WIDTH, midpoint =l_d()
    ld = LaneDetection(cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter, img_size, UNWARPED_SIZE, LANE_WIDTH, midpoint, auto)
    shift = ld.process_shift(img)
    shiline = ld.shiline
    return shift, shiline


def nkt(img):
    cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter, img_size, UNWARPED_SIZE, LANE_WIDTH, midpoint = l_d()
    ld = LaneDetection(cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter, img_size, UNWARPED_SIZE, LANE_WIDTH, midpoint, 'auto')
    img = ld.undistort(img)
    img_t = ld.warp(img)
    return img_t


def odom_person(left, right):
    cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter, img_size, UNWARPED_SIZE, LANE_WIDTH, midpoint = l_d()
    original = np.array([[left, right]], dtype=np.float32)
    converted = cv2.perspectiveTransform(original, perspective_transform)
    a = converted[0]
    b = np.array([UNWARPED_SIZE[1] / 2, UNWARPED_SIZE[0]])
    c = (a[0]+a[1])/2
    d = c-b
    d[0] = d[0]/pixels_per_meter[0]
    d[1] = -d[1]/pixels_per_meter[1]
    return d


def result_plt(tu, img):
    cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter, img_size, UNWARPED_SIZE, LANE_WIDTH, midpoint = l_d()
    roi_points = np.array(tu, dtype=np.int32)
    roi = np.zeros((UNWARPED_SIZE[0], UNWARPED_SIZE[1], 3), dtype=np.uint8)
    cv2.fillPoly(roi, [roi_points], (255, 0, 0))
    ld = LaneDetection(cam_matrix, dist_coeffs, perspective_transform, pixels_per_meter, img_size, UNWARPED_SIZE, LANE_WIDTH, midpoint, 'auto')
    tu_unwarped = ld.unwarp(roi)
    tu_img = cv2.addWeighted(img, 1, tu_unwarped, 1, 0.5)
    img_tu = cv2.cvtColor(tu_img, cv2.COLOR_BGR2RGB)
    return img_tu
