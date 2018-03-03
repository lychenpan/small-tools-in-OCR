# coding=utf-8
'''
Created on Mar 3, 2018
@author: chen_pan
One method to remove long vertical and horizontal line using morphological operations.
Consider the situation that lines may have overlap with the characters
some reference: https://stackoverflow.com/questions/33949831/whats-the-way-to-remove-all-lines-and-borders-in-imagekeep-texts-programmatic?rq=1
'''
from util import tools
import cv2
import numpy as np


def detect_long_hlines(img, w_est, h_est, debug=0):
    """Detect long horizontal lines and return the mask
    """
    img1 = img.copy()
    img1[img > 200] = 255
    img1[img <= 200] = 0

    kernel = np.ones((3, 1), np.uint8)
    img2 = cv2.erode(img1, kernel, iterations=1)
    # erode will increase the black region

    # Remove the characters;
    kernel = np.ones((1, int(w_est * 1.5)), np.uint8)
    img3 = cv2.dilate(img2, kernel, iterations=1)

    # extract long lines according to its bounding sizes
    bboxes, __ = tools.init_bboxes_v2(img3, save_contours=True)
    tools.calc_extra_info_bboxes(bboxes)
    contours = []
    for bbox in bboxes:
        if (bbox['w'] >= w_est * 3 and bbox['w'] / bbox['h'] > 10):
            contours.extend(bbox.get('contours', []))

    # more process with the long lines
    img_mask = np.zeros((img3.shape[0], img3.shape[1]), np.uint8)
    img_mask.fill(0)  # all black
    cv2.drawContours(img_mask, contours, -1, 255, thickness=-1)
    # it will fill the area bounded by the contours when thickness<0

    # The opposite of 'remove the characters'
    img_mask = cv2.dilate(img_mask, kernel, iterations=1)

    # img_coarse represent the coarse version of lines
    # it is the area got from morphological operations, not actual pixles
    # img_fine represent the fine version of lines;
    # after the bitwise with real binary image, the region will become real
    # pixels
    img_coarse = 255 - img_mask
    img_fine = cv2.bitwise_or(img1, img_coarse)

    bboxes, __ = tools.init_bboxes_v2(img_coarse)
    img_fine01 = img_fine.copy()
    img_fine01[img_fine == 0] = 1
    img_fine01[img_fine == 255] = 0
    img_coarse01 = img_coarse.copy()
    img_coarse01[img_coarse == 0] = 1
    img_coarse01[img_coarse == 255] = 0

    for bbox in bboxes:
        img_fine01_roi = tools.get_img_roi(img_fine01, bbox['rect'])
        img_coarse01_roi = tools.get_img_roi(img_coarse01, bbox['rect'])
        img_fine_sum = np.sum(img_fine01_roi, axis=0)
        img_coarse_sum = np.sum(img_coarse01_roi, axis=0)

        # for a pixel in the line, if its upper and lower pixels are all black
        # we can regard its a overlap with characters
        img_diff = img_coarse_sum - img_fine_sum
        img_diff_mask = np.empty(img_fine01_roi.shape, np.uint8)
        img_diff_mask.fill(255)
        img_diff_mask[:, img_diff <= 0] = 0
        mask_roi = tools.get_img_roi(img_mask, bbox['rect'])
        mask_roi = cv2.bitwise_and(mask_roi, img_diff_mask)
        tools.set_img_roi(img_mask, bbox['rect'], mask_roi)

    return img_mask


def detect_long_vlines(img, w_est, h_est):
    """Detect long vertical lines and return the mask;
       in vertical direction, we didn't consider the overlap situation
    """
    img1 = img.copy()
    img1[img > 200] = 255
    img1[img <= 200] = 0

    kernel = np.ones((1, 3), np.uint8)
    img2 = cv2.erode(img1, kernel, iterations=1)
    kernel = np.ones((int(h_est * 1.5), 1), np.uint8)
    img3 = cv2.dilate(img2, kernel, iterations=1)

    bboxes, __ = tools.init_bboxes_v2(img3, save_contours=True)
    tools.calc_extra_info_bboxes(bboxes)
    contours = []
    for bbox in bboxes:
        if (bbox['h'] >= h_est * 3 and bbox['h'] / bbox['w'] > 8):
            contours.extend(bbox.get('contours', []))

    img_mask = np.zeros((img3.shape[0], img3.shape[1]), np.uint8)
    img_mask.fill(0)
    cv2.drawContours(img_mask, contours, -1, 255, thickness=-1)
    img_mask = cv2.dilate(img_mask, kernel, iterations=1)

    return img_mask


def remove_long_lines(img_gray, w_est, h_est):
    img_mask1 = detect_long_hlines(img_gray, w_est, h_est, debug=0)
    img_mask2 = detect_long_vlines(img_gray, w_est, h_est)
    img_mask = cv2.bitwise_or(img_mask1, img_mask2)
    img_gray = cv2.bitwise_or(img_gray, img_mask)
    return img_gray

if __name__ == '__main__':
    import sys
    img = cv2.imread(
        sys.path[0] + "/line-test.jpg", cv2.IMREAD_GRAYSCALE)
    # the characters size of test image is (27,20)
    img_res = remove_long_lines(img, 20, 27)
    cv2.imwrite(sys.path[0] + "/test_res.jpg", img_res)
    pass
