'''
Created on Mar 3, 2018

@author: chen_pan
'''
import cv2


def set_img_roi(img, rect, img_small):
    """Set image roi

    @param img: Image to set ROI into
    @param rect: Region of the image to set
    @param img_small: The small image to set
    """
    img_resized = cv2.resize(img_small, (rect[2], rect[3]))
    img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]] = img_resized
    return img


def get_img_roi(img, rect):
    """Return image ROI by the rect

    @param img: A image to take ROI
    @param rect: A rect of (x, y, w, h)
    @return: Image ROI
    """
    assert len(img.shape) in [2, 3]

    if len(img.shape) == 2:
        return img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    elif len(img.shape) == 3:
        return img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2], :]


def init_bboxes_v2(img_gray, thresh_lower=128, save_contours=False):
    """Initialize bboxes from all of the connected components in img_gray
    the difference compared with upper method:  it didn't filter the bboxes 
    by the width and height

    @param img_gray: The gray image to initialize bboxes from
    @param thresh_lower: The lower threshold in opencv threshold function
    @return: bboxes, contours
    """
    __, img_inv = cv2.threshold(
        img_gray, thresh_lower, 255, cv2.THRESH_BINARY_INV)
    contours, hierachy = cv2.findContours(
        img_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if contours is None or hierachy is None:
        return [], []

    hierachy = hierachy.reshape(-1, 4)
    bboxes = []
    for i, contour in enumerate(contours):
        __, __, child, parent = hierachy[i]
        if parent == -1:
            bbox = {'rect': cv2.boundingRect(contour)}

            if save_contours:
                bbox['contours'] = [contour]
                while child != -1:
                    contour_child = contours[child]
                    bbox['contours'].append(contour_child)
                    child = hierachy[child][0]

            bboxes.append(bbox)

    return bboxes, contours


def calc_extra_info_bboxes(bboxes):
    """Calcuate extra information for bboxes
    left, right, top, bot, w, h, x_center, y_center of bbox are set
    Note the bboxes are changed

    @param bboxes: The bboxes
    """
    for bbox in bboxes:
        bbox['left'] = bbox['rect'][0]
        bbox['right'] = bbox['rect'][0] + bbox['rect'][2]
        bbox['top'] = bbox['rect'][1]
        bbox['bot'] = bbox['rect'][1] + bbox['rect'][3]
        bbox['w'] = bbox['rect'][2]
        bbox['h'] = bbox['rect'][3]
        bbox['x_center'] = (bbox['left'] + bbox['right']) / 2
        bbox['y_center'] = (bbox['top'] + bbox['bot']) / 2
