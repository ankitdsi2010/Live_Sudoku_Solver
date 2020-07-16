import cv2
import numpy as np

def preprocessImage(img_t):
    # apply Gaussian Blurring to reduce noise
    img_gb = cv2.GaussianBlur(img_t, (1,1), cv2.BORDER_DEFAULT)
    # applying Inverse Binary Threshold
    ret, thresh_inv = cv2.threshold(img_gb, 180, 255, cv2.THRESH_BINARY_INV)
    return thresh_inv

def find_area(img):
    # finds the contours on the board
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # finds the biggest area
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)
    for contour in contours:
        if cv2.contourArea(contour) > max_area:
            cnt = contour
            max_area = cv2.contourArea(contour)
    # draw approximate contour
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    poly_approx = cv2.approxPolyDP(cnt, epsilon, True)
    return poly_approx

def order_points(pts):
    # initialzie a list of coordinates
    rect = np.zeros((4,2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[0]
    rect[2] = pts[2]
    # the top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[3] = pts[3]
    rect[1] = pts[1]
    # return coordinates
    return rect

def four_point_transform(img_app, pts):
    # obtain ordered points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute width
    width_b = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
    width_t = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))
    maxWidth = max(int(width_b), int(width_t))
    # compute height
    height_b = np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))
    height_t = np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))
    maxHeight = max(int(height_b), int(height_t))
    # getting birds eye view of points
    td = np.array([[0,0], [0,maxHeight-1], [maxWidth-1,maxHeight-1], [maxWidth-1,0]], dtype="float32")
    # compute perspective matrix
    per_m = cv2.getPerspectiveTransform(rect, td)
    warped = cv2.warpPerspective(img_app, per_m, (maxWidth,maxHeight))
    # return image
    return warped




