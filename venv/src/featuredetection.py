# !/usr/bin/env python
# -*- coding:utf-8 -*- 

"""
author: yifeng
date  : 13.04.2022
"""
import cv2
import numpy as np
import open3d as o3d
from src.charuco_points import cha_l, cha_r

feature_params = {'maxCorners': 1000,
                  'quality_level': 0.3,
                  'minDistance': 7,
                  'blockSize': 7}

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def feature_detect(imgl, imgr, repro_l, repro_r, disparity, Q):

    imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB)
    imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
    rows, cols = imgl.shape[0:2]
    grayl = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
    grayr = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)

    # find the features in the left image
    dst1 = cv2.goodFeaturesToTrack(gray1, **feature_params)
    # convert the coordinate of the features to pixel and draw
    dst1 = np.int0(dst1)
    for i in dst1:
        x, y = i.ravel()  # 数组降维成一维数组（inplace的方式）
        cv2.circle(imgl, (x, y), 3, (0, 0, 255), -1)

    cv2.namedWindow('harris', cv2.WINDOW_FREERATIO)
    cv2.imshow('harris', img1)
    cv2.waitKey(0)

    exact_corners = cv2.cornerSubPix(gray1, dst1, (11, 11), (1, 1), criteria=criteria)

    # find the correspond features(points) in the right image
    features, ret, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, exact_corners, None, **lk_params)

    # convert to float
    dst1 = np.float32(dst1)
    features = np.float32(features)

    #
    F, mask = cv2.findFundamentalMat(dst1, features, method=cv2.FM_RANSAC, ransacReprojThreshold=0.9, confidence=0.99)
    ret_sr, H1, H2 = cv2.stereoRectifyUncalibrated(dst1, features, F, (rows, cols))

    K = cv2.getPerspectiveTransform(repro_l, repro_r)

    remap1 = cv2.warpPerspective(grayl, K, (cols, rows), flags=None, borderMode=None, borderValue=None)
    remap2 = cv2.warpPerspective(grayr, K, (cols, rows), flags=None, borderMode=None, borderValue=None)
    draw_line(remap1, remap2)

    cv2.imwrite(path + '\\' + 'result1.png', remap1)
    cv2.imwrite(path + '\\' + 'result2.png', remap2)

    points = cv2.reprojectImageTo3D(disparity, Q)

    np.savetxt(path + '\\' + 'points.txt', points)


def save_txt(points, save_path):
    filename = open(save_path + '\\' + 'points.txt', 'w')
    for k, v in points.items():
        filename.write(k + ':' + str(v))
        filename.write('\n')
    filename.close()


def draw_line(image1, image2):
    '''

    Parameters
    ----------
    image1: rectified left image
    image2: rectified right image

    Returns
    -------
    horizontal stacked image piar with polar lines.
    '''
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (2 * k)), (2 * width, line_interval * (2 * k)), (0, 255, 0),
                 thickness=2, lineType=cv2.LINE_AA)
        cv2.line(output, (0, line_interval * (2 * k + 1)), (2 * width, line_interval * (2 * k + 1)), (0, 0, 255),
                 thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow("withlines", output)
    # cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\withlines.png', output)
    cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\0422\\withlines.png', output)
    cv2.waitKey(8000)


def stereo_matchSGBM(left_image, right_image, down_scale=False):
    '''
    Calculate the depthmap use SGBM algorithm
    Parameters
    ----------
    left_image: the rectified left image
    right_image: the rectified right image
    down_scale:

    Returns
    -------
    Depthmap
    '''
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    # blockSize = [3, 5, 7, 15]
    # uniquenessRatui = [5, 10, 15]
    #
    blockSize = 3  # 3
    paraml = {'minDisparity': -64,
              'numDisparities': 10 * 16,  # 128
              'blockSize': blockSize,
              # 'P1': 8 * img_channels * blockSize ** 2,  # 8
              'P1': 100,
              # 'P2': 32 * img_channels * blockSize ** 2,  # 32
              'P2': 1000,
              'disp12MaxDiff': 2,  # 1 # non-positive vaalue to disable the check
              'preFilterCap': 31,
              'uniquenessRatio': 15,  # 15 don't set too large.
              'speckleWindowSize': 100,
              # 100  Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range
              'speckleRange': 2,  # 7
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # cread the SGBM instance
    left_matcher = cv2.StereoSGBM_create(**paraml)
    # paramr = paraml
    # paramr['minDisparity'] = -paraml['numDisparities']
    # paramr['minDisparity'] = -15*16
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Filter Parameters
    lmbda = 80000
    # sigma = [0.7, 1.2, 1.5]
    sigma = 1.2
    visual_multiplier = 1.0  # 1.0

    # wsl filter are used to smooth the depthmap
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # calculate the disparity map
    print('computing disparity...')
    displ = left_matcher.compute(left_image, right_image)  #.astype(np.float32)/16
    dispr = right_matcher.compute(right_image, left_image)  #.astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, left_image, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    cv2.imshow('Disparity Map', filteredImg)
    cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\0422\\depthmap.png', filteredImg)
    cv2.waitKey()
    cv2.destroyAllWindows()



img1 = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\22-04-21_15-07-572_1.png', -1)
# img1 = np.rot90(img1)
img2 = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\22-04-21_15-07-571_1.png', -1)
# img2 = np.rot90(img2)
rows, cols, channels = img1.shape
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

dst1 = cv2.goodFeaturesToTrack(gray1, 500, 0.01, 50, 7)

exact_corners = cv2.cornerSubPix(gray1, dst1, (11, 11), (1, 1),
                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
lk_params = {'winSize': (15, 15),
             'maxLevel': 9,
             'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03)}

features, ret, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, dst1, None, **lk_params)

# bb is the points can't find in image 2. There points will be deleted before stereoRectifyUncalibrated
bb = np.where(ret == 0)
print(bb[0])

dst1_new = np.delete(dst1, bb, axis=0)
print('points:', dst1_new.shape[0])
features_new = np.delete(features, bb, axis=0)
print('targets:', features_new.shape[0])
dst1_new = np.int0(dst1_new)  # 实际上也是np.int64
for i in dst1_new:
    x, y = i.ravel()  # 数组降维成一维数组（inplace的方式）
    cv2.circle(img1, (x, y), 7, (0, 0, 255), -1)

dst1_new = np.float32(dst1_new)  # 实际上也是np.int64
print()

cv2.namedWindow('harris', cv2.WINDOW_FREERATIO)
cv2.imshow('harris', img1)
cv2.waitKey(0)

features_new = np.int0(features_new)
for i in features_new:
    x, y = i.ravel()  # 数组降维成一维数组（inplace的方式）
    cv2.circle(img2, (x, y), 7, (0, 255, 0), -1)
features_new = np.float32(features_new)

cv2.namedWindow('harris2', cv2.WINDOW_FREERATIO)
cv2.imshow('harris2', img2)
cv2.waitKey(0)

print(dst1.shape[0])

# eg
F, mask = cv2.findFundamentalMat(dst1_new, features_new, method=cv2.FM_RANSAC, ransacReprojThreshold=0.9, confidence=0.99)
ret_sr, H1, H2 = cv2.stereoRectifyUncalibrated(dst1_new, features_new, F, (cols, rows))
print('H1', H1)
print('H2', H2)

# K = cv2.getPerspectiveTransform(get1, get2)

remap1 = cv2.warpPerspective(gray1, H1, (cols, rows), flags=None, borderMode=None, borderValue=None)
remap2 = cv2.warpPerspective(gray2, H2, (cols, rows), flags=None, borderMode=None, borderValue=None)

cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\0422\\result1.png', remap1)
cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\0422\\result2.png', remap2)

imgl = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\result1.png')
imgr = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\result2.png')
draw_line(imgl, imgr)

stereo_matchSGBM(imgl, imgr)

# Q = np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.32285745e+03],
#        [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.91700760e+03],
#        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 6.52395231e+03],
#        [ 0.00000000e+00,  0.00000000e+00, -6.68127053e-01, 1.14463130e+03]])

disparity = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\depthmap.png', 0)
# points = cv2.reprojectImageTo3D(disparity, Q)



