# !/usr/bin/env python
# -*- coding:utf-8 -*- 

"""
author: yifeng
date  : 10.04.2022
"""

import cv2
import numpy as np


def stereo_matchSGBM(left_image, right_image, save_path, down_scale=False):
    '''
    Calculate the depthmap use SGBM algorithm
    Parameters
    ----------
    save_path: save directory of the depth map
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
    blockSize = 9  # 3
    paraml = {'minDisparity': 50,
              'numDisparities': 20 * 16,  # 128
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,  # 8
              'P2': 32 * img_channels * blockSize ** 2,  # 32
              'disp12MaxDiff': -1,  # 1 # non-positive vaalue to disable the check
              'preFilterCap': 31,
              'uniquenessRatio': 10,  # 15
              'speckleWindowSize': 0,
              # 100  Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range
              'speckleRange': 2,  # 7
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # cread the SGBM instance
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    # paramr['minDisparity'] = -15*16
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # Filter Parameters
    lmbda = 80000
    # sigma = [0.7, 1.2, 1.5]
    sigma = 1.3
    visual_multiplier = 1.0  # 1.0

    # wsl filter are used to smooth the depthmap
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # calculate the disparity map
    print('computing disparity...')
    displ = left_matcher.compute(left_image, right_image)  # .astype(np.float32)/16
    dispr = right_matcher.compute(right_image, left_image)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, left_image, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    cv2.imshow('Disparity Map', filteredImg)
    cv2.imwrite(save_path + '\\' + 'depthmap.png', filteredImg)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # size = (left_image.shape[1], left_image.shape[0])
    # if down_scale == False:
    #     disparity_left = left_matcher.compute(left_image, right_image)
    #     disparity_right = right_matcher.compute(right_image, left_image)
    #
    # else:
    #     left_image_down = cv2.pyrDown(left_image)
    #     right_image_down = cv2.pyrDown(right_image)
    #     factor = left_image.shape[1] / left_image_down.shape[1]
    #
    #     disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
    #     disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
    #     disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
    #     disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
    #     disparity_left = factor * disparity_left
    #     disparity_right = factor * disparity_right
    #
    # # 真实视差（因为SGBM算法得到的视差是×16的）
    # trueDisp_left = disparity_left.astype(np.float32) / 16.
    # trueDisp_right = disparity_right.astype(np.float32) / 16.
    #
    # return trueDisp_left, trueDisp_right

def stereo_matchBM(left_image, right_image, save_path, down_scale=False):

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

    numDisparities = 15 * 16
    blockSize = 15

    paraml = {'minDisparity' : 0,
              'numDisparities' : 64,
              'blockSize' : 9  # 3
              }

    # cread the SGBM instance
    left_matcher = cv2.StereoBM_create(numDisparities, blockSize)
    paramr = paraml
    # paramr['minDisparity'] = -paraml['numDisparities']
    # paramr['minDisparity'] = -15*16
    # right_matcher = cv2.StereoBM_create(-numDisparities, blockSize)

    # Filter Parameters
    lmbda = 100000
    # sigma = [0.7, 1.2, 1.5]
    sigma = 1.3
    visual_multiplier = 1.0  # 1.0
    #
    # wsl filter are used to smooth the depthmap
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    left_image = cv2.cv.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_image = cv2.cv.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # calculate the disparity map
    print('computing disparity...')
    displ = left_matcher.compute(left_image, right_image)  # .astype(np.float32)/16
    # dispr = right_matcher.compute(right_image, left_image)  # .astype(np.float32)/16
    displ = np.int16(displ)
    # dispr = np.int16(dispr)
    # filteredImg = wls_filter.filter(displ, left_image, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=displ, dst=displ, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U);
    filteredImg = np.uint8(filteredImg)
    cv2.imshow('Disparity Map', filteredImg)
    cv2.imwrite(save_path + '\\' + 'depthmap.png', filteredImg)
    cv2.waitKey()
    cv2.destroyAllWindows()