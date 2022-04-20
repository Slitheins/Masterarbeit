# !/usr/bin/env python
# -*- coding:utf-8 -*- 

"""
author: yifeng
date  : 10.04.2022
"""

import cv2
import numpy as np
import glob
import time
import os
import matplotlib.pyplot as plt
import open3d as o3d

class StereoRectify:
    """
    Recitify the left- and right image pair. Draw the polar lines to find the corresponding feature points in the left- and in the right image.
    """

    def __init__(self, camera_model, save_path):

        self.cameraMatrix_l = camera_model['M1']
        self.cameraMatrix_r = camera_model['M2']
        self.dist_l = camera_model['dist1']
        self.dist_r = camera_model['dist2']
        self.R = camera_model['R']
        self.T = camera_model['T']
        self.dims = camera_model['dims']
        self.save_path = save_path
        print('self.dims:', self.dims)
        # 主点列坐标的差
        self.doffs = 0.0


    # 消除畸变
    def undistortion(self, image, camera_matrix, dist_coeff):
        undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)
        return undistortion_image

    @property
    def save_txt(self):
        filename = open(self.save_path + '\\' + 'rectify.txt', 'w')
        for k, v in self.camera_model.items():
            filename.write(k + ':' + str(v))
            filename.write('\n')
        filename.close()

    @property
    def get_rectify_transform(self):
        '''
        Get the rectified camera model. Preparing for the polar line's parallel correction.

        Returns
        -------
        rectify_model: contains the ????
        '''
        # 读取内参和外参

        # 计算校正变换
        # 记得对alpha调参，出来的图像可能是黑色的
        # Q是深度差异映射矩阵
        # flags can be set to 0 or 1024.
        # alpha can be set to -1 or [0, 1].
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.cameraMatrix_l, self.dist_l, self.cameraMatrix_r, self.dist_r,
                                                          (self.dims[1], self.dims[0]), self.R, self.T, flags=0, alpha=0.63
                                                          )
        # R1[0, :] *= -1
        # R2[0, :] *= -1

        # self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.cameraMatrix_r, self.dist_r, R2, P2,
        #                                                      (self.dims[1], self.dims[0]), cv2.CV_32FC1)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.cameraMatrix_l, self.dist_l, R1, P1,
                                                             self.dims, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.cameraMatrix_r, self.dist_r, R2, P2,
                                                             self.dims, cv2.CV_32FC1)

        # Q[3][2] = -Q[3][2]
        # Q[3][3] = 0

        rectify_model = dict([
            ('R1', R1),
            ('R2', R2),
            ('P1', P1),
            ('P2', P2),
            ('Q', Q),
            ('stereo_left_mapx', self.map1x),
            ('stereo_left_mapy', self.map1y),
            ('stereo_right_mapx', self.map2x),
            ('stereo_right_mapy', self.map2y)
        ])
        # print('rectify_model:',rectify_model)
        return rectify_model

    def rectify_image(self, img_l, img_r):
        '''
        rectify the image pair. Save and show the image pair.
        Parameters
        ----------
        grayl: gray image of left camera used for rectification
        grayr:　gray image of right camera used for rectification

        Returns
        -------

        '''
        # 其变矫正和立体矫正
        # 对图片重构，获取用于畸变矫正和立体矫正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        # 也可以用试试这句 rectifiedL = cv2.remap(imgL, left_map_x, left_map_y, cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
        remapped_img1 = cv2.remap(img_l, self.map1x, self.map1y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        remapped_img2 = cv2.remap(img_r, self.map2x, self.map2y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rectified_img1 = self.undistortion(remapped_img1, self.cameraMatrix_l, self.dist_l)
        rectified_img2 = self.undistortion(remapped_img2, self.cameraMatrix_r, self.dist_r)
        # rect_img1 = cv2.resize(rectified_img1, (1024, 768))
        cv2.imshow('rect_img1', rectified_img1)
        cv2.waitKey(3000)
        # rect_img2 = cv2.resize(rectified_img2, (1024, 768))
        cv2.imshow('rect_img2', rectified_img2)
        cv2.waitKey(3000)
        # cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\result1.png',
        #             rectified_img1)
        # cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\result2.png',
        #             rectified_img2)
        cv2.imwrite(self.save_path + '\\' + 'result1.png', rectified_img1)
        cv2.imwrite(self.save_path + '\\' + 'result2.png', rectified_img2)
        result = np.concatenate((rectified_img1, rectified_img2), axis=1)
        resize = cv2.resize(result, (1024, 384))
        # cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\result3.png',
        #             result)
        cv2.imwrite(self.save_path + '\\' + 'withoutlines.png', result)
        cv2.imshow("rec.png", resize)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()


    def draw_line(self, image1, image2):
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
            cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0),
                     thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("withlines", output)
        # cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\withlines.png', output)
        withline_save = self.save_path + '\\' + 'withlines.png'
        cv2.imwrite(withline_save, output)
        cv2.waitKey(8000)
        cv2.destroyAllWindows()