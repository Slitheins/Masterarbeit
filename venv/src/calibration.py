# !/usr/bin/env python
# -*- coding:utf-8 -*- 

"""
author: yifeng
date  : 23.02.2022
"""
# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
author: yifeng
date  : 16.02.2022
"""
import cv2
import numpy as np
import glob
import time
import os
from operator import attrgetter
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d


devices = list(i + 1 for i in range(6))


class Camera:
    """
    Creat folders for image and result steorage. Each folder stores the images of every camera pair.
    The result(include intrinsic and extrinsic parameters) and the rectified images will be saved here.
    """

    def __init__(self, device_num):
        '''
        Parameters
        ----------
        device_num: camera pair. 6 is the maximal number.
        '''
        self._id = device_num

    @property
    def get_id(self):
        '''

        Returns
        -------
        The ID of the camera pair
        '''
        return self._id

    @property
    def get_dir(self):
        '''
        Get the current and upper directories. Creat an 'image' folder to store calibration photos. Creat subfolders to store the images.
        The structure of the folder is like this.

        Returns
        path: the path of the folder which stores all the images of the camera pair
        path_resize_l: the path of the subfolder which stores all the images taken by the left camera
        path_resize_r: the path of the subfolder which stores all the images taken by the right camera
        -------

        '''
        # get the current direcotry
        cur_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        print(cur_dir)

        device = self.get_id
        print('--------------------------------------------------------------')
        print("preparing for the calibration of device #{}:".format(device))
        print('--------------------------------------------------------------')
        path = os.path.join(cur_dir, 'images', 'device' + str(device))
        path_resize_l = os.path.join(cur_dir, 'images', 'device' + str(device), 'result_l')
        path_resize_r = os.path.join(cur_dir, 'images', 'device' + str(device), 'result_r')
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path_resize_l):
            os.makedirs(path_resize_l)
        if not os.path.exists(path_resize_r):
            os.makedirs(path_resize_r)
        if os.path.isdir(path):
            print("The operating directory is:", path)
            print("Please make sure all photos taken by device #{} are stored here".format(device))
        if os.path.isdir(path_resize_l):
            print("The calibration photos of left camera will be saved:", path_resize_l)
        if os.path.isdir(path_resize_r):
            print("The calibration photos of right camera will be saved:", path_resize_r)
        return path, path_resize_l, path_resize_r


#############################################
class PhotoSet:
    """

    """
    img_suffix = ['png', 'jpg', 'jpeg']

    def __init__(self, path, save_dir_l, save_dir_r):
        '''

        Parameters
        ----------
        path: the path of the folder which stores all the images of the camera pair
        save_dir_l: the path of the subfolder which stores all the images taken by the left camera
        save_dir_r: the path of the subfolder which stores all the images taken by the right camera
        '''
        self._path = path
        self.left = 1
        # 找到path下的所有images
        self._save_dir_l = save_dir_l
        self._save_dir_r = save_dir_r
        self._images_l = []
        self._images_r = []

    def resize(self, img):
        '''
        Used to resize the original image.
        It is better not to resize the original images. Otherwise the result will be generated with large errors.
        Parameters
        ----------
        img: the image need to be resized.

        Returns
        -------
        '''
        img_resize = cv2.resize(img, (1024, 768))
        return img_resize

    @property
    def image_to_matrix(self, exist_dict = False):
        '''
        All the images taken by the same camera will be read as "np.array" format and stored together as a python "list".
        Returns
        -------

        '''
        if exist_dict == False:
            # image_list_l = sorted(glob.glob(self._path + '//' + '*_1.jpg'))
            image_list_l = sorted(glob.glob(self._path + '//' + '*2.png'))
            for i, image in enumerate(image_list_l):
                write_name = self._save_dir_l + '//' + 'Image_' + str(i) + '.png'
                img = cv2.imread(image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img = np.rot90(img, 3)
                # img = self.resize(img)
                # print('gray ', gray )
                cv2.imwrite(write_name, img)
                # 放到一个数组里
                self._images_l.append(img)
            print("{} photos in this directory are read".format(len(self._images_l)))

            # image_list_r = sorted(glob.glob(self._path + '//' + '*_2.jpg'))
            image_list_r = sorted(glob.glob(self._path + '//' + '*1.png'))
            for j, image in enumerate(image_list_r):
                write_name = self._save_dir_r + '//' + 'Image_' + str(j) + '.png'
                img = cv2.imread(image)
                # img = np.rot90(img, 3)
                # img = self.resize(img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img = np.uint8(np.clip((cv2.add(1 * gray, 40)), 0, 255))
                # img = self.resize(img)
                cv2.imwrite(write_name, img)
                # 放到一个数组里
                self._images_r.append(img)
        else:
            img_l = sorted(glob.glob(self._save_dir_l))
            img_r = sorted(glob.glob(self._save_dir_r))
            for i, image in enumerate(img_l):
                img = cv2.imread(image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self._images_l.append(img)
            for j, image in enumerate(img_r):
                img = cv2.imread(image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self._images_r.append(img)

        print("{} photos in this directory are read".format(len(self._images_r)))
        return self._images_l, self._images_r



#############################################

class CameraCalibration:
    """
    Pattern types: pattern = ["square", "circle", "charuco"]
    """

    def __init__(self, pattern_choice, image_list_l, image_list_r, save_path):
        '''
        Initialize the images used for calibration.
        Parameters
        ----------
        pattern_choice: 3 kinds of calibration patterns can be used: square patter, circle pattern and charuco pattern.
        image_list_l: python list include all the images as np.array which taken by the left camera
        image_list_r: python list include all the images as np.array which taken by the right camera
        '''
        # Initialize Function criterien.
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        # self.flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
        self.pattern = pattern_choice
        self._image_list_l = image_list_l
        self._image_list_r = image_list_r
        # used for storage of the object points
        self.objpoints = []
        # used for storage of the image points
        self.imgpoints_l = []
        self.imgpoints_r = []
        # used for storage of the charuco points
        self.allCorners_l = []
        self.allIds_l = []
        self.allCorners_r = []
        self.allIds_r = []
        self.save_path = save_path

    @staticmethod
    def gray_scale(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def resize(gray_image):
        resize = gray_image.shape[::-1]
        # print('resize',np.array(resize).shape)
        # resize = cv2.resize(resize, (720, 960))
        return resize

    @staticmethod
    def binarization(gray_image):
        ret, th1 = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY)
        # th1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imshow('binarization:', th1)
        return th1

    @staticmethod
    def blur(gray_image, kernel):
        blur_image = cv2.medianBlur(gray_image, kernel)
        return blur_image

    @staticmethod
    def filter(gray_image, kernel=3):
        kernel = np.zeros((kernel, kernel), np.float32)
        gray = cv2.filter2D(gray_image, -1, kernel)
        return gray

    def _square_calibrater(self, w_nums, h_nums, length):
        '''

        Parameters
        ----------
        w_nums: number of the inside row corners of the cheeboard
        h_nums: number of the inside colomn corners of the cheeboard
        length: distance of adjacent corner points in meters

        Returns
        -------
        intrinsic parameters of camera pairs:
        self.mtx_l: camera matrix of left camera
        self.dist_l: distortion matrix of left camera
        self.rvecs_l: rotation vector of extrinsic parameters
        self.tvecs_l: translation vector extrinsic parameters
        the same as the right camera
        '''
        print(r"chessboard with the height {} and the width {} is chosen".format(w_nums, h_nums))
        self.world_points = np.zeros((w_nums * h_nums, 3), np.float32)
        self.world_points[:, :2] = np.mgrid[0:w_nums, 0:h_nums].T.reshape(-1, 2)

        for fname in self._image_list_l:
            gray = self.gray_scale(fname)
            resize_l = self.resize(gray)
            ret, corners_l = cv2.findChessboardCorners(gray, (w_nums, h_nums), None)
            if ret:
                # Setting pixel found condition
                exact_corners = cv2.cornerSubPix(gray, corners_l, (5, 5), (-1, -1), self.criteria)
                self.objpoints.append(self.world_points * length)
                self.imgpoints_l.append(exact_corners)

                # draw and display corners
            #     cv2.drawChessboardCorners(fname, (w_nums, h_nums), corners_l, ret)
            #     cv2.imshow('findCorners', fname)
            #     cv2.waitKey(3000)
            # cv2.destroyAllWindows()

        for fname in self._image_list_r:
            gray = self.gray_scale(fname)
            resize_r = self.resize(gray)
            ret, corners_r = cv2.findChessboardCorners(gray, (w_nums, h_nums), None)
            if ret:
                # Setting pixel found condition
                exact_corners = cv2.cornerSubPix(gray, corners_r, (5, 5), (-1, -1), self.criteria)
                # self.objpoints.append(self.world_points)
                self.imgpoints_r.append(exact_corners)

                # draw and display corners
            #     cv2.drawChessboardCorners(fname, (w_nums, h_nums), corners_r, ret)
            #     cv2.imshow('findCorners2', fname)
            #     cv2.waitKey(3000)
            # cv2.destroyAllWindows()

        # left and right camera are calibrated separately to obtain the intrinsic parameters
        ret_l, self.mtx_l, self.dist_l, self.rvecs_l, self.tvecs_l = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, resize_l, None, None)
        ret_r, self.mtx_r, self.dist_r, self.rvecs_r, self.tvecs_r = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, resize_r, None, None)

        return resize_l


    def _charuco_calibrater(self, w_nums, h_nums, length):
        '''
        Creat ChArUco pattern and do the calibration

        Parameters
        ----------
        w_nums: number of the inside row corners of the ChArUco board
        h_nums: number of the inside colomn corners of the ChArUco board
        length: distance of adjacent corner points in meters

        Returns
        -------

        '''
        # creat ChArUco pattern from the ChArUco dictionary
        dictionary = cv2.aruco.getPredefinedDictionary(dict=cv2.aruco.DICT_6X6_250)
        board = cv2.aruco.CharucoBoard_create(squaresY=6, squaresX=6, squareLength=0.15, markerLength=0.075,
                                              dictionary=dictionary)
        img_board = board.draw(outSize=(4096, 4096), marginSize=None, borderBits=None)

        self.world_points = np.zeros((w_nums * h_nums, 3), np.float32)
        self.world_points[:, :2] = np.mgrid[0:w_nums, 0:h_nums].T.reshape(-1, 2)

        cameramatrix = np.array([[7.77777778e+03, 0., 1.526e+03],
                                 [0., 7.77777778e+03, 2.048e+03],
                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        flags = 0
        # # # 如果该标志被设置，那么就会固定输入的cameraMatrix和distCoeffs不变，只求解R,T,E,F.
        flags |= cv2.CALIB_FIX_INTRINSIC
        # # 根据用户提供的cameraMatrix和distCoeffs为初始值开始迭代
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # 迭代过程中不会改变焦距
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K6 + cv2.CALIB_FIX_K6
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # # # 切向畸变保持为零


        print(r"ChArUco with the height {} and the width {} is chosen".format(w_nums, h_nums))

        i = 0
        for fname in self._image_list_l:
            i = i + 1
            gray = self.gray_scale(fname)
            cv2.namedWindow('imshow', cv2.WINDOW_FREERATIO)
            cv2.waitKey(5000)
            resize_l = self.resize(gray)
            print('resize_l', resize_l)
            corners_l, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)
            # print("*" * 50)
            # print(rejected)
            # print('ids', ids)
            # print('corners_lkkkkkkkkkkkkkkkkkkkk', corners_l)
            # corners_l, ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(gray, cb, corners_l, ids, rejected,
            #                                                                     cameraMatrix=K, distCoeffs=dist_coef)
            # print('corners_l:', corners_l)
            if corners_l == None or len(corners_l) == 0:
                continue
            ret, charucoCorners_l, charucoIds_l = cv2.aruco.interpolateCornersCharuco(corners_l, ids, gray, board)
            print(i, ret)
            if corners_l is not None and charucoIds_l is not None:
                # Setting pixel found condition
                exact_corners = cv2.cornerSubPix(gray, charucoCorners_l, (11, 11), (-1, -1), self.criteria)
                # objpoints_L, imgpoints_L = cv2.aruco.getBoardObjectAndImagePoints(board, charucoCorners_l,
                #                                                                   charucoIds_l)
                # print('objpoints_L', objpoints_L)
                self.objpoints.append(self.world_points * length)
                self.allCorners_l.append(charucoCorners_l)
                self.allIds_l.append(charucoIds_l)

                # draw and display corners
                # cv2.aruco.drawDetectedMarkers(fname, corners_l, ids)
                # cv2.namedWindow('findCorners', cv2.WINDOW_FREERATIO)
                # cv2.imshow('findCorners', fname)
                # cv2.waitKey(3000)
            #     cv2.drawChessboardCorners(fname, (w_nums, h_nums), charucoCorners_l, ret)
            #     cv2.namedWindow('findCorners', cv2.WINDOW_FREERATIO)
            #     cv2.imshow('findCorners', fname)
            #     cv2.waitKey(50000)
            # cv2.destroyAllWindows()

        j = 0
        for fname in self._image_list_r:
            j = j+1
            gray = self.gray_scale(fname)
            resize_r = self.resize(gray)
            corners_r, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)
            if corners_r == None or len(corners_r) == 0:
                continue
            ret, charucoCorners_r, charucoIds_r = cv2.aruco.interpolateCornersCharuco(corners_r, ids, gray, board)
            print(j, ret)
            if corners_r is not None and charucoIds_r is not None:
                # Setting pixel found condition
                exact_corners = cv2.cornerSubPix(gray, charucoCorners_r, (11, 11), (-1, -1), self.criteria)
                self.allCorners_r.append(charucoCorners_r)
                self.allIds_r.append(charucoIds_r)

                # draw and display corners
                # cv2.aruco.drawDetectedMarkers(fname, corners_r, ids)
            #     cv2.drawChessboardCorners(fname, (w_nums, h_nums), charucoCorners_r, ret)
            #     cv2.namedWindow('findCorners2', cv2.WINDOW_FREERATIO)
            #     cv2.imshow('findCorners2', fname)
            #     cv2.waitKey(50000)
            # cv2.destroyAllWindows()

        # left and right camera are calibrated separately to obtain the intrinsic parameters
        # ret_l, self.mtx_l, self.dist_l, self.rvecs_l, self.tvecs_l = cv2.aruco.calibrateCameraCharuco(
        #     self.allCorners_l, self.allIds_l, board, (w_nums, h_nums), None, None, flags=0, criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-5))
        # ret_r, self.mtx_r, self.dist_r, self.rvecs_r, self.tvecs_r = cv2.aruco.calibrateCameraCharuco(
        #     self.allCorners_r, self.allIds_r, board, (w_nums, h_nums), None, None, flags=0, criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-5))
        ret_l, self.mtx_l, self.dist_l, self.rvecs_l, self.tvecs_l = cv2.calibrateCamera(
            self.objpoints, self.allCorners_l, resize_l, cameraMatrix=cameramatrix, distCoeffs=None, flags=0, criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 100, 1e-5))
        ret_r, self.mtx_r, self.dist_r, self.rvecs_r, self.tvecs_r = cv2.calibrateCamera(
            self.objpoints, self.allCorners_r, resize_r, cameraMatrix=cameramatrix, distCoeffs=None, flags=0, criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 100, 1e-5))
        # print(self.mtx_l, self.dist_l,self.mtx_r, self.dist_r)
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(self.objpoints, self.allCorners_l, self.allCorners_r,
                                                              self.mtx_l, self.dist_l, self.mtx_r, self.dist_r, (4096, 3072),
                                                              criteria=stereocalib_criteria, flags=flags)
        # self.camera_model = dict([('rms', ret), ('M1', self.mtx_l), ('M2', self.mtx_r), ('dist1', self.dist_l),
        #                           ('dist2', self.dist_r), ('R', R), ('T', T),
        #                           ('E', E), ('F', F), ('dims', resize_l)])
        self.camera_model = dict([('rms', ret), ('M1', M1), ('M2', M2), ('dist1', d1),
                                  ('dist2', d2), ('R', R), ('T', T),
                                  ('E', E), ('F', F), ('dims', resize_l)])
        print(self.camera_model)
        return resize_l


    def distortion(self, img):
        img_dis = cv2.imread(img)
        h, w = img_dis.shape[:2]
        newcameramatrix, roi = cv2.getOptimalNewMatrix(mtx, dist, (w,h), 0, (w, h))
        dst = cv2.undistort(img_dis, self.mtx, self.dist, None, newcameramatrix)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow("roi", dst)

    @property
    def validate_square(self):
        total_error = 0
        for i in range(len(self.objpoints)):
            img_reproject, _ = cv2.projectPoints(self.objpoints[i], self.rvecs_l[i], self.tvecs_l[i], self.mtx_l, self.dist_l)
            # print(img_reproject)
            # print('self:', self.allCorners_l[i])
            # error = cv2.norm(self.imgpoints_l[i], img_reproject, cv2.NORM_L2)
            error = cv2.norm(self.allCorners_l[i], img_reproject, cv2.NORM_L2)
            total_error += error * error
            print('error', i, error/np.sqrt(len(img_reproject)))
        print("rms error1: {}".format(np.sqrt(total_error / (len(self.objpoints) * len(img_reproject)))))

    @property
    def validate_square_2(self):
        total_error = 0
        for i in range(len(self.objpoints)):
            img_reproject, _ = cv2.projectPoints(self.objpoints[i], self.rvecs_r[i], self.tvecs_r[i], self.mtx_r,
                                                 self.dist_r)
            # error = cv2.norm(self.imgpoints_r[i], img_reproject, cv2.NORM_L2)
            error = cv2.norm(self.allCorners_r[i], img_reproject, cv2.NORM_L2)
            total_error += error * error
            error_single = error/np.sqrt(len(img_reproject))
            print('error', i, error/np.sqrt(len(img_reproject)))
        print("rms error2: {}".format(np.sqrt(total_error / (len(self.objpoints) * len(img_reproject)))))

    def _circle_calibrater(self, points_per_row, points_per_column, c_distance):
        '''

        Parameters
        ----------
        points_per_row: number of the row circle centers
        points_per_column: number of the column circle centers
        c_distance: distance of 2 adjacent circle centers in meters

        Returns
        -------

        '''
        # 目前的程序效果不好，simpleblob函数有很多可调的参数
        # 使用halcon标准的对称标定板时，圆的半径大小无所谓，可以自己设定，因为在提取圆心坐标时不涉及圆的半径
        print(r"circle_pattern with circles {} * {} is chosen".format(points_per_row, points_per_column))
        self.world_points = np.zeros((points_per_row * points_per_column, 3), np.float32)
        self.world_points[:, :2] = np.mgrid[:points_per_row, :points_per_column].T.reshape(-1, 2)
        i = 0
        for fname in self._image_list_l:
            i = i+1
            gray = self.gray_scale(fname)
            print('Picture brightness:', gray.mean())
            # gray = self.binarization(gray)
            thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)[1]
            resize_l = self.resize(thresh)
            # self.dims = resize_l[:2]
            ret, centers_l = cv2.findCirclesGrid(thresh, (points_per_row, points_per_column), None,
                                                 flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING))

            print(i, ret)
            if ret:
                # return centers
                self.objpoints.append(self.world_points * c_distance)
                exact_cornersl = cv2.cornerSubPix(thresh, centers_l, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_l.append(exact_cornersl)

                cv2.drawChessboardCorners(fname, (points_per_row, points_per_column), exact_cornersl, ret)
                cv2.namedWindow('result', cv2.WINDOW_FREERATIO)
                cv2.imshow('result', fname)
                cv2.waitKey(3000)
            cv2.destroyAllWindows()

            # Blobdetector used to draw circle and circle-centers
            # params = cv2.SimpleBlobDetector_Params()
            #
            # params.filterByArea = True
            # params.minArea = 100
            # params.maxArea = 300
            # #
            # params.filterByCircularity = True
            # params.minCircularity = 0.5
            #
            # params.filterByConvexity = True
            # params.minConvexity = 0.5
            #
            # params.filterByInertia = True
            # params.minInertiaRatio = 0.05
            #
            # blobDetector = cv2.SimpleBlobDetector_create(params)
            # keypoints = blobDetector.detect(gray)
            #
            # center_points = []
            # for keypoint in keypoints:
            #     center_points.append(keypoint.pt)
            #     # print(keypoint.pt)
            #     cv2.circle(fname, (int(keypoint.pt[0]), int(keypoint.pt[1])), 1, (0, 255, 0), 0)
            # with_keypoints = cv2.drawKeypoints(fname, keypoints, np.array([]), (0, 0, 255),
            #                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #
            #
            # cv2.namedWindow("blob", cv2.WINDOW_FREERATIO)
            # cv2.imshow("blob", with_keypoints)
            # # cv2.imshow("corners", thresh)
            # cv2.waitKey(5000)
            # cv2.destroyAllWindows()

        j = 0
        for fname in self._image_list_r:
            j = j+1
            gray = self.gray_scale(fname)
            print('Picture brightness:', gray.mean())
            # gray = self.binarization(gray)
            thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)[1]
            resize_r = self.resize(thresh)
            ret, centers_r = cv2.findCirclesGrid(thresh, (points_per_row, points_per_column), None,
                                                 flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING))

            print(j, ret)
            if ret:
                exact_cornersr = cv2.cornerSubPix(thresh, centers_r, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_r.append(exact_cornersr)

                cv2.drawChessboardCorners(fname, (points_per_row, points_per_column), exact_cornersr, ret)
                cv2.namedWindow('result2', cv2.WINDOW_FREERATIO)
                cv2.imshow('result2', fname)
                cv2.waitKey(3000)
            cv2.destroyAllWindows()

            # params = cv2.SimpleBlobDetector_Params()
            #
            # params.filterByArea = True
            # params.minArea = 100
            # params.maxArea = 300
            #
            # params.filterByCircularity = True
            # params.minCircularity = 0.5
            #
            # params.filterByConvexity = True
            # params.minConvexity = 0.5
            #
            # params.filterByInertia = True
            # params.minInertiaRatio = 0.05
            #
            # blobDetector = cv2.SimpleBlobDetector_create(params)
            # keypoints = blobDetector.detect(gray)
            #
            # # self.keypoinsr = keypoints
            # center_points = []
            # for keypoint in keypoints:
            #     center_points.append(keypoint.pt)
            #     cv2.circle(fname, (int(keypoint.pt[0]), int(keypoint.pt[1])), 1, (0, 255, 0), 0)
            #     # print("keypoints:", keypoint.pt)
            # with_keypoints = cv2.drawKeypoints(fname, keypoints, np.array([]), (0, 0, 255),
            #                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


            # cv2.namedWindow("blob2", cv2.WINDOW_FREERATIO)
            # cv2.imshow("blob2", with_keypoints)
            # cv2.waitKey(5000)
            # cv2.destroyAllWindows()


        ret_l, self.mtx_l, self.dist_l, self.rvecs_l, self.tvecs_l = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, resize_l, None, None)
        ret_r, self.mtx_r, self.dist_r, self.rvecs_r, self.tvecs_r = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, resize_r, None, None)
        # self.newCameraMatrix_l, newExtents_l = cv2.getOptimalNewCameraMatrix(self.mtx_l, self.dist_l, self.dims, 0.0)
        # self.newCameraMatrix_r, newExtents_r = cv2.getOptimalNewCameraMatrix(self.mtx_r, self.dist_r, self.dims, 0.0)
        print('ret', ret_l, ret_r)

        return resize_l


    def _circle_calibrater_noblob(self, points_per_row, points_per_column, c_distance):

        print(r"circle_pattern with circles {} * {} is chosen".format(points_per_row, points_per_column))
        self.world_points = np.zeros((points_per_row * points_per_column, 3), np.float32)
        self.world_points[:, :2] = np.mgrid[:points_per_row, :points_per_column].T.reshape(-1, 2)

        for i in (self._image_list_l, self._image_list_r):
            img_l = i[0]
            gray_l = self.gray_scale(img_l)
            print('Picture brightness:', gray_l.mean())
            # gray = self.binarization(gray)
            thresh_l = cv2.threshold(gray_l, 40, 255, cv2.THRESH_BINARY)[1]
            resize_l = self.resize(thresh_l)
            # 这里需要再加入二值化的命令吗????
            ret_l, centers_l = cv2.findCirclesGrid(thresh_l, (points_per_row, points_per_column), None,
                                                 flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING))
            img_r = i[1]
            gray_r = self.gray_scale(img_r)
            print('Picture brightness:', gray_r.mean())
            # gray = self.binarization(gray)
            thresh_r = cv2.threshold(gray_r, 40, 255, cv2.THRESH_BINARY)[1]
            resize_r = self.resize(thresh_r)
            # 这里需要再加入二值化的命令吗????
            ret_r, centers_r = cv2.findCirclesGrid(thresh_r, (points_per_row, points_per_column), None,
                                                 flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING))
            if ret_l & ret_r:
                # return centers
                self.objpoints.append(self.world_points * c_distance)
                exact_cornersl = cv2.cornerSubPix(thresh_l, centers_l, (11, 11), (-1, -1), self.criteria)
                exact_cornersr = cv2.cornerSubPix(thresh_r, centers_r, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_l.append(exact_cornersl)
                self.imgpoints_r.append(exact_cornersr)


                cv2.drawChessboardCorners(img_l, (points_per_row, points_per_column), exact_cornersl, ret_l)
                cv2.namedWindow('drawcenters', cv2.WINDOW_FREERATIO)
                cv2.imshow('drawcenters', img_l)

                cv2.drawChessboardCorners(img_r, (points_per_row, points_per_column), exact_cornersr, ret_r)
                cv2.namedWindow('drawcenters2', cv2.WINDOW_FREERATIO)
                cv2.imshow('drawcenters2', img_r)
                cv2.waitKey(3000)
            cv2.destroyAllWindows()

        ret_l, self.mtx_l, self.dist_l, self.rvecs_l, self.tvecs_l = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, resize_l, None, None)
        ret_r, self.mtx_r, self.dist_r, self.rvecs_r, self.tvecs_r = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, resize_r, None, None)
        # self.newCameraMatrix_l, newExtents_l = cv2.getOptimalNewCameraMatrix(self.mtx_l, self.dist_l, (self.w, self.h), 0.0)
        # self.newCameraMatrix_r, newExtents_r = cv2.getOptimalNewCameraMatrix(self.mtx_r, self.dist_r, (self.w, self.h), 0.0)
        print('ret', ret_l, ret_r)

        return resize_l

    # def _circle_calibrater_2(self, points_per_row, points_per_column, c_distance):
    #     # use hough function to detect circle centers
    #     # Poor detection results
    #     print(r"circle_pattern with circles {} * {} is chosen".format(points_per_row, points_per_column))
    #     self.world_points = np.zeros((points_per_row * points_per_column, 3), np.float32)
    #     self.world_points[:, :2] = c_distance * np.mgrid[0:points_per_row, 0:points_per_column].T.reshape(-1, 2)
    #     for fname in self._image_list_l:
    #         gray = self.gray_scale(fname)
    #         gray = self.blur(gray, 7)
    #         resize = self.resize(gray)
    #         circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10,
    #                                    param1=80, param2=20, minRadius=0, maxRadius=50)
    #         circles = np.uint16(np.around(circles))
    #         print("circles:", circles)
    #         print("shape of circles:", circles.shape)
    #         self.objpoints.append(self.world_points)
    #         self.imgpoints.append(circles)
    #
    #         for i in circles[0, :]:
    #             cv2.circle(fname, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #             cv2.circle(fname, (i[0], i[1]), 2, (0, 0, 255), 2)
    #
    #             # sorted(circles, key=attrgetter(i[0], i[1]))
    #
    #         # with_keypoints = cv2.drawKeypoints(fname, circles, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    #         cv2.imshow('findCorners', fname)
    #         cv2.waitKey(3000)
    #         cv2.destroyAllWindows()
    #
    #     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    #         self.objpoints, self.imgpoints, resize, None, None)
    #     print("intrinsic :", mtx)
    #     print("distortion :", dist)
    #     self.mtx = mtx
    #     self.dist = dist
    #     self.rvecs = rvecs
    #     self.tvecs = tvecs
    #     return self.mtx, self.dist

    def stereocalibrate(self, dims):
        '''

        Parameters
        ----------
        dims: dimention of the calibrated image, the format is w * h

        Returns
        -------
        camera model contains the intrinsic parameters of the single camera and the extrinsic parameters of the stereocamera pair.
        '''

        flags = 0
        # # # 如果该标志被设置，那么就会固定输入的cameraMatrix和distCoeffs不变，只求解R,T,E,F.
        # flags |= cv2.CALIB_FIX_INTRINSIC
        # # 根据用户提供的cameraMatrix和distCoeffs为初始值开始迭代
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # 迭代过程中不会改变焦距
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # # # 切向畸变保持为零
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_FIX_K3
        print('dims:', dims[0], dims[1])

        # calib_flags = \
        #     cv2.CALIB_FIX_INTRINSIC + \
        #     cv2.CALIB_ZERO_TANGENT_DIST + \
        #     cv2.CALIB_USE_INTRINSIC_GUESS + \
        #     cv2.CALIB_SAME_FOCAL_LENGTH

        # calib_flags = \
        #     cv2.CALIB_FIX_ASPECT_RATIO + \
        #     cv2.CALIB_ZERO_TANGENT_DIST + \
        #     cv2.CALIB_USE_INTRINSIC_GUESS + \
        #     cv2.CALIB_SAME_FOCAL_LENGTH + \
        #     cv2.CALIB_RATIONAL_MODEL + \
        #     cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5

        calib_flags2 = \
            cv2.CALIB_RATIONAL_MODEL + \
            cv2.CALIB_FIX_K6

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        stereocalib_criteria2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-3)

        # ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(self.objpoints, self.imgpoints_l, self.imgpoints_r,
        #                                                       self.mtx_l, self.dist_l, self.mtx_r, self.dist_r, dims,
        #                                                       criteria=stereocalib_criteria,
        #                                                       flags=flags)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(self.objpoints, self.allCorners_l, self.allCorners_r,
                                                              self.mtx_l, self.dist_l, self.mtx_r, self.dist_r, dims,
                                                              flags=cv2.CALIB_RATIONAL_MODEL, criteria=stereocalib_criteria)

        self.camera_model = dict([('rms', ret), ('M1', M1), ('M2', M2), ('dist1', d1),
                                  ('dist2', d2), ('R', R), ('T', T),
                                  ('E', E), ('F', F), ('dims', dims)])
        # return self.camera_model

    @ property
    def save_txt(self):
        filename = open(self.save_path + '\\' + 'intrinsic.txt', 'w')
        for k, v in self.camera_model.items():
            filename.write(k + ':' + str(v))
            filename.write('\n')
        filename.close()


    def method_manager(self, *args):
        '''

        Parameters
        ----------
        args: please define the number of the row points(corners), coloum points(corners) and the distance of the adjacent points(corners) in meter.

        Returns
        -------
        The camera model parameters.
        '''
        # *args代表的是标定板的长宽等参数
        if self.pattern == "square":
            dims = self._square_calibrater(args[0], args[1], args[2])
            # self.distortion()
        elif self.pattern == "charuco":
            dims = self._charuco_calibrater(args[0], args[1], args[2])
            # self.validate_square()
            # self.validate_square_2()
        else:
            dims= self._circle_calibrater(args[0], args[1], args[2])
            # self.validate_square()
            # self.validate_square_2()
        # self.stereocalibrate(dims)
        # self.save_txt
        return self.camera_model


class StereoRectify():
    '''
    Recitify the image pair. Draw the polar lines to find the corresponding feature points in left image and right image.
    '''

    def __init__(self, camera_model):
        self.cameraMatrix_l = camera_model['M1']
        self.cameraMatrix_r = camera_model['M2']
        self.dist_l = camera_model['dist1']
        self.dist_r = camera_model['dist2']
        self.R = camera_model['R']
        self.T = camera_model['T']
        self.dims = camera_model['dims']
        print('self.dims:', self.dims)
        # 主点列坐标的差
        self.doffs = 0.0


    # 消除畸变
    def undistortion(self, image, camera_matrix, dist_coeff):
        undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)
        return undistortion_image

    def save_txt(self, path):
        filename = open(path + 'entrinsic.txt', 'w')
        for k, v in camera1.items():
            filename.write(k + ':' + str(v))
            filename.write('\n')
        filename.close()

    def get_rectify_transform(self):
        '''
        Get the rectified camera model. Preparing for the parallel correction.8

        Returns
        -------
        rectify_model
        '''
        # 读取内参和外参

        # 计算校正变换
        # 记得对alpha调参，出来的图像可能是黑色的
        # Q是深度差异映射矩阵
        # R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.cameraMatrix_l, self.dist_l, self.cameraMatrix_r, self.dist_r,
        #                                                   (self.dims[0], self.dims[1]), self.R, self.T, flags = 1024, alpha = -1
        #                                                   )
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.cameraMatrix_l, self.dist_l, self.cameraMatrix_r,
                                                          self.dist_r,
                                                          (4096, 3072), self.R, self.T, flags=0,
                                                          alpha=-1
                                                          )
        # R1[0, :] *= -1
        # R2[0, :] *= -1

        # R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.cameraMatrix_l, self.dist_l, self.cameraMatrix_r,
        #                                                   self.dist_r,
        #                                                   (self.dims[1], self.dims[0]), self.R, self.T, cv2.CALIB_ZERO_DISPARITY, 0, (0, 0))
        # R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.cameraMatrix_l, self.dist_l, self.cameraMatrix_r, self.dist_r,
        #                                                   self.dims, self.R, self.T, flags=0, alpha=-1)
        # R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.cameraMatrix_l, self.dist_l, self.cameraMatrix_r,
        #                                                   self.dist_r, self.dims, self.R, self.T, 1, (0,0))

        # self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.cameraMatrix_l, self.dist_l, R1, P1, (self.dims[1], self.dims[0]), cv2.CV_16SC2)
        # self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.cameraMatrix_r, self.dist_r, R2, P2, (self.dims[1], self.dims[0]), cv2.CV_16SC2)
        # self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.cameraMatrix_l, self.dist_l, R1, P1,
        #                                                      self.dims, cv2.INTER_NEAREST)
        # self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.cameraMatrix_r, self.dist_r, R2, P2,
        #                                                      self.dims, cv2.INTER_NEAREST)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.cameraMatrix_l, self.dist_l, R1, P1,
                                                             (4096, 3072), cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.cameraMatrix_r, self.dist_r, R2, P2,
                                                             (4096, 3072), cv2.CV_32FC1)
        # self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.cameraMatrix_l, self.dist_l, R1, P1,
        #                                                      self.dims, cv2.CV_32FC1)
        # self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.cameraMatrix_r, self.dist_r, R2, P2,
        #                                                      self.dims, cv2.CV_32FC1)

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
            ('stereo_right_mapy', self.map2y),
            ('roi1', roi1),
            ('roi2', roi2)
        ])
        print('rectify_model:',rectify_model)
        # self.save_txt(self, path)
        return rectify_model

    def rectify_image(self, grayl, grayr):
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
        rectified_img1 = cv2.remap(grayl, self.map1x, self.map1y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rectified_img2 = cv2.remap(grayr, self.map2x, self.map2y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        # rectified_img1 = self.undistortion(rectified_img1, self.cameraMatrix_l, self.dist_l)
        # rectified_img2 = self.undistortion(rectified_img2, self.cameraMatrix_r, self.dist_r)
        # rectified_img1 = cv2.remap(grayl, self.map1x, self.map1y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        # rectified_img2 = cv2.remap(grayr, self.map2x, self.map2y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
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
        cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\0422\\result1.png',
                    rectified_img1)
        cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\0422\\result2.png',
                    rectified_img2)
        result = np.concatenate((rectified_img1, rectified_img2), axis=1)
        resize = cv2.resize(result, (1024, 384))
        # cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\result3.png',
        #             result)
        cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\0422\\result3.png',
                    result)
        cv2.imshow("rec.png", result)
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
            cv2.line(output, (0, line_interval * (2*k )), (2 * width, line_interval * (2*k )), (0, 255, 0),
                     thickness=2, lineType=cv2.LINE_AA)
            cv2.line(output, (0, line_interval * (2*k + 1)), (2 * width, line_interval * (2*k + 1)), (0, 0, 255),
                     thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("withlines", output)
        # cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\withlines.png', output)
        cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\0422\\withlines.png', output)
        cv2.waitKey(8000)


    def stereo_matchSGBM(self, left_image, right_image, down_scale=False):
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
        blockSize = 7 #3
        paraml = {'minDisparity': -16,
                  'numDisparities': 32*16,  #128
                  'blockSize': 5,
                  'P1': 8 * img_channels * blockSize ** 2,  #8
                  'P2': 24 * img_channels * blockSize ** 2,  #32
                  'disp12MaxDiff': 1,  #1 # non-positive vaalue to disable the check
                  'preFilterCap': 31,
                  'uniquenessRatio': 10,   #15
                  'speckleWindowSize': 100, #100  Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range
                  'speckleRange': 4, #7
                  'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
                  }

        # cread the SGBM instance
        left_matcher = cv2.StereoSGBM_create(**paraml)
        paramr = paraml
        # paramr['minDisparity'] = -paraml['numDisparities']
        # paramr['numDisparities'] = paraml['minDisparity']
        # paramr['minDisparity'] = -15*16
        right_matcher = cv2.StereoSGBM_create(**paramr)

        # Filter Parameters
        lmbda = 80000
        # sigma = [0.7, 1.2, 1.5]
        sigma = 1.2
        visual_multiplier = 1.0 # 1.0

        # wsl filter are used to smooth the depthmap
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        # calculate the disparity map
        print('computing disparity...')
        displ = left_matcher.compute(left_image, right_image)  # .astype(np.float32)/16
        cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\0422\\dl.png', displ)
        dispr = right_matcher.compute(right_image, left_image)  # .astype(np.float32)/16
        cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\0422\\dr.png', dispr)
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, left_image, None, dispr)  # important to put "imgL" here!!!

        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        cv2.imshow('Disparity Map', filteredImg)
        cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\0422\\depthmap.png', filteredImg)
        cv2.waitKey()
        cv2.destroyAllWindows()


cam_5 = Camera(5)
path, path_l, path_r = cam_5.get_dir
print('paths:', path, path_l, path_r)
read_all = PhotoSet(path, path_l, path_r)
imma1, imma2 = read_all.image_to_matrix
chacal = CameraCalibration("charuco", imma1, imma2, path)
camera5 = chacal.method_manager(5, 5, 0.15)
print('camera5:', camera5)


def preprocess(img1, img2):
    # 彩色图->灰度图
    if(img1.ndim == 3):#判断为三维数组
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if(img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2

# path = 'C:\\Users\\wyfmi\\Desktop\\0422'
stereo5 = StereoRectify(camera5)
rectify_model = stereo5.get_rectify_transform()
#
print(rectify_model)

imgl = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\22-04-21_15-20-442.png')
imgl = np.rot90(imgl, 3)
imgr = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\22-04-21_15-20-441.png')
imgr = np.rot90(imgr, 3)

stereo5.rectify_image(imgl, imgr)

imgl = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\result1.png')
imgr = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\result2.png')
stereo5.draw_line(imgl, imgr)

stereo5.stereo_matchSGBM(imgl, imgr)


