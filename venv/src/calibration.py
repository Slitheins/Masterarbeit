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
            image_list_l = sorted(glob.glob(self._path + '//' + '*1.png'))
            for i, image in enumerate(image_list_l):
                write_name = self._save_dir_l + '//' + 'Image_' + str(i) + '.png'
                img = cv2.imread(image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.rot90(img, 1)
                # img = self.resize(img)
                # print('gray ', gray )
                cv2.imwrite(write_name, img)
                # 放到一个数组里
                self._images_l.append(img)
            print("{} photos in this directory are read".format(len(self._images_l)))

            # image_list_r = sorted(glob.glob(self._path + '//' + '*_2.jpg'))
            image_list_r = sorted(glob.glob(self._path + '//' + '*2.png'))
            for j, image in enumerate(image_list_r):
                write_name = self._save_dir_r + '//' + 'Image_' + str(j) + '.png'
                img = cv2.imread(image)
                img = np.rot90(img, 1)
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
                                                              self.mtx_l, self.dist_l, self.mtx_r, self.dist_r, (4096,3072),
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
        Get the rectified camera model. Preparing for the parallel correction.

        Returns
        -------
        rectify_model
        '''
        # 读取内参和外参

        # 计算校正变换
        # 记得对alpha调参，出来的图像可能是黑色的
        # Q是深度差异映射矩阵
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.cameraMatrix_l, self.dist_l, self.cameraMatrix_r, self.dist_r,
                                                          (self.dims[0], self.dims[1]), self.R, self.T, flags = 0, alpha = -1
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
        # self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.cameraMatrix_l, self.dist_l, R1, P1,
        #                                                      (self.dims[1], self.dims[0]), cv2.CV_32FC1)
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
        rectified_img1 = self.undistortion(rectified_img1, self.cameraMatrix_l, self.dist_l)
        rectified_img2 = self.undistortion(rectified_img2, self.cameraMatrix_r, self.dist_r)
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


    def depth_map(self, imgL, imgR, sigma=0.7):
        """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
        # SGBM Parameters -----------------
        window_size = 31  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

        left_matcher = cv2.StereoSGBM_create(
            minDisparity= 50,
            numDisparities= 10 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=window_size,
            P1=8 * 3 * window_size ** 2,
            # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff= 12, # default value = 1
            uniquenessRatio= 10,
            speckleWindowSize= 50, # opencv example = 100
            speckleRange= 32, # oepncv example = 10
            preFilterCap= 63, # default =15, opencv example=63
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        # left_matcher = cv2.StereoSGBM_create(
        #     minDisparity=0,
        #     numDisparities=5 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        #     blockSize=5, # from 5 - 21
        #     P1=8 * 3 * window_size, # 3是图像的通道数
        #     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        #     P2=32 * 3 * window_size,
        #     disp12MaxDiff=12,
        #     uniquenessRatio=10,
        #     speckleWindowSize=80,
        #     speckleRange=32,
        #     preFilterCap=15, # default value=15
        #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        # )
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        # FILTER Parameters
        lmbda = 80000
        visual_multiplier = 6   # 6

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)

        wls_filter.setSigmaColor(sigma)
        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\0422\\depthmap.png', filteredImg)

        return filteredImg

    def stereo_match(self, imgL, imgR):
        # disparity range is tuned for 'aloe' image pair
        window_size = 15
        min_disp = 150
        num_disp = 96 - min_disp
        stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                       numDisparities=num_disp,
                                       blockSize=16,
                                       P1=8 * 3 * window_size ** 2,
                                       P2=32 * 3 * window_size ** 2,
                                       disp12MaxDiff=1,
                                       uniquenessRatio=10,
                                       speckleWindowSize=150,
                                       speckleRange=32
                                       )

        # print('computing disparity...')
        disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

        # print('generating 3d point cloud...',)
        h, w = imgL.shape[:2]
        # f = 0.8 * w  # guess for focal length
        Q = np.float32([[1, 0, 0, -0.5 * w],
                        [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                        [0, 0, 0, -f],  # so that y-axis looks up
                        [0, 0, 1, 0]])
        points = cv2.reprojectImageTo3D(disp, Q)
        colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        mask = disp > disp.min()
        out_points = points[mask]
        out_colors = colors[mask]
        append_ply_array(out_points, out_colors)

        disparity_scaled = (disp - min_disp) / num_disp
        disparity_scaled += abs(np.amin(disparity_scaled))
        disparity_scaled /= np.amax(disparity_scaled)
        disparity_scaled[disparity_scaled < 0] = 0
        return np.array(255 * disparity_scaled, np.uint8)



# 实例化相机并用棋盘格标定
# cam_1 = Camera(1)
# path, path_l, path_r = cam_1.get_dir
# read_all = PhotoSet(path, path_l, path_r)
# imma1, imma2 = read_all.image_to_matrix
# square = CameraCalibration("square", imma1, imma2)
# camera1 = square.method_manager(9, 6, 2.5000000372529030e-02, path)
# print('camera1:', camera1)


# 实例化相机并用圆点格标定
# cam_2 = Camera(2)
# path = cam_2.get_dir()
# read_all = PhotoSet(path)
# image_matrix = read_all.read_images()
# circle = Calibration("circle", image_matrix)
# a, b = circle._circle_calibrater(4, 11, 0.02)
# id_1 = cam_1.get_id
# cali = Calibration("circle")
# cali.method_manager()

cam_5 = Camera(5)
path, path_l, path_r = cam_5.get_dir
print('paths:', path, path_l, path_r)
read_all = PhotoSet(path, path_l, path_r)
imma1, imma2 = read_all.image_to_matrix
chacal = CameraCalibration("charuco", imma1, imma2, path)
camera5 = chacal.method_manager(5, 5, 0.15)
print('camera5:', camera5)

# camera5 = {'rms': 0.3340139547504995, 'M1': np.array([[8.11194256e+03, 0.00000000e+00, 1.50573679e+03],
#        [0.00000000e+00, 8.11337077e+03, 2.26124973e+03],
#        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), 'M2': np.array([[8.11194256e+03, 0.00000000e+00, 1.50573679e+03],
#        [0.00000000e+00, 8.11337077e+03, 2.26124973e+03],
#        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), 'dist1': np.array([[-1.99060644e+00,  8.17715084e+01,  1.76126084e-03,
#         -8.81308791e-04, -1.21146285e+01, -1.95354945e+00,
#          8.25324858e+01, -1.29870567e+01,  0.00000000e+00,
#          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#          0.00000000e+00,  0.00000000e+00]]), 'dist2': np.array([[-1.32890753e+01,  2.99335988e+02,  2.10385460e-03,
#         -8.33790090e-04,  2.27295026e+02, -1.32241701e+01,
#          2.98734234e+02,  2.49206056e+02,  0.00000000e+00,
#          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#          0.00000000e+00,  0.00000000e+00]]), 'R': np.array([[ 0.98520661,  0.00321211,  0.17134065],
#        [ 0.00391106,  0.99914246, -0.0412194 ],
#        [-0.17132612,  0.04127975,  0.9843492 ]]), 'T': np.array([[-1.50150311],
#        [-0.09902166],
#        [ 0.08425395]]), 'E': np.array([[ 0.01663547, -0.08826928, -0.09399899],
#        [-0.17423916,  0.06225231,  1.49243951],
#        [ 0.09168432, -1.49989745,  0.0788575 ]]), 'F': np.array([[ 4.77968876e-09, -2.53613720e-08, -1.72314760e-04],
#        [-5.00534199e-08,  1.78830731e-08,  3.52631002e-03],
#        [ 3.19676412e-04, -3.49807466e-03,  1.00000000e+00]]), 'dims': (3072, 4096)}

# 这组数据是目前最准的 极线齐的不像话
# camera5 = {'rms': 0.3340139547504995, 'M1': np.array([[8.22169440e+03, 0.00000000e+00, 1.50206854e+03],
#        [0.00000000e+00, 8.24432071e+03, 2.46090682e+03],
#        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), 'M2': np.array([[8.22169440e+03, 0.00000000e+00, 1.50206854e+03],
#        [0.00000000e+00, 8.24432071e+03, 2.46090682e+03],
#        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), 'dist1': np.array([[-8.49267507e-02, -3.19915593e-02,  4.18965592e-03,  8.06757093e-05,
#    0.00000000e+00]]), 'dist2': np.array([[-0.04726039, -0.196313,    0.0101606,   0.00119035,  0.        ]]), 'R': np.array([[ 0.98625401, -0.00168698,  0.16522768],
#  [ 0.00901614,  0.9990076,  -0.04361798],
#  [-0.16499012,  0.04450812,  0.98529046]]), 'T': np.array([[-1.52199684],
#  [ 0.03340957],
#  [ 0.11968841]]), 'E': np.array([[-0.00659138, -0.11808263,  0.0381387 ],
#        [-0.13307127,  0.06753931,  1.5193848 ],
#        [-0.04667286, -1.52043004,  0.06086624]]), 'F': np.array([[-1.77814094e-09, -3.17674641e-08,  1.65437316e-04],
#        [-3.57998176e-08,  1.81200583e-08,  3.36985007e-03],
#        [-1.27470717e-05, -3.35985487e-03,  1.00000000e+00]]), 'dims': (3072, 4096)}


# 下面的测试程序是可用的
# cam_3 = Camera(3)
# path, path_l, path_r = cam_3.get_dir()
# read_all = PhotoSet(path, path_l, path_r)
# imma1, imma2 = read_all.image_to_matrix()
# circlecal = CameraCalibration("circle", imma1, imma2)
# camera1 = circlecal.method_manager(7, 7, 0.1)
# print(camera1)
# # Kreispunkte(33 Paare)
# camera1 = {'rms':0.6160846063201201,
#            'M1':np.array([[8.63936278e+03, 0.00000000e+00, 2.14338706e+03],
#                           [0.00000000e+00, 8.66314910e+03, 1.37603525e+03],
#                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
#            'M2': np.array([[8.55432393e+03, 0.00000000e+00, 2.22009028e+03],
#                            [0.00000000e+00, 8.57157453e+03, 1.62701783e+03],
#                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
#            'dist1': np.array([[1.08458092e-03, -1.46765523e+00, -4.31746568e-03, 8.08144707e-04, 1.26288403e+01]]),
#            'dist2': np.array([[-4.01207645e-02, -4.86349653e-01, 3.39117551e-03, -9.56593330e-05, 4.24637107e+00]]),
#            'R': np.array([[0.99943117, -0.00303017, -0.03358791],
#                           [-0.00371039, 0.98002908, -0.19881962],
#                           [0.03351959, 0.19883115, 0.97946037]]),
#            'T': np.array([[-0.11844889], [1.6333303], [0.04970491]]),
#            'E': np.array([[0.05493298, 0.27604469, 1.6096646], [0.05364699, 0.02340072, 0.11434651],[-1.63196172, -0.11113409, 0.07841012]]),
#            'F': np.array([[9.33392036e-09, 4.67752622e-08, 2.27854392e-03],
#                           [9.09706623e-09, 3.95722786e-09, 1.42573854e-04],
#                           [-2.40758814e-03, -2.71374248e-04, 1.00000000e+00]]),
#            'dims': (4096, 3072)}

# # CHARUCO 41 Paare
# camera1 =  {'rms': 0.6321732817249623,
#             'M1': np.array([[8.03114872e+03, 0.00000000e+00, 2.28022179e+03],
#                        [0.00000000e+00, 8.03197699e+03, 1.40290208e+03],
#                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
#             'M2': np.array([[7.80678115e+03, 0.00000000e+00, 2.40782644e+03],
#                            [0.00000000e+00, 7.82023187e+03, 1.23287350e+03],
#                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
#             'dist1': np.array([[-5.90615219e-02, -1.13864245e+00, -1.24115531e-03, 2.31612909e-03,  9.10422101e+00]]),
#             'dist2': np.array([[-0.1567977 ,  0.18735536, -0.00396937, -0.0019885 , -0.10360508]]),
#             'R': np.array([[ 9.99127195e-01, -5.91683547e-03, -4.13502019e-02],
#                            [-3.14116954e-04,  9.88824838e-01, -1.49081658e-01],
#                            [ 4.17701984e-02,  1.48964528e-01,  9.87959928e-01]]),
#             'T': np.array([[-0.12983819],
#                        [ 1.53116863],
#                        [-0.18807782]]),
#             # 'T': np.array([[-1.53116863],
#             #             [0.12983819],
#             #             [0.18807782]]),
#             'E': np.array([[ 0.06389814,  0.41406583,  1.4846943 ],
#                           [-0.1824903 ,  0.02045411,  0.13605198],
#                           [-1.52979143, -0.11932755,  0.08267062]]),
#             'F': np.array([[ 1.29363598e-08,  8.38201598e-08,  2.26691809e-03],
#                            [-3.68821332e-08,  4.13344395e-09,  2.99130996e-04],
#                            [-2.40352056e-03, -3.95498761e-04,  1.00000000e+00]]), 'dims': (4096, 3072)}

# # CHARUCO 27 Paare
camera1 =  {'rms': 0.23029411670690533,
            'M1': np.array([[7.76926961e+03, 0.00000000e+00, 2.18344842e+03],
                            [0.00000000e+00, 7.77186856e+03, 1.47653162e+03],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            'M2': np.array([[7.78745513e+03, 0.00000000e+00, 2.12119110e+03],
                            [0.00000000e+00, 7.78760033e+03, 1.58820078e+03],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            'dist1': np.array([[ 2.15411484e+01, -7.52984445e-01, -4.77100616e-04,
                                -1.11004567e-03,  6.28611368e+01,  2.16371640e+01,
                                 2.20579706e+00,  5.95536409e+01,  0.00000000e+00,
                                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                 0.00000000e+00,  0.00000000e+00]]),
            'dist2': np.array([[-8.21074720e+00, -7.73014559e+01,  3.25460836e-03,
                                 4.04255109e-06, -7.28655606e+00, -8.11190274e+00,
                                -7.80827618e+01, -1.51635306e+01,  0.00000000e+00,
                                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                 0.00000000e+00,  0.00000000e+00]]),
            'R': np.array([[ 0.99921903, -0.00389242,  0.03932152],
                           [-0.00215945,  0.98826975,  0.15270309],
                           [-0.03945466, -0.15266875,  0.98748954]]),
            'T': np.array([[ 0.09087192], [-1.45123094], [ 0.14659083]]),
            'E': np.array([[ 0.05757437,  0.07668633, -1.45546024],
                           [ 0.15006167,  0.01330271, -0.0839709 ],
                           [ 1.44990134,  0.08415718,  0.07094103]]),
            'F': np.array([[ 1.21159856e-08,  1.61325201e-08, -2.42990829e-03],
                           [ 3.15784784e-08,  2.79844190e-09, -2.10369283e-04],
                           [ 2.30024029e-03,  9.92057380e-05,  1.00000000e+00]]), 'dims': (4096, 3072)}

# Charuco camera pair 2
camera7 = {'rms': 0.6001401637562398,
           'M1': np.array([[8.15066773e+03, 0.00000000e+00, 2.27520316e+03],
                       [0.00000000e+00, 8.15066773e+03, 1.41852113e+03],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
           'M2': np.array([[8.25986570e+03, 0.00000000e+00, 2.34313795e+03],
                       [0.00000000e+00, 8.25986570e+03, 1.52131470e+03],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
           'dist1': np.array([[-6.44207147e-02, -2.11755525e+00, -1.21034426e-03, 2.50408304e-03,  2.63932819e+01]]),
           'dist2': np.array([[-0.10662915,  0.04702318, -0.00432018,  0.00430638, -0.4923346 ]]),
           'R': np.array([[ 0.99985754,  0.01520622, -0.00732576],
                       [-0.01640778,  0.97745836, -0.21048977],
                       [ 0.00395987,  0.21057999,  0.97756861]]),
           'T': np.array([[0.02664127],
                       [1.47886014],
                       [0.02899261]]),
           'E': np.array([[ 0.00961336,  0.0875876 ,  1.49388785],
                       [ 0.22885449, -0.00212801, -0.02772121],
                       [-1.47908659,  0.00355286,  0.00522606]]),
           'F': np.array([[ 9.15849137e-09,  8.34432770e-08,  1.14608474e-02],
                       [ 2.18025935e-07, -2.02731674e-09, -7.08432920e-04],
                       [-1.19921357e-02, -1.64477318e-04,  1.00000000e+00]]),
           'dims': (4096, 3072)}
camera4 = {'rms': 0.6001401637562398,
           'M1': np.array([[8.15066773e+03, 0.00000000e+00, 2.27520316e+03],
                       [0.00000000e+00, 8.15066773e+03, 1.41852113e+03],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
           'M2': np.array([[8.25986570e+03, 0.00000000e+00, 2.34313795e+03],
                       [0.00000000e+00, 8.25986570e+03, 1.52131470e+03],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
           'dist1': np.array([[-6.44207147e-02, -2.11755525e+00, -1.21034426e-03,
                            2.50408304e-03,  2.63932819e+01]]),
           'dist2': np.array([[-0.10662915,  0.04702318, -0.00432018,  0.00430638, -0.4923346 ]]),
           'R': np.array([[ 0.99985754,  0.01520622, -0.00732576],
                       [-0.01640778,  0.97745836, -0.21048977],
                       [ 0.00395987,  0.21057999,  0.97756861]]),
           'T': np.array([[0.02664127],
                       [1.47886014],
                       [0.22899261]]),
           'E': np.array([[ 0.00961336,  0.0875876 ,  1.49388785],
                       [ 0.22885449, -0.00212801, -0.02772121],
                       [-1.47908659,  0.00355286,  0.00522606]]),
           'F': np.array([[ 9.15849137e-09,  8.34432770e-08,  1.14608474e-02],
                       [ 2.18025935e-07, -2.02731674e-09, -7.08432920e-04],
                       [-1.19921357e-02, -1.64477318e-04,  1.00000000e+00]]),
           'dims': (4096, 3072)}

# camera5 = {'rms': 0.6184895242917753, 'M1': np.array([[8.27834552e+03, 0.00000000e+00, 1.54767841e+03],
#        [0.00000000e+00, 8.28051220e+03, 1.79568282e+03],
#        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), 'M2': np.array([[8.27834552e+03, 0.00000000e+00, 1.54767841e+03],
#        [0.00000000e+00, 8.28051220e+03, 1.79568282e+03],
#        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), 'dist1': np.array([[-0.13254596,  0.35538618, -0.00261707, -0.00201904,  0.        ]]), 'dist2': np.array([[-0.06980564, -0.22430307, -0.00541761, -0.00367005,  0.        ]]), 'R': np.array([[ 0.98021208,  0.0169093 , -0.19722667],
#        [-0.01758684,  0.99984392, -0.00168424],
#        [ 0.19716741,  0.00511951,  0.98035647]]), 'T': np.array([[ 1.49422874],
#        [-0.02040696],
#        [ 0.1239421 ]]), 'E': np.array([[-0.00184384, -0.12402722, -0.01979735],
#        [-0.17312367, -0.00555394, -1.48932149],
#        [-0.00627561,  1.49434059, -0.00654144]]), 'F': np.array([[ 1.77847340e-09,  1.19599160e-07, -5.94352578e-05],
#        [ 1.66942745e-07,  5.35424881e-09,  1.16209512e-02],
#        [-2.52418703e-04, -1.21237214e-02,  1.00000000e+00]]), 'dims': (3072, 4096)}


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
# # imgl = cv2.imread('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\22-02-25_12-45-24_1.jpg', 0)
# # imgr = cv2.imread('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\22-02-25_12-45-24_2.jpg', 0)
# imgl = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\22-04-21_15-19-321.png', -1)
# imgr = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\22-04-21_15-19-322.png', -1)
# imgl = cv2.imread('C:\\Users\\wyfmi\\Downloads\\opencv-4.5.5\\opencv-4.5.5\\samples\\data\\aloeL.jpg')
# imgr = cv2.imread('C:\\Users\\wyfmi\\Downloads\\opencv-4.5.5\\opencv-4.5.5\\samples\\data\\aloeR.jpg')

imgl = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\22-04-21_15-20-441 - Kopie.png')
# imgl = np.rot90(imgl)
imgr = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\22-04-21_15-20-442 - Kopie.png')
# imgr = np.rot90(imgr)
# imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
# imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
# iml_, imr_ = preprocess(imgl, imgr)
stereo5.rectify_image(imgl, imgr)

# # imgl = cv2.imread('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\result1.png')
# # imgr = cv2.imread('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\result2.png')
imgl = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\result1.png')
imgr = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\result2.png')
stereo5.draw_line(imgl, imgr)
# imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
# imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
# iml_, imr_ = preprocess(imgl, imgr)
stereo5.stereo_matchSGBM(imgl, imgr)
# stereo5.depth_map(imgl, imgr)


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
    blockSize = 15  # 3
    paraml = cv2.StereoSGBM_create(
        minDisparity=-64,
        numDisparities=15 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=9,
        # P1=8 * img_channels * blockSize ** 2,
        P1=100,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        # P2=32 * img_channels * blockSize ** 2,
        P2=1000,
        disp12MaxDiff=-1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # cread the SGBM instance
    left_matcher = cv2.StereoSGBM_create(matcher_left=left_matcher)
    # paramr = paraml
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
    displ = left_matcher.compute(left_image, right_image)  #.astype(np.float32)/16
    dispr = right_matcher.compute(right_image, left_image)  #.astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, left_image, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    cv2.imshow('Disparity Map', filteredImg)
    cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-03-18_Kalibrierbilder\\depthmap.png', filteredImg)
    cv2.waitKey()
    cv2.destroyAllWindows()

def stereo_matchBM(left_image, right_image, down_scale=False):
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


    num = 5
    blockSize = 15
    units = 0.001
    fx = 0
    baseline = 0

    paraml = {'minDisparity' : 0,
              'numDisparities' : num * 16,
              'blockSize' : 9  # 3
              }

    # cread the SGBM instance
    left_matcher = cv2.StereoBM_create(numDisparities, blockSize)
    # paramr = paraml
    # paramr['minDisparity'] = -paraml['numDisparities']
    # paramr['minDisparity'] = -15*16
    # right_matcher = cv2.StereoBM_create(-numDisparities, blockSize)

    # Filter Parameters
    # lmbda = 100000
    # sigma = [0.7, 1.2, 1.5]
    # sigma = 1.3
    # visual_multiplier = 1.0  # 1.0
    #
    # wsl filter are used to smooth the depthmap
    # wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    # wls_filter.setLambda(lmbda)
    # wls_filter.setSigmaColor(sigma)

    # calculate the disparity map
    print('computing disparity...')
    displ = left_matcher.compute(left_image, right_image)  # .astype(np.float32)/16
    # dispr = right_matcher.compute(right_image, left_image)  # .astype(np.float32)/16
    depth = np.zeros(shape=left.shape).astype(float)
    depth[disparity > 0] = (fx * baseline) / (units * disparity[disparity > 0])
    # displ = np.int16(displ)
    # dispr = np.int16(dispr)
    # filteredImg = wls_filter.filter(displ, left_image, None, dispr)  # important to put "imgL" here!!!

    # filteredImg = cv2.normalize(src=displ, dst=displ, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U);
    # filteredImg = np.uint8(filteredImg)
    # cv2.imshow('Disparity Map', filteredImg)
    # cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-03-18_Kalibrierbilder\\depthmap.png', filteredImg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


def stereo_match(imgL, imgR):
    # disparity range is tuned for 'aloe' image pair
    window_size = 5
    min_disp = 100
    num_disp = 10 * 16
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=16,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=150,
                                   speckleRange=32
                                   )

    # print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16

    # print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    # f = 0.8 * w  # guess for focal length
    # Q = np.float32([[1, 0, 0, -0.5 * w],
    #                 [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
    #                 [0, 0, 0, -f],  # so that y-axis looks up
    #                 [0, 0, 1, 0]])

    Q = np.array([[1, 0, 0, -2.048e+03],
	              [0, 1, 0, -1.536e+03],
	              [0, 0, 0, 7.78e+03],
	              [0, 0, -6.67e-01,0.2214]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    # append_ply_array(out_points, out_colors)

    disparity_scaled = (disp - min_disp) / num_disp
    disparity_scaled += abs(np.amin(disparity_scaled))
    disparity_scaled /= np.amax(disparity_scaled)
    disparity_scaled[disparity_scaled < 0] = 0
    cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-03-18_Kalibrierbilder\\depthmap.png', 255 * disparity_scaled)
    return np.array(255 * disparity_scaled, np.uint8)


# imgl = np.array(imgl, dtype = np.uint8)
# imgr = np.array(imgr, dtype = np.uint8)
# #
#
# stereo1.draw_line(imgl, imgr)
# imgl = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\result1.png')
# imgr = cv2.imread('C:\\Users\\wyfmi\\Desktop\\0422\\result2.png')
# imgl = cv2.imread('C:\\Users\\wyfmi\\Downloads\\opencv-4.5.5\\opencv-4.5.5\\samples\\data\\aloeL.jpg')
# imgr = cv2.imread('C:\\Users\\wyfmi\\Downloads\\opencv-4.5.5\\opencv-4.5.5\\samples\\data\\aloeR.jpg')
# stereo_matchSGBM(imgl, imgr)

# # depth_img = stereo3.depth_map(imgl, imgr)
# stereo_matchSGBM(iml_, imr_)
# d = stereo_match(iml_, imr_)
# disp = cv2.imread('C:\\Users\\wyfmi\\Desktop\\2022-03-18_Kalibrierbilder\\depthmap.png', 0)
# disp, _ = stereo3.stereoMatchSGBM(imgl, imgr)
# depthmap = cv2.imread('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\depthmap.png', 0)
# depthmap = cv2.imread('C:\\Users\\wyfmi\\Desktop\\2022-03-18_Kalibrierbilder\\depthmap.png', 0)
# cv2.imshow('depth_map1', depthmap)
# cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\left.png', disp)
# cv2.namedWindow("disparity",0)
# cv2.imshow("disparity", disp)


# cv2.waitKey(8000)
#
# cv2.imshow('depth_map2', r)
# cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\right.png', r)
# cv2.waitKey(8000)
# cv2.imshow('depth_map', depth_img)
# cv2.waitKey(8000)
# plt.imshow(depth_img, cmap='plasma')
# plt.colorbar()
# plt.show()

# cv2.imshow('depth_map2', right)
# cv2.waitKey(8000)
# pointImage = cv2.reprojectImageTo3D(depthmap.astype(np.float32)/16, rectify_model['Q'] )
# print('3D data:', pointImage)
# print('process finished')
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(pointImage)
# # point_cloud.colors = o3d.utility.Vector3dVector(colors / 255)
# o3d.visualization.draw_geometries([point_cloud])

def hw3ToN3(points):
    height, width = points.shape[0:2]

    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)

    points_ = np.hstack((points_1, points_2, points_3))

    return points_

def DepthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    size = rows * cols

    points_ = hw3ToN3(points_3d)
    print('points_:', points_, points_.shape)
    colors_ = hw3ToN3(colors).astype(np.int64)
    print('colors_:', colors_.shape)

    # 颜色信息
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)

    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)

    # 将坐标+颜色叠加为点云数组
    pointcloud = np.hstack((points_, rgb)).astype(np.float32)

    # 删掉一些不合适的点
    X = pointcloud[:, 0]
    Y = -pointcloud[:, 1]
    Z = -pointcloud[:, 2]

    # remove_idx1 = np.where(Z <= 0)
    # remove_idx2 = np.where(Z > 15000)
    # remove_idx3 = np.where(X > 10000)
    # remove_idx4 = np.where(X < -10000)
    # remove_idx5 = np.where(Y > 10000)
    # remove_idx6 = np.where(Y < -10000)
    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)

    remove_idx = np.hstack((remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

    pointcloud_1 = np.delete(pointcloud, remove_idx, 0)

    return points_, rgb

def view_cloud(pointcloud):
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_array(pointcloud)

    try:
        visual = pcl.pcl_visualization.CloudViewing()
        visual.ShowColorACloud(cloud)
        v = True
        while v:
            v = not (visual.WasStopped())
    except:
        pass

# points_3d = cv2.reprojectImageTo3D(disp, rectify_model['Q'])
#
# print('points_3d:', points_3d.shape)
# img = cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB)
# print('img:', img.shape)
# # colors = img.reshape(-1, 3)
# pointcloud_, colors_ = DepthColor2Cloud(points_3d, img)
# # view_cloud(pointcloud)
#
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(pointcloud_)
# point_cloud.colors = o3d.utility.Vector3dVector((colors_) / 255)
# o3d.visualization.draw_geometries([point_cloud])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cam_6 = Camera(6)
# path, path_l, path_r = cam_6.get_dir
# read_all = PhotoSet(path, path_l, path_r)
# imma1, imma2 = read_all.image_to_matrix
# square = CameraCalibration("circle", imma1, imma2, path)
# camera1 = square.method_manager(7, 7, 0.1)
# print('camera1:', camera1)

# cml = np.array([[],[],[]])
# cmr = np.array([[],[],[]])
