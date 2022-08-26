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
import json
import pickle

devices = list(i + 1 for i in range(6))

############################################

class Camera:
    """

    Create folders for image and result storage. Each folder stores several pairs of images taken by a camera pair.
    Each calibration result, including intrinsic and extrinsic parameters.
    And the corresponding rectified image pair will be saved here.

    """

    def __init__(self, device_num):
        """Constructor function.

        Parameters
        ----------
        device_num : int
            ID of one camera pair. Here are 12 camera pairs. 12 is the largest available ID.

        """

        self._id = device_num

    @property
    def get_id(self):
        """Get the ID of camera pair.

        Returns
        -------
        self._id : int
            ID of the camera pair.

        """

        return self._id

    @property
    def get_dir(self):
        """Get the directories to storage the images.

        Folder operation. Get the current and upper directories.
        Create a corresponding folder under a certain ID to store the images that used for calibration.
        The structure of the folder is like this.

        Returns
        -------
        path : string
            the path of the folder which stores all the images of the camera pair.
        path_resize_l : string
            path of the sub folder which stores all the images taken by the left camera.
        path_resize_r : string
            path of the sub folder which stores all the images taken by the right camera.

        """

        # get the current working directory.
        cur_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        # get the ID of camera pair. There are 12 pairs of stereo camera pair.
        device = self.get_id
        print('--------------------------------------------------------------')
        print("preparing for the calibration of device #{}:".format(device))
        print('--------------------------------------------------------------')
        # get the path of the directory to store the calibration images.
        path = os.path.join(cur_dir, 'images', 'device' + str(device))
        # create 2 sub folders to store the left- and right photos individually.
        # The intrinsic reference matrix of the left and right cameras will be calculated separately first.
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
            print("The calibration photos of left camera will be saved in the folder:", path_resize_l)
        if os.path.isdir(path_resize_r):
            print("The calibration photos of right camera will be saved in the folder:", path_resize_r)
        return path, path_resize_l, path_resize_r


#############################################

class PhotoSet:
    """Convert the images to python numpy arrays.

    Images operation. Store the photos taken by the left and right cameras to the corresponding sub folders.
    Read these two sets of photos separately and concatenate them as a numpy array.

    """

    def __init__(self, path, save_dir_l, save_dir_r):
        """Constructor function.

        Parameters
        ----------
        path : string
            path of the folder which stores all the images of the camera pair.
        save_dir_l : string
            path of the sub folder which stores all the images taken by the left camera.
        save_dir_r : string
            path of the sub folder which stores all the images taken by the right camera.
        """

        # Image storage folder.
        self._path = path
        # Build 2 sub folders to store the left and right images.
        # Build 2 lists to store the image matrices.
        self._save_dir_l = save_dir_l
        self._save_dir_r = save_dir_r
        self._images_l = []
        self._images_r = []

    def resize(self, img):
        """Change the image resolution to smaller.

        Used to resize the original image.
        In order to preserve the calibration accuracy, it is not recommended to resize the original images.
        The resized image can lead to subsequent calibration accuracy and disparity map calculations with large errors.

        Parameters
        ----------
        img : list(array)
            the images need to be resized.

        Returns
        -------
        img_resize : list(array)
            the resized image with a lower resolution.

        """
        img_resize = cv2.resize(img, (1024, 768))
        return img_resize

    def image_to_matrix(self, exist_dict=False):
        """Convert and save the images in a list of array.

        All the images taken by the same camera will be read as "np.array" format and stored as a python "list".

        .. important::
            Please notice the layout of the camera pair. If the cameras are horizontal placed, no need to rotate the
            images. If the camera pair is vertical placed, the images need to be rotated for 90 degrees or 270 degrees.

        .. image:: /_static/cameralayout1.png
            :scale: 90 %
            :align: center
            :name: horizontal layout.

        |

        .. image:: /_static/cameralayout2.png
            :scale: 90 %
            :align: center
            :name: vertical layout.

        |

        Returns
        -------
        self._images_l : list(array)
            an array containing all the image of the left camera.
        self._images_r : list(array)
            an array containing all the image of the right camera.

        """

        if exist_dict == False:
            # Read all the photos under the sub folder of left camera.
            # Please note the format of the image (jpg or png).
            image_list_l = sorted(glob.glob(self._path + '//' + '*_2.jpg'))
            for i, image in enumerate(image_list_l):
                write_name = self._save_dir_l + '//' + 'Image_' + str(i) + '.png'
                img = cv2.imread(image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Rotation angle need to be determined according to the camera layout.
                # If the stereo camera are horizontally aligned, images need not to be rotated.
                # If the stereo camera are vertically aligned, images need to be rotated 90 or 270 degrees.
                img = np.rot90(img, 3)
                cv2.imwrite(write_name, img)
                # Append all the image matrix as a list.
                self._images_l.append(img)
            print("{} photos in left directory are read".format(len(self._images_l)))

            # Read all the photos under the sub folder of left camera.
            image_list_r = sorted(glob.glob(self._path + '//' + '*_1.jpg'))
            for j, image in enumerate(image_list_r):
                write_name = self._save_dir_r + '//' + 'Image_' + str(j) + '.png'
                img = cv2.imread(image)
                img = np.rot90(img, 3)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(write_name, img)
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

        print("{} photos in right directory are read".format(len(self._images_r)))
        return self._images_l, self._images_r


#############################################

class CameraCalibration:
    """Intrinsic and extrinsic calibration.

    Intrinsic camera calibration. intrinsic parameters, such as camera matrix and distortion will be determined.
    Stereo camera calibration. extrinsic parameters, such as Rotation and Translation between the camera pair
    will be calculated.
    After calibration, the intrinsic and extrinsic will be saved as ”.txt“ and ”.pickle“ files.
    ”.txt“ is used for reading. ”.pickle“ is easy for data extraction.
    Pattern selections: pattern = ["square", "circle", "charuco"]

    """

    def __init__(self, pattern_choice, image_list_l, image_list_r, save_path):
        """Constructor function.

        Initialize the images used for calibration.

        Parameters
        ----------
        pattern_choice : string
            3 kinds of calibration patterns can be used: square patter, circle pattern and charuco pattern.
        image_list_l : list(array)
            python list includes all the images as "np.array" which taken by the left camera.
        image_list_r : list(array)
            python list includes all the images as "np.array" which taken by the right camera.

        """
        # Initialize Function criteria for intrinsic and stereo(extrinsic) calibration.
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        self.pattern = pattern_choice
        self._image_list_l = image_list_l
        self._image_list_r = image_list_r
        # used for storage of the object points in thr world coordination.
        # Suitable for "square", "circle" and "charuco" patterns.
        self.obj_points = []
        # used for storage of the image points in the image coordination. Suitable for "square", "circle" patterns.
        self.img_points_l = []
        self.img_points_r = []
        # used for storage of the image points and ID of the "charuco" pattern in the image coordination.
        self.all_corners_l = []
        self.allIds_l = []
        self.all_corners_r = []
        self.allIds_r = []
        self.save_path = save_path

    @staticmethod
    def gray_scale(img):
        """Convert image to grayscale.

        Returns
        -------
        gray : array
            image of grayscale.

        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def resize(gray_image):
        """Change image size to image shape.

        Different from OpenCV's cv2.resize function, this function is not used to modify the original size of the image.

        Returns
        -------
        resize : array
            shape of the image.

        """
        resize = gray_image.shape[::-1]
        return resize

    @staticmethod
    def binarization(gray_image):
        """Apply fixed-level thresholding to a multiple-channel array.

        Returns
        -------
        th1 : array
            image after binarization.

        """
        ret, th1 = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY)
        # th1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imshow('binarization:', th1)
        return th1

    def _square_calibration(self, w_nums, h_nums, length):
        """Intrinsic and Extrinsic calibration with square pattern.

        Please make sure the number of left and right images are the same.
        Please make sure the number of every pair of images are the same.
        please make sure the predefined world point numbers are the same as the detected point numbers.

        Parameters
        ----------
        w_nums : int
            number of the inside row corners of the chessboard.
        h_nums : int
            number of the inside column corners of the chessboard.
        length : int
            distance of adjacent corner points in meters.

        Returns
        -------
        intrinsic parameters of camera pairs:
        self.mtx_l : 3 x 3 matrix
            camera matrix of left camera.
        self.dist_l : 1 x 5 vector
            distortion matrix of left camera.
        self.rvecs_l : This one is not needed in this project.
            rotation vector of extrinsic parameters.
        self.tvecs_l : This one is not needed in this project.
            translation vector extrinsic parameters.
        resize_l :
            the shape of the image.

        The right camera is the same.

        """
        print(r"chessboard with the height {} and the width {} is chosen".format(w_nums, h_nums))
        # Create corner points in the world coordination.
        self.world_points = np.zeros((w_nums * h_nums, 3), np.float32)
        self.world_points[:, :2] = np.mgrid[0:w_nums, 0:h_nums].T.reshape(-1, 2)

        # The ground truth of the camera intrinsic matrix.
        # The calculation of the intrinsic parameters will be based on the iteration of this matrix.
        # Assume the image height > width.
        # If  width > height, term [0][0] and [1][2] need to be changed.
        camera_matrix = np.array([[7.77777778e+03, 0., 1.526e+03],
                                  [0., 7.77777778e+03, 2.048e+03],
                                  [0.0, 0.0, 1.0]])

        for f_name in self._image_list_l:
            gray = self.gray_scale(f_name)
            resize_l = self.resize(gray)
            ret, corners_l = cv2.findChessboardCorners(gray, (w_nums, h_nums), None)
            if ret:
                # Setting pixel found condition
                exact_corners = cv2.cornerSubPix(gray, corners_l, (5, 5), (-1, -1), self.criteria)
                self.obj_points.append(self.world_points * length)
                self.img_points_l.append(exact_corners)

            # # Draw and display the detected corners at the calibration pattern.
            #     cv2.drawChessboardCorners(f_name, (w_nums, h_nums), corners_l, ret)
            #     cv2.imshow('findCorners', f_name)
            #     cv2.waitKey(3000)
            # cv2.destroyAllWindows()

        for f_name in self._image_list_r:
            gray = self.gray_scale(f_name)
            resize_r = self.resize(gray)
            ret, corners_r = cv2.findChessboardCorners(gray, (w_nums, h_nums), None)
            if ret:
                # Setting pixel found condition
                exact_corners = cv2.cornerSubPix(gray, corners_r, (5, 5), (-1, -1), self.criteria)
                self.img_points_r.append(exact_corners)

            #     cv2.drawChessboardCorners(f_name, (w_nums, h_nums), corners_r, ret)
            #     cv2.imshow('findCorners2', f_name)
            #     cv2.waitKey(3000)
            # cv2.destroyAllWindows()

        # Left and right camera are calibrated separately to obtain the intrinsic parameters.
        # See OpenCV tutorial for the usage of the functions.
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
        ret_l, self.mtx_l, self.dist_l, self.rvecs_l, self.tvecs_l = cv2.calibrateCamera(
            self.obj_points, self.img_points_l, resize_l, cameraMatrix=camera_matrix, distCoeffs=None, flags=0,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 100, 1e-5))
        ret_r, self.mtx_r, self.dist_r, self.rvecs_r, self.tvecs_r = cv2.calibrateCamera(
            self.obj_points, self.img_points_r, resize_r, cameraMatrix=camera_matrix, distCoeffs=None, flags=0,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 100, 1e-5))
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(self.obj_points, self.img_points_l, self.img_points_r,
                                                              self.mtx_l, self.dist_l, self.mtx_r, self.dist_r,
                                                              (3072, 4096),
                                                              criteria=self.criteria_stereo,
                                                              flags=cv2.CALIB_FIX_INTRINSIC)
        self.camera_model = dict([('rms', ret), ('M1', M1), ('M2', M2), ('dist1', d1),
                                  ('dist2', d2), ('R', R), ('T', T),
                                  ('E', E), ('F', F), ('dims', resize_l)])

        return resize_l

    def _charuco_calibration(self, w_nums, h_nums, length):
        """Intrinsic and Extrinsic calibration with charuco pattern.
        Create ChArUco pattern and implement the intrinsic and extrinsic calibration

        Parameters
        ----------
        w_nums : int
            number of the inside row corners of the ChArUco board.
        h_nums : int
            number of the inside column corners of the ChArUco board.
        length : int
            distance of adjacent corner points in meters.

        Returns
        -------
        The same as in Function "_square_calibration".

        """
        # Create ChArUco pattern from the ChArUco dictionary. See OpenCV tutorial.
        # https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gaf5d7e909fe8ff2ad2108e354669ecd17
        dictionary = cv2.aruco.getPredefinedDictionary(dict=cv2.aruco.DICT_6X6_250)
        board = cv2.aruco.CharucoBoard_create(squaresY=6, squaresX=6, squareLength=0.15, markerLength=0.075,
                                              dictionary=dictionary)
        img_board = board.draw(outSize=(4096, 4096), marginSize=None, borderBits=None)

        self.world_points = np.zeros((w_nums * h_nums, 3), np.float32)
        self.world_points[:, :2] = np.mgrid[0:w_nums, 0:h_nums].T.reshape(-1, 2)

        # The ground truth of the camera intrinsic matrix.
        # As described in Function "_square_calibration", stereo camera layout need to be noticed.
        cameramatrix = np.array([[7.77777778e+03, 0., 1.526e+03],
                                 [0., 7.77777778e+03, 2.048e+03],
                                 [0.0, 0.0, 1.0]])

        # Set flags to OpenCV Function "stereoCalibrate".
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC

        print(r"ChArUco with the height {} and the width {} is chosen".format(w_nums, h_nums))

        i = 0
        for f_name in self._image_list_l:
            i = i + 1
            gray = self.gray_scale(f_name)
            resize_l = self.resize(gray)
            print('resize_l', resize_l)
            corners_l, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)
            if corners_l == None or len(corners_l) == 0:
                continue
            ret, charucoCorners_l, charucoIds_l = cv2.aruco.interpolateCornersCharuco(corners_l, ids, gray, board)
            print(i, ret)
            if corners_l is not None and charucoIds_l is not None:
                # Setting pixel found condition
                exact_corners = cv2.cornerSubPix(gray, charucoCorners_l, (11, 11), (-1, -1), self.criteria)
                self.obj_points.append(self.world_points * length)
                self.all_corners_l.append(charucoCorners_l)
                self.allIds_l.append(charucoIds_l)

            # # Draw detected features on image.
            #     cv2.drawChessboardCorners(f_name, (w_nums, h_nums), charucoCorners_l, ret)
            #     cv2.namedWindow('findCorners', cv2.WINDOW_FREERATIO)
            #     cv2.imshow('findCorners', f_name)
            #     cv2.waitKey(5000)
            # cv2.destroyAllWindows()

        j = 0
        for f_name in self._image_list_r:
            j = j + 1
            gray = self.gray_scale(f_name)
            resize_r = self.resize(gray)
            corners_r, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)
            if corners_r == None or len(corners_r) == 0:
                continue
            ret, charucoCorners_r, charucoIds_r = cv2.aruco.interpolateCornersCharuco(corners_r, ids, gray, board)
            print(j, ret)
            if corners_r is not None and charucoIds_r is not None:
                # Setting pixel found condition
                exact_corners = cv2.cornerSubPix(gray, charucoCorners_r, (11, 11), (-1, -1), self.criteria)
                self.all_corners_r.append(charucoCorners_r)
                self.allIds_r.append(charucoIds_r)

                # # Draw detected features on image.
                #     cv2.drawChessboardCorners(f_name, (w_nums, h_nums), charucoCorners_l, ret)
                #     cv2.namedWindow('findCorners', cv2.WINDOW_FREERATIO)
                #     cv2.imshow('findCorners', f_name)
                #     cv2.waitKey(5000)
                # cv2.destroyAllWindows()

        # Left and right camera are calibrated separately to obtain the intrinsic parameters.
        ret_l, self.mtx_l, self.dist_l, self.rvecs_l, self.tvecs_l = cv2.calibrateCamera(
            self.obj_points, self.all_corners_l, resize_l, cameraMatrix=cameramatrix, distCoeffs=None, flags=0,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 100, 1e-5))
        ret_r, self.mtx_r, self.dist_r, self.rvecs_r, self.tvecs_r = cv2.calibrateCamera(
            self.obj_points, self.all_corners_r, resize_r, cameraMatrix=cameramatrix, distCoeffs=None, flags=0,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 100, 1e-5))

        # Stereo calibration to obtain the extrinsic parameters.
        # Setting iteration termination conditions.
        # See OpenCV tutorial for the usage of the functions.
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(self.obj_points, self.all_corners_l, self.all_corners_r,
                                                              self.mtx_l, self.dist_l, self.mtx_r, self.dist_r,
                                                              resize_l,
                                                              criteria=stereocalib_criteria, flags=flags)

        self.camera_model = dict([('rms', ret), ('M1', M1), ('M2', M2), ('dist1', d1),
                                  ('dist2', d2), ('R', R), ('T', T),
                                  ('E', E), ('F', F), ('dims', resize_l)])
        print(type(self.camera_model))

        return resize_l

    def distortion(self, img, mtx, dist):
        """Refine the camera matrix.
        Refine the obtained camera matrix.
        Get the cropped valid area of the images.

        Parameters
        ----------
        mtx : 3 x 3  matrix
            camera intrinsic matrix.
        dist : 1 x 5 vector
            distortion vector.

        Returns
        -------
        newcameramatrix : 3 x 3  matrix
            the refined camera matrix.

        """
        img_dis = cv2.imread(img)
        h, w = img_dis.shape[:2]
        newcameramatrix, roi = cv2.getOptimalNewMatrix(mtx, dist, (w, h), 0, (w, h))
        dst = cv2.undistort(img_dis, self.mtx, self.dist, None, newcameramatrix)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv2.imshow("roi", dst)
        return newcameramatrix

    @property
    def validate_square(self):
        """Compute and print RMS errors of each left image.

        RMS is used for indicate the accuracy of the detected features on the calibration pattern.
        Calculate and display the re-projection error to each image of left camera.

        """
        total_error = 0
        for i in range(len(self.obj_points)):
            img_reproject, _ = cv2.projectPoints(self.obj_points[i], self.rvecs_l[i], self.tvecs_l[i], self.mtx_l,
                                                 self.dist_l)
            error = cv2.norm(self.all_corners_l[i], img_reproject, cv2.NORM_L2)
            total_error += error * error
            print('error', i, error / np.sqrt(len(img_reproject)))
        print("rms error1: {}".format(np.sqrt(total_error / (len(self.obj_points) * len(img_reproject)))))

    @property
    def validate_square_2(self):
        """Compute and print RMS errors of each right image.

        RMS is used for indicate the accuracy of the detected features on the calibration pattern.
        Calculate and display the re-projection error to each image of right camera.

        """
        total_error = 0
        for i in range(len(self.obj_points)):
            img_reproject, _ = cv2.projectPoints(self.obj_points[i], self.rvecs_r[i], self.tvecs_r[i], self.mtx_r,
                                                 self.dist_r)
            error = cv2.norm(self.all_corners_r[i], img_reproject, cv2.NORM_L2)
            total_error += error * error
            error_single = error / np.sqrt(len(img_reproject))
            print('error', i, error / np.sqrt(len(img_reproject)))
        print("rms error2: {}".format(np.sqrt(total_error / (len(self.obj_points) * len(img_reproject)))))

    def _circle_calibration(self, points_per_row, points_per_column, c_distance):
        """Intrinsic and Extrinsic calibration with circle pattern.
        Calibration with circle pattern.
        Blob detector can be set to enable or disable as required.
        The detailed OpenCV functions information can be referred to OpenCV website.
        https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gaf5d7e909fe8ff2ad2108e354669ecd17

        Parameters
        ----------
        points_per_row : int
            number of the row circle centers.
        points_per_column : int
            number of the column circle centers.
        c_distance : int
            distance of 2 adjacent circle centers in meters.

        Returns
        -------
        The same as in Function "_square_calibration".

        """

        print(r"circle_pattern with circles {} * {} is chosen".format(points_per_row, points_per_column))
        self.world_points = np.zeros((points_per_row * points_per_column, 3), np.float32)
        self.world_points[:, :2] = np.mgrid[:points_per_row, :points_per_column].T.reshape(-1, 2)
        i = 0
        for f_name in self._image_list_l:
            i = i + 1
            gray = self.gray_scale(f_name)
            print('Picture brightness:', gray.mean())
            # gray = self.binarization(gray)
            thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)[1]
            resize_l = self.resize(thresh)
            ret, centers_l = cv2.findCirclesGrid(thresh, (points_per_row, points_per_column), None,
                                                 flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING))

            print(i, ret)
            if ret:
                # return centers
                self.obj_points.append(self.world_points * c_distance)
                exact_cornersl = cv2.cornerSubPix(thresh, centers_l, (11, 11), (-1, -1), self.criteria)
                self.img_points_l.append(exact_cornersl)

                cv2.drawChessboardCorners(f_name, (points_per_row, points_per_column), exact_cornersl, ret)
                cv2.namedWindow('result', cv2.WINDOW_FREERATIO)
                cv2.imshow('result', f_name)
                cv2.waitKey(3000)
            cv2.destroyAllWindows()

            # # Blobdetector used to draw circle and circle-centers.
            # # To perform a calibration with a blob detector, activate the remarks.
            # # Params need to be tested according to the size of the calibration pattern.
            # # Blob detector creation.
            # params = cv2.SimpleBlobDetector_Params()
            # # Set filters by area, circularity and other criterien.
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
            # # Visualize the circle centre.
            # center_points = []
            # for keypoint in keypoints:
            #     center_points.append(keypoint.pt)
            #     # print(keypoint.pt)
            #     cv2.circle(f_name, (int(keypoint.pt[0]), int(keypoint.pt[1])), 1, (0, 255, 0), 0)
            # with_keypoints = cv2.drawKeypoints(f_name, keypoints, np.array([]), (0, 0, 255),
            #                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #
            #
            # cv2.namedWindow("blob", cv2.WINDOW_FREERATIO)
            # cv2.imshow("blob", with_keypoints)
            # # cv2.imshow("corners", thresh)
            # cv2.waitKey(5000)
            # cv2.destroyAllWindows()

        j = 0
        for f_name in self._image_list_r:
            j = j + 1
            gray = self.gray_scale(f_name)
            print('Picture brightness:', gray.mean())
            # gray = self.binarization(gray)
            thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)[1]
            resize_r = self.resize(thresh)
            ret, centers_r = cv2.findCirclesGrid(thresh, (points_per_row, points_per_column), None,
                                                 flags=(cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING))

            print(j, ret)
            if ret:
                exact_cornersr = cv2.cornerSubPix(thresh, centers_r, (11, 11), (-1, -1), self.criteria)
                self.img_points_r.append(exact_cornersr)

                cv2.drawChessboardCorners(f_name, (points_per_row, points_per_column), exact_cornersr, ret)
                cv2.namedWindow('result2', cv2.WINDOW_FREERATIO)
                cv2.imshow('result2', f_name)
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
            #     cv2.circle(f_name, (int(keypoint.pt[0]), int(keypoint.pt[1])), 1, (0, 255, 0), 0)
            #     # print("keypoints:", keypoint.pt)
            # with_keypoints = cv2.drawKeypoints(f_name, keypoints, np.array([]), (0, 0, 255),
            #                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # cv2.namedWindow("blob2", cv2.WINDOW_FREERATIO)
            # cv2.imshow("blob2", with_keypoints)
            # cv2.waitKey(5000)
            # cv2.destroyAllWindows()


        ret_l, self.mtx_l, self.dist_l, self.rvecs_l, self.tvecs_l = cv2.calibrateCamera(
            self.obj_points, self.img_points_l, resize_l, None, None)
        ret_r, self.mtx_r, self.dist_r, self.rvecs_r, self.tvecs_r = cv2.calibrateCamera(
            self.obj_points, self.img_points_r, resize_r, None, None)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(self.obj_points, self.img_points_l, self.img_points_r,
                                                              self.mtx_l, self.dist_l, self.mtx_r, self.dist_r,
                                                              resize_l,
                                                              criteria=self.criteria_stereo,
                                                              flags=cv2.CALIB_FIX_INTRINSIC)
        self.camera_model = dict([('rms', ret), ('M1', M1), ('M2', M2), ('dist1', d1),
                                  ('dist2', d2), ('R', R), ('T', T),
                                  ('E', E), ('F', F), ('dims', resize_l)])

        print('ret', ret_l, ret_r)

        return resize_l

    @property
    def save_txt(self):
        """Save the intrinsic and extrinsic parameters as ".txt" file.

        Save the intrinsic and extrinsic parameters as ".txt" file.
        The txt file facilitates a manual confirmation of the intrinsic and extrinsic matrices.

        """
        filename = open(self.save_path + '\\' + 'intrinsic.txt', 'w')
        for k, v in self.camera_model.items():
            filename.write(k + ':' + str(v))
            filename.write('\n')
        print("Intrinsic parameters are saved as txt file.")
        filename.close()

    @property
    def save_pickle(self):
        """Save the intrinsic and extrinsic parameters as ".pickle" file.

        Save the intrinsic and extrinsic parameters as ".pickle" file.
        The pickle file facilitates the extraction of the parameters during running of codes.

        """
        save_file = open(self.save_path + '\\' + 'intrinsic.pickle', 'wb')
        pickle.dump(self.camera_model, save_file)
        print("Intrinsic parameters are saved as pickle file.")
        save_file.close()

    def method_manager(self, *args):
        """Manage the implementation of calibration.

        The calibration function can be invoked automatically by specifying the size of the calibration pattern.
        3 kinds of calibration patterns can be chosen.

        Parameters
        ----------
        args :
            please define the number of the row points(corners), column points(corners)
            and the distance of the adjacent points(corners) with the unit meter.

        Returns
        -------
        The camera model parameters.

        """
        # *args contains the size of the pattern.
        # arg[0] and arg[1] are the width and length. arg[2] are the distance of corners or circle centres.
        if self.pattern == "square":
            dims = self._square_calibration(args[0], args[1], args[2])

        elif self.pattern == "charuco":
            dims = self._charuco_calibration(args[0], args[1], args[2])
            # self.validate_square()
            # self.validate_square_2()
        else:
            dims = self._circle_calibration(args[0], args[1], args[2])
            # self.validate_square()
            # self.validate_square_2()
        self.save_txt
        self.save_pickle
        return self.camera_model


#####################################################
class StereoRectify:
    """Rectify the image pair using the obtained intrinsic and extrinsic parameters.

    Rectification of the image pair.
    Draw the epi-polar lines to simplify the search of the corresponding feature points in left and right image.

    """

    def __init__(self, camera_model, path):
        """Constructor function.

        """
        self.cameraMatrix_l = camera_model['M1']
        self.cameraMatrix_r = camera_model['M2']
        self.dist_l = camera_model['dist1']
        self.dist_r = camera_model['dist2']
        self.R = camera_model['R']
        self.T = camera_model['T']
        self.dims = camera_model['dims']
        # Save path for parallel correction images. The
        self.path = path
        print('self.dims:', self.dims)

    def get_rectify_transform(self):
        """Calculate the rectified parameters to implement the parallel correction.

        Get the rectified camera model. Preparing for the parallel correction.

        Returns
        -------
        rectify_model : dictionary

        """
        # Rectify the image pair.
        # See OpenCV function description.
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.cameraMatrix_l, self.dist_l, self.cameraMatrix_r,
                                                          self.dist_r,
                                                          self.dims, self.R, self.T, flags=0,
                                                          alpha=0
                                                          )

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.cameraMatrix_l, self.dist_l, R1, P1,
                                                             self.dims, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.cameraMatrix_r, self.dist_r, R2, P2,
                                                             self.dims, cv2.CV_32FC1)

        self.rectify_model = dict([
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
        print('rectify_model:', self.rectify_model)
        # # self.save_txt(self, path)
        # return self.rectify_model, self.cameraMatrix_l, self.rectify_model['Q']

    def rectify_image(self, grayl, grayr):
        """Implement the parallel rectification.

        rectify the image pair. Save and show them.

        Parameters
        ----------
        grayl : array
            gray image of left camera used for rectification.
        grayr : array
            gray image of right camera used for rectification.

        """
        # Remap the un-rectified images to new images.
        rectified_img1 = cv2.remap(grayl, self.map1x, self.map1y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rectified_img2 = cv2.remap(grayr, self.map2x, self.map2y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # # Image show.
        # cv2.imshow('rect_img1', rectified_img1)
        # cv2.waitKey(3000)
        # cv2.imshow('rect_img2', rectified_img2)
        # cv2.waitKey(3000)

        cv2.imwrite(self.path + '//' + 'result1.png', rectified_img1)
        cv2.imwrite(self.path + '//' + 'result2.png', rectified_img2)
        result = np.concatenate((rectified_img1, rectified_img2), axis=1)
        resize = cv2.resize(result, (1024, 384))

        cv2.imwrite(self.path + '//' + 'result3.png', result)
        # Image show.
        cv2.imshow("rec.png", resize)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def draw_line(self, image1, image2):
        """Draw lines to present the result of the rectified images.

        Parameters
        ----------
        image1 : array
            rectified left image.
        image2 : array
            rectified right image.

        Returns
        -------
        horizontal stacked image pair with polar lines.

        """
        height = max(image1.shape[0], image2.shape[0])
        width = image1.shape[1] + image2.shape[1]

        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[0:image1.shape[0], 0:image1.shape[1]] = image1
        output[0:image2.shape[0], image1.shape[1]:] = image2

        # Draw equally spaced parallel lines.
        # For a clear display, these parallel lines will be represented at intervals in red and green.
        # The interval of the lines are 50 pixels.
        line_interval = 50
        for k in range(height // line_interval):
            cv2.line(output, (0, line_interval * (2 * k)), (2 * width, line_interval * (2 * k)), (0, 255, 0),
                     thickness=2, lineType=cv2.LINE_AA)
            cv2.line(output, (0, line_interval * (2 * k + 1)), (2 * width, line_interval * (2 * k + 1)), (0, 0, 255),
                     thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("withlines", output)
        cv2.imwrite(self.path + '//' + 'withlines.png', output)
        cv2.waitKey()
        cv2.destroyAllWindows()

    @property
    def save_txt(self):
        """Save the intrinsic and extrinsic parameters as ".txt" file.

        """
        filename = open(self.path + '\\' + 'extrinsic.txt', 'w')
        for k, v in self.rectify_model.items():
            filename.write(k + ':' + str(v))
            filename.write('\n')
        print("Extrinsic parameters are saved as txt file.")
        filename.close()

    @property
    def save_pickle(self):
        """Save the intrinsic and extrinsic parameters as ".pickle" file.

        """
        save_file = open(self.path + '\\' + 'extrinsic.pickle', 'wb')
        pickle.dump(self.rectify_model, save_file)
        print("Extrinsic parameters are saved as pickle file.")
        save_file.close()

def write_txt0(file, path):
    filename = open(path + '\\' + 'intrinsic.txt', 'w')
    for k, v in file.items():
        filename.write(k + ':' + str(v) + ',')
        filename.write('\n')
    filename.close()

def write_txt(file, path):
    filename = open(path + '\\' + 'extrinsic.txt', 'w')
    for k, v in file.items():
        filename.write(k + ':' + str(v) + ',')
        filename.write('\n')
    filename.close()

def store(dict, path):
    json_str = json.dumps(dict)
    with open(path + '\\' + 'intrinsic.json', 'w') as json_file:
        # 将字典转化为字符串
        # json_str = json.dumps(data)
        # fw.write(json_str)
        # 上面两句等同于下面这句
        json_file.write(json_str)


if __name__ == '__main__':
    """
    Implementation the intrinsic calibration and extrinsic calibration.
    Please make sure that the images of the camera pair to be calibrated have been saved in the correct folder. 
    Please refer to the directory structure.
    File named format is "device_x", "x" indicates the serial number of the camera pair.
    """

    print('Please choose the ID of the camera pair to be calibrated.')
    a = input("device:")
    start = time.time()
    device_name = 'camera_' + a
    device_name = Camera(a)

    # Reads the images used for calibration.
    # Saves the images taken by the left and right cameras separately.
    path, path_l, path_r = device_name.get_dir

    # Read images as arrays.
    read_all = PhotoSet(path, path_l, path_r)
    imma1, imma2 = read_all.image_to_matrix

    # Three types of calibration plates can be selected for calibration.
    # "square", "charuco" and "circle" can be chosen.
    chacal = CameraCalibration("charuco", imma1, imma2, path)

    # Entering the parameters of the calibration pattern.
    camera_model = chacal.method_manager(5, 5, 0.15)
    print("The intrinsic camera matrix of device_{}:".format(a), camera_model)

    # Execute stereo calibration.
    stereo_device = 'stereo' + a
    stereo_device = StereoRectify(camera_model, path)
    stereo_device.get_rectify_transform()
    stereo_device.save_txt
    stereo_device.save_pickle
    # Preparing for the parallel correction.
    # Please note that the images used for parallel correction should be saved as "left.jpg" and "right.jpg".
    # Of course, if you use other formats of photos, such as ".png", the following lines needs to be changed.
    for_rectify_l = cv2.imread(path + '\\' + 'left.jpg', 0)
    for_rectify_l = np.rot90(for_rectify_l, 3)
    for_rectify_r = cv2.imread(path + '\\' + 'right.jpg', 0)
    for_rectify_r = np.rot90(for_rectify_r, 3)

    # Verify that images are read correctly.
    if for_rectify_l is None or for_rectify_r is None:
        print("Check path")
        exit()

    # Perform parallel correction. Save correction results for disparity map generation.
    stereo_device.rectify_image(for_rectify_l, for_rectify_r)
    stereo_device.draw_line(cv2.imread(path + '\\' + 'result1.png'), cv2.imread(path + '\\' + 'result2.png'))

    print("Camera calibration took %.3f sec.\n" % (time.time() - start))






