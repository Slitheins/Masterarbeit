# !/usr/bin/env python
# -*- coding:utf-8 -*- 

"""
author: yifeng
date  : 13.04.2022
"""
import cv2
import numpy as np
import open3d as o3d

feature_params = {'maxCorners' : 1000,
                  'quality_level' : 0.3,
                  'minDistance' : 7,
                  'blockSize' : 7}

lk_params = {'winSize' : (15,5),
             'maxLevel' : 2,
             'criteria' : (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

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

    output = np.zeros((height, width), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0),
                 thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow("withlines", output)
    # cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\withlines.png', output)
    cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-03-18_Kalibrierbilder\\withlines.png', output)
    cv2.waitKey(80000)

def hw3ToN3(points):
    height, width = points.shape[0:2]

    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)

    points_ = np.hstack((points_1, points_2, points_3))

    return points_


# 深度、颜色转换为点云
def DepthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    print('rows, cols:', rows, cols)
    size = rows * cols

    points_ = hw3ToN3(points_3d)
    colors_ = hw3ToN3(colors).astype(np.int64)

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

    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack((remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

    pointcloud_1 = np.delete(pointcloud, remove_idx, 0)

    return pointcloud_1



# img1 = cv2.imread('C:\\Users\\wyfmi\\Desktop\\2022-03-18_Kalibrierbilder\\22-03-17_13-15-54_1.jpg', 0)
# img2 = cv2.imread('C:\\Users\\wyfmi\\Desktop\\2022-03-18_Kalibrierbilder\\22-03-17_13-15-54_2.jpg', 0)
img1 = cv2.imread('C:\\Users\\wyfmi\\Downloads\\opencv-4.5.5\\opencv-4.5.5\\samples\\data\\left01.jpg')
img2 = cv2.imread('C:\\Users\\wyfmi\\Downloads\\opencv-4.5.5\\opencv-4.5.5\\samples\\data\\right01.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
rows, cols, channels = img1.shape
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

dst1 = cv2.goodFeaturesToTrack(gray1, 100, 0.01, 30)
# dst2 = cv2.goodFeaturesToTrack(gray2, 500, 0.01, 30)
dst1 = np.int0(dst1)  # 实际上也是np.int64
# dst2 = np.int0(dst2)
for i in dst1:
    x, y = i.ravel()  # 数组降维成一维数组（inplace的方式）
    cv2.circle(img1, (x, y), 3, (0, 0, 255), -1)
# for i in dst2:
#     x, y = i.ravel()  # 数组降维成一维数组（inplace的方式）
#     cv2.circle(img2, (x, y), 3, (0, 255, 0), -1)

dst1 = np.float32(dst1)  # 实际上也是np.int64
# dst2 = np.float32(dst2)


cv2.namedWindow('harris', cv2.WINDOW_FREERATIO)
cv2.imshow('harris', img1)
cv2.waitKey(0)
# cv2.namedWindow('harris2', cv2.WINDOW_FREERATIO)
# cv2.imshow('harris2', img2)
# cv2.waitKey(0)

exact_corners = cv2.cornerSubPix(gray1, dst1, (11, 11), (1, 1),
                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

features, ret, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, dst1, None, **lk_params)
features = np.float32(features)

# eg
F, mask = cv2.findFundamentalMat(dst1, features, method=cv2.FM_RANSAC, ransacReprojThreshold=0.9, confidence=0.99)
ret_sr, H1, H2 = cv2.stereoRectifyUncalibrated(dst1, features, F, (rows, cols))
print('H1', H1)
print('H2', H2)
# print(type(dst1))
# print(type(dst2))
# dst1 = np.reshape(dst1, (-1, 2))
# features = np.reshape(features, (-1, 2))
# get1 = dst1[0:4, :, :]
# get2 = features[0:4, :, :]
# get1 = np.float32([[1244.3438, 1557.8981], [1383.5292, 1557.9946], [1246.2388, 1420.1682], [1385.0676, 1420.1768]]) # charuco
# get2 = np.float32([[912.126, 1614.6339], [1052.332, 1613.647], [910.7861, 1474.6659], [1051.0118, 1473.9277]]) # charuco
get1 = np.float32([[2102.0972, 937.55426], [2033.5266, 1004.94385], [2034.0977, 871.0303], [1965.6803, 938.29767]])
get2 = np.float32([[1768.7593, 1048.3353], [1699.5786, 1117.5253], [1699.3003, 978.6675], [1630.3837, 1047.9603]])
K = cv2.getPerspectiveTransform(get1, get2)
# K = np.array([[0.99943117, -0.00303017, -0.03358791],
#                           [-0.00371039, 0.98002908, -0.19881962],
#                           [0.03351959, 0.19883115, 0.97946037]])
remap1 = cv2.warpPerspective(gray1, K, (cols, rows), flags=None, borderMode=None, borderValue=None)
remap2 = cv2.warpPerspective(gray2, K, (cols, rows), flags=None, borderMode=None, borderValue=None)

cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-03-18_Kalibrierbilder\\result1.png', remap1)
cv2.imwrite('C:\\Users\\wyfmi\\Desktop\\2022-03-18_Kalibrierbilder\\result2.png', remap2)

Q = np.array([[1, 0, 0, -2.048e+03],
              [0, 1, 0, -1.536e+03],
              [0, 0, 0, 7.78e+03],
              [0, 0, -6.67e-01, 0.2214]])

disparity = cv2.imread('C:\\Users\\wyfmi\\Desktop\\2022-03-18_Kalibrierbilder\\depthmap.png', 0)
points = cv2.reprojectImageTo3D(disparity, Q)

# pointcloud = hw3ToN3(points)

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('点 (%d, %d) 的三维坐标 (%f, %f, %f)' % (x, y, points[y, x, 0], points[y, x, 1], points[y, x, 2]))
        dis = ((points[y, x, 0] ** 2 + points[y, x, 1] ** 2 + points[y, x, 2] ** 2) ** 0.5) / 1000
        print('点 (%d, %d) 距离左摄像头的相对距离为 %0.3f m' % (x, y, dis))

    # 显示图片


cv2.namedWindow("disparity", 0)
cv2.imshow("disparity", disparity)
cv2.setMouseCallback("disparity", onMouse, 0)


# pointcloud = DepthColor2Cloud(points, img1)[:, 0:3]

# path = 'C:\\Users\\wyfmi\\Desktop\\2022-03-18_Kalibrierbilder'
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(pointcloud)
# point_cloud.colors = o3d.utility.Vector3dVector(colors / 255)
# o3d.visualization.draw_geometries([point_cloud])

def save_txt(points, save_path):
    filename = open(save_path + '\\' + 'points.txt', 'w')
    for k, v in points.items():
        filename.write(k + ':' + str(v))
        filename.write('\n')
    filename.close()

print(points.shape)
# np.savetxt(path + '\\' + 'points.txt', points)

cv2.namedWindow('reprojection1', cv2.WINDOW_FREERATIO)
cv2.imshow('reprojection1', remap1)
cv2.waitKey(0)
cv2.namedWindow('reprojection2', cv2.WINDOW_FREERATIO)
cv2.imshow('reprojection2', remap2)
cv2.waitKey(0)

# result = np.concatenate((rectified_img1, rectified_img2), axis=1)
# resize = cv2.resize(result, (1024, 384))

draw_line(remap1, remap2)