import cv2
import numpy as np
import time


def stereo_matchSGBM(path, imgL, imgR):
    """Disparity calculating using SGBM algorithm.

    Calculate the disparity map use SGBM algorithm.

    Parameters
    ----------
    left_image : array
        the rectified left image.
    right_image : array
        the rectified right image.

    Returns
    -------
    Disparity map.

    .. important::
        Considering that the physical installation parameters of each camera pair are different, and the different
        capture times of the images used for calibration result in varying image brightness and sharpness, the parameters
        for generating disparity maps shall be tuned accordingly. The key parameters to be tuned are minDisparity,
        numDisparities and blockSize.


    """
    if imgL.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3

    # Parameter setting can be referred to OpenCV.
    # https://docs.opencv.org/4.x/df/d6c/ximgproc_8hpp.html
    # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
    blockSize = 15  # 3
    # Create the SGBM instance for the left image.
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=5 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        # blockSize=15,
        # # P1 and P2 can be defined using blockSize.
        # P1=8 * img_channels * blockSize ** 2,
        P1=600,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        # P2=32 * img_channels * blockSize ** 2,
        P2=2400,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Create the SGBM instance for the right image.
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Set lambda parameters for wsl filter.
    lmbda = 70000
    # sigma = [0.7, 1.2, 1.5]
    sigma = 1
    visual_multiplier = 1.3  # 1.0

    # Creat the wls filter instance used for disparity map smoothing.
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)

    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # Calculating the disparity maps.
    displ = left_matcher.compute(imgL, imgR).astype(np.float32) / 16.0
    dispr = right_matcher.compute(imgR, imgL).astype(np.float32) / 16.0

    # Improve the smoothness of disparity maps.
    filtered_img = wls_filter.filter(displ, imgL, None, dispr)
    # Invert the format of the image storage to unit8.
    disparity = np.uint8(filtered_img)

    # disp = cv2.normalize(src=filteredImg, dst=filteredImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite(path + '//' + 'disparity.png', disparity)

    cv2.imshow("disp_map", disparity)
    cv2.resizeWindow("disp_map", 1024, 768)
    cv2.waitKey()
    cv2.destroyAllWindows()


