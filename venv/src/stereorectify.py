# !/usr/bin/env python
# -*- coding:utf-8 -*- 

"""
author: yifeng
date  : 10.04.2022
"""

import cv2
import numpy as np


class RectifyImages:
    """Rectification of the image pair.

    Rectification of the image pair.
    Draw the epi-polar lines to simplify the search of the corresponding feature points in left and right image.
    Please note that the used image pairs need to have overlapping parts,
    so that the subsequent point cloud registration can be implemented.

    """

    def __init__(self, rectify_model, path):
        """Constructor function.

        """
        self.map1x = rectify_model['stereo_left_mapx']
        self.map1y = rectify_model['stereo_left_mapy']
        self.map2x = rectify_model['stereo_right_mapx']
        self.map2y = rectify_model['stereo_right_mapy']
        # Save path for parallel correction images.
        self.path = path

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
