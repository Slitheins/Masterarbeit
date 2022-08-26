# !/usr/bin/env python
# -*- coding:utf-8 -*- 

"""
author: yifeng
date  : 16.03.2022
"""
from cv2 import aruco as ar
import cv2
import numpy as np

# make ChArUco board
dictionary = cv2.aruco.getPredefinedDictionary(dict=cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(squaresY=6, squaresX=6, squareLength=0.031, markerLength=0.0155, dictionary=dictionary)

img_board = board.draw(outSize=(4096, 4096), marginSize=None, borderBits=None)
cv2.imwrite(filename='charuco3.png', img=img_board, params=None)