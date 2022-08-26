Sequence of code execution
==========================

This document is used to describe the steps to perform stereo camera calibration and point cloud registration of adjacent camera pairs.


----------------------------------------------

The layout of the camera pairs are as follows.

.. image:: /_static/cameralayout.png
    :scale: 50 %
    :align: center


------------------------------------

The execution steps are as follows.

1. Separate calibrations for each camera pairs.                                                    -src/calibration.py
2. Parallel correction using images/device2/left.jpg and right.jpg.                                -src/disp_generation.py
3. Disparity map generation.                                                                       -src/disp_generation.py
4. Point cloud generation.                                                                         -src/pcd_generation.py
5. Point cloud preprocessing.                                                                      -src/pl_preprocessing.py
6. Point cloud registration.                                                                       -src/pcd_registration.py
7. At last, you can use the main function to merge the point cloud of any adjacent image pair.     -main.py

Notes
-----

All of the above programs can run individually upon request. To start the program you need to activate the sentence marked as a comment following separation statements '#########'.

https://github.com/Slitheins/Masterarbeit/blob/master/README.md
