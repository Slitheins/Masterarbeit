Introduction to the use of mian.py
==================================


4 steps will be executed.

Step 1 : the parallel correction of image pairs obtained from a stereo camera pair using intrinsic/extrinsic matrices.

Step 2 : disparity map generation using SGBM algorithm.

Step 3 : point cloud generation using intrinsic and Q matrices.

Step 4 : point cloud registration using homogenous matrix H.

Notes
-----
Before starting the main function, each camera pair needs to be calibrated for intrinsic and extrinsic parameters obtaining. Subsequently, registration of the point clouds of the neighboring camera pairs needs to be done.
The image pairs for registering should have overlapping parts, i.e., the same objects need to be present in the image pairs at the same time.
Intrinsic, extrinsic, Q and H matrices can be loaded from previous saved pickle files.

https://github.com/Slitheins/Masterarbeit/blob/master/README.md
