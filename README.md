# Masterarbeit

## Code description

For a detailed description of the program, please refer to the following file.
path is: Masterarbeit/venv/doc/_build/html/genindex.html

## Folder structure

Please create a folder structured as follows：
Each folder named deviceX stores the images used for calibration and parallel correction.

```
├─ __init__.py
├─ images
│    ├─ __init__.py
│    ├─ device1
│    ├─ device10
│    ├─ device11
│    ├─ device12
│    ├─ device2
│    │    ├─ 22-03-17_13-07-51_1.jpg    // calibration image
│    │    ├─ 22-03-17_13-07-51_2.jpg    // calibration image
│    │    ├─ 22-03-17_13-08-03_1.jpg    // calibration image
│    │    ├─ 22-03-17_13-08-03_2.jpg    // calibration image
│    │    ├─ 22-03-17_13-08-13_1.jpg    // calibration image
│    │    ├─ 22-03-17_13-08-13_2.jpg    // calibration image
│    │    ├─ extrinsic.pickle           // results of calibration.py
│    │    ├─ extrinsic.txt              // results of calibration.py
│    │    ├─ intrinsic.pickle           // results of calibration.py
│    │    ├─ intrinsic.txt              // results of calibration.py
│    │    ├─ left.jpg                   // parallel correction image
│    │    ├─ pcd2.ply                   // result of pcd_generation.py
│    │    └─ right.jpg                  // parallel correction image
│    ├─ device3
│    ├─ device4
│    ├─ device5
│    ├─ device6
│    ├─ device7
│    ├─ device8
│    └─ device9
├─ main.py
└─ src
       ├─ __init__.py
       ├─ calibration.py                 // camera calibration
       ├─ disp_generation.py             // disparity map generation
       ├─ pcd_generation.py              // point cloud generation
       ├─ pcd_registration.py            // point cloud registration
       ├─ pl_preprocessing.py            // preprocessing of point cloud
       ├─ registration_result_2_3.pickle // result of pcd_registration.py with device 2 and device 3.
       ├─ registration_result_2_3.txt    // result of pcd_registration.py  with device 2 and device 3.
       └─ stereorectify.py               // image pair parallel correction
```

## Sequence of code execution

1. Separate calibrations for each camera pairs.                                                    -src/calibration.py
2. Parallel correction using images/device2/left.jpg and right.jpg.                                -src/disp_generation.py
3. Disparity map generation.                                                                       -src/disp_generation.py
4. Point cloud generation.                                                                         -src/pcd_generation.py 
5. Point cloud preprocessing.                                                                      -src/pl_preprocessing.py 
6. Point cloud registration.                                                                       -src/pcd_registration.py
7. At last, you can use the main function to merge the point cloud of any adjacent image pair.     -main.py
