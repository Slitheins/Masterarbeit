import pickle
import time
import open3d as o3d
import copy
import numpy as np
import os

class Registration:
    """Implementation of point cloud registration.

    Implementation of 2 point clouds. RANSAC and 2 kinds of ICP(point-to-point and point-to-plane) algorithms are used.
    The FPFH features need to be calculated first.
    The result of registration will be saved as 4 x 4 matrix in ".txt" and ".pickle" files.

    """

    def __init__(self, pcd_1, pcd_2, path, filename_1, filename_2):
        """Constructor function.

        """
        self.pcd_1 = pcd_1
        self.pcd_2 = pcd_2
        self.path = path
        self.filename_1 = filename_1
        self.filename_2 = filename_2


    def FPFH_Compute(self, pcd):
        """FPFH features calculation.

        Pass in the point cloud data and calculate FPFH.

        Parameters
        ----------
        pcd : ply
            point cloud for feature calculation.

        Returns
        -------
        pcd_fpfh : array
            FPFH features.

        """
        # Prameters of kdtree for estimating the radius of the normal.
        radius_normal = 0.03
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        # Estimate 1 parameter of the normal, using a hybrid kdtree, taking up to 30 neighbors in the radius
        # Kdtree parameter for estimating the radius of FPFH features.
        radius_feature = 0.06
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        # Calculating FPFH features using the kdtree search method.
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
                                                                   o3d.geometry.KDTreeSearchParamHybrid(
                                                                       radius=radius_feature, max_nn=100))
        return pcd_fpfh


    def execute_global_registration(self, source, target, source_fpfh, target_fpfh):
        """Implement the global registration using RANSAC.

        Point Cloud global registration.

        Parameters
        ----------
        source : ply
            source point cloud.
        target : ply
            target point cloud.
        source_fpfh: FPFH features of source point cloud.
        target_fpfh: FPFH features of target point cloud.

        Returns
        -------
        ransac_result : 4X4 matrix
            the results of the global registration.

        """
        # Threshold of distance. It is 0.5 times the size of the voxel.
        distance_threshold = 0.04 * 1.5
        print("we use a liberal distance threshold %.3f." % distance_threshold)
        # Execute global registration using the ransac algorithm.
        self.ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1))
        print("+ '_'", self.ransac_result)
        return self.ransac_result


    def draw_registration_result(self, source, target, transformation):
        """Visulization the registration rsult.

        Visulization of global registration.
        Target and source point cloud are painted with green and red colours.

        Parameters
        ----------
        source : ply
            source point cloud.
        target : ply
            target point cloud.
        transformation : 4 x 4 matrix
            result of registration.

        """
        # The function ”paint_uniform_color“ changes the point cloud.
        # So copy.deepcoy is invoked to copy and protect the original point cloud.
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        # Point cloud painting.
        source_temp.paint_uniform_color([1, 0, 0])  # Point cloud painting
        target_temp.paint_uniform_color([0, 1, 0])
        source_temp.transform(transformation)

    def execute_local_registration(self, source, target, result_ransac, p_2_p=True):
        """Implement the local registration using ICP.

        Implement the local registration by using point-to-point ICP or point-to-plane ICP.

        Parameters
        ----------
        source : ply
            source point cloud.
        target : ply
            target point cloud.
        result_ransac : 4 x 4 matrix
            result of ransac registration.
        p_2_p : bool
            True: use point-to-point ICP.
            False: use point-to-plane ICP.

        Returns
        -------
        icp.transformation :  4 X 4 matrix
            results of the global registration.

        """
        start = time.time()
        if p_2_p==True:
            print("Apply point-to-point ICP")
            self.icp = o3d.pipelines.registration.registration_icp(
                source, target, 0.5, result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),    # implemente the point-to-point ICP.
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))  # set the maximum iteration number.
            print("ICP point-to-point registration took %.3f sec.\n" % (time.time() - start))
            print(self.icp)  # display ICP result.
            print("Transformation of ICP is:")
            print(self.icp.transformation)  # display registration transformation.
        else:
            print("Apply point-to-plane ICP")
            self.icp = o3d.pipelines.registration.registration_icp(
                    source, target, 0.5, result_ransac.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(), # implemente the point-to-plane ICP.
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)) # set the maximum iteration number.
            print("ICP point-to-plane registration took %.3f sec.\n" % (time.time() - start))
            print(self.icp)
            print("Transformation of ICP is:")
            print(self.icp.transformation)
        self.ICP = dict([('ICP', self.icp.transformation)
        ])
        return self.icp.transformation

    @property
    def save_txt(self):
        """Save the transformation matrix of ICP as ".txt" file.

        """
        filename = open(self.filename_1, 'w')
        # filename.write(str(self.ransac_result))
        # filename.write('\n')
        for k, v in self.ICP.items():
            filename.write(k + ':' + str(v))
            filename.write('\n')
        print("ICP result is saved as txt file.")
        filename.close()

    @property
    def save_pickle(self):
        """Save the transformation matrix of ICP as ".pickle" file.

        """
        save_file = open(self.filename_2, 'wb')
        pickle.dump(self.ICP, save_file)
        print("ICP result is saved as pickle file.")
        save_file.close()

    def get_global_local(self):
        pass


def draw_registration_result(source, target, transformation):
    """ Visulization the registration rsult.

    Visulization of global registration.
    Target and source point cloud are painted with green and red colours.

    Parameters
    ----------
    source : ply
        source point cloud.
    target : ply
        target point cloud.
    transformation : 4 x 4 matrix
        result of registration.

    """
    # The function ”paint_uniform_color“ changes the point cloud.
    # So copy.deepcoy is invoked to copy and protect the original point cloud.
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # Point cloud painting.
    source_temp.paint_uniform_color([1, 0, 0])  # Point cloud painting
    target_temp.paint_uniform_color([0, 1, 0])
    source_temp.transform(transformation)

####################################################################################################################
# # The following sentences are used for point cloud registration.
# # If you want to registrate 2 point clouds, please activate the following sentence.

# print('Please choose the ID of the camera pair to be registered.')
# Nr1 = input("registrate_pcd_1:")
# Nr2 = input("registrate_pcd_2:")
# start = time.time()
# registrate_ID_1, registrate_ID_2 = 'pcd_' + Nr1, 'pcd_' + Nr2
# print("The selected camera pairs are:", registrate_ID_1, registrate_ID_2)
#
#
# # get the current working directory.
# cur_dir = os.path.abspath(os.path.dirname((__file__)))
# # get upper directory.
# upper = os.path.abspath(os.path.join(os.getcwd(), ".."))
# pcd_1_path = os.path.join(upper, 'images', 'device' + str(Nr1), str(registrate_ID_1) + '.ply')
# pcd_2_path = os.path.join(upper, 'images', 'device' + str(Nr2), str(registrate_ID_2) + '.ply')

#
# filename_txt = os.path.join(cur_dir, 'registration_result' + '_' + str(Nr1) + '_' + str(Nr2) + '.txt')
# filename_pickle = os.path.join(cur_dir, 'registration_result' + '_' + str(Nr1) + '_' + str(Nr2) + '.pickle')
# print("saved file name：", filename_txt, filename_pickle)
#
# source = o3d.io.read_point_cloud(pcd_1_path)
# target = o3d.io.read_point_cloud(pcd_2_path)
#
# registrate = Registration(source, target, cur_dir, filename_txt, filename_pickle)
# source_fpfh = registrate.FPFH_Compute(source)
# target_fpfh = registrate.FPFH_Compute(target)
#
# voxel_size = 0.002
# source = source.voxel_down_sample(voxel_size)
# target = target.voxel_down_sample(voxel_size)
#

# # Implementation of global registration
# start = time.time()
#
# result_ransac = registrate.execute_global_registration(source, target, source_fpfh, target_fpfh)
# registrate.execute_local_registration(source, target, result_ransac, p_2_p=True)
# registrate.save_txt
# registrate.save_pickle
#
#
# print("Global registration took %.3f sec.\n" % (time.time() - start))
# print(result_ransac)
# # Rotation and translation of the source point cloud to the target point cloud
# # draw_registration_result(source, target, result_ransac.transformation)
# # draw_registration_result(source, target, icp_p2p.transformation)
