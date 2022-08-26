import cv2
import numpy as np
import open3d as o3d
import os
import time
# from open3d.open3d.geometry import voxel_down_sample, estimate_normals

# pcd = o3d.io.read_point_cloud("C:\\Users\\Nutzer\\Desktop\\0422\\crop1 - Cloud.ply")
# pcd = o3d.io.read_point_cloud("C:\\Users\\Nutzer\\Desktop\\0422\\pointcloud1_change.ply")
# # pcd = o3d.io.read_point_cloud("C:\\Users\\Nutzer\\Desktop\\rectify\\camera1\\pointcloud.ply")
# # pcd = o3d.io.read_point_cloud(path + '\\' + "pointcloud.ply")
# xyz = np.asarray(pcd.points)
# print("\nOriginal pl without filtering has # to %d" % (np.asarray(pcd.points).shape[0]))
#
# # show the original point cloud.
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(xyz)
# o3d.visualization.draw_geometries([pcd], window_name='original', width=800, height=600)


class PointcloudPrep:
    """Preprocessing of point cloud.

    Implementation the point cloud downsampling and filtering.
    Voxel downsampling, uni downsampling and statistical filtering, radius filtering can be chosen.
    Normally, voxel downsampling + statistaical- + radius filtering are used sequentially.

    """

    def __init__(self, pcd, path):
        """Constructor function.

        Parameters
        ----------
        pcd : ply
            Point clouds that need to be pre-processed.
        path : string
            Path to save point cloud.

        """
        self.pcd = pcd
        self.path = path


    @property
    def voxel_down_filter(self):
        """Voxel down sampling.

        Implement the voxel down sampling.

        Returns
        -------
        points : ply
            point cloud after voxel down sampling.

        """

        # The value of voxel_size needs to be tuned according to the requirements.
        voxel_size = 0.005
        # Implementation of voxel down sampling.
        down_sample_pcd = self.pcd.voxel_down_sample(voxel_size)
        print("\nVoxel Scale %f, # of points %d" % (voxel_size, np.asarray(down_sample_pcd.points).shape[0]))
        # visualization.
        o3d.visualization.draw_geometries([down_sample_pcd], window_name="voxel", width=800, height=600)
        return down_sample_pcd


    @staticmethod
    def uni_down_filter(pcd):
        """Uniform down sampling.

        Implement the uniform down sampling filtering.

        Returns
        -------
        down_sample_pcd : ply
            point cloud after uniform sampling.

        """

        # The value of every_k_points needs to be tuned according to the requirements.
        every_k_points = 10
        # Implementation of uniform down sampling.
        down_sample_pcd = pcd.uniform_down_sample(every_k_points=every_k_points)
        print("\nUni Scale %f, # of points %d" % (every_k_points, np.asarray(down_sample_pcd.points).shape[0]))
        # visualization.
        o3d.visualization.draw_geometries([down_sample_pcd], window_name="uni_down", width=800, height=600)
        return down_sample_pcd


    def statistical_removal(self, pcd):
        """Implement statistical filtering.

        remove the outlier points with statistical outlier remover filter.

        Parameters
        ----------
        pcd : ply
            point cloud.

        Returns
        -------
        inlier : ply
            point cloud after statistical filtering.

        """

        # Num_neighbors and std_ratio need to be tuned according to the requirements.
        num_neighbors = 200  # Number of K-neighborhood points.
        std_ratio = 0.1  # The standard deviation multiplier.
        cl, ind1 = pcd.remove_statistical_outlier(nb_neighbors=num_neighbors, std_ratio=std_ratio)
        inlier = self.display_inlier_outlier(pcd, ind1, stat=True)

        return inlier

    def radius_removal(self, pcd):
        """Implement radius filtering.

        remove the outlier points with radius outlier remover filter.

        Parameters
        ----------
        pcd : ply
            point cloud.

        Returns
        -------
        inlier : ply
            point cloud after statistical filtering.

        """
        # Statistical_outlier_removal.
        nb_points = 10  # Number of neighborhood points.
        radius = 0.05  # Radius size.
        cl, ind2 = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        inlier = self.display_inlier_outlier(pcd, ind2, stat=False)

        return inlier

        # o3d.io.write_point_cloud(path + '\\' + 'afterfilter.ply', inliner)


    @staticmethod
    def display_inlier_outlier(pcd, ind, stat=True):
        """Display inlier and outlier of the point cloud.

        Display inlier and outlier of the point cloud.
        The outliers are pained with yellow, the remaining inlier points are gray.

        Parameters
        ----------
        pcd : ply
            point cloud.
        ind : list(int)
            the Serial Number of outliers.
        stat : bool
            True: show statistical filtering result.
            False: show radius filtering result.

        Returns
        -------
        inlier_cloud : ply
            point cloud after filtering.

        """
        print("Outlier removal will be implemented")
        inlier_cloud = pcd.select_by_index(ind)
        outlier_cloud = pcd.select_by_index(ind, invert=True)
        # Painting the outliers with yellow.
        outlier_cloud.paint_uniform_color([1, 0.706, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        print("Showing outliers (yellow) and inliers (gray): ")
        print("\npoints %d" % (np.asarray(inlier_cloud.points).shape[0]))
        if stat==True:
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="statistical_outlier", width=800,
                                               height=600)
        else:
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="radius_outlier", width=800,
                                               height=600)

        return inlier_cloud

    @staticmethod
    def point_display(pcd):
        """Display point cloud.

        Point cloud display.

        Parameters
        ----------
        pcd : ply
            point cloud.

        """
        # pcd = o3d.io.read_point_cloud(path + '//' + 'pointcloud_change5555555.ply')
        xyz = np.asarray(pcd.points)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz)
        o3d.visualization.draw_geometries([pcd], window_name='point cloud', width=800, height=600)

####################################################################################
# # The following sentences are used for point cloud preprocessing.
# # If you want to implemente voxel down sampling, statistical and radius filtering,
# # please activate the following sentence.

# print('Please enter the camera ID of the point cloud that needs to be preprocessed.')
# a = input("Pointcloud:")
# pl_name = 'pointcloud' + a
# print("plname:", pl_name)
# # get the current working directory.
# cur_dir = os.path.abspath(os.path.dirname((__file__)))
# print(cur_dir)
# # get upper directory.
# upper = os.path.abspath(os.path.join(os.getcwd(), ".."))
# name = os.path.join(upper, 'images', 'device' + str(a), str(pl_name) + '.ply')
# print("namenamename:", name)
# print("upper:", upper)
# pcd = o3d.io.read_point_cloud(name)
# path = os.path.join(upper, 'images', 'device' + str(a))
# pl5 = PointcloudPrep(pcd, path)
# voxel_down = pl5.voxel_down_filter
# stat = pl5.statistical_removal(voxel_down)
# radi = pl5.radius_removal(stat)

# down_sample_pcd = down_sample_filter(pcd)
# outlier_removal = outlier_removal(down_sample_pcd)
# outlier_removal2 = outlier_removal(outlier_removal, stat=False)
