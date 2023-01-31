import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from opals import pyDM
from itertools import cycle


class VisualizeODM:

    def __init__(self):
        self.pcd = None
        # define for toggling
        self.colors = None
        self.index = 0
        self.high = 1
        self.attribute_name = None
        self.colormap_list = cycle(['jet', 'summer', 'winter', 'hot', 'gray', 'PiYG', 'coolwarm', 'RdYlBu'])
        self.cm = plt.get_cmap(self.colormap_list.__next__())

    @classmethod
    def initialize_key_to_callback(cls):
        """
        Sets extra keys to callback, to be available for all VisualizeO3D

        :return: key_to_callback dictionary

        :rtype: dict
        """
        key_to_callback = {}

        return key_to_callback
    #
    # @classmethod
    # def toggle_black_white_background(cls, vis):
    #     """
    #     Change background from white to black and vise versa
    #
    #     :param vis: an open3d visualization object
    #
    #     :type vis: open3d.Visualizer
    #
    #     """
    #     opt = vis.get_render_option()
    #     if np.array_equal(opt.background_color, np.ones(3)):
    #         opt.background_color = np.zeros(3)
    #     else:
    #         opt.background_color = np.ones(3)
    #     return False



    # @classmethod
    # def visualize_odm(cls, odm):
    #     """
    #     Visualizes odm and its attributes using open3d visualization
    #
    #     :param odm:
    #
    #     :return:
    #     """
    #     import DM_tools
    #     pcd, np_dict = DM_tools.odm2o3d(odm, attributes_dict='all')
    #
    #     key_to_callback = cls.initialize_key_to_callback()
    #     if not pcd.has_colors():
    #
    #     if not (colors is None):
    #         if isinstance(colors, ColorProperty):
    #             colors_ = colors.rgb
    #         elif isinstance(colors, np.ndarray):
    #             colors_ = colors
    #         if colors_.max() > 1:
    #             colors_ /= 255
    #         pcd.data.colors = o3d.utility.Vector3dVector(colors_)
    #
    #     if drawCoordianteFrame:
    #         if coordinateFrameOrigin == 'min':
    #             cf = o3d.geometry.create_mesh_coordinate_frame(size=coordinateFrameSize,
    #                                                            origin=pcd.ToNumpy().min(axis=0) - originOffset)
    #         else:
    #             cf = o3d.geometry.create_mesh_coordinate_frame(size=coordinateFrameSize)
    #         o3d.visualization.draw_geometries_with_key_callbacks([pcd.data, cf], key_to_callback)
    #     else:
    #         o3d.visualization.draw_geometries_with_key_callbacks([pcd.data], key_to_callback)


    def visualize_odm(self, odm, attributes_list='all'):
        """
        Visualize property classes

        :param odm: odm path
        :param attributes_list: list of attributes to be shown. If not sent, all available attributes will be loaded

        :type odm: str
        :type attributes_list: list

        """

        import DM_tools
        self.pcd, np_dict = DM_tools.odm2o3d(odm, attributes_list)

        key_to_callback = self.initialize_key_to_callback()

        self.__prepare_attributes(np_dict)

        key_to_callback[ord('A')] = self.toggle_attributes_colors
        key_to_callback[ord('C')] = self.toggle_colormaps

        o3d.visualization.draw_geometries_with_key_callbacks([self.pcd], key_to_callback)

    def __prepare_attributes(self, attributes_dict):
        """
        Prepares property for coloring

        :param attributes_dict: a dictionary with the attributes according to which the colors will be applied

        :return: colors by property
        """
        from numpy import ndarray


        colors_new = []
        attribute_name = []
        for name in attributes_dict:
            colors_new.append(self.__make_color_array(attributes_dict[name]))
            attribute_name.append(name)

        self.high = len(attributes_dict)

        self.colors = colors_new
        self.attribute_name = attribute_name

    def toggle_attributes_colors(self, vis):
        """
        Change visualization colors according to attributes of the property

        :param vis:

        :type vis: open3d.Visualizer

        """

        if self.index >= self.high - 1:
            self.index = 0
        else:
            self.index += 1

        print(self.attribute_name[self.index])
        colors = self.__apply_colormap(self.colors[self.index])
        self.pcd.colors = colors
        vis.update_geometry(self.pcd)

    def toggle_colormaps(self, vis):
        """
        changes between colormaps
        :param vis:
        :return:
        """
        colorname = self.colormap_list.__next__()
        print(colorname)
        self.cm = plt.get_cmap(colorname)

        new_colors = self.__apply_colormap(np.asarray(self.colors)[self.index])
        self.pointset.data.colors = new_colors
        vis.update_geometry(self.pointset.data)

    def __apply_colormap(self, array):
        """
        changes the array to a given colormap

        :param array: the array of colors to transform

        :type array: np.array
        :type colormap: plt.colormap

        :return: the array in the new colormap
        """

        colored = self.cm(array)
        R = colored[:, 0, 0].flatten()
        G = colored[:, 0, 1].flatten()
        B = colored[:, 0, 2].flatten()

        return o3d.utility.Vector3dVector(np.vstack((R, G, B)).T)

    @classmethod
    def __make_color_array(cls, array):
        """
        Prepare an array to be used as a color array for visualization

        :param array: array of numbers to be converted into color array

        :type array: numpy.array

        :return: an open3d vector
        :rtype: o3d.Vector3D
        """
        size_array = array.shape

        # check the dimension of the array, if only a list of numbers, it should be transformed into a 2D array (nx3)
        if len(size_array) <= 2:
            rgb = np.ones((size_array[0], 3))
            try:
                rgb *= array[:, None]
            except:
                rgb *= array

        elif len(size_array) == 3:
            rgb = array.reshape((size_array[0] * size_array[1], 3))

        else:
            rgb = array

        # normalize to range [0...1]
        rgb_normed = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-12)

        return o3d.utility.Vector3dVector(rgb_normed)





