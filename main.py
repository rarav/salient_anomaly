from opals import Import, Export
from opals import TerrainFilter as tf
from opals.AddInfo import AddInfo
import DM_tools
import numpy as np
import DM_o3d_Visualize
import open3d as o3d

import sys

if __name__ == '__main__':

    folder = '/home/reuma/ownCloud_TU/data/Pielach/Laser/VQ-880-G/'
    file = '115357_0'
    odm =  file + '.odm'

    # define output format
    exp = Export.Export()
    exp.oFormat = 'LAS_1.4_2Classes.xml'

    filterize=False

    if filterize:
        Import.Import(inFile=folder+file+'.laz', outFile=odm).run()

        # reverse z to filter out point under the ground
        # DM_tools.reverseZ(odm)
        # filter1 = tf.TerrainFilter()
        # # filter1.robustInterpolation.filterThresholds = [0.06, 0.1, 0.5]
        # filter1.robustInterpolation.gridSize = 0.1
        # # filter1.robustInterpolation.gridSize = 0.1
        # filter1.inFile = odm
        # filter1.run()

        # initialize classification
        AddInfo(inFile=odm, attribute='_Classification1=Classification').run()
        AddInfo(inFile=odm, attribute='Classification=0*Classification').run()

        # return z upwards and filter terrain again
        # DM_tools.reverseZ(odm)
        filter2 = tf.TerrainFilter()
        filter2.robustInterpolation.gridSize = 0.25
        filter2.inFile = odm
        filter2.run()

    pcd, np_dict = DM_tools.odm2o3d(odm, attributes_dict=['Classification', ])
    pts = np.asarray(pcd.points)
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(pts[np_dict['Classification']==2, :])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_filtered,
                                                                voxel_size=0.2)
    o3d.visualization.draw_geometries([voxel_grid])
    # pcd.colors = np_dict['Classification']
    # DM_o3d_Visualize.VisualizeODM().visualize_odm(odm, attributes_list=['Classification'])

    # save to las
    exp.outFile = folder + file + '_filtered.las'
    exp.inFile = odm
    exp.run()

    print('Done!')


