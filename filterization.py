from opals import Import, Export, Bounds
from opals import TerrainFilter as tf
from opals.AddInfo import AddInfo
import DM_tools, my_tools
import numpy as np
import DM_o3d_Visualize
import open3d as o3d
from matplotlib.path import Path


import geopandas as gpd
import os, glob

if __name__ == '__main__':

    base_dir = r'../data/Pielach/2021-03-09-VQ-880-GH/'
    laz_folder = 'laz/'
    shp_folder = os.path.join(base_dir,  'shpfiles/')
    odm_folder = os.path.join(base_dir,  'odm/')

    files = glob.glob(base_dir + laz_folder + '*.laz')
    intersection_files = []

    filterize = True # for debugging purposes

    # filterization parameters
    filter2 = tf.TerrainFilter()
    filter2.robustInterpolation.gridSize = 0.3
    filtered_folder = os.path.join(base_dir, f'filtered_{filter2.robustInterpolation.gridSize}/')

    z_thresh_up = 263 # z higher than this value will be filterized
    z_thresh_down = 257 # z lower than this value will be filterized
    # define output format
    exp = Export.Export()
    exp.oFormat = 'LAS_1.4_2Classes.xml'

    boulders_bounds = gpd.read_file(shp_folder + 'boulders.shp')  # bounds by Gottfried (known bounds)
    boulders_xy = [line.geometry.boundary.xy[1] for id, line in boulders_bounds]
    print('hello')
    # create odm and bounds file for all las/laz
    #---------------------------------------------
    for base_file in files:
        print(f'Check boundaries files for {os.path.basename(base_file)}')
        if not os.path.exists(odm_folder):
            os.makedirs(odm_folder)

        filename = os.path.basename(base_file)[:-4]
        odm =  os.path.join(odm_folder, filename + '.odm')
        shp =  os.path.join(shp_folder, filename + '.shp')

        # check if bounds file exist (if it is, there is an odm)
        if os.path.isfile(odm):
            continue
        else:
            print(f"Create odm file for {os.path.basename(base_file)})")
            Import.Import(inFile= base_file, outFile=odm).run()

        # if os.path.isfile(shp):
        #     continue
        # else:
        #     print(f"Create boundary shapefile file for {os.path.basename(base_file)})")
        #     Bounds.Bounds(odm, shp).run()

    # check intersections between odm and known boulders
    files_odm = glob.glob(odm_folder + '/*.odm')
    for odm in files_odm:
        filename = os.path.basename(odm)[:-4]
        # shpfile = shp_folder + filename + '.shp'
        # 2. check intersections
        # las_bounds = gpd.GeoDataFrame.from_file(shpfile)  # bounds of the las file

        cboulders = []  # current boulders
        for index1, boulder in boulders_bounds.iterrows():
            # for index2, las_bound in las_bounds.iterrows():
            #     if boulder['geometry'].intersects(las_bound['geometry']):

            xyz_dict = DM_tools.odm2numpy(odm)
            xyz = np.vstack((xyz_dict['x'], xyz_dict['y'], xyz_dict['z'])).T
            # xy_flat = xyz[:,:2]
            boulder_path = Path(np.asarray(boulder['geometry']))
            xy_in = boulder_path.contains_point(xyz)

                    # cboulders.append({'geometry': boulder['geometry'].intersects(las_bound['geometry'])})
                    # intersection_files.append(filename)
                    # intersection_las = my_tools.create_las(())

        if filterize:
            print(f"fliterize {base_file}")
            # initialize classification
            AddInfo(inFile=odm, attribute='_Classification1=Classification').run()
            AddInfo(inFile=odm, attribute='Classification=0*Classification').run()

            filter2.inFile = odm
            filter2.run()

        pcd, np_dict = DM_tools.odm2o3d(odm, attributes_dict=['Classification', ])
        pts = np.asarray(pcd.points)
        pts_filtered = pts[np_dict['Classification']==2, :]

        # filter by height
        ind_up = pts_filtered[:,2]< z_thresh_up
        pts_filtered = pts_filtered[ind_up, :]
        ind_down = pts_filtered[:, 2] > z_thresh_down
        pts_filtered = pts_filtered[ ind_down, :]

        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(pts_filtered)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_filtered,
                                                                    voxel_size=0.2)
        o3d.visualization.draw_geometries([pcd_filtered])
        # pcd.colors = o3d.utility.Vector3dVector(np_dict['Classification'])
        # DM_o3d_Visualize.VisualizeODM().visualize_odm(odm, attributes_list=['Classification'])

        new_las = my_tools.create_las(pts_filtered)

        if not os.path.exists(filtered_folder):
            os.makedirs(filtered_folder)

        new_las.write(os.path.join(filtered_folder,
                                   f"{os.path.basename(base_file)[:-4]}.las"))
        # # save to las
        # exp.outFile = base_file + '_filtered.las'
        # exp.inFile = odm
        # exp.run()

    print('Done!')


