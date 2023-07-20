# Utility functions which do not require OPALS

import laspy, os
import numpy as np
import geopandas as gpd

def create_las(points, fields = None):
    """
    Creates new las file with fields

    :type fields: dict
    :type las_points: np.ndarray
    """
    new_hdr = laspy.LasHeader(version="1.4", point_format=6)
    if fields:
        extrabytesparam = [laspy.ExtraBytesParams(name=name, type='float64') for name in fields.keys()]
        new_hdr.add_extra_dims(extrabytesparam)
        # new_hdr.add_extra_dims(laspy.ExtraBytesParams(name=name, type='float64'))
    # new_las = laspy.create(point_format=6)
    # new_las.header = new_hdr
    new_las = laspy.LasData(new_hdr)
    new_las.x = points[:,0]
    new_las.y = points[:,1]
    new_las.z = points[:,2]
    if fields:
        for name in fields.keys():
            new_las[name] = fields[str(name)]
    # new_las.__dict__.update(fields)

    return new_las

def shapefile2numpy(shapefile, ndim=2):
    """
    Transforms a shapefile to numpy array of coordinates with n-dimensions

    :param shapefile: path to shapefile
    :param ndim: number of dimensions (default 2)

    :type shapefile: str
    :type ndim: int

    :return: numpy array holding the coordinates from the shapefile

    :rtype: np.ndarray
    """
    shp = gpd.read_file(shapefile)  # known bounds
    a = []
    for index, row in shp.iterrows():
        for p in list(row['geometry'].exterior.coords):
            a.append(p)
    return np.asarray(a).reshape(-1, ndim)

def derive_boundingbox(path, files):
    """
    Extract the bounding boxes of a list of point clouds

    :param path: path to the point cloud file(s)
    :param files: names of the point cloud files

    :return:
    """
    from opals import Bounds, Import
    bounds_path = os.path.join(path, '../../test_inf/bounds')
    if not os.path.isdir(bounds_path):
        os.mkdir(bounds_path)

    [Import.Import(os.path.join(path, pcl), outFile=os.path.join( path , 'odm', pcl[:-4] +'.odm')).run() for pcl in files]
    [Bounds.Bounds(inFile=os.path.join(path, 'odm', pcl), oFormat='shp', outFile=os.path.join(bounds_path, pcl[:-4] +'.shp')).run() for pcl in os.listdir(os.path.join(path, 'odm'))]
    return bounds_path
def merge_shp(path):
    """
    Merges shapefiles in a path

    :param path:
    :return:
    """

    import geopandas as gpd

    aoi_files = [gpd.read_file(os.path.join(path, f)) for f in os.listdir(path) if f.endswith('.shp') ]
    merged = gpd.pd.concat(aoi_files)
    path_to_merged = os.path.join(path, 'bounds.shp')
    merged.to_file(path_to_merged)
    return path_to_merged+'bounds.shp'


if __name__ == '__main__':

    path = '../../ownCloud_TU/data/cave/Test'
    folders = os.listdir(path)
    for folder in folders:
        path1 = os.path.join(path, folder)
        files = [ f for f in os.listdir(path1) if f.endswith('.las') ]
        bounds_path = derive_boundingbox(path1, files)
    merge_shp(os.path.join(path, '../test_inf/bounds'))

