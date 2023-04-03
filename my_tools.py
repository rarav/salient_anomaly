# Utility functions which do not require OPALS

import laspy
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
