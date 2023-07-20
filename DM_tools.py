from __future__ import print_function

from opals import pyDM
import sys, os, warnings, datetime
import numpy as np
from tqdm import tqdm


# def numpy2odm(py_dict,  odm):
#     """
#     Converts a numpy array to an odm
#
#     :param py_dict: numpy dictionary which holds: 'x', 'y', 'z' - points' coordinates and other points' attributes
#     :param odm: path + filename of new odm
#
#     :type py_dict: dict
#     :type odm: str
#
#     """
#     #TODO function not finished!
#
    # # create an attribute layout for generating points with the corresponding attributes
    # lf = pyDM.AddInfoLayoutFactory()
    #
    # for name in py_dict:
    #     if name == 'x' or 'y' or 'z':
    #         continue
    #     else:
    #         lf.addColumn(pyDM.ColumnSemantic.GPSTime)
    #         lf.addColumn(pyDM.ColumnSemantic.Amplitude)
    #         lf.addColumn(pyDM.ColumnSemantic.EchoWidth)
    # layout = lf.getLayout()
    #
    # # now create 100 random points
    # print("Generate 100 random points")
    # pts = []
    # for i in range(100):
    #     pt = pyDM.Point(layout)
    #     pt.x = random.uniform(-10, 10)
    #     pt.y = random.uniform(-10, 10)
    #     pt.z = random.uniform(-10, 10)
    #     pt.info().set(0, 132 + i / 1000.)  # GPSTime
    #     pt.info().set(1, random.uniform(1, 342.3))  # Amplitude
    #     pt.info().set(2, random.uniform(0.5, 20))  # EchoWidth
    #     pts.append(pt)
    #
    # # write points to pyDM 'test.pyDM'
    # print("Write points into manager '" + odm + "'")
    #
    # # pyDM.Datamanager.create parameter: file, threadSafety(bool)
    # dm = pyDM.Datamanager.create(odm, False)
    # if not dm:
    #     print("Unable to create ODM '" + odm + "'")
    #     sys.exit(1)
    #
    # for pt in pts:
    #     dm.addPoint(pt)
    # dm.save()


def odm2numpy(odm, attributes_list=None, print_attributes=False):
    """
    Converts an odm including attributes of interest into numpy objects

    The script uses the NumpyConverter functionality of pyDM. Coordinates and attributes are converted as separate numpy array
    and stored in a single dictionary object (script works in python 2 and 3)

    :param odm: path to odm
    :param attributes_list: attributes of interest that should be converted. If "all" is sent, all attributes will be converted
    :param print_attributes: print all attributes in the odm

    :type odm: str
    :type attributes_list: list of str
    :type print_attributes: bool

    :return: numpy dictionary
    :rtype: dict
    """
    # pyDM.Datamanager.load parameter: filename(string), readOnly(bool) threadSafety(bool)
    dm = pyDM.Datamanager.load(odm, True, False)
    if not dm:
        print("Unable to open ODM '" + odm + "'")
        sys.exit(1)

    #query attribute layout that is stored within the odm
    stat = dm.getAddInfoStatistics()
    dmLayout = stat.layout()

    #output odm attribute layout
    if print_attributes:
        print("Attribute list of the ODM:")
        for i in range(dmLayout.columns()):
            name = dmLayout.name(i)
            print(i,"\t", dmLayout.name(i)," "*(25-len(name)), dmLayout.type(i))

    # if "all" is sent, all attributes are converted
    if attributes_list=='all':
        attributes_list = []
        for i in range(dmLayout.columns()):
            name = dmLayout.name(i)
            attributes_list.append(name)

    # build layout for attributes of interest (subset of odm attributes)
    lf = pyDM.AddInfoLayoutFactory()
    if attributes_list is not None:
        for attribute in attributes_list:
            type, inDM = lf.addColumn(dm, attribute,   True); assert inDM == True
    layout = lf.getLayout()

    print("Get odm points as numpy object...")
    #create dictionary of numpy objects
    numpyDict = pyDM.NumpyConverter.createNumpyDict(dm.sizePoint(),layout,True)
    pointindex = dm.getPointIndex()

    #fill numpy dictionary with all leafs of the ODM point index
    rowIdx = 0
    count = float(pointindex.sizeLeaf())
    for idx,leaf in tqdm(enumerate(pointindex.leafs()), desc='points conversion'):
        # print("%5.1f%% finished" % (idx/count*100.) )
        rowIdx += pyDM.NumpyConverter.fillNumpyDict(numpyDict,rowIdx,leaf)

    # print("100.0% finished.",rowIdx,"points have been converted")

    # list all values
    print("Output converted numpy arrays")
    if sys.version_info >= (3, 0):
        for attr, value in numpyDict.items():
            print(attr,"->", value)
    else:
        for attr, value in numpyDict.iteritems():
            print(attr,"->", value)
    return numpyDict

def odm2o3d(odm, attributes_dict=None, print_attributes=False):
    """
    Converts odm to open3d using the coordinates as coordinates
    Other features are returned as numpy dict, if required in attributes_dict

    :param odm: path to the odm
    :param attributes_dict: attributes of interest that should be converted
    :param print_attributes: print all attributes in the odm

    :type odm: str
    :type attributes_dict: list (of strings)
    :type print_attributes: bool

    :return: open3d pcl object, numpy dict of the desired attributes
    """
    import open3d as o3d

    np_dict = odm2numpy(odm, attributes_list=attributes_dict, print_attributes=print_attributes)
    xyz = np.vstack((np_dict['x'], np_dict['y'], np_dict['z']))
    del np_dict['x']
    del np_dict['y']
    del np_dict['z']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.T)

    return pcd, np_dict

def __polylgon2DM(polypoints):
    """
    Creates a polygon to be added to an  odm

    :param polypoints: points that consist the polyline

    :type polypoints: np.ndarray
    :type odm: str

    :return: a pyDM Polygon object
    :rtype: pyDM.PolygonFactory

    """

    polygons = pyDM.PolygonFactory()

    for p in polypoints:
        polygons.addPoint(*p)

    return polygons
def polygons2odm(polygons_list, odm):
    """
    Converts a list of polygons to odm

    :param polygons_list: a list of polygons, each composed of a list of points
    :param odm: the path to odm to be created (or appended, if exists)

    :return: odm path
    :rtype: str
    """

    dm = pyDM.Datamanager.load(odm, False, False)
    if not dm:
        print("Unable to open ODM '" + odm + "' \n Creating instead")
        dm = pyDM.Datamanager.create(odm, False)

    for polypoints in polygons_list:
        p = __polylgon2DM(polypoints)
        dm.addPolygon(p.getPolygon())

    return dm


def reverseZ(odm):
    """
    Reverses Z ordinate (saves to the same odm)

    :param odm: path to odm

    :type odm: str

    """
    # pyDM.Datamanager.create parameter: file, threadSafety(bool)
    dm = pyDM.Datamanager.load(odm, False)
    if not dm:
        print("Unable to create ODM '" + odm + "'")
        sys.exit(1)

    # change the z-ordinate
    for p in tqdm(dm.points(), desc='Reversing z', total=dm.sizePoint()):
        pt = p.clone()
        pt.z = -pt.z
        dm.replacePoint(pt)
    dm.save()
    print('z was reversed')

#--------------- QA - pearson coefficient computation ---------------
class correlation_kernel(pyDM.KernelPointEx):
    """callback object computing an user-defined attribute '_K_nonparam'"""
    _radius = None
    _pcl2 = None
    _attribute = None

    def __init__(self):
        """
        initialize the kernel process, and set constant parameters

        """

        # initialize the base class (mandatory!)
        if sys.version_info >= (3, 0):
            super().__init__()
        else:
            super(correlation_kernel, self).__init__()
        return

    def setArgs(self, attribute, radius):
        """
        Sets the class private arguments the second pcl and the feature

        :param attribute: the feature to evaluate the correlation to
        :param radius: radius of search

        :type pcl2: kdtree
        :type dm2: odm
        :type attribute: str
        :type nncount: int
        :type radius: float
        """
        self._feature = attribute
        self._search_dist = radius

    def tileFilter(self):
        return None
    def releaseLeaf(self):
        """
        """
        return

    def leafChanged(self, leaf, *args, **kwargs):
        """callback notifies about a changed leaf"""
        # print("process tile with id = %d" % leaf.id())
        return

    def process(self, pt, neighbours):
        r"""
        callback for computing a point's cross correlation

        """
        # extract feature values from each point neighbourhood (both pcls)
        neighbours_dict = neighbours.asNumpyDict()

        pts1_idx = np.where(neighbours_dict['FileId'] == 1)[0]
        pts2_idx = np.where(neighbours_dict['FileId'] == 2)[0]

        # match array sizes (assuming neighbours are sorted from closest):
        if pts1_idx.shape[0] > pts2_idx.shape[0]:
            pts1_idx = pts1_idx[:pts2_idx.shape[0]]

        elif pts2_idx.shape[0] > pts1_idx.shape[0]:
            pts2_idx = pts2_idx[:pts1_idx.shape[0]]

        f2 = neighbours_dict[self._feature][pts2_idx]
        f1 = neighbours_dict[self._feature][pts1_idx]

        f2[np.abs(f2) < 1e-10] = 0
        f1[np.abs(f1) < 1e-10] = 0
        f2[np.abs(f2) > 1e+10] = 1
        f1[np.abs(f1) > 1e+10] = 1

        # normalize the values
        f1_normalized = (f1 - np.min(f1)) / (np.max(f1) - np.min(f1) + 0.00001)
        f2_normalized = (f2 - np.min(f2)) / (np.max(f2) - np.min(f2) + 0.00001)

        f2_mean = f2_normalized - np.mean(f2_normalized)
        f1_mean = f1_normalized - np.mean(f1_normalized)

        # compute the cross correlation
        r_enum = np.abs((f1_mean).dot(f2_mean.T))
        r_denom = np.sqrt(f1_mean.dot(f1_mean.T)) * np.sqrt((f2_mean.dot(f2_mean.T)))
        r = r_enum/(r_denom + 0.000001)

        if np.abs(r) < 1e-10:
            r=0.
        elif np.isnan(r):
            r=0.
            print('NCC is nan, set to zero')
        # print(r)
        # r= np.round(r, decimals=2)
        pt.info().set(2, r)
        print(r)

        # compute the local sparsity
        s1_local = 1/np.mean(f1_normalized +0.000001)
        s2_local = 1/np.mean(f2_normalized)+0.000001
        s_relative = s1_local/s2_local
        pt.info().set(3, s1_local)
        pt.info().set(4,s_relative)
        return True

def DM_correlate(merged_odm, attribute, radius, **kwargs):
    r"""
    Compute the Pearson correlation coefficient of a specific feature between two point clouds.

    .. math:
        r_{xy}={\frac {\sum _{i=1}^{n}(x_{i}-{\bar {x}})(y_{i}-{\bar {y}})}{{\sqrt {\sum _{i=1}^{n}(x_{i}-{\bar {x}})^{2}}}{\sqrt {\sum _{i=1}^{n}(y_{i}-{\bar {y}})^{2}}}}}}

    with :math:`x_i` the feature value from the first point cloud, and :math:`y_i` the feature value from second point cloud

    The result is inserted into the odm of pcl1 under a new feature "_pearson_pcl2", where "pcl2" is the name of the pcl2 filename

    :param merged_odm: path to the odm file that includes both pcls
    :param attribute: the feature for which the (normalized) cross correlation should be evaluated
    :param radius: the radius of the sphere in which the cross correlation should be evalutated.

    :type pcl1: str
    :type pcl2: str
    :type attribute: str
    :type radius: float
    :type nncount: int

    :return:
    """

    try:
        dm = pyDM.Datamanager.load(merged_odm + '.odm', False, False)

    except IOError as e:
        print(e)

    # initialize an empty layout
    lf = pyDM.AddInfoLayoutFactory()
    lf.addColumn(pyDM.ColumnSemantic.FileId)  # column 0
    lf.addColumn(pyDM.ColumnType.float_, '_' + attribute)  # column 1
    lf.addColumn(pyDM.ColumnType.double_, "_pearson")  # column 2
    lf.addColumn(pyDM.ColumnType.double_, "_s_local")  # column 3
    lf.addColumn(pyDM.ColumnType.double_, "_s_relative")  # column 4
    layout = lf.getLayout()

    # create spatial selection
    # define the shape in which the neighbourhood will be searched
    search_shape = 'sphere(r=' + str(radius) + ')'
    query = pyDM.QueryDescriptor(search_shape)  # creates the query according to the shape defined

    # define a kd-tree to enable search within pcl2
    # kdtree = pyDM.PointIndexLeaf(pyDM.IndexType.kdtree,3,True)  # index type, dimension, thread safety
    # kdtree.addPoint(dm2.points())

    r = correlation_kernel()
    r.setArgs(attribute='_' + attribute, radius=radius)
    start = datetime.datetime.now()

    # create processor according to which the kernel will run (sends a point and its neighbours to the kernel)
    processFilter = pyDM.Filter('Generic[FileId==1]')
    processor = pyDM.ProcessorEx(dm, query, processFilter , layout, False, None, None, False)
    print("Computing cross correlation... ")
    print('Process started at', datetime.datetime.now().strftime('%H:%M:%S'))

    processor.run(r)  # perform computation

    diff = datetime.datetime.now() - start
    print('Process ended at', datetime.datetime.now().strftime('%H:%M:%S'))
    print("Done. Cross correlation took %.2f [s]" % diff.total_seconds())

    # save manager object
    print("Save...")
    dm.save()

    return dm
