from __future__ import print_function

from opals import pyDM
import sys
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

def o3d2odm(points, odm, attributes_dict=None, print_attributes=False):
    """
    Converts odm to open3d using the coordinates as coordinates
    Other features are returned as numpy dict, if required in attributes_dict

    :param points: points to convert to the odm
    :param odm: path to the odm
    :param attributes_dict: attributes of interest that should be converted
    :param print_attributes: print all attributes in the odm

    :type odm: str
    :type attributes_dict: dict (of strings)
    :type print_attributes: bool

    :return: open3d pcl object, numpy dict of the desired attributes
    """
    import open3d as o3d

    # pyDM.Datamanager.load parameter: filename(string), readOnly(bool) threadSafety(bool)
    dm = pyDM.Datamanager.load(odm, False, False)
    if not dm:
        print("Unable to open ODM '" + odm + "'")
        dm = pyDM.Datamanager.create(odm, False)
        if not dm:
            print("Unable to create ODM '" + odm + "'")
            sys.exit(1)

    # build layout for retrieving echo width values (subset of odm attributes)
    lf = pyDM.AddInfoLayoutFactory()
    for att in attributes_dict.key():
        lf.addColumn(pyDM.ColumnType.float_, att)

    layoutRead = lf.getLayout()
    layoutWrite = lf.getLayout()

    print("Get odm echo width as numpy object...")
    # create dictionary of numpy objects
    numpyDict = pyDM.NumpyConverter.createNumpyDict(dm.sizePoint(), layoutRead, False)
    print("len(numpyDict)=", len(numpyDict))
    pointindex = dm.getPointIndex()

    # fill numpy dictionary with all leafs of the ODM point index
    rowIdx = 0
    count = float(pointindex.sizeLeaf())
    for idx, leaf in enumerate(pointindex.leafs()):
        print("%5.1f%% finished" % (idx / count * 100.))
        rowIdx += pyDM.NumpyConverter.fillNumpyDict(numpyDict, rowIdx, leaf)

    print("100.0% finished.", rowIdx, "values have been converted")

    print("\nCompute min max echo width for single echos using numpy...")
    mask = numpyDict["NrOfEchos"] == 1  # generate single echo mask
    minValue = (numpyDict["EchoWidth"][mask]).min()
    maxValue = (numpyDict["EchoWidth"][mask]).max()
    print("\tmin=%.2f" % minValue)
    print("\tmax=%.2f" % maxValue)

    # compute parameter of linear transform function
    k = 1. / (maxValue - minValue)
    d = -minValue * k
    normalizedEchoWidth = numpyDict["EchoWidth"] * k + d

    # check if linear transform function
    minCheck = (normalizedEchoWidth[mask]).min();
    assert abs(minCheck) < 1e-10
    maxCheck = (normalizedEchoWidth[mask]).max();
    assert abs(maxCheck - 1) < 1e-10

    # store normalized echo width values within the odm
    storeDict = {}
    storeDict["_normalizedEchoWidth"] = normalizedEchoWidth
    print("\nStore normalised echo width values in ODM...")
    for idx, leaf in enumerate(pointindex.leafs()):
        print("%5.1f%% finished" % (idx / count * 100.))

        # pyDM.NumpyConverter.setFromNumpyDict( numpyDict, translators, leaf, layout, filter = None)
        # for details on the function, please refer to the docu
        pyDM.NumpyConverter.setFromNumpyDict(storeDict, [], leaf, layoutWrite)  # we don't need translators in this case

        # mark leaf as changed/dirty that it will be written do disk again
        leaf.setChanged(True)

    print("100.0% finished.")

    print("\nSave odm...")
    dm.save()
    print("finished")


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
