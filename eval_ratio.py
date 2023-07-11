import laspy
import numpy as np
import os
from opals import Import, Export
from DM_tools import DM_correlate

if __name__ == "__main__":
    path = '/home/rarav/ownCloud/data/cave/'
    pcl1 = os.path.join(path, 'cave_saly_vae_16_32_005')
    pcl2 = os.path.join(path, 'cave2_saly_plane_24')

    # define output format
    exp = Export.Export()
    exp.oFormat = 'LAS_1.4_curvature_saliency.xml'

    # define search radius
    rad = 0.3

    if not os.path.isfile(pcl1 + '.odm'):
        Import.Import(inFile= pcl1 + '.las', outFile=pcl1[:-3] + 'odm').run()
    if not os.path.isfile( pcl2 + '.odm'):
        Import.Import(inFile=pcl2 + '.las', outFile=pcl2[:-3] + 'odm').run()

    pcl1_ncc = DM_correlate(pcl1,pcl2, 'saliency', rad)






