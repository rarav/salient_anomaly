from opals import Import, Export
from opals import TerrainFilter as tf
from opals.AddInfo import AddInfo

import os, glob

if __name__ == '__main__':

    base_dir = r'Data/developing/2021-03-09-VQ-880-GH/'
    laz_folder = 'laz/'
    shp_folder = 'shpfiles/'
    odm_folder = os.path.join(base_dir,  'odm/')

    exp = Export.Export()
    exp.oFormat = 'LAS_1.4_ptRecFmt_6.xml'

    files = glob.glob(odm_folder + '*.odm')
    aoi_file = glob.glob(base_dir + shp_folder + 'nb3.shp')

    z_thresh_max = 261  # maximal z, the rest will be filterized
    z_thresh_min = 257  # minimum z, the rest will be filterized


    for base_file in files:
        filename = os.path.basename(base_file)[:-4]

        exp.inFile = base_file
        exp.outFile = os.path.join(r'Data/training/2021-03-09-VQ-880-GH/',  filename + '_nb3.las')
        exp.filter = f'Region[{aoi_file[0]}] AND (Generic[Z > {z_thresh_min} AND Z < {z_thresh_max}] ' \
                     f'AND Generic[Classification == 2])'
        exp.run()

print('Done!')


