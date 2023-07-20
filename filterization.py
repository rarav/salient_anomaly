from opals import Import, Export
from opals import TerrainFilter as tf
from opals.AddInfo import AddInfo

import os, glob

if __name__ == '__main__':

    base_dir = r'Data/developing/2021-03-09-VQ-880-GH/'
    laz_folder = 'laz/'
    odm_folder = os.path.join(base_dir,  'odm/')

    files = glob.glob(base_dir + laz_folder + '*.laz')
    intersection_files = []

    filterize = True # for debugging purposes

    # filterization parameters
    filter2 = tf.TerrainFilter()
    filter2.robustInterpolation.gridSize = 0.3
    filtered_folder = os.path.join(base_dir, f'filtered_{filter2.robustInterpolation.gridSize}/')

    z_thresh_max = 262 # maximal z, the rest will be filterized
    z_thresh_min = 257 # minimum z, the rest will be filterized
    # define output format
    exp = Export.Export()
    exp.oFormat = 'LAS_1.4_ptRecFmt_6.xml'

    for base_file in files:
        if not os.path.exists(odm_folder):
            os.makedirs(odm_folder)

        filename = os.path.basename(base_file)[:-4]
        odm =  os.path.join(odm_folder, filename + '.odm')

        Import.Import(inFile=base_file, outFile=odm, filter=f'Generic[Z > {z_thresh_min} AND Z < {z_thresh_max} ]').run()

        print(f"fliterize {base_file}")
        # initialize classification
        AddInfo(inFile=odm, attribute='_Classification1=Classification').run()
        AddInfo(inFile=odm, attribute='Classification=0*Classification').run()

        filter2.inFile = odm
        filter2.run()

        if not os.path.exists(filtered_folder):
            os.makedirs(filtered_folder)

        # save to las
        exp.outFile = os.path.join(filtered_folder,
                                   f"{os.path.basename(base_file)[:-4]}.las")
        exp.inFile = odm
        exp.filter = f'Generic[Classification == 2]  '
        exp.run()

    print('Done!')


