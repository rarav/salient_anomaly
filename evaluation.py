import os
import pickle
import my_tools as mt
from matplotlib import pyplot as plt
import numpy as np


def plot_training_curve(root):
    metrics = {'training': [], 'validation': [], 'testing': []}

    for file in sorted(os.listdir(root), key=lambda x: int(x.split('-')[1].split('.')[0])):
        print(os.path.join(root, file))
        with open(os.path.join(root, file), 'rb') as f:
            score = pickle.load(f)
        metrics[file.split('-')[0]].append(score)

    plt.plot(metrics['training'], label='Training loss', color='C1')
    plt.plot(metrics['validation'], label='Validation loss', color='C2')
    plt.plot(metrics['testing'], label='Testing loss', color='C3')
    plt.legend()
    plt.show()


def plot_prediction(path):
    print('Showing file:', path)
    with open(path, 'rb') as f:
        grids = pickle.load(f)

    shell, pred, ref = grids
    print('Min/max/std of predictions:', np.min(pred), np.max(pred), np.std(pred))
    pred = np.clip((pred * 50 - 25).astype(int)[0], 0, 25)  # smaller -> more points

    tools.visualize_dense_grid(shell, 'shell', colorize=False)
    tools.visualize_dense_grid(pred, 'pred', colorize=False)
    tools.visualize_dense_grid(ref, 'ref')
    plt.show()

def print_metrics(file):
    with open(os.path.join(file), 'rb') as fl:
        sal_sal, sal_reg, ratio = pickle.load(fl)

    print('sal_sal:', sal_sal)
    print('sal_reg:', sal_reg)
    print('ratio:', ratio)

def average_runs(root):
    # create a dict with all runs
    runs = {}

    for folder in os.listdir(root):
        if not os.path.isdir(os.path.join(root, folder)):
            continue
        w, f, run = folder.split('_')[1:]

        best = 0.0
        sal = 0.0
        reg = 0.0
        for file in os.listdir(os.path.join(root, folder, 'metrics')):
            if not file.endswith('.loss') or not file.startswith('validation'):
                continue
            # print(file)
            with open(os.path.join(root, folder, 'metrics', file), 'rb') as fl:
                sal_sal, sal_reg, ratio = pickle.load(fl)
                if ratio > best:
                    best = ratio
                    sal = sal_sal
                    reg = sal_reg

        # store as int
        runs[(int(w), int(f), int(run))] = best
        print('Run', folder, 'best ratio:', best, 'sal:', sal, 'reg:', reg)

    print(runs.keys())

    # Plot table where x is the parameter w and y is the parameter f. Display mean and std of runs.
    w_values = [16,24,32]
    f_values = [8,16,32]
    num_runs = 5

    # set numpy print precision to 2
    np.set_printoptions(precision=2)

    table_mean = np.zeros((len(w_values), len(f_values)))
    table_std = np.zeros((len(w_values), len(f_values)))
    for w in w_values:
        for f in f_values:
            table_mean[w_values.index(w), f_values.index(f)] = np.mean([runs[(w, f, run+1)] for run in range(num_runs)])
            table_std[w_values.index(w), f_values.index(f)] = np.std([runs[(w, f, run+1)] for run in range(num_runs)])

    print(table_mean)
    print(table_std)

    #plt.imshow(table_mean)
    #plt.show()

def prepare_odm(pcl1, pcl2):
    """
    Prepares the odm for cross correlation

    :param pcl1: first point cloud(s)
    :param pcl2: second point cloud. If None, pcl1 is already a merged odm

    :return: name of the odm

    :type pcl1: list (str)
    :type pcl2: str
    """
    from opals import Import

    if pcl2 is None:
        return pcl1  # means that the file sent is a merged odm (without extension)

    path2, pcl2_name = os.path.split(pcl2)
    path1, pcl1_name = os.path.split(pcl1)

    if type(pcl1) is list:
        inFile = pcl1
        inFile.append(pcl2)
        path1, pcl1_name = os.path.split(pcl1[0])
        path_to_bounds = os.path.join(path1, 'bounds/bounds.shp')
    else:
        path1, pcl1_name = os.path.split(pcl1)
        inFile = [pcl1, pcl2]
        path_to_bounds = os.path.join(path1, 'bounds', pcl1_name[-9:-6] + '.shp')

    outFile = os.path.join(path1, pcl1_name[:-5] + '_' + pcl2_name[:-4])
    Import.Import(inFile=inFile, filter=f'Region[{path_to_bounds}]',outFile=outFile + '.odm').run()

    return outFile
def cross_correlate(pcl1, pcl2=None):
    """
    Cross correlate between two results of saliency using cross correlation (computes the cross correlation coefficient)

    :param pcl1: path to file after saliency computation
    :param pcl2: path to file after saliency computation

    exports a new las file with the cross correlation coefficient, local and relative sparsity values as
    attributes of the first cloud

    :return: none

    :type pcl1: list (str)
    :type pcl2: str, optional
    """
    from DM_tools import DM_correlate
    from opals import Export
    # define output format
    exp = Export.Export()
    exp.oFormat = 'LAS_1.4_saliency_pearson.xml'

    # define search radius
    rad = 1

    outFile = prepare_odm(pcl1, pcl2)
    DM_correlate(outFile, 'saliency', rad)
    exp.outFile = outFile + '.las'
    exp.inFile = outFile+ '.odm'
    exp.filter = 'Generic[FileId==1]'
    exp.run()

if __name__ == '__main__':
    path = '../../ownCloud_TU/data/cave'
    # plot_training_curve('./runs/results/vae_small/metrics')
    # average_runs('./runs/vae/results')
    # plot_prediction('./runs/vae/results/vae_24_16_1/training_samples/shell_pred_ref_25-1.pickle')
    path1 = os.path.join(path, 'test_inf')
    pcl2 = os.path.join(path, 'cave_saly_hand.las')

    files = os.listdir(path1)

    path_files = [os.path.join(path1, file) for file in files if file.endswith('.las')]
    # mt.derive_boundingbox(path_files)
    for file in path_files:
        # pcl1 = os.path.join(path1,file)
        cross_correlate(file,pcl2 )
        print(f'ended cross correelation with {file}')
    print('Done')
        # cross_correlate(pcl2, pcl1)
        # print('Evaluation finished')