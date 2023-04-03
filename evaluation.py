import pickle
import tools
from matplotlib import pyplot as plt
import numpy as np
import laspy as lp


def plot_prediction(path):
    print(path)
    with open(path, 'rb') as file:
        grids = pickle.load(file)

    shell, pred, ref = grids
    print(pred[0, 5, 5, 5], pred[0, 16, 16, 16])
    pred = np.clip((pred * 5000 - 200).astype(int)[0], 0, 25)  # smaller -> more points

    tools.visualize_dense_grid(shell, 'shell')
    tools.visualize_dense_grid(pred, 'pred')
    tools.visualize_dense_grid(ref, 'ref')
    plt.show()


def test_las(file):
    pcl = lp.read(file)
    print(pcl.header.offsets)
    print(pcl.xyz[0])


if __name__ == '__main__':
    plot_prediction('./runs/results/vae_base/outputs/0_training/shell_pred_ref_11-0.pickle')
    # test_las('./Data/test/120114_0.las')
    # test_las('./runs/inference/120114_0_saly.las')
