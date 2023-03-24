import pickle
import tools
from matplotlib import pyplot as plt
import numpy as np


def plot_prediction(path):
    print(path)
    with open(path, 'rb') as file:
        grids = pickle.load(file)

    shell,pred,ref = grids
    print(pred[0,5,5,5])
    pred = np.clip((pred*50-25).astype(int)[0],0,25)
    print(pred.shape)
    tools.visualize_dense_grid(shell,'shell')
    tools.visualize_dense_grid(pred,'pred')
    tools.visualize_dense_grid(ref,'ref')
    plt.show()


if __name__ == '__main__':
    plot_prediction('./runs/results/vae_base/outputs/0_training/shell_pred_ref_1-1.pickle')