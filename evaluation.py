import os
import pickle
import tools
from matplotlib import pyplot as plt
import numpy as np


def plot_training_curve(root):
    metrics = {'training': [], 'validation': [], 'testing': []}

    for file in sorted(os.listdir(root),key=lambda x: int(x.split('-')[1].split('.')[0])):
        print(os.path.join(root, file))
        with open(os.path.join(root, file),'rb') as f:
            score = pickle.load(f)
        metrics[file.split('-')[0]].append(score)

    plt.plot(metrics['training'], label='Training loss', color='C1')
    plt.plot(metrics['validation'], label='Validation loss', color='C2')
    plt.plot(metrics['testing'], label='Testing loss', color='C3')
    plt.legend()
    plt.show()


def plot_prediction(path):
    print(path)
    with open(path, 'rb') as f:
        grids = pickle.load(f)

    shell, pred, ref = grids
    print(pred[0, 5, 5, 5], pred[0, 16, 16, 16])
    print(np.min(pred), np.max(pred), np.std(pred))
    pred = np.clip((pred * 50 - 25).astype(int)[0], 0, 25)  # smaller -> more points

    tools.visualize_dense_grid(shell, 'shell')
    tools.visualize_dense_grid(pred, 'pred')
    tools.visualize_dense_grid(ref, 'ref')
    plt.show()


if __name__ == '__main__':
    plot_training_curve('./runs/results/vae_small/metrics')
    plot_prediction('./runs/results/vae_small/training_samples/shell_pred_ref_9-1.pickle')
