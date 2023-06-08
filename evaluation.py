import os
import pickle
import tools
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






if __name__ == '__main__':
    # plot_training_curve('./runs/results/vae_small/metrics')
    # average_runs('./runs/vae/results')
    plot_prediction('./runs/vae/results/vae_24_16_1/training_samples/shell_pred_ref_25-1.pickle')
