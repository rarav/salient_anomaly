import os, datetime, sys, time
from os.path import exists, join as pjoin

import matplotlib.pyplot as plt
import torch

import config
import datamanagement
import model
import tools
from datamanagement import TrainingDataset, EvalDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional
import torch.nn as nn
from torch import optim
from functools import partial
from tools import *
import pickle


class ExperimentHandler:
    vae: nn.Module  # ------------------ variational auto encoder
    opt: object  # --------------------- optimizer
    cf: config.Config  # --------------- configuration object (see args.py)
    e: int  # -------------------------- current epoch (int)
    e_run: int  # ---------------------- num. of trained epochs in run (different from e if training was interrupted)
    e_es: int  # ----------------------- num. of epochs since model improvement (e.g. increase of validation score)
    F: FolderDict  # ------------------- dictionary of folders for saving/loading
    top_scores: dict  # ---------------- dictionary of best scores
    train_start_time: float  # --------- training start time
    epoch_start_time: float  # --------- start time of current epoch
    device: str  # --------------------- device to use ('cpu' or 'cuda')
    block3d: torch.nn.Module

    tds: Dataset
    tdl: DataLoader
    vds: EvalDataset
    vdl: DataLoader
    tsds: EvalDataset
    tsdl: DataLoader

    # ============================================================================================================ INIT

    def __init__(self, cf: config.Config):
        """Create ExperimentHandler based on configuration object.

        Initializes ExperimentHandler and corresponding attributes. Sets up folder structure and auxiliary variables.
        :param cf: Configuration object.
        """
        self.cf = cf
        print(f'\n\033[31m{cf.OUTPUTS.FOLDER}\033[0m\n')
        self.make_output_folders()
        self.store_config()
        self.init_cuda()
        self.init_aux_vars()

    def make_output_folders(self):
        """Create folders for storing results.

        When using folder dicts, the corresponding folders are created on the first usage.
        This prevents creating unused folders.
        """
        root = self.cf.OUTPUTS.FOLDER
        os.makedirs(root, exist_ok=True)
        self.F = FolderDict({
            'metrics': pjoin(root, 'confusion_matrices'), 'checkpoints': pjoin(root, 'checkpoints'),
            'train': pjoin(root, 'outputs/0_training'), 'validation': pjoin(root, 'outputs/1_validation'),
            'testing': pjoin(root, 'outputs/2_testing'),
        })

    def store_config(self):
        """Stores the full yaml file including inherited attributes to the output folder.

        The stored yaml file may be reused to repeat an experiment.
        """
        now_s = current_datetime_as_str() + '.yaml'
        with open(pjoin(self.cf.OUTPUTS.FOLDER, now_s), 'w+') as f:
            f.write(f"# python main.py {' '.join(sys.argv[1:])}\n")
            f.write(str(self.cf))

    def init_cuda(self):
        """Defines which GPU should be used.

        Switches to CPU if cf.CUDA = -1 or if no GPU is available.
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cf.CUDA)
        if torch.cuda.is_available() and int(self.cf.CUDA) >= 0:
            print(f'Using GPU nr. {self.cf.CUDA}: {torch.cuda.get_device_name(0)}')
            self.device = 'cuda'
            torch.cuda.empty_cache()
        else:
            print('Using CPU only!')
            self.device = 'cpu'

    def init_aux_vars(self):
        """Sets up all auxiliary variables.

        Auxiliary variables are used for computing/storing metrics, tracking epoch, compute weighted loss.
        Note that the confusion matrix (self.CM) is reused multiple times.
        """
        self.e, self.e_es = -1, -1
        self.train_start_time = -1.0
        self.top_scores = {'training': 0.0, 'validation': 0.0, 'testing': 0.0}

        self.block3d = nn.Conv3d(1, 1, (3, 3, 3), padding=1, bias=False)
        torch.nn.init.constant_(self.block3d.weight, 1.0 / (3 * 3 * 3))
        for param in self.block3d.parameters():  # mask is not updated
            param.requires_grad = False
        self.block3d.to(self.device)

        # =============================================================================================== EXPERIMENT SETUPS

    def prepare_datasets(self):
        """ Sets up datasets and data loaders for training, validation and testing."""

        cf = self.cf
        DL_v = partial(DataLoader, batch_size=cf.TRAIN.BTSZ, shuffle=False)  # --- Used for val/test sets
        DL_t = partial(DataLoader, batch_size=cf.TRAIN.BTSZ, shuffle=True, num_workers=cf.TRAIN.NUM_WK,
                       prefetch_factor=cf.TRAIN.PREFETCH_FACTOR, persistent_workers=True, pin_memory=True)  # --- train

        if cf.PATHS.TRAIN:
            self.tds = TrainingDataset(cf)
            self.tdl = DL_t(dataset=self.tds)
        if cf.PATHS.VALIDATION:
            self.vds = EvalDataset(cf, 'validation')
            self.vdl = DL_v(dataset=self.vds)
        if cf.PATHS.TEST:
            self.tsds = EvalDataset(cf, 'test')
            self.tsdl = DL_v(dataset=self.tsds)

    def load_checkpoint(self):
        """Loads a checkpoint.

        If LOAD_FROM is specified in the config, the models in the respective checkpoint will be used.
        Else the 'latest.pt' model will be used if it exists (e.g. for continuing training)
        If 'latest.pt' does not exist, the models from LOAD_INIT will be used (e.g. for domain adaptation)
        or default initializations are used (training from scratch or pre-trained weights).
        """

        load = torch.load if torch.cuda.is_available() else partial(torch.load, map_location='cpu')
        load_model = self.cf.CHECKPOINTS.LOAD_FROM

        if load_model:  # --------------------------------------- Case 1: Explicit model given by LOAD_FROM
            assert exists(load_model), f"Checkpoint cf.CHECKPOINTS.LOAD_FROM does not exist: {load_model}"
            print('Loading seg model from cf.CHECKPOINTS.LOAD_FROM: ', load_model)
            return load(load_model)
        else:
            checkpoint_file = pjoin(self.F['checkpoints'], 'latest.pt')
            if exists(checkpoint_file):  # ----------------------- Case 2: Loading a native checkpoint
                print('Loading native checkpoint:', checkpoint_file)
                return load(checkpoint_file)

    def restore_checkpoint(self, checkpoint):
        """Initializes network(s) from a checkpoint.

        If weights for networks are missing or incompatible, they will be skipped.
        :param checkpoint: The loaded checkpoint file to be restored.
        """

        cf = self.cf
        if checkpoint:
            shape_mismatch = False
            if cf.CHECKPOINTS.LOAD:
                try:
                    state_dict_to_load = checkpoint[f'state_dict']
                    target_state_dict = self.vae.state_dict()
                    target_keys = target_state_dict.keys()
                    to_drop = []
                    for k, v in state_dict_to_load.items():
                        if k not in target_keys:
                            print(f"Parameter {k} is missing")
                            continue
                        if target_state_dict[k].shape != v.shape:
                            print(f"Removing param set {k} due to size mismatch")
                            to_drop.append(k)
                            shape_mismatch = True
                    for k in to_drop:
                        state_dict_to_load.pop(k)
                    self.vae.load_state_dict(state_dict_to_load)
                except ValueError:
                    print(f'Network incompatible')
                except KeyError:
                    print(f'Network not in checkpoint')
            if cf.CHECKPOINTS.LOAD_OPT and not shape_mismatch:
                try:
                    state_dict_to_load = checkpoint[f'optimizer_state_dict']
                    self.opt.load_state_dict(state_dict_to_load)
                except ValueError:
                    print(f'Optimizer for incompatible')
                except KeyError:
                    print(f'Optimizer for not in checkpoint')

            self.e = checkpoint['epoch']
            self.e_es = checkpoint['es_epochs']

            stored_top_scores = checkpoint['top_scores']
            for entry in stored_top_scores.keys():
                self.top_scores[entry] = stored_top_scores[entry]

            print(f"Restored checkpoint from epoch {self.e}. Top scores: {self.top_scores}")

    def init_network(self, make_optimizer=True):
        """Initializes a 3DVAE and (optionally) its optimizer.

        Used to generate network based on configuration file.
        :param make_optimizer: Creates the optimizer for the network (not required for evaluation).
        """

        self.vae = model.Autoencoder3DRes(self.cf, self.device)
        if not make_optimizer: return self.vae

        # --------------------------------------------------------------------- Setup optimizer:
        hprm = self.cf.TRAIN.VAE
        btsz = self.cf.TRAIN.BTSZ
        prms = self.vae.parameters()

        if hprm.OPTIM == 'sgd':
            self.opt = optim.SGD(prms, lr=hprm.LR, momentum=hprm.BETA1, weight_decay=hprm.WDEC)
            print(f"Opt.: SGD: lr {hprm.LR} M {hprm.BETA1} [wdec={hprm.WDEC}, bs={btsz}]")
        elif hprm.OPTIM == 'adam':
            self.opt = optim.Adam(prms, lr=hprm.LR, betas=(hprm.BETA1, hprm.BETA2), weight_decay=hprm.WDEC)
            print(f"Opt.: ADAM: lr {hprm.LR} B1/2 {hprm.BETA1}/{hprm.BETA2} [wdec={hprm.WDEC}, bs={btsz}]")
        else:
            raise NotImplementedError(f"Optimizer {hprm.OPTIM} is not supported")

        return self.vae, self.opt

    # ======================================================================================================= AUXILIARY

    def print_training_progress(self):
        """Print information about training progress (epoch, early stopping, runtime, ETA).

        Nothing is printed if verbosity is set to 0.
        """
        cf = self.cf
        if self.train_start_time < 0:  # Setup at first call of the function
            self.train_start_time = self.epoch_start_time = time.time()
            self.e_run = -1

        self.e_run += 1
        run_time = time.time() - self.train_start_time
        it_time = (time.time() - self.epoch_start_time) / cf.TRAIN.IT_P_EP
        self.epoch_start_time = time.time()
        eta_sek = int(run_time / self.e_run * (cf.TRAIN.N_EP_MAX - self.e + 1)) if self.e_run > 0 else 0
        eta_hr, eta_min, eta_sec = sek2hms(eta_sek)

        print(f"\n[{current_datetime_as_str()} / {int(run_time // 60)} min]\
         epoch {self.e}/{cf.TRAIN.N_EP_MAX}, erl.st. cnt. {self.e_es}/{cf.TRAIN.EARLY_STOPPING_EP}\
        , estimated remaining training time is {eta_hr}:{eta_min} ({int(it_time * 1000)} ms/batch)")

    def print_num_params(self):
        """Prints the number of parameters for the VAE."""

        params = list(self.vae.parameters())
        pp = np.sum([np.prod(list(P.size())) for P in params])
        print(f'Model has {pp} prms in {len(params)} vars ({int(pp * 4 / 1000 / 1000 * 10) / 10} MB)')

    def check_finished(self):
        """Checks if training is finished.

        This function should be called after restoring a checkpoint and before each epoch to check
        if training is already finished. Considers max epochs and early stopping. The function returns
        True if training is finished and False otherwise.
        """

        return (0 < self.cf.TRAIN.N_EP_MAX <= self.e) or (0 < self.cf.TRAIN.EARLY_STOPPING_EP <= self.e_es)

    # ====================================================================================================== EVALUATION

    def evaluate_on_subset(self, subset):
        if subset == 'validation':
            if not self.cf.PATHS.VALIDATION: return
            dl = self.vdl
        else:
            if not self.cf.PATHS.TEST: return
            dl = self.tsdl

        recon_errors = []
        for batch in dl:
            dense = batch['dense'].to(self.device, non_blocking=True)
            shell = batch['shell'].to(self.device, non_blocking=True)
            with torch.no_grad():
                recons = self.vae(shell)
                loss = self.loss(recons, dense, keep_batch_axis=True)
            recon_errors.extend(tensor2numpy(loss))

        print("Average reconstruction error:", np.mean(np.array(recon_errors)))

    # ========================================================================================================= WRITING

    def may_save_model(self, f_name: str, net_only=False):
        """Saves the current model status if cf.CHECKPOINTS.SAVE is True.

        :param f_name: Name of file to write to (is appended to the 'checkpoint' folder)
        :param net_only: Don't save optimizer.
        """

        cf = self.cf
        if not cf.CHECKPOINTS.SAVE:
            print("Skipping to save network")
            return

        if not net_only:
            save_dict = {'epoch': self.e, 'es_epochs': self.e_es, 'top_scores': self.top_scores}
            save_dict[f'state_dict'] = self.vae.state_dict()
            save_dict[f'optimizer_state_dict'] = self.opt.state_dict()
        else:
            save_dict = {f'state_dict': self.vae.state_dict()}

        print(f"Saving network to {f_name}")
        torch.save(save_dict, pjoin(self.F['checkpoints'], f_name))

    def save_training_samples(self, input_shells, predictions, references):
        """ Save the first (max=4) training samples in a batch.

        """
        root = self.F['train']
        for b in range(min(input_shells.shape[0], 4)):
            file_name = f'shell_pred_ref_{self.e}-{b}.pickle'
            grids = [tools.tensor2numpy(input_shells[b]),
                     tools.tensor2numpy(predictions[b]),
                     tools.tensor2numpy(references[b])]
            with open(pjoin(root, file_name), 'wb') as file:
                pickle.dump(grids, file)

    # ======================================================================================================== TRAINING

    def loss(self, reconstruction, reference, keep_batch_axis=False):
        """Setup loss according to the configuration and sets initial weights for classes if provided.

        Reference is encoded as (number of points - 1). Voxels containing a single point are ignored in the loss.
        The remaining voxels are clipped to 0 (free) or 1 (occupied).
        Predictions are bound to the range 0 - 1 by the sigmoid function.
        Loss is average square error for voxels that are not ignored.
        """

        if self.cf.TRAIN.LOSS.TYPE == 'wae':
            # pos_preds = (reconstruction > 0.5).float()
            # sure = (torch.abs(reconstruction - 0.5) > 0.1).float()
            norm_ref = torch.clip(reference, 0, 1)
            # correct = ((norm_ref == pos_preds)).float()
            # ignore = sure * correct
            # consider = (1 - ignore)
            weights = (reference != 0).float()  # torch.ones_like(reference)  # , 1, 2)
            # weights += (reference > 1).float() # double weight if at least two hits in voxel
            weights = weights.detach()  # consider *

            # ref_blurred = self.block3d(torch.unsqueeze(norm_ref, 1))[:, 0, :, :]
            # ref_max = torch.maximum(ref_blurred, norm_ref)
            abs_diff = (reconstruction[:, 0, :, :] - norm_ref)

            # return torch.sum(torch.abs(abs_diff) * weights) / torch.sum(weights)
            if not keep_batch_axis:
                return torch.sum(torch.pow(abs_diff, 2) * weights) / torch.sum(weights)
            else:
                return torch.sum(torch.pow(abs_diff, 2) * weights, dim=(1, 2, 3)) / torch.sum(weights, dim=(1, 2, 3))

    def train_vae(self):
        vae, vae_opt = self.init_network()
        self.print_num_params()
        self.restore_checkpoint(self.load_checkpoint())
        if self.check_finished(): return
        self.prepare_datasets()

        while not self.check_finished():
            self.e += 1
            self.e_es += 1
            self.print_training_progress()
            vae.train()
            avl = []
            for i, batch in enumerate(self.tdl):
                shell = batch['shell'].to(self.device, non_blocking=True)
                dense = batch['dense'].to(self.device, non_blocking=True)

                recons = vae(shell)
                loss = self.loss(recons, dense)
                vae_opt.zero_grad()
                loss.backward()
                vae_opt.step()

                if not i % 50:
                    print(f'\r {i}/{self.cf.TRAIN.IT_P_EP} - Loss: {loss.item():.3f}', end='', flush=True)
                    avl.append(loss.item())
            print(f' - average loss: {np.mean(avl):.3f}')
            vae.eval()
            self.save_training_samples(shell, recons, dense)
            # self.quick_eval_seg_on_training_batch()
            # self.may_update_weights(self.CM)
            self.evaluate_on_subset('validation')
            self.evaluate_on_subset('testing')
            self.may_save_model('latest.pt')

    def inference(self):
        vae = self.init_network(make_optimizer=False)
        self.print_num_params()
        self.restore_checkpoint(self.load_checkpoint())

        ds = datamanagement.PclDataset(self.cf)
        dl = DataLoader(dataset=ds, batch_size=self.cf.TRAIN.BTSZ, shuffle=False)

        recon_errors = []
        cnt = 0
        start = time.time()

        for batch in dl:
            cnt += self.cf.TRAIN.BTSZ
            if (cnt/self.cf.TRAIN.BTSZ) % 10 == 0:
                crnt = time.time()
                diff = crnt - start
                print(f'\r Points: {cnt}/{len(ds)} time (sek): {diff:.1f}/{diff/cnt*len(ds):.1f}',end='',flush=True)
            dense = batch['dense'].to(self.device, non_blocking=True)
            shell = batch['shell'].to(self.device, non_blocking=True)

            with torch.no_grad():
                recons = vae(shell)
                loss = self.loss(recons, dense, keep_batch_axis=True)
            recon_errors.extend(tensor2numpy(loss))

        ds.save_with_saliency(recon_errors)
        print("Average reconstruction error:", np.mean(np.array(recon_errors)))

        # may save model and results (as pcl?)

    def test_dataloader(self):
        self.prepare_datasets()

        # for j in range(10):
        #     start = time.time()
        #     for i in range(100):
        #         self.tds.get_random_voxel_crop()
        #     print(f'loading 100 crops took {time.time() - start} seconds.')
        #

        for i, batch in enumerate(self.tdl):
            shell = batch['shell'].to(self.device, non_blocking=True)
            dense = batch['dense'].to(self.device, non_blocking=True)

            visualize_dense_grid(tensor2numpy(shell[0]), 'shell')
            visualize_dense_grid(tensor2numpy(dense[0]), 'dense')
            plt.show()


def run_experiment(cf_path):
    c = config.Config(cf_path)

    c.check_consistency(config.Config('./Documentation/_config_documentation.yaml'))
    H = ExperimentHandler(c)

    if c.MODE == 'train_vae':
        H.train_vae()

    if c.MODE == 'inference':
        H.inference()

    if c.MODE == 'test_dataloader':
        H.test_dataloader()

    del c, H
    print('Experiment done!\n')


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python main.py path/to/config.yaml or use experiment_scheduler.py"
    run_experiment(sys.argv[1])
