# DOCUMENTATION:
# # BASE: ~CONFIG/../local_paths.yaml  # Config file to inherit from
#
# PATHS: # ----------------- Contains the absolute or relative data paths to dataset folders
#   TRAIN: ""
#   VAL_SAL: ""
#   VAL_REG: ""
#   TEST_SAL: ""
#   TEST_REG: ""
#
# MODE: "" # --------------- Experiment mode ['train_vae', 'inference', 'test_dataloader']
# CUDA: "" # --------------- CUDA device to use: str(ID)
#
# DATA: # ------------------- Data configuration
#   SHELL_SIZE: 3 # ----------- Width of the shell [vx]
#   VOXEL_SIZE: 0.2 # --------- Side length of each voxel [m]
#   IN_SIZE: 32 # ------------- Raster side length of voxel grids [vx]
#
# AUG: # -------------------- Online data augmentation
#   ROTATE: True # ------------ Apply random rotations along height axis?
#   FLIP: True # -------------- Apply random flipping?
#
# VAE_MODEL: # -------------- Variational auto encoder network
#   TYPE: "unet" # ------------ Architecture type ['unet', 'resnet', 'plane']
#   F_START: 16 # ------------- Number of features in first layer of the network
#   HAS_PARAMS: True # -------- Does the network have parameters?
#
# TRAIN: # ------------------ Parameters regarding network training
#   IT_P_EP: 1000 # ----------- Number of training iterations per epoch
#   N_EP_MAX: 100 # ----------- Maximum number of epochs
#   EARLY_STOPPING_EP: 5 # ---- Number of epochs after which training is stopped if val recon error is not decreasing
#   BTSZ: 16 # ---------------- Batch size
#   NUM_WK: 2 # --------------- Number of workers for loading training data
#   PREFETCH_FACTOR: 8 # ------ Parameter for training data loaders
#   LOSS: # ------------------- Supervised classification loss
#     TYPE: "wae" # ------------- Loss to use in ['wae', 'dice']
#   VAE: # -------------------- Variational auto encoder optimizer configuration
#     OPTIM: "adam" # ----------- Optimiser of Segmentation network in ['sgd', 'adam']
#     LR: 0.002 # --------------- Learning rate of Segmentation model
#     BETA1: 0.9 # -------------- Momentum
#     BETA2: 0.999 # ------------ 2nd momentum for ADAM
#     WDEC: 0.0 # --------------- Weight decay (0.0 = Off)
#
# TEST: # ------------------- Parameters for testing
#   NUM_POINTS: 10000 # ------- Number of points to evaluate
#   STRIDE: 1 # --------------- Load only every i-th point in MODE='inference'
#   PATH: "" # ---------------- Path to the point cloud to evaluate in MODE='inference'
#   OUT_PATH: "" # ------------ Path to store the resulting point cloud to in MODE='inference'
#
# CHECKPOINTS: # ------------ Settings for loading / saving models
#   LOAD_FROM: "" # ----------- Path to checkpoint (when loading for testing).
#   SAVE: True # -------------- Save checkpoints?
#   LOAD: True # -------------- Load checkpoint?
#   LOAD_OPT: True # ---------- Load optimizer?
#
# OUTPUTS: # ---------------- Settings for all outputs (e.g. images)
#   SAVE_TRAIN: False # ------- Save training examples
#   SAVE_METRICS: False # ----- Save validation predictions
#   FOLDER: "~CONFIG/def" # --- Root path to outputs (exp. name)

# variants
runs = [1, 2, 3, 4, 5]  # Repetitions, added to the end of the experiment name
features = [8, 16, 32]  # variants for F_START
sizes = [16, 24, 32]  # variants for IN_SIZE
rotations = [True, ]  # variants for ROTATE

# fixed variables
BASE = "~CONFIG/../local_paths.yaml"
MODE = "train_vae"
CUDA = "0"
SHELL_SIZE = 3
VOXEL_SIZE = 0.2
IN_SIZE = 16
FLIP = True
HAS_PARAMS = True
VAE_TYPE = "unet"

IT_P_EP= 1000
N_EP_MAX= 100
EARLY_STOPPING_EP= 10
BTSZ= 16
NUM_WK= 4
PREFETCH_FACTOR= 16
L_TYPE= "dice"
OPTIM= "adam"
LR= 0.0001
BETA1= 0.0
BETA2= 0.999
WDEC= 0.0
NUM_POINTS= 5000

LOAD_FROM= ""
SAVE= True
LOAD= True
LOAD_OPT= True

SAVE_TRAIN= True
SAVE_METRICS= True

VERSION= 1.17

# create and save yaml files for each variant
for run in runs:
    for feature in features:
        for size in sizes:
            for rotation in rotations:
                out_file = f'./train_vae_{size}_{feature}_{run}.yaml'
                FOLDER = f"~CONFIG/results/vae_{size}_{feature}_{run}"

                # open out_file
                with open(out_file, 'w') as f:
                    # write lines according to template
                    f.write(f'BASE: {BASE}\n\n')
                    f.write(f'MODE: {MODE}\n')
                    f.write(f'CUDA: {CUDA}\n\n')
                    f.write(f'DATA:\n')
                    f.write(f'  SHELL_SIZE: {SHELL_SIZE}\n')
                    f.write(f'  VOXEL_SIZE: {VOXEL_SIZE}\n')
                    f.write(f'  IN_SIZE: {size}\n\n')
                    f.write(f'AUG:\n')
                    f.write(f'  ROTATE: {rotation}\n')
                    f.write(f'  FLIP: {FLIP}\n\n')
                    f.write(f'VAE_MODEL:\n')
                    f.write(f'  TYPE: {VAE_TYPE}\n')
                    f.write(f'  F_START: {feature}\n')
                    f.write(f'  HAS_PARAMS: {HAS_PARAMS}\n\n')
                    f.write(f'TRAIN:\n')
                    f.write(f'  IT_P_EP: {IT_P_EP}\n')
                    f.write(f'  N_EP_MAX: {N_EP_MAX}\n')
                    f.write(f'  EARLY_STOPPING_EP: {EARLY_STOPPING_EP}\n')
                    f.write(f'  BTSZ: {BTSZ}\n')
                    f.write(f'  NUM_WK: {NUM_WK}\n')
                    f.write(f'  PREFETCH_FACTOR: {PREFETCH_FACTOR}\n')
                    f.write(f'  LOSS:\n')
                    f.write(f'    TYPE: {L_TYPE}\n')
                    f.write(f'  VAE:\n')
                    f.write(f'    OPTIM: {OPTIM}\n')
                    f.write(f'    LR: {LR}\n')
                    f.write(f'    BETA1: {BETA1}\n')
                    f.write(f'    BETA2: {BETA2}\n')
                    f.write(f'    WDEC: {WDEC}\n\n')
                    f.write(f'TEST:\n')
                    f.write(f'  NUM_POINTS: {NUM_POINTS}\n\n')
                    f.write(f'CHECKPOINTS:\n')
                    f.write(f'  LOAD_FROM: {LOAD_FROM}\n')
                    f.write(f'  SAVE: {SAVE}\n')
                    f.write(f'  LOAD: {LOAD}\n')
                    f.write(f'  LOAD_OPT: {LOAD_OPT}\n\n')
                    f.write(f'OUTPUTS:\n')
                    f.write(f'  SAVE_TRAIN: {SAVE_TRAIN}\n')
                    f.write(f'  SAVE_METRICS: {SAVE_METRICS}\n')
                    f.write(f'  FOLDER: {FOLDER}\n\n')
                    f.write(f'VERSION: {VERSION}\n')

                # close out_file
                f.close()

                # print confirmation
                print(f'Created {out_file}')

print('Done!')

