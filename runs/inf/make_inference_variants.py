import os
BASE= "\"~CONFIG/../train_vae_24_16_1.yaml\""
DATA_PATH = "/home/reuma/ownCloud_TU/data/Pielach/Data/training"
FOLDER = f"~CONFIG/../../inference"

CUDA = '\"0\"'

MODE = '\"inference\"'
BTSZ = 64

NUM_POINTS = -1
STRIDE = 1

SAVE = 'False'
LOAD = 'True'
LOAD_OPT = 'False'


files = os.listdir(DATA_PATH)
base_name = os.path.split(BASE)


# create and save yaml files for each variant
for file_name in files:
    if not file_name.endswith('.las'):
        continue
    out_file = f'inf_vae_{file_name[:-4]}_{base_name[1][10:-8]}.yaml'
    LOAD_FROM = f'\"~CONFIG/../results/vae_{base_name[1][10:-6]}/checkpoints/validation.pt\"'
    # open out_file
    with open(out_file, 'w') as f:
        # write lines according to template
        f.write(f'BASE: {BASE}\n\n')
        f.write(f'MODE: {MODE}\n')
        f.write(f'CUDA: {CUDA}\n\n')
        f.write(f'TRAIN:\n')
        f.write(f'  BTSZ: {BTSZ}\n\n')
        f.write(f'TEST:\n')
        f.write(f'  NUM_POINTS: {NUM_POINTS}\n')
        f.write(f'  STRIDE: {STRIDE}\n')
        f.write(f'  PATH: "{DATA_PATH}/{file_name}"\n')
        f.write(f'  OUT_PATH: \"{FOLDER}/{os.path.split(out_file)[1][:-5]}.las\" \n\n')
        f.write(f'CHECKPOINTS:\n')
        f.write(f'  LOAD_FROM: {LOAD_FROM}\n')
        f.write(f'  SAVE: {SAVE}\n')
        f.write(f'  LOAD: {LOAD}\n')
        f.write(f'  LOAD_OPT: {LOAD_OPT}\n\n')
        f.write(f'OUTPUTS:\n')
        f.write(f'  FOLDER: {FOLDER}\n\n')

    # close out_file
    f.close()

    # print confirmation
    print(f'Created {out_file}')

print('Done!')

