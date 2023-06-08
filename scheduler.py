import os, sys
from subprocess import  check_output
from training import run_experiment

# Note: run this script from "/code/": python scheduler.py

# =======================================================

def execute(cmd):
    # with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
    #     for line in p.stdout:
    #         print(line, end='')  # process line here
    out = check_output(cmd, shell=True)
    print(out.decode('ascii'))  # 'out' is a byte string


def run_config(config):
    # execute(f'python ./main.py {config}')
    run_experiment(config)


def run_folder(root):
    yfiles = [f for f in os.listdir(root) if f.endswith('.yaml')]
    yfiles.sort()
    print('running folder', root, yfiles)
    for file in yfiles:
        print(file)
        run_config(os.path.join(root,file))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(sys.argv)
        print("Please provide folder(s)")
        print("->  python experiment_scheduler.py ./rel/path/to/root/ ...")
        exit()

    for folder in sys.argv[1:]:
        run_folder(folder)
    exit()
