import os

from yaml import load, Loader, dump


def joined_yaml(base: dict, new: dict):
    """ Update this config with base yaml file.
    Shared arguments will be overwritten by self

    :param base: base yaml object to be joined
    :param new: new yaml object to be joined
    :return raw joined text
    """

    joined = {}

    for key, v in base.items():
        if key == 'BASE': continue
        if isinstance(v, dict):
            if key in new.keys():  # ----------------------------------------------------------------------- update node
                joined[key] = joined_yaml(v, new[key])
            else:  # ------------------------------------------------------------------------------------ keep base node
                joined[key] = v.copy()
        else:
            if key in new.keys():
                joined[key] = new[key]
            else:
                joined[key] = v

    for key, v in new.items():
        if key == 'BASE': continue
        if key in base.keys(): continue
        if isinstance(v, dict):
            joined[key] = v.copy()
        else:
            joined[key] = v

    return joined


class Config(object):
    # NOTE: The inital attributes are only required for type hints in the IDE!
    # The helper function 'make_prototype_from_yaml' can be used to generate the respective code.
    # START OF AUTOMATED CODE
    class PROT_PATHS(object):
        TRAIN: str
        VALIDATION: str
        TEST1: str
        TEST2: str

    PATHS = PROT_PATHS()

    MODE: str
    CUDA: str

    class PROT_DATA(object):
        SHELL_SIZE: int
        VOXEL_SIZE: float
        IN_SIZE: int

    DATA = PROT_DATA()

    class PROT_AUG(object):
        ROTATE: bool
        FLIP: bool

    AUG = PROT_AUG()

    class PROT_VAE_MODEL(object):
        TYPE: str
        F_START: int
        HAS_PARAMS: bool

    VAE_MODEL = PROT_VAE_MODEL()

    class PROT_TRAIN(object):
        SD_SET: str
        IT_P_EP: int
        N_EP_MAX: int
        EARLY_STOPPING_EP: int
        BTSZ: int
        NUM_WK: int
        PREFETCH_FACTOR: int

        class PROT_LOSS(object):
            TYPE: str

        LOSS = PROT_LOSS()

        class PROT_VAE(object):
            OPTIM: str
            LR: float
            BETA1: float
            BETA2: float
            WDEC: float

        VAE = PROT_VAE()

    TRAIN = PROT_TRAIN()

    class PROT_TEST(object):
        NUM_POINTS: int
        STRIDE: int
        PATH: str
        OUT_PATH: str

    TEST = PROT_TEST()

    class PROT_CHECKPOINTS(object):
        LOAD_FROM: str
        SAVE: bool
        LOAD: bool
        LOAD_OPT: bool

    CHECKPOINTS = PROT_CHECKPOINTS()

    class PROT_OUTPUTS(object):
        SAVE_TRAIN: bool
        SAVE_METRICS: bool
        FOLDER: str

    OUTPUTS = PROT_OUTPUTS()

    # END OF AUTOMATED CODE

    def __init__(self, path=None, yaml=None, root=""):
        """ Generate config object.
        Either path or yaml dict has to be provided.
        All configurations can be accessed in object style.
        BASE config will only be considered if the path is given.

        :param path: path to a config file
        :param yaml: yaml data as dict
        :param root: root of the config (will be automatically identified if path is given and used to replace ~CONFIG)
        """

        assert (path is None) ^ (yaml is None), "Either Path or YAML dict has to be provided"
        if not path is None:
            print('init config from',path)
            with open(path, 'r') as file:
                self.absfile = os.path.abspath(path)
                self.root = '/'.join(self.absfile.split(os.sep)[:-1])
                raw = file.read()
                raw = raw.replace("~CONFIG", self.root)
                raw = raw.replace("\t", "    ")
                self.yaml = load(raw, Loader)
        else:
            self.yaml = yaml
            self.root = root

        if 'BASE' in self.yaml.keys() and not path is None:
            # print(self.yaml['BASE'])
            base = Config(path=self.yaml['BASE'])
            self.yaml = joined_yaml(base.yaml, self.yaml)

        self.__set_attributes_from_yaml__()

    def __set_attributes_from_yaml__(self):
        """Transform yaml to attributes """

        for key, v in self.yaml.items():
            if key == 'BASE': continue
            if isinstance(v, (list, tuple)):
                new_v = [Config(yaml=x, root=self.root) if isinstance(x, dict) else x for x in v]
            else:
                new_v = Config(yaml=v, root=self.root) if isinstance(v, dict) else v
            setattr(self, key, new_v)

    def __str__(self):
        return dump(self.yaml)

    def __repr__(self):
        return dump(self.yaml)

    def check_consistency(self, prot: object):
        """Will throw AssertionError if a declared attribute is missing in prototype!

        :param prot: Prototype config object to compare this instance to.

        Note that no type-checks are done as some variables can be given as simple type or list!
        Not all arguments defined in the prototype have to be provided! (May lead to runtime errors)
        Exemplary usage:
         C = Config('../exp/tests/test_config.yaml')
         PROT = Config('./Documentation/_config_documentation.yaml')
         C.check_consistency(PROT)
        """
        for att in dir(self):
            if att.startswith('__'): continue
            if not att in dir(prot):
                print(f"WARNING: Node {att} is not listed in prototype!")
                continue
            att_v = self.__getattribute__(att)
            if issubclass(type(att_v), Config):
                att_v.check_consistency(prot.__getattribute__(att))


def make_prototype_from_yaml(path):
    """Creates the attribute code for using the config in the ide

    .
    """

    def append_node(node, indentation):
        code = ""
        for key, v in node.items():
            if isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, dict):
                        code += " " * indentation + f"class PROT_{key}(object):\n"
                        code += append_node(x, indentation + 4)
                        code += " " * indentation + f"{key} = PROT_{key}()\n\n"
                    else:
                        code += " " * indentation + f"{key}: {type(x).__name__}\n"
            else:
                if isinstance(v, dict):
                    code += " " * indentation + f"class PROT_{key}(object):\n"
                    code += append_node(v, indentation + 4)
                    code += " " * indentation + f"{key} = PROT_{key}()\n\n"
                else:
                    code += " " * indentation + f"{key}: {type(v).__name__}\n"
        return code

    with open(path, 'r') as file:
        raw = file.read()
        raw = raw.replace("\t", "    ")
        yaml = load(raw, Loader)

    code = append_node(yaml, 0)
    print(code)


if __name__ == '__main__':
    # If the exemplary yaml file for documentation was changed, run this file to update the code in the Config class
    make_prototype_from_yaml('./Documentation/_config_documentation.yaml')
