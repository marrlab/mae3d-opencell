from omegaconf import OmegaConf
from pathlib import Path
import random
import torch
import torch.backends.cudnn as cudnn
import warnings


def get_conf(conf_file: str):
    conf_file = Path(conf_file)
    if not conf_file.exists():
        raise FileNotFoundError(f"Config file does not exist: {conf_file}")

    conf = OmegaConf.load(conf_file)
    OmegaConf.resolve(conf)  # resolves ${...} like run_name, ckpt_dir

    # Optional: derive paths safely
    conf.output_dir = str(Path(conf.output_dir) / conf.run_name)
    conf.ckpt_dir = str(Path(conf.output_dir) / "ckpts")

    # Optional defaults (only if missing)
    if not hasattr(conf, "num_samples") or conf.num_samples is None:
        conf.num_samples = 4

    return conf

def set_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        # np.random.seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        cudnn.benchmark = True
