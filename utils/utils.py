import os
import yaml

import torch
from datetime import datetime

def use_best_hyperparams(args, dataset_name):
    best_params_file_path = "best_hyperparams.yml"
    with open(best_params_file_path, "r") as file:
        hyperparams = yaml.safe_load(file)

    for name, value in hyperparams[dataset_name].items():
        print(f' {name}: {value}')
        if hasattr(args, name):
            setattr(args, name, value)
        else:
            raise ValueError(f"Trying to set non existing parameter: {name}")

    return args


def get_available_accelerator():
    if torch.cuda.is_available():
        return "gpu"
    # Keep the following commented out as some of the operations
    # we use are currently not supported by mps
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     return "mps"
    else:
        return "cpu"

def log_file(net_to_print, dataset_to_print, args):
    log_file_name = dataset_to_print+'_'+net_to_print+'_'+args.conv_type+'K'+str(args.k_plus)+'_lay'+str(args.layer)+'_lr'+str(args.lr)+'_split'+str(args.num_runs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name_with_timestamp = f"{log_file_name}_{timestamp}.log"

    log_directory = "~/Documents/Benlogs/"  # Change this to your desired directory
    log_directory = os.path.expanduser(log_directory)

    return log_directory, log_file_name_with_timestamp






