from pathlib import Path
import os
import csv
import torch

from library import DS_PATH


def path_check_and_fix(pathfile):
    '''
    pathfile is assumed to be abs path
    '''
    try:
        path2check = os.path.abspath(os.path.join(pathfile, ".."))
        Path(path2check).mkdir(parents=True, exist_ok=True)
        return pathfile
    except Exception as e:
        print("Error when attempting to fix path!")
        print(e)
        raise


def save2csv(csv_data, filepath=None):
    if filepath:
        print("savings to csv...")
        filepath = path_check_and_fix(filepath)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
    else:
        print("not saving, please insert filepath")


def save_torch_model(state, filepath=None):
    if filepath:
        filepath = path_check_and_fix(filepath)
        torch.save(state, filepath)
    else:
        print("not saving, please insert filepath")
