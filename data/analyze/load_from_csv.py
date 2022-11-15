import os
from typing import List, Tuple
import config

def load_anff_from_csv(path, ignore_head=True) -> List[Tuple[int, str, float]]:
    datas = []
    head = True
    with open(os.path.join(config.AimFold,path), "r") as fileR:
        for line in fileR:
            if ignore_head and head:
                head = False
                continue

            head = False
            data = line.split(",")
            if len(data[0].strip()) < 1:
                continue
            datas.append((int(data[0].strip()), data[1].strip(), float(data[2].strip())))

    return datas

def load_count_from_csv(path, ignore_head=True) -> List[Tuple[float, int]]:
    datas = []
    head = True
    with open(os.path.join(config.AimFold,path), "r") as fileR:
        for line in fileR:
            if ignore_head and head:
                head = False
                continue

            head = False
            data = line.split(",")
            if len(data[0].strip()) < 1:
                continue
            datas.append((float(data[0].strip()), int(data[1].strip())))

    return datas