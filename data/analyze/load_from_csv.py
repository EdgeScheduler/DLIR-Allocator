import os
from typing import List, Tuple
import config

def load_from_csv(path, ignore_head=True) -> List[Tuple[int, str, float]]:
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
