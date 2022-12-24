from logging import exception
import os
import json
import math
import config

def Ratio(limit, total):
    if (total < limit):
        return 1.0

    return total / limit


def Run(path, max_count=1000):
    os.makedirs(os.path.join(config.AimFold, path),exist_ok=True)

    datas = []
    try:
        for i in range(max_count):
            with open(os.path.join(config.DataPath, path, str(i) + ".json"), "r") as fileR:
                data = json.load(fileR)
                datas.append((data["tag"], data["model_name"], Ratio(data["limit_cost_by_ms"], data["total_cost_by_ms"])))
    except exception as ex:
        print("warning: count(%s) < %d, but we had deal data that found (=%d)." % (path, max_count, len(datas)))

    with open(os.path.join(config.AimFold, path, "radios.csv"), "w") as fileW:
        print("id, model-name, radio", file=fileW)
        for data in datas:
            print(data[0], ",", data[1], ",", data[2], file=fileW)

    datas.sort(key=lambda value: value[2])
    with open(os.path.join(config.AimFold, path, "radios_sort.csv"), "w") as fileW:
        print("id, model-name, radio", file=fileW)
        for data in datas:
            print(data[0], ",", data[1], ",", data[2], file=fileW)

def main():
    for bench_fold in config.BenchFoldNames:
        for env in config.RunEnvs:
            Run(str(os.path.join(bench_fold, env)), config.MaxCount)

if __name__ == "__main__":
    main()
