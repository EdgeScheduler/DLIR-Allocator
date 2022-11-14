# run: python3 -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/main.py" >data/analyze/csv/data.csv
# run append: python3 -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/main.py" >>data/analyze/csv/data.csv

from logging import exception
import sys,os
import json

def Run(fold_index=0,append=True,count=1000):
    fold_names=["allocator","raw"]

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"csv/data.csv"), "a" if append else "w") as filep:
        if not append:
            print("id, recv-time,finish-time, type, model-name, limit, wait, execute, total",file=filep)
        try:
            for i in range(count):
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../",fold_names[fold_index],str(i)+".json"),"r") as fp:
                    data=json.load(fp)
                    print(data["tag"],",",data["recv_time"],",",data["finish_time"],",",fold_names[fold_index] ,",",data["model_name"],",",data["limit_cost_by_ms"],",",data["wait_cost"],",",data["execute_cost"],",",data["total_cost_by_ms"],file=filep)
        except exception as ex:
            print("error:",ex)
