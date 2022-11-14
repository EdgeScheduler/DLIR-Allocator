from audioop import avg
from itertools import count
from logging import exception
import sys,os
import json
import math

def SLO(limit):
    return 10*limit

def Eva(limit,total):
    if(total<limit):
        return 1

    return total/limit

filters=["vgg19","googlenet","resnet50","squeezenetv1"]

def Run(end_name="",count=1000):
    allocator_data=[]
    raw_data=[]

    datas={}
    try:
        for i in range(count):
            if "ANTT" not in datas:
                datas["ANTT"]={}

            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"csv","data-"+end_name,"allocator",str(i)+".json"),"r") as fp:
                data=json.load(fp)
                # if filter is None or data["model_name"]==filter:
                #     allocator_data.append(Eva(float(data["limit_cost_by_ms"]), float(data["total_cost_by_ms"])))
                if data["model_name"] not in datas:
                    datas[data["model_name"]]={}

                if "allocator" not in datas[data["model_name"]]:
                    datas[data["model_name"]]["allocator"]=[]

                datas[data["model_name"]]["allocator"].append(float(data["total_cost_by_ms"]))
                
                if "allocator" not in datas["ANTT"]:
                    datas["ANTT"]["allocator"]=[]
                datas["ANTT"]["allocator"].append(Eva(float(data["limit_cost_by_ms"]), float(data["total_cost_by_ms"])))

            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"csv","data-"+end_name,"raw",str(i)+".json"),"r") as fp:
                data=json.load(fp)
                # if filter is None or data["model_name"]==filter:
                #     raw_data.append(Eva(float(data["limit_cost_by_ms"]), float(data["total_cost_by_ms"])))

                if data["model_name"] not in datas:
                    datas[data["model_name"]]={}

                if "raw" not in datas[data["model_name"]]:
                    datas[data["model_name"]]["raw"]=[]

                datas[data["model_name"]]["raw"].append(float(data["total_cost_by_ms"]))

                if "raw" not in datas["ANTT"]:
                    datas["ANTT"]["raw"]=[]
                datas["ANTT"]["raw"].append(Eva(float(data["limit_cost_by_ms"]), float(data["total_cost_by_ms"])))
        return datas

                
    except exception as ex:
        print("error:",ex)
        return None



def avg(list):
    return sum(list)/len(list)

datas=Run("1000-200-ban")
sum_allocator=0
sum_raw=0
for model_name in datas:
    print(model_name+":")
    print("raw:", avg(datas[model_name]["raw"]))
    sum_raw+=sum(datas[model_name]["raw"])

    print("allocator:", avg(datas[model_name]["allocator"]))
    sum_allocator+=sum(datas[model_name]["allocator"])

    print("improve:",100*(1-avg(datas[model_name]["allocator"])/avg(datas[model_name]["raw"])),"%")
    print()

print("total:")
print("raw:", sum_raw/1000)
print("allocator:", sum_allocator/1000)
print("improve:",100*(1-sum_allocator/sum_raw),"%")

print()

print("ANTT:")
print("raw:", avg(datas["ANTT"]["raw"]))
print("allocator:", avg(datas["ANTT"]["allocator"]))
print("improve:",100*(1-avg(datas["ANTT"]["allocator"])/avg(datas["ANTT"]["raw"])),"%")