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

filter=None

def Run(end_name="",count=1000):
    allocator_data=[]
    raw_data=[]

    try:
        for i in range(count):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"csv","data-"+end_name,"allocator",str(i)+".json"),"r") as fp:
                data=json.load(fp)
                if filter is None or data["model_name"]==filter:
                    allocator_data.append(Eva(float(data["limit_cost_by_ms"]), float(data["total_cost_by_ms"])))

            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"csv","data-"+end_name,"raw",str(i)+".json"),"r") as fp:
                data=json.load(fp)
                if filter is None or data["model_name"]==filter:
                    raw_data.append(Eva(float(data["limit_cost_by_ms"]), float(data["total_cost_by_ms"])))
                
    except exception as ex:
        print("error:",ex)
        return

    allocator_data=sorted(allocator_data)
    raw_data=sorted(raw_data)

    min_value=int(10*min(allocator_data[0],raw_data[0]))/10
    max_value=math.ceil(10*max(allocator_data[-1],raw_data[-1]))/10

    slide=int(10*(max_value-min_value))/500
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"csv","data-"+end_name,"analyze-"+(filter if filter is not None else "") + ".csv"),"w") as fp:
        print("evaluate",", ","allocator",", ","raw",file=fp)
        print(min_value,", ",0,", ",0,file=fp)
        current=min_value
        index_allocator=0
        index_raw=0

        while current<max_value:
            current+=slide

            before_allocator=index_allocator
            while index_allocator<len(allocator_data) and current>=allocator_data[index_allocator]:
                index_allocator+=1
            
            before_raw=index_raw
            while index_raw<len(raw_data) and current>=raw_data[index_raw]:
                index_raw+=1

            print(round(100*current)/100,", ",index_allocator-before_allocator,", ",index_raw-before_raw,file=fp)

Run("1000-400")