from load_from_csv import load_anff_from_csv
import config
import os
import json
import numpy as np

def main():
    for bench_fold in config.BenchFoldNames:
        for env in config.RunEnvs:
            datas=load_anff_from_csv(str(os.path.join(bench_fold, env, "radios.csv")), ignore_head=True)
            datas.sort(key=lambda value: value[2])

            # data classification
            kind_datas={}
            for data in datas:
                if data[1] not in kind_datas:
                    kind_datas[data[1]]=[]
                
                kind_datas[data[1]].append(data)

            cal_info={}
            for model_name in kind_datas:
                cal_info[model_name]={}
                data=[v[2] for v in kind_datas[model_name]]
                cal_info[model_name]["min"]=min(data)
                cal_info[model_name]["avg"]=sum(data)/len(data)
                cal_info[model_name]["max"]=max(data)
                cal_info[model_name]["std"]=np.std(data,ddof=1)
                

            data=[v[2] for v in datas]
            cal_info["total"]={}
            cal_info["total"]["min"]=min(data)
            cal_info["total"]["avg"]=sum(data)/len(data)
            cal_info["total"]["max"]=max(data)
            cal_info["total"]["std"]=np.std(data,ddof=1)
            
            with open(os.path.join(config.AimFold,bench_fold,env,"statistic.json"),"w") as fp:
                json.dump(cal_info,fp,indent=4)

            # counter with classification
            statistic_data={}
            max_length=0
            for model_name in kind_datas:
                items=kind_datas[model_name]

                current_count=0
                current_max=1000+int(config.Slide*1000)

                result=[(1.0,0)]
                for item in items:
                    while item[2]*1000>current_max:
                        result.append((current_max/1000, current_count))
                        current_count=0
                        current_max+=int(config.Slide*1000)

                    current_count+=1
                result.append((current_max/1000, current_count))

                statistic_data[model_name]=result
                max_length=max(max_length,len(result))

            total=[]
            for i in range(max_length):
                up=1.0
                count=0
                for model_name in statistic_data:
                    if i<len(statistic_data[model_name]):
                        up=statistic_data[model_name][i][0]
                        count+=statistic_data[model_name][i][1]

                total.append((up,count))

            # block
            os.makedirs(os.path.join(config.AimFold,bench_fold,env,"models","block"), exist_ok=True)
            for model_name in statistic_data:
                with open(os.path.join(config.AimFold,bench_fold,env,"models","block",model_name+"_block.csv"), "w") as fp:
                    print("limit, count", file=fp)
                    for item in statistic_data[model_name]:
                        print(item[0],", ",item[1], file=fp)

            with open(os.path.join(config.AimFold,bench_fold,env,"total_block.csv"), "w") as fp:
                print("limit, count", file=fp)
                for item in total:
                    print(item[0],", ",item[1], file=fp)
            
            # accumulate
            os.makedirs(os.path.join(config.AimFold,bench_fold,env,"models","accumulate"), exist_ok=True)
            for model_name in statistic_data:
                with open(os.path.join(config.AimFold,bench_fold,env,"models","accumulate",model_name+"_accumulate.csv"), "w") as fp:
                    print("limit, count", file=fp)
                    sum_count=0
                    for item in statistic_data[model_name]:
                        sum_count+=item[1]
                        print(item[0],", ",sum_count, file=fp)
            
            with open(os.path.join(config.AimFold,bench_fold,env,"total_accumulate.csv"), "w") as fp:
                print("limit, count", file=fp)
                sum_count=0
                for item in total:
                    sum_count+=item[1]
                    print(item[0],", ", sum_count, file=fp)


if __name__ == "__main__":
    main()