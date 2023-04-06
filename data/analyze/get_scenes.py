import json
import config
import os

catalogue={}
with open(os.path.join(config.AimFold,"catalogue.json"),"r") as fp:
    catalogue = json.load(fp)

statistic={}
models=set()
types=set()
for scene in catalogue:
    statistic[scene]={}
    for bench in config.BenchFoldNames:
        statistic[scene][bench]={}
        
        with open(os.path.join(config.AimFold,bench,scene,"statistic.json"),"r") as fp:
            info = json.load(fp)

            statistic[scene][bench]=info
            models|=set(list(info.keys()))

            for k,v in info.items():
                types|=set(list(v.keys()))

models.discard("total")
models=list(models)+["total"]
types=list(types)

# values={
#     "16130607092908398826":1,
#     "16043504136821049534":0,
#     "791273470220973308":4,
#     "1231353308209120479":2,
#     "6246902654124171765":50,
#     "9624955804878585299":3,
#     "9684811068443324387":5,
#     "14416590567720251897":6,
#     "16770780636103774362":7
# }

values = {
    "11145471626182783555": 1,
    "11164575861383691666": 12,
    "11204759012807403925": 7,
    "11637442584803965078": 13,
    "14297953843126044557": 3,
    "14416590567720251897": 2,
    "14476283230313702212": 8,
    "15754402714772727911": 5,
    "16770780636103774362": 4,
    "3030279626028234749": 11,
    "9624955804878585299": 10,
    "9882370101184157663": 6,
    "9938926441397847365": 9
}



names={
    "DLIR":"DLIR",
    "OYST":"OYST", 
    "FIFO":"ClockWork",
    "BNST":"PREMA",  
    "PARALLEL":"MPS"
}

for scene in catalogue:
    for type in types:
        with open(os.path.join(config.AimFold,"total",scene,"scene_"+str(values[scene])+"_"+type+".csv"),"w") as fp:
            print(", ".join(["kind"]+models),file=fp)

            for bench in config.BenchFoldNames:
                data=[]
                data.append(names[bench])

                for model in models:
                    data.append(str(statistic[scene][bench][model][type]))
                print(", ".join(data),file=fp)

max_min=float("inf")
max_max=-100

avg_min=float("inf")
avg_max=-100

std_min=float("inf")
std_max=-100

print("max:")
values_max=[]
for scene in catalogue:
    if values[scene] not in [2,4,6,8,10,12]:
        continue
    
    for bench in ["FIFO","BNST"]:
        values_max.append(1- statistic[scene]["DLIR"]["total"]["max"]/statistic[scene][bench]["total"]["max"])
        max_min=min(max_min,1- statistic[scene]["DLIR"]["total"]["max"]/statistic[scene][bench]["total"]["max"])
        max_max=max(max_max,1- statistic[scene]["DLIR"]["total"]["max"]/statistic[scene][bench]["total"]["max"])
print(max_min,"~",max_max, sum(values_max)/len(values_max))

print("avg:")
values_avg=[]
for scene in catalogue:
    if values[scene] not in [2,4,6,8,10,12]:
        continue

    for bench in ["FIFO","BNST"]:
        values_avg.append(1- statistic[scene]["DLIR"]["total"]["avg"]/statistic[scene][bench]["total"]["avg"])
        avg_min=min(avg_min,1- statistic[scene]["DLIR"]["total"]["avg"]/statistic[scene][bench]["total"]["avg"])
        avg_max=max(avg_max,1- statistic[scene]["DLIR"]["total"]["avg"]/statistic[scene][bench]["total"]["avg"])
print(avg_min,"~",avg_max, sum(values_avg)/len(values_avg))

print("std:")
values_std=[]
for scene in catalogue:
    if values[scene] not in [2,4,6,8,10,12]:
        continue

    for bench in ["FIFO","BNST"]:
        values_std.append(1- statistic[scene]["DLIR"]["total"]["std"]/statistic[scene][bench]["total"]["std"])
        std_min=min(std_min,1- statistic[scene]["DLIR"]["total"]["std"]/statistic[scene][bench]["total"]["std"])
        std_max=max(std_max,1- statistic[scene]["DLIR"]["total"]["std"]/statistic[scene][bench]["total"]["std"])

print(std_min,"~",std_max,sum(values_std)/len(values_std))


max_min=float("inf")
max_max=-100

avg_min=float("inf")
avg_max=-100

std_min=float("inf")
std_max=-100


print("----only BNST")
print("max:")
values_max=[]
for scene in catalogue:
    if values[scene] not in [2,4,6,8,10,12]:
        continue

    for bench in ["BNST"]:
        values_max.append(1- statistic[scene]["DLIR"]["total"]["max"]/statistic[scene][bench]["total"]["max"])
        max_min=min(max_min,1- statistic[scene]["DLIR"]["total"]["max"]/statistic[scene][bench]["total"]["max"])
        max_max=max(max_max,1- statistic[scene]["DLIR"]["total"]["max"]/statistic[scene][bench]["total"]["max"])
print(max_min,"~",max_max, sum(values_max)/len(values_max))

print("avg:")
values_avg=[]
for scene in catalogue:
    if values[scene] not in [2,4,6,8,10,12]:
        continue

    for bench in ["BNST"]:
        values_avg.append(1- statistic[scene]["DLIR"]["total"]["avg"]/statistic[scene][bench]["total"]["avg"])
        avg_min=min(avg_min,1- statistic[scene]["DLIR"]["total"]["avg"]/statistic[scene][bench]["total"]["avg"])
        avg_max=max(avg_max,1- statistic[scene]["DLIR"]["total"]["avg"]/statistic[scene][bench]["total"]["avg"])
print(avg_min,"~",avg_max, sum(values_avg)/len(values_avg))

print("std:")
values_std=[]
for scene in catalogue:
    if values[scene] not in [2,4,6,8,10,12]:
        continue

    for bench in ["BNST"]:
        values_std.append(1- statistic[scene]["DLIR"]["total"]["std"]/statistic[scene][bench]["total"]["std"])
        std_min=min(std_min,1- statistic[scene]["DLIR"]["total"]["std"]/statistic[scene][bench]["total"]["std"])
        std_max=max(std_max,1- statistic[scene]["DLIR"]["total"]["std"]/statistic[scene][bench]["total"]["std"])

print(std_min,"~",std_max,sum(values_std)/len(values_std))