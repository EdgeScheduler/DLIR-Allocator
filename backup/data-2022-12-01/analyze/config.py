import os
import json

Slide=0.1
DataPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
BenchFoldNames = ["DLIR","BNST", "FIFO", "OYST", "PARALLEL"]
AimFold = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../BenchMark")
MaxCount = 1000
Catalogue=None
RunEnvs=[]

with open(os.path.join(DataPath,"catalogue.json"),"r") as fp:
    Catalogue=json.load(fp)
    RunEnvs=list(Catalogue.keys())

    os.makedirs(AimFold,exist_ok=True)
    with open(os.path.join(AimFold,"catalogue.json"),"w") as fpw:
        json.dump(Catalogue,fpw,indent=4)


def GetDescribe(env: str)->str:
    describe=""
    for key,value in Catalogue[env].items():
        describe+=", "
        describe+=str(key)
        describe+="="
        describe+=str(value)

    
    return describe[2:]
