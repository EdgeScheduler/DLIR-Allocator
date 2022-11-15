import os

Slide=0.1
DataPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
BenchFoldNames = ["DLIR","BNST", "FIFO", "OYST", "PARALLEL"]
RunEnvs = [""]
AimFold = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../BenchMark")
MaxCount = 1000