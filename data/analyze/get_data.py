# from data.analyze.Run_childs import Run
from data.analyze.Run import Run
import sys

def main():
    Run(0,False)
    Run(1,append=True)

if __name__=="__main__":
    main()


