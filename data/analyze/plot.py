import matplotlib.pyplot as plt
import os
import numpy as np
import config
from load_from_csv import load_count_from_csv
from typing import List, Tuple
from matplotlib.ticker import FuncFormatter

def mycolor(index)->str:
    colors_ = ['red','green','blue','m','y','k']

    if index<len(colors_) and index>=0:
        return colors_[index]
    else:
        return 'w'

def ToPercent(value, position):
    return '%1.0f'%(100*value) + '%'

# [ [[""],["total"]], [["models/accumulate"], ["googlenet","resnet50","squeezenetv1","vgg19"]] ]
def PlotAccumulates(benchs: list, envs: list, blocks: list):
    # 开始画图
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False      # 解决保存图像时'-'显示为方块的问题
    for env in envs:
        total_xs=[]
        total_ys_list=[]
        total_labels=[]
        for bench in benchs:
            plt.figure()
            xs=[]
            ys_list=[]
            labels=[]
            for block in blocks:
                for sub_path in block[0]:
                    for name in block[1]:
                        datas=load_count_from_csv(os.path.join(bench,env,sub_path,name+"_accumulate.csv"), ignore_head=True)
                        if len(xs)<len(datas):
                            xs=[v[0] for v in datas]

                        ys_list.append([v[1]/datas[-1][1] for v in datas])
                        labels.append(name)

                        if name=="total":
                            if len(total_xs)<len(datas):
                                total_xs=[v[0] for v in datas]

                            total_ys_list.append([v[1]/datas[-1][1] for v in datas])
                            total_labels.append(bench)
            
            for i in range(len(labels)):
                plt.plot(xs,ys_list[i]+[ys_list[i][-1]]*(len(xs)-len(ys_list[i])), linewidth=(1.0 if labels[i]=="total" else 0.5),color=mycolor(i),label=labels[i],linestyle=("-" if labels[i]=="total" else "--"))
            
            plt.gca().yaxis.set_major_formatter(FuncFormatter(ToPercent))
            plt.ylabel('累计完成比例')
            plt.xlabel('响应比')
            plt.legend()
            plt.savefig(os.path.join(config.AimFold,bench,env,"accumulate.svg"), dpi=300,format="svg")
        
        plt.figure()
        for i in range(len(total_labels)):
            plt.plot(xs,total_ys_list[i]+[total_ys_list[i][-1]]*(len(xs)-len(total_ys_list[i])), linewidth=(1.0 if labels[i]=="DLIR" else 0.5),color=mycolor(i),label=total_labels[i], linestyle=("-" if total_labels[i]=="DLIR" else "--"))

        plt.gca().yaxis.set_major_formatter(FuncFormatter(ToPercent))
        plt.ylabel('累计完成比例')
        plt.xlabel('响应比')
        plt.legend()
        plt.savefig(os.path.join(config.AimFold,env,"accumulate.svg"), dpi=300,format="svg")

def main():
    PlotAccumulates(config.BenchFoldNames,config.RunEnvs,[[[""],["total"]], [["models/accumulate"], ["googlenet","resnet50","squeezenetv1","vgg19"]]])

if __name__ == "__main__":
    main()