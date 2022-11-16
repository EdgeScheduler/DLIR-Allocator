import matplotlib.pyplot as plt
import os
import numpy as np
import config
import json
from load_from_csv import load_count_from_csv, load_anff_from_csv
from typing import List, Tuple
from matplotlib.ticker import FuncFormatter

TotalWidth=0.8

def mycolor(index)->str:
    colors_ = ['red','green','blue','m','y','k']

    if index<len(colors_) and index>=0:
        return colors_[index]
    else:
        return 'w'

def ToPercent(value, position):
    return '%1.0f'%(100*value) + '%'

# [ [[""],["total"]], [["models/accumulate"], ["googlenet","resnet50","squeezenetv1","vgg19"]] ]
def PlotAccumulates(benchs: list, envs: list):
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

            blocks= [[[""],["total"]]]
            info={}
            with open(os.path.join(config.AimFold,bench,env,"statistic.json"),"r") as fp:
                info=json.load(fp)

            models=list(info.keys())
            models.remove("total")
            blocks.append([["models/accumulate"],models])

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
            os.makedirs(os.path.join(config.AimFold,bench,env),exist_ok=True)
            plt.savefig(os.path.join(config.AimFold,bench,env,"accumulate.svg"), dpi=300,format="svg")
        
        plt.figure()
        for i in range(len(total_labels)):
            plt.plot(xs,total_ys_list[i]+[total_ys_list[i][-1]]*(len(xs)-len(total_ys_list[i])), linewidth=(1.0 if labels[i]=="DLIR" else 0.5),color=mycolor(i),label=total_labels[i], linestyle=("-" if total_labels[i]=="DLIR" else "--"))

        plt.gca().yaxis.set_major_formatter(FuncFormatter(ToPercent))
        plt.ylabel('累计完成比例')
        plt.xlabel('响应比')
        plt.legend()
        os.makedirs(os.path.join(config.AimFold,"total",env),exist_ok=True)
        plt.savefig(os.path.join(config.AimFold,"total",env,"accumulate.svg"), dpi=300,format="svg")

# [ [[""],["total"]], [["models/accumulate"], ["googlenet","resnet50","squeezenetv1","vgg19"]] ]
def PlotRange(benchs: list, envs: list):
    # 开始画图
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False      # 解决保存图像时'-'显示为方块的问题
    for env in envs:
        # total_bar_width= TotalWidth/len(benchs)
        # total_xs=np.arange(2)-(TotalWidth-total_bar_width)/2
        total_data={}
        for bench in benchs:
            plt.figure()

            info={}
            with open(os.path.join(config.AimFold,bench,env,"statistic.json"),"r") as fp:
                info=json.load(fp)
            total_data[bench]=info["total"]

            bar_width= TotalWidth/len(info)
            xs=np.arange(3)-(TotalWidth-bar_width)/2
            for idx,name in enumerate(info):
                plt.bar(xs+bar_width*idx,np.array([info[name]["min"], info[name]["avg"], info[name]["max"]])-1, width=bar_width, label=name)
            plt.xticks(np.arange(3), ["min", "avg", "max"])

            plt.ylabel('平均响应比-1')
            plt.legend()
            os.makedirs(os.path.join(config.AimFold,bench,env),exist_ok=True)
            plt.savefig(os.path.join(config.AimFold,bench,env,"range.svg"), dpi=300,format="svg")

            plt.figure()
            avgs=[info[key]["std"] for key in info]
            plt.bar(np.arange(len(info)),avgs, width=0.5, color="orange")
            
            plt.axhline(y=info["total"]["std"], color='r', linestyle='--', linewidth=0.5)
            plt.xticks(np.arange(len(info)), list(info.keys()))

            plt.ylabel('响应比标准差')
            os.makedirs(os.path.join(config.AimFold,bench,env),exist_ok=True)
            plt.savefig(os.path.join(config.AimFold,bench,env,"std.svg"), dpi=300,format="svg")
        
        plt.figure()
        # for idx,name in enumerate(total_data):
            # plt.bar(total_xs+total_bar_width*idx,[total_data[name]["avg"],  total_data[name]["std"]], width=total_bar_width, label=name)
        avgs=[total_data[key]["avg"] for key in total_data]
        plt.bar(np.arange(len(total_data)),np.array(avgs)-1, width=0.5, color="orange")
        
        plt.axhline(y=total_data["DLIR"]["avg"]-1, color='r', linestyle='--', linewidth=0.5)
        plt.xticks(np.arange(len(total_data)), list(total_data.keys()))

        plt.ylabel('平均响应比-1')
        os.makedirs(os.path.join(config.AimFold,"total",env),exist_ok=True)
        plt.savefig(os.path.join(config.AimFold,"total",env,"total_avg.svg"), dpi=300,format="svg")

        plt.figure()
        # for idx,name in enumerate(total_data):
            # plt.bar(total_xs+total_bar_width*idx,[total_data[name]["avg"],  total_data[name]["std"]], width=total_bar_width, label=name)
        avgs=[total_data[key]["std"] for key in total_data]
        plt.bar(np.arange(len(total_data)),avgs, width=0.5, color="orange")
        
        plt.axhline(y=total_data["DLIR"]["std"], color='r', linestyle='--', linewidth=0.5)
        plt.xticks(np.arange(len(total_data)), list(total_data.keys()))

        plt.ylabel('响应比标准差')
        os.makedirs(os.path.join(config.AimFold,"total",env),exist_ok=True)
        plt.savefig(os.path.join(config.AimFold,"total",env,"total_std.svg"), dpi=300,format="svg")


def main():
    # PlotAccumulates(config.BenchFoldNames,config.RunEnvs,[[[""],["total"]], [["models/accumulate"], ["googlenet","resnet50","squeezenetv1","vgg19"]]])
    PlotAccumulates(config.BenchFoldNames,config.RunEnvs)
    PlotRange(config.BenchFoldNames,config.RunEnvs)

if __name__ == "__main__":
    main()