#!/bin/bash

SHELL_PATH=$(cd $(dirname $0) && pwd )

RootPath=$SHELL_PATH/..

time=$(date +%F~%H:%M:%S)

if [ -f "$RootPath/data/catalogue.json" ];then
    mkdir -p $RootPath/abandon/$time
    echo "mv $RootPath/data/catalogue.json $RootPath/abandon/$time/"
    mv $RootPath/data/catalogue.json $RootPath/abandon/$time/
fi

for mode in "OYST" "BNST" "FIFO" "PARALLER" "DLIR"
do
    if [ -d "$RootPath/data/$mode" ];then
        mkdir -p $RootPath/abandon/$time
        echo "$RootPath/data/$mode $RootPath/abandon/$time/"
        mv $RootPath/data/$mode $RootPath/abandon/$time/
    fi
done

if [ -d "$RootPath/data/BenchMark" ];then
    mkdir -p $RootPath/abandon/$time
    echo "mv $RootPath/data/BenchMark $RootPath/abandon/$time/"
    mv $RootPath/data/BenchMark $RootPath/abandon/$time/
fi