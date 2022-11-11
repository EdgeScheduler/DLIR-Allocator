#!/bin/bash

SHELL_PATH=$(cd $(dirname $0) && pwd )

echo $SHELL_PATH
RootPath=$SHELL_PATH/../

mkdir -p $RootPath/build
rm -rf $RootPath/build

mkdir -p $RootPath/build && cd $RootPath/build

for mode in "OYST_MODE" "BNST_MODE" "FIFO_MODE" "PARALLER_MODE" "DLIR_MODE"
do
    echo "build $mode..."
    mkdir -p $RootPath/build/$mode && cd $RootPath/build/$mode
    cmake ../../ -DCOMPILE_MODE="$mode"
    make -j20
done