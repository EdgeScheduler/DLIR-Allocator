Count=1000
DefaultLambda=200

SHELL_PATH=$(cd $(dirname $0) && pwd )
RootPath=$SHELL_PATH/..

for args in "" "--vgg19=150 --resnet50=300" "--vgg19=300 --resnet50=150 --googlenet=200 --squeezenetv1=200" "--vgg19=150 --resnet50=300 --googlenet=300 --squeezenetv1=150" "--vgg19=300 --resnet50=150 --googlenet=150 --squeezenetv1=300"
do
    for mode in "DLIR-Allocator" "BNST-Allocator" "FIFO-Allocator" "PARALLER-Allocator" "OYST-Allocator"
    do
        echo "run $mode..."
        # echo "$RootPath/bin/release/$mode --count=$Count --lambda=$DefaultLambda  $args"
        $RootPath/bin/release/$mode --count=$Count --lambda=$DefaultLambda  $args
    done

    python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/get_ratios.py"
    python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/get_statistics.py"
    python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/plot.py"
done