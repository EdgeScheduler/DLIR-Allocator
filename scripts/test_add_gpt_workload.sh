# 此脚本用于测量平均lambda变化时的系统性能

Count=1000
DefaultLambda=200

SHELL_PATH=$(cd $(dirname $0) && pwd )
RootPath=$SHELL_PATH/..

Run()
{
    echo "avg(λ)=$1"
    for mode in "DLIR-Allocator" "BNST-Allocator" "FIFO-Allocator" "PARALLER-Allocator" "OYST-Allocator"
    do
        echo ""
        echo "run($(date)): $RootPath/bin/release/$mode --count=$Count --lambda=$1 "
        $RootPath/bin/release/$mode --count=$Count --lambda=$1
    done
}

RunUrgency()
{
    for args in "--vgg19=$1 --resnet50=$1 --yolov2=$1 --gpt128v2=$1 --googlenet=$1"
    do
        for mode in "DLIR-Allocator" "BNST-Allocator" "FIFO-Allocator" "PARALLER-Allocator" "OYST-Allocator"
        do
            echo ""
            echo "run($(date)): $RootPath/bin/release/$mode --count=$Count --lambda=$1 $args"
            $RootPath/bin/release/$mode --count=$Count --lambda=$1  $args
        done
    done
}

# for args in "" "--vgg19=150 --resnet50=300" "--vgg19=300 --resnet50=150 --yolov2=200 --gpt128v2=200" "--vgg19=150 --resnet50=300 --yolov2=300 --gpt128v2=150" "--vgg19=300 --resnet50=150 --yolov2=150 --gpt128v2=300"
# do
#     for mode in "DLIR-Allocator" "BNST-Allocator" "FIFO-Allocator" "PARALLER-Allocator" "OYST-Allocator"
#     do
#         echo "run $mode..."
#         # echo "$RootPath/bin/release/$mode --count=$Count --lambda=$DefaultLambda  $args"
#         $RootPath/bin/release/$mode --count=$Count --lambda=$DefaultLambda  $args
#     done

#     python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/get_ratios.py"
#     python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/get_statistics.py"
#     python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/plot.py"
# done

for lambda in 105 110 115 120 125 130 135 140 145 150 155 160 165
do
    Run $lambda
done

python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/get_ratios.py"
python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/get_statistics.py"
python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/plot.py"