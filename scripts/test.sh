Count=1000
DefaultLambda=200

SHELL_PATH=$(cd $(dirname $0) && pwd )
RootPath=$SHELL_PATH/..

Run()
{
    
    echo "avg(λ)=$1"
    for args in "--vgg19=$1 --resnet50=$1 --googlenet=$1 --squeezenetv1=$1" "--vgg19=$(echo "scale=0;$1*1.25/1"|bc -l) --resnet50=$(echo "scale=0;$1*0.75/1"|bc -l) --googlenet=$1 --squeezenetv1=$1" "--vgg19=$(echo "scale=0;$1*0.75/1"|bc -l) --resnet50=$(echo "scale=0;$1*1.25/1"|bc -l) --googlenet=$1 --squeezenetv1=$1" "--vgg19=$(echo "scale=0;$1*0.75/1"|bc -l) --resnet50=$(echo "scale=0;$1*1.25/1"|bc -l) --googlenet=$(echo "scale=0;$1*1.25/1"|bc -l) --squeezenetv1=$(echo "scale=0;$1*0.75/1"|bc -l)" "--vgg19=$(echo "scale=0;$1*1.25/1"|bc -l) --resnet50=$(echo "scale=0;$1*0.75/1"|bc -l) --googlenet=$(echo "scale=0;$1*0.75/1"|bc -l) --squeezenetv1=$(echo "scale=0;$1*1.25/1"|bc -l)"  "--vgg19=$(echo "scale=0;$1*1.25/1"|bc -l) --resnet50=$(echo "scale=0;$1*1.25/1"|bc -l) --googlenet=$(echo "scale=0;$1*0.75/1"|bc -l) --squeezenetv1=$(echo "scale=0;$1*0.75/1"|bc -l)"  "--vgg19=$(echo "scale=0;$1*0.75/1"|bc -l) --resnet50=$(echo "scale=0;$1*0.75/1"|bc -l) --googlenet=$(echo "scale=0;$1*1.25/1"|bc -l) --squeezenetv1=$(echo "scale=0;$1*1.25/1"|bc -l)"
    do
        for mode in "DLIR-Allocator" "BNST-Allocator" "FIFO-Allocator" "PARALLER-Allocator" "OYST-Allocator"
        do
            echo ""
            echo "run($(date)): $RootPath/bin/release/$mode --count=$Count --lambda=$1 $args"
            $RootPath/bin/release/$mode --count=$Count --lambda=$DefaultLambda  $args
        done
    done

    python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/get_ratios.py"
    python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/get_statistics.py"
    python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/plot.py"
}

# for args in "" "--vgg19=150 --resnet50=300" "--vgg19=300 --resnet50=150 --googlenet=200 --squeezenetv1=200" "--vgg19=150 --resnet50=300 --googlenet=300 --squeezenetv1=150" "--vgg19=300 --resnet50=150 --googlenet=150 --squeezenetv1=300"
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

for lambda in 100 150 200 250
do
    Run $lambda
done

python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/get_ratios.py"
python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/get_statistics.py"
python -u "/home/onceas/yutian/DLIR-Allocator/data/analyze/plot.py"