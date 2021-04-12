#!/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
epoch=150
dataset="Harvard"
concat="False"
model_name=("HSCNN HSI_Network HyperReconNet Attention_HSI_None Dense_HSI")
block_num=9


while getopts b:e:d:c:m:bn: OPT
do
    echo "$OPTARG"
    case $OPT in
        b) batch_size=$OPTARG ;;
        e) epoch=$OPTARG ;;
        d) dataset=$OPTARG ;;
        c) concat=$OPTARG ;;
        m) model_name=$OPTARG ;;
        bn) block_num=$OPTARG ;;
        *) echo "Usage: $CMDNAME [-b batch size] [-e epoch]" 1>&2
            exit 1;;
    esac
done


echo $batch_size
echo $epoch
echo $dataset
echo $block_num


model_name=( `echo $model_name | tr ' ' ' '` )
for name in $model_name[@]; do
    echo $name
done
for name in $model_name[@]; do
    python train_sh.py -b $batch_size -e $epoch -d $dataset -c $concat -m $name -bn $block_num
done