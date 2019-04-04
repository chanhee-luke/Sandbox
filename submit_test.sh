#!/bin/csh

#$ -M csong1@nd.edu
#$ -m abe
#$ -q gpu@qa-titanx-011
#$ -l gpu_card=1
#$ -N test

#module load conda

#echo ". /opt/crc/c/conda/miniconda2/4.5.4/etc/profile.d/conda.sh" >> ~/.bashrc
#conda activate
#conda activate pytorch
#conda list
module load pytorch
python3 -c "import torch;print(torch.__version__)"
pip3 install torchtext

setenv CUDA_VISIBLE_DEVICES 0

fsync $SGE_STDOUT_PATH &

cd /scratch365/csong1/Sandbox

python3 preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
time python3 train.py -data data/demo -save_model demo-model -gpuid 0 -epochs 10

foreach var (demo-model*[1-2][0-9].pt)
    echo $var
    time python3 translate.py -gpu 0 -model $var -src data/src-test.txt -output $var\_pred.en.txt
    echo "\n"
end
