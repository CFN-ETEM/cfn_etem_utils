#!/bin/bash 

which python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=1"
export MKL_NUM_STRIPES=1
export NUMEXPR_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=''

ulimit -u 524288
ulimit -n 100000
ulimit -a

nprocs=16
ipcluster start -n ${nprocs} --profile-dir=ipypar &
sleep 5
wait_ipp_engines -e ${nprocs} --ipp_dir=ipypar

echo `date` "Start conversion at ", `date`
read_K2 -g ~/Documents/data/Converter_Project/Capture6_.gtg -b 100 --out_dir converted_image --ipp_dir=ipypar
echo `date` "Finished conversion at ", `date`

ipcluster stop --profile-dir=ipypar
sleep 3

