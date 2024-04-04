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

nprocs=64
ipcluster start -n ${nprocs} --profile-dir=ipypar &
sleep 5
wait_ipp_engines -e ${nprocs} --ipp_dir=ipypar

echo `date` "Start conversion at ", `date`
mkdir drift_corrected_image
drift_correction -s converted_image -t drift_corrected_image --rebase_shift 10 --rebase_steps 4000 -d -p --fps 100
echo `date` "Finished conversion at ", `date`

ipcluster stop --profile-dir=ipypar
sleep 3

