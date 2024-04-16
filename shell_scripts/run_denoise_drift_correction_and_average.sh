#!/bin/bash

if [[ $# -ne 2 ]]
    then
        echo Usage: run_denoise_drift_correction_and_average.sh DM4_FILENAME UDVD_ATH
    else
        dm4_fn=$1
        udvd_path=$2
        echo Using the python environment at `which python`
    fi


