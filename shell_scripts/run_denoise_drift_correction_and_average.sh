#!/bin/bash


if [[ $# -ne 2 ]]
    then
        echo Usage: run_denoise_drift_correction_and_average.sh DM4_FILENAME UDVD_PATH
    else
        dm4_fn=$1
        udvd_path=$2
        echo Using the python environment at `which python`

        echo Starting unpack DM4 file at `date`
        convert_stacked_dm4_to_tiff -s ${dm4_fn} -t indv_images

        echo Starting denoising at `date`
        udvd_denoising -s indv_images -t denoised_images --udvd_path ${udvd_path}

        echo Starting drift finding at `date`
        drift_correction -s denoised_images -t dc_images -d --no_crop

        echo Starting averaging at `date`
        drift_corrected_average -s indv_images -d drift_corrections.json -t avg_images

        echo Finished everything at `date`
        echo Please see the average image inside the "avg_images" folder
    fi


