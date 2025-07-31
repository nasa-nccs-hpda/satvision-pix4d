#!/bin/bash
base_dir="/raid/jacaraba/convection"
required_dirs=("band" "dataset_name" "date_created" "goes_imager_projection" "latitude" "longitude" "t" "time" "time_coverage_end" "time_coverage_start" "x" "x_image" "y" "y_image")
output_file="missing_dirs.log"

> "$output_file"  # clear old log

find "$base_dir" -type d -name "*.zarr" | while read -r zarr; do
    missing=()
    for d in "${required_dirs[@]}"; do
        if [ ! -d "$zarr/$d" ]; then
            missing+=("$d")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "$zarr is missing: ${missing[*]}"
        echo "$zarr is missing: ${missing[*]}" >> "$output_file"
    fi
done

