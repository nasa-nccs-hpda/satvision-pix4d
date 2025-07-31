#!/bin/bash
base_dir="/raid/jacaraba/convection"
required_dirs=("__xarray_dataarray_variable__" "t")
output_file="missing_dirs_band.log"

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

