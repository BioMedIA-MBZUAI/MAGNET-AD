#!/bin/bash

# Define input and output directories
input_folder="./fMRI-Registered"
output_folder="./fMRI-Processed"
mkdir -p "$output_folder"

# Define constants
TR=2  # Replace with your repetition time in seconds
sigma_smooth=6  # Spatial smoothing FWHM in mm
highpass_sigma=25  # High-pass filter sigma (use TR in seconds to calculate for a cutoff of 0.01 Hz)

# Loop through all fMRI files in the input folder in reversed order
for fmri_file in $(ls "$input_folder"/*.nii.gz); do
    # Extract the base name for each file
    base_name=$(basename "$fmri_file" .nii.gz)
    
    # Define final output file to check if already processed
    final_output_file="$output_folder/${base_name}_clean.nii.gz"
    
    # Check if the final output file exists
    if [ -f "$final_output_file" ]; then
        echo "Skipping $base_name, already processed."
        continue
    fi
    
    echo "Processing $base_name..."

    # Slice Timing Correction
    slicetimer -i "$fmri_file" -o "$output_folder/${base_name}_stc.nii.gz" --repeat=$TR
    
    if [ ! -f "$output_folder/${base_name}_stc.nii.gz" ]; then
        echo "Error: Slice timing correction failed for $base_name."
        continue
    fi

    # Motion Correction
    mcflirt -in "$output_folder/${base_name}_stc.nii.gz" -out "$output_folder/${base_name}_mc.nii.gz" -plots -mats -refvol 0
    
    if [ ! -f "$output_folder/${base_name}_mc.nii.gz" ]; then
        echo "Error: Motion correction failed for $base_name."
        continue
    fi

    # Spatial Smoothing
    fslmaths "$output_folder/${base_name}_mc.nii.gz" -s $sigma_smooth "$output_folder/${base_name}_smooth.nii.gz"
    
    if [ ! -f "$output_folder/${base_name}_smooth.nii.gz" ]; then
        echo "Error: Smoothing failed for $base_name."
        continue
    fi

    # Temporal Filtering (High-pass filtering at 0.01 Hz)
    fslmaths "$output_folder/${base_name}_smooth.nii.gz" -bptf $highpass_sigma -1 "$output_folder/${base_name}_filtered.nii.gz"
    
    if [ ! -f "$output_folder/${base_name}_filtered.nii.gz" ]; then
        echo "Error: Temporal filtering failed for $base_name."
        continue
    fi

    # Optional: Nuisance Regression
    nuisance_regressors="${input_folder}/${base_name}_nuisance.txt"
    if [ -f "$nuisance_regressors" ]; then
        fsl_glm -i "$output_folder/${base_name}_filtered.nii.gz" -d "$nuisance_regressors" -o "$final_output_file"
    else
        cp "$output_folder/${base_name}_filtered.nii.gz" "$final_output_file"
    fi

    echo "Finished processing $base_name."
done

echo "All fMRI files processed and saved in $output_folder."
