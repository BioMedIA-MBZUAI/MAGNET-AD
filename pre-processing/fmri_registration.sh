#!/bin/bash

# Define the root folder containing all patient subfolders and the T1 reference image
root_folder="./"
ref_image="./T1_Scans"  # T1-weighted reference image
output_folder="./fMRI-Registered"
temp_folder="./temp"  # Temporary folder for intermediate files
mkdir -p "$output_folder"
mkdir -p "$temp_folder"

# Loop over all fMRI files in patient subdirectories that start with A4_MR_fMRI and end with .nii.gz
find "$root_folder" -type f -name "A4_MR_fMRI*.nii.gz" | while read -r fmri_file; do
    # Extract the patient identifier from the fMRI file path (assuming the patient subfolder contains this ID)
    patient_id=$(basename "$(dirname "$fmri_file")")

    #check if output file already exists
    base_name=$(basename "$fmri_file" .nii.gz)
    output_image="$output_folder/${base_name}_in_t1_space.nii.gz"
    if [ -f "$output_image" ]; then
        echo "Output file already exists for $fmri_file, skipping..."
        continue
    fi
    
    # Construct the path to the corresponding T1 file for this patient
    t1_image="${ref_image}/A4_MR_T1_${patient_id}.nii.gz"

    # Check if the T1 image exists for this patient
    if [ ! -f "$t1_image" ]; then
        echo "No T1-weighted image found for patient $patient_id, skipping..."
        continue
    fi
    
    # Extract the base name of the fMRI file without the extension
    base_name=$(basename "$fmri_file" .nii.gz)
    
    # Define the temporary single-volume fMRI file path
    single_volume_file="$temp_folder/${base_name}_vol0.nii.gz"
    
    # Extract the first volume from the 4D fMRI file
    fslroi "$fmri_file" "$single_volume_file" 0 1
    if [ ! -f "$single_volume_file" ]; then
        echo "Failed to extract the first volume from $fmri_file, skipping..."
        continue
    fi

    # Define the output file paths
    output_image="$output_folder/${base_name}_in_t1_space.nii.gz"
    output_matrix="$output_folder/${base_name}_to_t1.mat"
    
    # Run FLIRT registration using the patient-specific T1 image
    flirt -in "$single_volume_file" -ref "$t1_image" -out "$output_image" -omat "$output_matrix" -dof 12

    echo "Registered $fmri_file to $t1_image and saved to $output_image"
done

# Clean up temporary files
rm -rf "$temp_folder"
