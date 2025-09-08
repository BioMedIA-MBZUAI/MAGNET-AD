#%% Import important packages
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from brainles_hd_bet.run import run_hd_bet
import os

# Define input and output paths
dataset_path = '../Raw_Scans'
output_path = './Preprocessed_Scans'  # New output folder path
mni_template_path = '../MNI152_T1_1mm.nii.gz'

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Read the MNI template image
print(f"Reading the MNI template image {os.path.basename(mni_template_path)}---->>>")
fixed_image = sitk.ReadImage(mni_template_path, sitk.sitkFloat32)

# Sort the files in the dataset path in reverse order
for patient in sorted(os.listdir(dataset_path), reverse=True):
    if not patient.endswith('.nii.gz'):
        continue  # Skip non-image files if present

    print(f"\nWorking with case: {patient}  >>>>>>>>>>")
    
    # Define input file path
    moving_img_path = os.path.join(dataset_path, patient)
    
    # Extract the base name without the ".nii.gz" extension
    base_name = os.path.splitext(os.path.splitext(patient)[0])[0]
    
    # Define output file paths
    registered_img_path = os.path.join(output_path, f"{base_name}_MNI.nii.gz")
    brain_image_path = os.path.join(output_path, f"{base_name}_MNI_brain.nii.gz")
    corrected_N4_image_path = os.path.join(output_path, f"{base_name}_MNI_brain_N4.nii.gz")

    try:
        # Load the moving image
        print(f"Reading the moving image {os.path.basename(moving_img_path)}---->>>")
        moving_image = sitk.ReadImage(moving_img_path, sitk.sitkFloat32)

        # Registration to MNI template
        if not os.path.isfile(registered_img_path):
            print("Registration to MNI 152 template--->>>>>>")
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image, moving_image,
                sitk.AffineTransform(3), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.01)
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, 
                                                              convergenceMinimumValue=1e-6, convergenceWindowSize=10)
            registration_method.SetOptimizerScalesFromPhysicalShift()
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            registration_method.SetInitialTransform(initial_transform, inPlace=False)
            
            final_transform = registration_method.Execute(
                sitk.Cast(fixed_image, sitk.sitkFloat32), 
                sitk.Cast(moving_image, sitk.sitkFloat32)
            )
            
            resampled_moving = sitk.Resample(moving_image, fixed_image, final_transform, 
                                             sitk.sitkLinear, 0.0, moving_image.GetPixelID())
            
            sitk.WriteImage(resampled_moving, registered_img_path)
        else:
            print(f"Skip registration....The file {os.path.basename(registered_img_path)} already exists--->>>>>>")
            continue
        
        # Brain extraction using HD-BET
        if not os.path.isfile(brain_image_path):
            print("Run HD-BET for brain extraction --->>>>>>")        
            run_hd_bet(mri_fnames=registered_img_path, output_fnames=brain_image_path, device=0) 
        else:
            print(f"Skip HD-BET....The file {os.path.basename(brain_image_path)} already exists--->>>>>>")
        
        # N4 bias field correction
        if not os.path.isfile(corrected_N4_image_path):
            print("Run N4 bias field correction--->>>>>>")
            inputImage = sitk.ReadImage(brain_image_path, sitk.sitkFloat32)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrected_image = corrector.Execute(inputImage)
            sitk.WriteImage(corrected_image, corrected_N4_image_path)
        else:
            print(f"Skip N4....The file {os.path.basename(corrected_N4_image_path)} already exists--->>>>>>")
        
        print(f".....Finished with case: {patient}  >>>>>>>>>>")
    
    except Exception as e:
        print(f"Error processing case {patient}: {str(e)}. Skipping to the next case.")
