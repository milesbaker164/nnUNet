import os
import json
from tqdm import tqdm
import SimpleITK as sitk

def create_dataset_json(output_dir, modality_names, labels, dataset_name):
    """
    Create dataset.json required by nnU-Net
    """
    dataset_json = {
        "name": "ROI Dataset",
        "description": "Converted ROI segmentation dataset",
        "reference": "None",
        "licence": "None",
        "release": "1.0",
        "tensorImageSize": "4D",
        "modality": {str(i): mod for i, mod in enumerate(modality_names)},
        "labels": labels,
        "task": dataset_name,
        "numTraining": 0,
        "numTest": 0,
        "training": [],
        "test": []
    }
    with open(os.path.join(output_dir, "dataset.json"), 'w') as f:
        json.dump(dataset_json, f, indent=4)
    return dataset_json

def setup_directories(base_dir):
    """
    Create nnU-Net directory structure
    """
    dirs = {
        'imagesTr': os.path.join(base_dir, 'imagesTr'),
        'labelsTr': os.path.join(base_dir, 'labelsTr'),
        'imagesTs': os.path.join(base_dir, 'imagesTs')
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def get_file_list(directory, extension=".tif"):
    """
    Get list of files with specific extension in directory
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted([f for f in os.listdir(directory) if f.lower().endswith(extension.lower())])

def convert_tif_to_nifti(tif_path, output_path):
    """
    Convert a TIF image to NIfTI format
    """
    img = sitk.ReadImage(tif_path)
    sitk.WriteImage(img, output_path)
    return output_path

def convert_to_nnunet_format(input_data_dir, input_mask_dir, output_base_dir, dataset_name="Task001_ROISegmentation"):
    """
    Convert existing data to nnU-Net format
    """
    # Create the main dataset directory
    dataset_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Setup directory structure
    dirs = setup_directories(dataset_dir)

    # Initialize dataset parameters
    modality_names = ["CT"]  # Modify if needed
    labels = {
        "0": "background",
        "1": "roi"
    }

    # Create and initialize dataset.json
    dataset_json = create_dataset_json(dataset_dir, modality_names, labels, dataset_name)

    # Get list of all .tif files
    data_files = get_file_list(input_data_dir, ".tif")
    mask_files = get_file_list(input_mask_dir, ".tif")

    if len(data_files) != len(mask_files):
        raise ValueError(f"Number of images ({len(data_files)}) and masks ({len(mask_files)}) do not match.")

    # Convert each case
    for idx, (data_file, mask_file) in enumerate(tqdm(zip(data_files, mask_files), total=len(data_files))):
        case_id = f"case_{idx:04d}"

        # Source and destination paths
        src_data_path = os.path.join(input_data_dir, data_file)
        dst_data_path = os.path.join(dirs['imagesTr'], f"{case_id}_0000.nii.gz")
        src_mask_path = os.path.join(input_mask_dir, mask_file)
        dst_mask_path = os.path.join(dirs['labelsTr'], f"{case_id}.nii.gz")

        # Convert TIFF files to NIfTI format
        convert_tif_to_nifti(src_data_path, dst_data_path)
        convert_tif_to_nifti(src_mask_path, dst_mask_path)

        # Update dataset.json
        dataset_json['training'].append({
            "image": f"./imagesTr/{case_id}_0000.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz"
        })

    # Update dataset statistics
    dataset_json['numTraining'] = len(data_files)

    # Save updated dataset.json
    with open(os.path.join(dataset_dir, "dataset.json"), 'w') as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Dataset converted successfully to: {dataset_dir}")
    print(f"Total training cases: {len(data_files)}")

if __name__ == "__main__":
    # Set your paths here
    input_data_dir = os.path.expanduser("/Users/Bmbaker/Documents/ROI_AI_TRAINING_DATA/images")
    input_mask_dir = os.path.expanduser("/Users/Bmbaker/Documents/ROI_AI_TRAINING_DATA/mask")
    output_base_dir = os.path.expanduser("/Users/Bmbaker/Documents/roi ai code/nnUNet_raw")  # Where to store the converted dataset

    # Convert the dataset
    convert_to_nnunet_format(input_data_dir, input_mask_dir, output_base_dir)
