import os
import subprocess


def download():
    # Define the folder name
    folder_name = "nyu-depth-v2"

    # Check if the folder exists
    if not os.path.exists(folder_name):
        print(f"The folder '{folder_name}' does not exist. Downloading dataset...")
        try:
            # Execute the Kaggle command to download the dataset
            subprocess.run(
                ["kaggle", "datasets", "download", "soumikrakshit/nyu-depth-v2"],
                check=True,
            )
            print("Dataset downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while downloading the dataset: {e}")
    else:
        print(f"The folder '{folder_name}' already exists.")

    import os
    import zipfile

    # Define the zip file and the target extraction folder
    zip_file = "nyu-depth-v2.zip"
    extract_folder = "nyu-depth-v2"

    # Check if the folder already exists
    if os.path.exists(extract_folder):
        print(f"The folder '{extract_folder}' already exists. Skipping extraction.")
    else:
        # Check if the zip file exists
        if os.path.exists(zip_file):
            print(f"Found zip file: {zip_file}. Extracting contents...")
            try:
                # Open the zip file
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    # Extract all files to the target folder
                    zip_ref.extractall(extract_folder)
                print(f"Extraction completed. Files extracted to: {extract_folder}")
            except zipfile.BadZipFile:
                print("Error: The file is not a valid zip file.")
        else:
            print(
                f"Zip file '{zip_file}' not found. Please ensure the file is downloaded."
            )
