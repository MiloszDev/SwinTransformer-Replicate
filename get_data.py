import zipfile
import os

# Specify the path to the zip file and the directory to extract to
zip_file_path = './data/stanford-dogs-dataset.zip'
extract_to_path = './data'

# Check if the zip file exists before attempting to extract
if os.path.exists(zip_file_path):
    # Open the zip file and extract its contents
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    
    print(f"Extracted to {os.path.abspath(extract_to_path)}")
else:
    print(f"Zip file not found: {zip_file_path}")
