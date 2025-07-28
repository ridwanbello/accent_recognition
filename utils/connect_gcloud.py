# First create a bucket on https://console.cloud.google.com

# Then on Googlre Colab

# Install the cloud storage package
!pip install -q google-cloud-storage

# Authenticate with gmail account
from google.colab import auth
auth.authenticate_user()

# Import necessary modules
import os
from google.cloud import storage

# Move files and folders to Google Cloud Storage
def upload_folder_to_gcs(local_folder_path, bucket_name, gcs_folder_path=""):
    """Move folder to Google Cloud"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_folder_path)
            blob_path = os.path.join(gcs_folder_path, relative_path).replace("\\", "/")  # Ensure GCS-friendly paths
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file_path)
            print(f"✅ Uploaded {local_file_path} → gs://{bucket_name}/{blob_path}")

# Sample move of "South African folder" to bucket name "afrispeech-bucket"
local_folder = "/content/south_africa"
bucket_name = "afrispeech-bucket"
gcs_destination = "datasets/south_africa"

upload_folder_to_gcs(local_folder, bucket_name, gcs_destination)