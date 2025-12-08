from google.cloud import storage
import tensorflow as tf
import os

def upload_model(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"
    # client = storage.Client()
    # bucket = client.bucket(BUCKET_NAME)
    # blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


def load_model(bucket_name, source_blob_name, destination_file_name):
    """Downloads a model file from the bucket and loads it as a TensorFlow model."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The ID of your GCS object
    # source_blob_name = "storage-object-name"
    # The path to save the file locally
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    print("bucket correct")
    blob = bucket.blob(source_blob_name)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)

    # Download the model file
    blob.download_to_filename(destination_file_name)

    print(
        f"Model {source_blob_name} downloaded to {destination_file_name}."
    )

    # Load and return the TensorFlow model
    model = tf.keras.models.load_model(destination_file_name)
    return model
