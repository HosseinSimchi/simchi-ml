import os
from minio import Minio
from minio.error import S3Error

def upload_file_to_minio():
    # Read from environment variables (Docker-friendly)
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "minio1:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "simchi")
    secret_key = os.getenv("MINIO_SECRET_KEY", "Hossein!7175")
    bucket_name = os.getenv("MINIO_BUCKET", "tfmodels")
    file_path = os.getenv("MODEL_PATH", "models/tfmodels.keras")
    object_name = os.getenv("OBJECT_NAME", "tfmodels.keras")

    # Create MinIO client (S3 API → port 9000 ONLY)
    minio_client = Minio(
        minio_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )

    try:
        # Create bucket if it does not exist
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created.")

        # Upload model file
        minio_client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=file_path
        )

        print(
            f"File '{file_path}' uploaded as '{object_name}' "
            f"to bucket '{bucket_name}'."
        )

    except S3Error as e:
        print(f"MinIO error: {e}")
    except FileNotFoundError:
        print(f"Model file not found: {file_path}")

if __name__ == "__main__":
    upload_file_to_minio()
