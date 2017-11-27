from os.path import join, isfile
import glob
import os
from azure.storage.blob import BlockBlobService
import datetime


def load_file_from_blob(container, file_name, dest):
    print("Starting download of {}...".format(file_name))
    if os.path.isfile(dest):
        print("File {} already exists, skipping download from Azure Blob.".
              format(dest))
        return False

    blob_service = get_blob_service()
    print("container {0}, file_name {1}, dest {2}".format(
        container, file_name, dest))
    blob_service.get_blob_to_path(container, file_name, dest)
    return True


def get_blob_service():
    storage_account_name = os.environ['STORAGE_ACCOUNT_NAME']
    storage_account_key = os.environ['STORAGE_ACCOUNT_KEY']
    return BlockBlobService(
        account_name=storage_account_name, account_key=storage_account_key)
