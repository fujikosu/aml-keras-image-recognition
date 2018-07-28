import os
from azure.storage.blob import BlockBlobService
import logging

logger = logging.getLogger('azure_utils')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def load_file_from_blob(container, file_name, dest_path):
    logger.info("Starting download of {}...".format(file_name))
    if dest_path.is_file():
        logger.info(
            "File {} already exists, skipping download from Azure Blob.".
            format(dest_path))
        return False

    blob_service = get_blob_service()
    logger.info("container: {0}, file_name: {1}, destination: {2}".format(
        container, file_name, dest_path))
    blob_service.get_blob_to_path(container, file_name, str(dest_path))
    return True


def get_blob_service():
    storage_account_name = os.environ['STORAGE_ACCOUNT_NAME']
    storage_account_key = os.environ['STORAGE_ACCOUNT_KEY']
    return BlockBlobService(
        account_name=storage_account_name, account_key=storage_account_key)
