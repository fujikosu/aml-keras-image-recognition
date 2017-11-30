from shutil import unpack_archive
import os


def unzip_file(filepath, destination):
    print("Unzipping {0} to {1}".format(filepath, destination))
    if os.path.isdir(destination):
        print("Unzipped directory {} already exists, skipping unzipping.".format(destination))
        # print("Unzipping {0} to {1}".format(filepath, destination))
    else:
        unpack_archive(filepath, destination)
    # with zipfile.ZipFile(filepath, "r") as zip_ref:
    #     # print(zip_ref.namelist())
    #     # for name in zip_ref.namelist():
    #     #     # print(name)
    #     #     if name.endswith('/'):
    #     #         os.makedirs(os.path.join(destination, name), exist_ok=True)
    #     #     else:
    #     #         zip_ref.extract(name, path=os.path.join(destination, name))

    #     for f in zip_ref.namelist():
    #         if not os.path.basename(f):
    #             os.mkdir(f)
    #         else:
    #             with open(f, 'wb') as uzf:
    #                 uzf.write(zip_ref.read(f))
    # zip_ref.extractall(destination)
