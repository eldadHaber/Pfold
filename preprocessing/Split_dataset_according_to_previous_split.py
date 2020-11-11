import glob
import os
import random
from shutil import copyfile

if __name__ == "__main__":
    # cov_folder = "./../data/cov/"
    dataset = 'f:/final_dataset_1d_all/'
    dataset_train = 'f:/final_dataset_1d_all_train/'
    dataset_validate = 'f:/final_dataset_1d_all_validate/'

    dataset_train_template = 'f:/final_dataset_1d_train/'
    dataset_validate_template = 'f:/final_dataset_1d_validate/'

    os.makedirs(dataset_train)
    os.makedirs(dataset_validate)

    search_command = dataset + "*.npz"
    main_files = [f for f in glob.glob(search_command)]

    search_command = dataset_train_template + "*.npz"
    train_templates = [f for f in glob.glob(search_command)]

    search_command = dataset_validate_template + "*.npz"
    validate_templates = [f for f in glob.glob(search_command)]

    for i,file in enumerate(main_files):
        if file in train_templates:
            path = dataset_train
        elif file in validate_templates:
            path = dataset_validate
        else:
            print("problem with file {:}".format(file))
        filename = os.path.basename(file)
        fileout = os.path.join(path,filename)
        copyfile(file, fileout)
