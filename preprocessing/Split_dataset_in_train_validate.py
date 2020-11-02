import glob
import os
import random
from shutil import copyfile

if __name__ == "__main__":
    # cov_folder = "./../data/cov/"
    dataset = 'f:/final_dataset_1d/'

    dataset_train = 'f:/final_dataset_1d_train/'
    dataset_validate = 'f:/final_dataset_1d_validate/'

    os.makedirs(dataset_train)
    os.makedirs(dataset_validate)

    nvalidate = 200
    ntrain = -1

    search_command = dataset + "*.npz"
    main_files = [f for f in glob.glob(search_command)]

    random.shuffle(main_files)

    for i,file in enumerate(main_files):
        if i < nvalidate:
            path = dataset_validate
        else:
            path = dataset_train
        filename = os.path.basename(file)
        fileout = os.path.join(path,filename)
        copyfile(file, fileout)

