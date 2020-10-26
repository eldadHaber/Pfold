import os
import numpy as np
import glob

def create_bookkeeper(files,folder_out,max_size):
    """
    max_size should be given in megabytes.
    """

    #make sure the files are sorted
    n = len(files)
    filesizes = np.empty(n)
    for i,file in enumerate(files):
        filesizes[i] = os.path.getsize(file) / 1024 / 1024


    nb = 0
    size = 0
    files_i = []
    ids = []
    for i,(file,filesize) in enumerate(zip(files,filesizes)):
        if filesize + size > max_size:
            #Kick off the files as a new batch
            filename_book = "{:}book_{:}".format(folder_out,nb)
            np.savez(filename_book,files=files_i,ids=ids)
            files_i = []
            ids = []
            size = 0
            nb += 1
        files_i.append(os.path.basename(file))
        size += filesize
        ids.append(i)
    filename_book = "{:}book_{:}".format(folder_out, nb)
    np.savez(filename_book, files=files_i, ids=ids)
    return

if __name__ == "__main__":
    # MSA_folder = "./../data/MSA/"
    MSA_folder = "F:/Globus/raw/"
    search_command = MSA_folder + "*.a2m.gz"
    a2mfiles = [f for f in sorted(glob.glob(search_command))]
    # bookkeeping = "./../data/bookkeeping/"
    bookkeeping = "F:/Globus/bookkeeping/"

    max_size = 100
    create_bookkeeper(a2mfiles, bookkeeping, max_size)
    print("done")


