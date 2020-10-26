import os
import paramiko
import numpy as np
import fnmatch

def get_data_batch(remote_booking_folder,remote_data_folder,local_book,local_data_in):
    booking_found = False
    bookfiles_all = sftp.listdir(remote_booking_folder)

    pattern = '*.npz'
    bookfiles = fnmatch.filter(bookfiles_all, pattern)
    for bookfile in bookfiles:
        try:
            bookkeeper = os.path.join(remote_booking_folder, bookfile)
            bookkeeper_busy = "{:}.running".format(bookkeeper)
            sftp.rename(bookkeeper, bookkeeper_busy)
            booking_found = True
        except:
            continue
        break

    if booking_found:
        local_book_file = os.path.join(local_book,os.path.basename(bookkeeper_busy))
        sftp.get(bookkeeper_busy, local_book_file)
        book = np.load(local_book_file)
        files_to_get = book['files']
        ids = book['ids']

        for file in files_to_get:
            fullfile_remote = os.path.join(remote_data_folder, file)  # NOTE that this might cause trouble between systems
            fullfile_local = os.path.join(local_data_in, file)
            sftp.get(fullfile_remote, fullfile_local)
    else:
        print("No more bookings")
        ids = None
        files_to_get = None
        bookkeeper_busy = None

    bookfiles_running = fnmatch.filter(bookfiles_all, "*.npz.running")
    bookfiles_done = fnmatch.filter(bookfiles_all, "*.npz.done")
    print("{:} bookings found. {:} done, {:} running, {:} available".format(len(bookfiles_all), len(bookfiles_done), len(bookfiles_running), len(bookfiles)))
    return ids, files_to_get, bookkeeper_busy


def send_data_batch(remote_result_folder,local_data_out,bookkeeper_name=None):
    for root, dirs, files in os.walk(local_data_out):
        for fname in files:
            full_fname = os.path.join(root, fname)
            full_remote = os.path.join(remote_result_folder, fname)
            sftp.put(full_fname, full_remote)
    if bookkeeper_name is not None:
        bookkeeper_name_done = "{:}.done".format(os.path.splitext(bookkeeper_name)[0])
        sftp.rename(bookkeeper_name, bookkeeper_name_done)
    return


def clean_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))
    return

server_tera = '142.103.36.194'

server = '137.82.107.147'
username = 'tb'
password = '123'

local_data_in = './../data/input/'
local_data_out = './../data/output/'
local_book = './../data/'

remote_data_folder = './MSA/'
remote_booking_folder = './bookkeeping/'
remote_result_folder = './cov/'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
ssh.connect(server, username=username, password=password)
sftp = ssh.open_sftp()

ids, files_to_get, bookkeeper_name = get_data_batch(remote_booking_folder,remote_data_folder,local_book,local_data_in)

# Here we do the analysis

#Next we wish to transfer output files
send_data_batch(remote_result_folder,local_data_out,bookkeeper_name)

clean_folder(local_data_in)
clean_folder(local_data_out)

sftp.close()
ssh.close()




