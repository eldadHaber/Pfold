import os
import time

import paramiko
import numpy as np
import fnmatch


class FastTransport(paramiko.Transport):
    def __init__(self, sock):
        super(FastTransport, self).__init__(sock)
        self.window_size = 2147483647
        self.packetizer.REKEY_BYTES = pow(2, 40)
        self.packetizer.REKEY_PACKETS = pow(2, 40)


def establish_connection(server=None,username=None,password=None):


    # server_tera = '142.103.36.194'
    if server is None:
        server = '137.82.107.147'
    if username is None:
        username = 'tb'
    if password is None:
        password = '123'

    # ssh = paramiko.SSHClient()
    # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    # ssh.connect(server, username=username, password=password)
    # sftp = ssh.open_sftp()


    ssh = FastTransport((server, 22))
    ssh.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(ssh)

    return ssh, sftp


def get_data_batch(remote_booking_folder,remote_data_folder,local_book,local_data_in, server=None, user=None, pas=None, MAX_RETRIES=10):
    booking_found = False
    ssh, sftp = establish_connection(server=server, username=user, password=pas)
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

        filesizes = 0
        t0 = time.time()
        for file in files_to_get:
            retry = 0
            fullfile_remote = os.path.join(remote_data_folder, file)  # NOTE that this might cause trouble between systems
            fullfile_local = os.path.join(local_data_in, file)
            while retry < MAX_RETRIES:
                try:
                    sftp.get(fullfile_remote, fullfile_local)
                    break
                except:
                    retry += 1
                    ssh, sftp = establish_connection(server=server, username=user, password=pas)

            filesizes += os.path.getsize(fullfile_local) / 1024 / 1024
        t1 = time.time()
    else:
        print("No more bookings")
        ids = None
        files_to_get = None
        bookkeeper_busy = None
    sftp.close()
    ssh.close()

    bookfiles_running = fnmatch.filter(bookfiles_all, "*.npz.running")
    bookfiles_done = fnmatch.filter(bookfiles_all, "*.npz.done")
    print("{:} bookings found. {:} done, {:} running, {:} available".format(len(bookfiles_all), len(bookfiles_done), len(bookfiles_running), len(bookfiles)))
    print("Transfered {:} files totalling {:2.2f} Mb, took {:2.2f} s. Speed {:2.2f} Mb/s ".format(len(files_to_get), filesizes, t1-t0, filesizes / (t1-t0)))
    return ids, files_to_get, bookkeeper_busy




def get_data(sftp ,remote_data_folder,local_data_in):
    files_to_get = sftp.listdir(remote_data_folder)

    filesizes = 0
    t0 = time.time()
    for file in files_to_get:
        fullfile_remote = os.path.join(remote_data_folder, file)  # NOTE that this might cause trouble between systems
        fullfile_local = os.path.join(local_data_in, file)
        sftp.get(fullfile_remote, fullfile_local)
        filesizes += os.path.getsize(fullfile_local) / 1024 / 1024
    t1 = time.time()

    print("Transfered {:} files totalling {:2.2f} Mb, took {:2.2f} s. Speed {:2.2f} Mb/s ".format(len(files_to_get), filesizes, t1-t0, filesizes / (t1-t0)))
    return



def send_data_batch(remote_result_folder,local_data_out,bookkeeper_name=None, server=None, user=None, pas=None, MAX_RETRIES=10):
    filesizes = 0
    t0 = time.time()
    nf = 0
    ssh, sftp = establish_connection(server=server, username=user, password=pas)
    for root, dirs, files in os.walk(local_data_out):
        for fname in files:
            retry = 0
            nf += 1
            full_fname = os.path.join(root, fname)
            full_remote = os.path.join(remote_result_folder, fname)
            while retry < MAX_RETRIES:
                try:
                    sftp.put(full_fname, full_remote)
                    break
                except:
                    retry += 1
                    ssh, sftp = establish_connection(server=server, username=user, password=pas)
            filesizes += os.path.getsize(full_fname) / 1024 / 1024
    t1 = time.time()
    if bookkeeper_name is not None:
        bookkeeper_name_done = "{:}.done".format(os.path.splitext(bookkeeper_name)[0])
        sftp.rename(bookkeeper_name, bookkeeper_name_done)
    sftp.close()
    ssh.close()
    print("Transfered {:} files totalling {:2.2f} Mb, took {:2.2f} s. Speed {:2.2f} Mb/s ".format(nf, filesizes, t1-t0, filesizes / (t1-t0)))
    return


def clean_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))
    return



if __name__ == "__main__":
    local_data_in = './../data/input/'
    local_data_out = './../data/output/'
    local_book = './../data/'

    remote_data_folder = './MSA/'
    remote_booking_folder = './bookkeeping/'
    remote_result_folder = './cov/'


    server_tera = '142.103.36.194'
    user = 'tboesen'
    pas = 'Sl1ndertex'

    MAX_RETRIES = 10

    port = 22

    remote_file = '/home/tboesen/test/1.gz'
    local_file = 'F:/test/1.gz'
    local_file2 = 'F:/test/2.gz'

    # sftp_file = "/somefolder/somefile.txt"
    # local_file = "/somefolder/somewhere/here.txt"
    ssh_conn = sftp_client = None
    # username = "username"
    # password = "password"

    start_time = time.time()

    for retry in range(MAX_RETRIES):
        try:
            ssh_conn = paramiko.Transport((server_tera, port))
            ssh_conn.connect(username=user, password=pas)
            # method 1 using sftpfile.get and settings window_size, max_packet_size
            window_size = pow(4, 12)  # about ~16MB chunks
            max_packet_size = pow(4, 12)
            sftp_client = paramiko.SFTPClient.from_transport(ssh_conn, window_size=window_size,
                                                             max_packet_size=max_packet_size)
            # get_data(sftp_client, remote_result_folder, local_data_out)
            t0 = time.time()
            sftp_client.get(remote_file, local_file)
            t1 = time.time()
            # method 2 breaking up file into chunks to read in parallel
            sftp_client = paramiko.SFTPClient.from_transport(ssh_conn)
            filesize = sftp_client.stat(remote_file).st_size
            chunksize = pow(4, 12)  # <-- adjust this and benchmark speed
            chunks = [(offset, chunksize) for offset in range(0, filesize, chunksize)]
            with sftp_client.open(remote_file, "rb") as infile:
                with open(local_file2, "wb") as outfile:
                    for chunk in infile.readv(chunks):
                        outfile.write(chunk)
            t2 = time.time()
            break
        except (EOFError, paramiko.ssh_exception.SSHException, OSError) as x:
            retry += 1
            print("%s %s - > retrying %s..." % (type(x), x, retry))
            time.sleep(abs(retry) * 10)
            # back off in steps of 10, 20.. seconds
        finally:
            if hasattr(sftp_client, "close") and callable(sftp_client.close):
                sftp_client.close()
            if hasattr(ssh_conn, "close") and callable(ssh_conn.close):
                ssh_conn.close()

    print("Loading File %s Took %d seconds " % (remote_file, time.time() - start_time))
    print("method 1 {:2.2f}, method 2 {:2.2f}".format(t1-t0,t2-t1))



    # ssh, sftp = establish_connection(server=server_tera, username=user, password=pas)
    #
    # remote_result_folder = '/home/tboesen/test/'
    # local_data_out = 'F:/test/'
    #
    # # send_data_batch(sftp, remote_result_folder,local_data_out)
    #
    # get_data(sftp, remote_result_folder, local_data_out)



    #
    # ids, files_to_get, bookkeeper_name = get_data_batch(sftp, remote_booking_folder,remote_data_folder,local_book,local_data_in)
    #
    # sftp.close()
    # ssh.close()
    #
    #
    # # Here we do the analysis
    #
    # #Next we wish to transfer output files
    # ssh, sftp = establish_connection()
    #
    # send_data_batch(sftp, remote_result_folder,local_data_out,bookkeeper_name)
    #
    # sftp.close()
    # ssh.close()
    #
    #
    # clean_folder(local_data_in)
    # clean_folder(local_data_out)
    #




