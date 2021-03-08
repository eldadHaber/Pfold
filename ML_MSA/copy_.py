import glob
import os
import time
from pathlib import Path
from subprocess import call

class CTError(Exception):
    def __init__(self, errors):
        self.errors = errors

try:
    O_BINARY = os.O_BINARY
except:
    O_BINARY = 0
READ_FLAGS = os.O_RDONLY | O_BINARY
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | O_BINARY
BUFFER_SIZE = 128*1024

def copyfile(src, dst):
    try:
        fin = os.open(src, READ_FLAGS)
        stat = os.fstat(fin)
        fout = os.open(dst, WRITE_FLAGS, stat.st_mode)
        for x in iter(lambda: os.read(fin, BUFFER_SIZE), b""):
            os.write(fout, x)
    finally:
        try: os.close(fin)
        except: pass
        try: os.close(fout)
        except: pass



import torch

folder_in = 'F:/Globus/raw/'
# folder_in = 'F:/Globus/raw_subset/'
folder_ref = 'F:/MSAS/'
# folder_out = 'D:/raw_MSA/'
# folder_out = 'F:/MSAS_10000_20000/'



search_command = folder_in + "*.a2m.gz"
a2mfiles = [f for f in sorted(glob.glob(search_command))]
max_seq_len = 600
min_seq_len = 1
max_samples = 30000
unk_idx = 20
max_files_to_copy = 10000
start_file_idx = 30000
folder_out = "D:/raw_MSAS_{:}_{:}/".format(start_file_idx,start_file_idx+max_files_to_copy)
os.makedirs(folder_out,exist_ok=True)
#
search_command = folder_out + "*.pt"
outfiles = [f for f in sorted(glob.glob(search_command))]
# for i, file in enumerate(torchfiles):
#     t0 = time.time()
#     data = torch.load(file)
#     msa = data['msas']
#     seq = data['seq']
#     t1 = time.time()
#     print("Time taken {:2.2f}".format(t1-t0))
#


# base_infiles = Path(Path(a2mfiles).stem).stem
base_infiles = [Path(Path(x).stem).stem for x in a2mfiles]

# base_outfiles = Path(outfiles).stem
base_outfiles = [Path(x).stem for x in outfiles]

files_in = [x for x in base_infiles if x not in base_outfiles]

extension = '.a2m.gz'

nfiles = len(files_in)
files_in = files_in[start_file_idx:start_file_idx+max_files_to_copy]
t0 = time.time()
for i, file_in in enumerate(files_in):
    file_in_full = "{:}{:}{:}".format(folder_in,file_in,extension)
    file_out_full = "{:}{:}{:}".format(folder_out,file_in,extension)
    # call(['xcopy', file_in_full, file_out_full, '/K/O/X'])
    copyfile(file_in_full, file_out_full)

