import os.path as osp
import lmdb
import pyarrow as pa
import torch.utils.data as data
import copy

from srcOld.dataloader_utils import SeqFlip


class Dataset_lmdb(data.Dataset):
    '''
    Reads an lmdb database.
    Expects the data to be packed as follows:
    (features, target, mask)
        features = (seq, pssm, entropy)
        target = (dist, omega, phi, theta)

    '''
    def __init__(self, db_path, transform=None, target_transform=None, mask_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path), max_readers=1,
                             readonly=True, lock=False,
                             readahead=False, meminit=False)


        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform
        # self.nfeatures = 84

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        features = copy.deepcopy(unpacked[0])

        targets = copy.deepcopy(unpacked[1])
        # coords = copy.deepcopy(unpacked[2])

        # mask = copy.deepcopy(unpacked[2])


        if isinstance(self.transform.transforms[0], SeqFlip):
            self.transform.transforms[0].reroll()
        if self.transform is not None:
            features = self.transform(features)

        if self.target_transform is not None:
            distances, coords = self.target_transform(targets)

        # if self.mask_transform is not None:
        #     mask = self.mask_transform(mask)

        return features, distances, coords

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
