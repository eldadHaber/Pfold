import glob

import numpy as np
import torch.utils.data as data
from supervised.dataloader_utils import SeqFlip, DrawFromProbabilityMatrix, MaskRandomSubset, convert_seq_to_onehot, \
    convert_1d_features_to_2d, ConvertCoordToDists, Random2DCrop
from supervised.config import config as c


class Dataset_npz(data.Dataset):
    '''
    Reads a folder full of npz files, and treats it as a database.
    npz file variables recognized:
        Required variables:
            AA_LIST[List](k): List translating k amino acids to the numbers [0:k].
            log_units[int]: Gives the log10 units that coordinates are saved in. (picometer = -12, nanometer = -9...)
            id[str]: PDB id.
        Optional variables:
            seq[vector of ints]: Gives the amino acid configuration of the protein according to the AA_Dict
            pssm[Matrix of floats]: Gives the position specific scoring matrix for the protein
            entropy:
            R_CAlpha:
            R_Cbeta:
            R_N:
            cov:
            contact:


    '''
    def __init__(self, folder,  flip_protein, i_seq, i_pssm, i_entropy, i_inpaint, i_cov, i_contact, o_rCa,o_rCb, o_rN, o_dist, AA_list, log_units, use_weight):
        """
        log_units[int]: Gives the log10 units that coordinates/distances should be returned in. (picometer = -12, nanometer = -9...)

        """
        search_command = folder + "*.npz"
        npzfiles = [f for f in glob.glob(search_command)]
        self.folder = folder
        self.files = npzfiles

        self.Flip = SeqFlip(flip_protein)
        self.i_seq = i_seq
        self.i_pssm = i_pssm
        self.i_entropy = i_entropy
        self.i_inpaint = i_inpaint
        self.i_cov = i_cov
        self.i_contact = i_contact
        self.use_weight = use_weight
        self.o_rCa = o_rCa
        self.o_rCb = o_rCb
        self.o_rN = o_rN
        self.o_dist = o_dist
        self.AA_list = list(AA_list)
        self.log_units = log_units
        self.coord_to_dist = ConvertCoordToDists()
        self.seq_mask = MaskRandomSubset()

        # We run the first example to get the input and output channels and perform sanity checks
        arg = self.__getitem__(0)
        self.chan_in = arg[0][0].shape[0]
        self.chan_out = 3*(self.o_rCa+self.o_rCb+self.o_rN)
        try:
            c['network_args']['chan_in'] = self.chan_in
            c['network_args']['chan_out'] = self.chan_out
        except:
            pass

        return

    def __getitem__(self, index):
        data = np.load(self.files[index], allow_pickle=True)
        AA_list = list(data['AA_LIST'])
        log_units = data['log_units']
        scaling = 10.0 ** (log_units - self.log_units)
        assert AA_list == self.AA_list, "The data was saved with a different amino acid list than the one you are currently using."

        w = np.empty((1,1),dtype=np.float32)
        if self.use_weight:
            w[0] = data['weight']
        else:
            w[0] = 1

        coords = ()
        if self.o_rCa:
            coords += ((data['rCa']*scaling).astype('float32'),)
        if self.o_rCb:
            coords += ((data['rCb']*scaling).astype('float32'),)
        if self.o_rN:
            coords += ((data['rN']*scaling).astype('float32'),)

        features = ()
        if self.i_seq:
            features += (convert_seq_to_onehot(data['seq']),)
        if self.i_pssm:
            features += (data['pssm'],)
        if self.i_entropy:
            features += (data['entropy'],)
        if self.i_inpaint:
            r, m = self.seq_mask(coords[0])
            features += (r,m[None,:],)

        if self.i_cov:
            cov = data['cov']
            cov = cov.reshape(-1,cov.shape[-1])
            features += (cov,)
        if self.i_contact:
            contact = data['contact']
            features += (contact.reshape(-1,contact.shape[-1]),)

        features = np.concatenate(features, axis=0).astype('float32')
        self.Flip.reroll()
        features = self.Flip(features)
        coords = self.Flip(coords)
        protein_id = data['id']




        if self.o_dist:
            distances = self.coord_to_dist(coords)
            return (features,), distances, coords, (protein_id,), w
        else:
            return (features,), coords, (protein_id,), w



    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.folder + ')'
