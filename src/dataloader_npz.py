import glob

import numpy as np
import torch.utils.data as data

from src.dataloader_utils import SeqFlip, DrawFromProbabilityMatrix, MaskRandomSubset, convert_seq_to_onehot, \
    convert_1d_features_to_2d, ConvertCoordToDists, Random2DCrop


class Dataset_npz(data.Dataset):
    '''
    Reads a folder full of npz files, and treats its as a database.
    Expects the data to be packed as follows:
    (features, target, mask)
        features = (seq, pssm, entropy)
        target = (r1, r2, r3)

    '''
    def __init__(self, folder, seq_flip_prop=0.5, chan_out=3, feature_dim=1, i_seq=False, i_pssm=False, i_entropy=False, i_cov=False, i_cov_all=False, i_contact=False, inpainting=False, random_crop=False, cross_dist=False):

        search_command = folder + "*.npz"
        npzfiles = [f for f in glob.glob(search_command)]
        self.folder = folder
        self.files = npzfiles

        self.Flip = SeqFlip(seq_flip_prop)
        self.chan_out = chan_out
        self.draw_seq_from_pssm = False
        self.draw = DrawFromProbabilityMatrix(fraction_of_seq_drawn=0.2)
        self.draw_prob = 0.5
        self.seq_mask = MaskRandomSubset()
        self.feature_dim = feature_dim
        self.i_seq = i_seq
        self.i_pssm = i_pssm
        self.i_entropy = i_entropy
        self.i_cov = i_cov
        self.i_cov_all = i_cov_all
        self.i_contact = i_contact
        self.inpainting = inpainting
        self.chan_in = self.calculate_chan_in()
        self.coord_to_dist = ConvertCoordToDists()
        self.crop = Random2DCrop()
        self.use_crop = random_crop
        self.use_cross_dist = cross_dist

    # def mask_and_label(self, token_ids: List[int]):
    #     """
    #     Masks a sequence string according to the rules set out by BERT
    #     :return:
    #     """
    #     labels = []
    #
    #     for i, token in enumerate(token_ids):
    #         prob = random.random()
    #         if prob < 0.15 and token != 0:
    #             prob *= 6.666666666
    #             # prob /= 0.15
    #
    #             # 80% randomly change token to mask token
    #             if prob < 0.8:
    #                 token_ids[i] = self.mask_token
    #
    #             # 10% randomly change token to random token
    #             elif prob < 0.9:
    #                 token_ids[i] = self.get_random_base_token()
    #
    #             # 10% randomly stay with current token
    #
    #             labels.append(token)
    #
    #         else:
    #             labels.append(0)  # Padding
    #     return token_ids, labels


    def __getitem__(self, index):
        data = np.load(self.files[index])
        features_1d += (convert_seq_to_onehot(data['seq']),)


        self.Flip.reroll()
        features = self.Flip(features)
        coords = self.Flip(coords)

        if self.use_cross_dist: # In this case, we put all the coordinates into one long coordinate array.
            nc = len(coords)
            nl = coords[0].shape[1]
            coord_long = (np.concatenate(coords, axis=1),)
            dist_long = self.coord_to_dist(coord_long)
            distances = ()
            for i in range(nc):
                for j in range(nc):
                    distances += (dist_long[0][i*nl:(i+1)*nl,j*nl:(j+1)*nl],)
        else:
            distances = self.coord_to_dist(coords)

        if self.feature_dim == 2 and self.use_crop:
            # Random 64x64 crop of the data
            self.crop.randomize(features.shape[-1])
            features = self.crop(features)
            distances = self.crop(distances)
            coords = (coords[0][:,0:64],) # Note that the coordinates are useless in the 2D case!

        return features, distances, coords

    def __len__(self):
        return len(self.files)


    def match_target_channels(self,target):
        if self.chan_out == 3:
            target = (target[0],)
        elif self.chan_out == 6:
            target = target[0:2]
        elif self.chan_out == 9:
            pass
        else:
            raise NotImplementedError("chan_out is {}, which is not implemented".format(self.chan_out))
        return target


    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.folder + ')'
