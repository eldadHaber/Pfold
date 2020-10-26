import glob
import os

import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import matplotlib
# matplotlib.use('TkAgg') #TkAgg

from srcOld.dataloader import PadCollate
from srcOld.dataloader_npz import Dataset_npz
from srcOld.dataloader_utils import ConvertPnetFeaturesTo1D, ConvertCoordToDists
from srcOld.loss import loss_tr_tuples
from srcOld.utils import move_tuple_to
from srcOld.visualization import compare_distogram, plotfullprotein

if __name__ == "__main__":
    dataset_in = "./../data/clean_pnet_test/"
    dataset_out = "./../data/inpainted_pnet_test/"
    network = "./../pretrained_networks/inpaint_transformer.pt"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 1

    # Get metadata for dataset_in
    transform_test = transforms.Compose([ConvertPnetFeaturesTo1D()])
    transform_target_test = transforms.Compose([ConvertCoordToDists()])

    dataset = Dataset_npz(dataset_in, transform=transform_test, target_transform=transform_target_test,
                               chan_in=25, chan_out=3, draw_seq_from_pssm=False, mask_random_seq=False)

    dl = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size,len(dataset)), shuffle=False, num_workers=0, collate_fn=PadCollate(),
                                           drop_last=False)


    # Load network and set it in evaluate mode
    net = torch.load(network)
    net.eval()
    net.to(device)

    # Check output folder is non-existent, and then create it
    os.makedirs(dataset_out,exist_ok=True)

    plot_results = True
    with torch.no_grad():
        for i,(seq, dists,mask, coords) in enumerate(dl):
            seq = seq.to(device, non_blocking=True)
            dists = move_tuple_to(dists, device, non_blocking=True)
            coords = move_tuple_to(coords, device, non_blocking=True)
            dists_pred, coords_pred = net(seq)
            _, coords_pred_tr, coords_tr = loss_tr_tuples(coords_pred, coords, return_coords=True)
            if plot_results:
                compare_distogram(dists_pred, dists)
                plotfullprotein(coords_pred_tr, coords_tr)
            #Now save the data to the output folder





