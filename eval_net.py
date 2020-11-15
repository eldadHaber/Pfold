import os

import torch

from src.dataloader import PadCollate
from src.dataloader_npz import Dataset_npz
from src.optimization import net_prediction

if __name__ == '__main__':
    network = 'F:/results/run_amazon/2020-11-09_19_31_14/network.pt'
    dataset = 'f:/final_dataset_1d_validate/'
    dataset_out = './results/figures/'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 1

    i_entropy = False
    feature_dim = 1
    i_seq = True
    i_pssm = False
    i_cov = False
    i_cov_all = False
    i_contact = False
    inpainting = False


    dataset_test = Dataset_npz(dataset, feature_dim=feature_dim, seq_flip_prop=0, i_seq=i_seq, i_pssm=i_pssm,
                               i_entropy=i_entropy, i_cov=i_cov, i_cov_all=i_cov_all, i_contact=i_contact,
                               inpainting=inpainting)

    pad_modulo = 8

    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=PadCollate(pad_modulo=pad_modulo),
                                           drop_last=False)


    # Load network and set it in evaluate mode
    net = torch.load(network)
    net.eval()
    net.to(device)

    # Check output folder is non-existent, and then create it
    os.makedirs(dataset_out,exist_ok=True)

    net_prediction(net, dl_test, device=device, plot_results=False, save_results="{:}/".format(dataset_out))

