import matplotlib.pyplot as plt
import numpy as np
import torch

def ind2sub(idx, cols):
	r = idx // cols
	c = idx - r * cols
	return r, c



def sub2ind(asize, rows, cols):
    s1 = asize[1]
    return rows*s1 + cols

def createQuadTreeFromImage(A, tol):

    m1 = A.shape[0]
    m2 = A.shape[1]
    maxbsz = m1
    print(m1)
    S = torch.ones(m1,m2)
    Amin = A.view(-1)
    Amax = A.view(-1)

    cnt = 1
    for k in range(np.log2(maxbsz)):
        bsz = 2**k #[0:log2(maxbsz) - 1]:
        i, j = torch.where(S == bsz)
        mi = torch.remainder(i - 1, 2 * bsz)
        mj = torch.remainder(j - 1, 2 * bsz)
        I = torch.where((mi == 0) & (mj == 0))

        if I.nelement() != 0:
            I00 = S.shape[1]*i[I] + j[I]
            I01 = S.shape[1]*(i[I] + bsz) + j[I]
            I10 = S.shape[1]*i[I] + ( j[I] + bsz)
            I11 = S.shape[1]*(i[I] + bsz) + (j[I] + bsz)

            Amin[I00] = torch.tensor([Amin[I00].min(), Amin[I01].min(), Amin[I10].min(), Amin[I11].min(), 2]).min()
            Amax[I00] = torch.tensor([Amax[I00].max(), Amax[I01].max(), Amin[I10].max(), Amax[I11].max(), 2]).max()

            Ic = torch.where((Amax[I00] - Amin[I00]) <= tol)

            S[I00[Ic]] = 2 * bsz
            S[I01[Ic]] = 0
            S[I10[Ic]] = 0
            S[I11[Ic]] = 0

    return S, A