


def convertCoordToDistAngles(rN, rCa, rCb, mask=None):
    '''
    data should be coordinate data in pnet format, meaning that each amino acid is characterized
    by a 3x3 matrix, which are the coordinates of r1,r2,r3=N,Calpha,Cbeta.
    This lite version, only computes the angles in along the sequence
    (upper one-off diagonal of the angle matrices of the full version)
    '''
    seq_len = rN.shape[0]
    # Initialize distances and angles

    d = torch.zeros([seq_len, seq_len])
    phi = torch.zeros([seq_len, seq_len])
    omega = torch.zeros([seq_len, seq_len])
    theta = torch.zeros([seq_len, seq_len])

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if mask is not None and (mask[i] == 0 or mask[j] == 0):
                continue

            r1i = rN[i, :]  # N1  atom
            r2i = rCa[i, :]  # Ca1 atom
            r3i = rCb[i, :]  # Cb1 atom
            r1j = rN[j, :]  # N2  atom
            r2j = rCa[j, :]  # Ca2 atom
            r3j = rCb[j, :]  # Cb2 atom

            # Compute distance Cb-Cb
            vbb = r3j - r3i
            d[i, j] = torch.norm(vbb)
            d[j, i] = d[i, j]

            # Compute phi
            v1 = r2i - r3i  # Ca1 - Cb1
            v2 = r3i - r3j  # Cb1 - Cb2
            # phi[i,j] = torch.acos(torch.dot(v1,v2)/torch.norm(v1)/torch.norm(v2))
            phi[i, j] = torch.dot(v1, v2) / torch.norm(v1) / torch.norm(v2)

            v1 = r2j - r3j  # Ca2 - Cb2
            v2 = r3j - r3i  # Cb2 -Cb1
            # phi[j, i] = torch.acos(torch.dot(v1,v2)/torch.norm(v1)/torch.norm(v2))
            phi[j, i] = torch.dot(v1, v2) / torch.norm(v1) / torch.norm(v2)

            # Thetas
            v1 = r1i - r2i  # N1 - Ca1
            v2 = r2i - r3i  # Ca1 - Cb1
            v3 = r3i - r3j  # Cb1 - Cb2
            theta[i, j] = ang2plain(v1, v2, v2, v3)

            v1 = r1j - r2j  # N2 - Ca2
            v2 = r2j - r3j  # Ca2 - Cb2
            v3 = r3j - r3i  # Cb2 - Cb1
            theta[j, i] = ang2plain(v1, v2, v2, v3)

            # Omega
            v1 = r2i - r3i  # Ca1 - Cb1
            v2 = r3i - r3j  # Cb1 - Cb2
            v3 = r3j - r2j  # Cb2 - Ca2
            omega[i, j] = ang2plain(v1, v2, v2, v3)
            omega[j, i] = omega[i, j]

    return d, omega, phi, theta



def convertCoordToDistAnglesVec(rN, rCa, rCb, mask=None):
    # Vectorized

    # Get D
    D = torch.sum(rCb ** 2, dim=1).unsqueeze(1) + torch.sum(rCb ** 2, dim=1).unsqueeze(0) - 2 * (rCb @ rCb.t())
    M = mask.unsqueeze(1) @  mask.unsqueeze(0)
    D = torch.sqrt(torch.relu(M*D))

    # Get Upper Phi
    # TODO clean Phi to be the same as OMEGA
    V1x = rCa[:, 0].unsqueeze(1) - rCb[:, 0].unsqueeze(1)
    V1y = rCa[:, 1].unsqueeze(1) - rCb[:, 1].unsqueeze(1)
    V1z = rCa[:, 2].unsqueeze(1) - rCb[:, 2].unsqueeze(1)
    V2x = rCb[:, 0].unsqueeze(1) - rCb[:, 0].unsqueeze(1).t()
    V2y = rCb[:, 1].unsqueeze(1) - rCb[:, 1].unsqueeze(1).t()
    V2z = rCb[:, 2].unsqueeze(1) - rCb[:, 2].unsqueeze(1).t()
    # Normalize them
    V1n = torch.sqrt(V1x**2 + V1y**2 + V1z**2)
    V1x = V1x/V1n
    V1y = V1y/V1n
    V1z = V1z/V1n
    V2n = torch.sqrt(V2x**2 + V2y**2 + V2z**2)
    V2x = V2x/V2n
    V2y = V2y/V2n
    V2z = V2z/V2n
    # go for it
    PHI = M*(V1x * V2x + V1y * V2y + V1z * V2z)
    indnan = torch.isnan(PHI)
    PHI[indnan] = 0.0

    # Omega
    nat = rCa.shape[0]
    V1 = torch.zeros(nat, nat, 3)
    V2 = torch.zeros(nat, nat, 3)
    V3 = torch.zeros(nat, nat, 3)
    # Ca1 - Cb1
    V1[:,:,0] = (rCa[:,0].unsqueeze(1) - rCb[:,0].unsqueeze(1)).repeat((1,nat))
    V1[:,:,1] = (rCa[:,1].unsqueeze(1) - rCb[:,1].unsqueeze(1)).repeat((1, nat))
    V1[:,:,2] = (rCa[:,2].unsqueeze(1) - rCb[:,2].unsqueeze(1)).repeat((1, nat))
    # Cb1 - Cb2
    V2[:,:,0] = rCb[:,0].unsqueeze(1) - rCb[:,0].unsqueeze(1).t()
    V2[:,:,1] = rCb[:,1].unsqueeze(1) - rCb[:,1].unsqueeze(1).t()
    V2[:,:,2] = rCb[:,2].unsqueeze(1) - rCb[:,2].unsqueeze(1).t()
    # Cb2 - Ca2
    V3[:,:,0] = (rCb[:,0].unsqueeze(0) - rCa[:,0].unsqueeze(0)).repeat((nat,1))
    V3[:,:,1] = (rCb[:,1].unsqueeze(0) - rCa[:,1].unsqueeze(0)).repeat((nat,1))
    V3[:,:,2] = (rCb[:,2].unsqueeze(0) - rCa[:,2].unsqueeze(0)).repeat((nat,1))

    OMEGA     = M*ang2plainMat(V1, V2, V2, V3)
    indnan = torch.isnan(OMEGA)
    OMEGA[indnan] = 0.0

    # Theta
    V1 = torch.zeros(nat, nat, 3)
    V2 = torch.zeros(nat, nat, 3)
    V3 = torch.zeros(nat, nat, 3)
    # N - Ca
    V1[:,:,0] = (rN[:,0].unsqueeze(1) - rCa[:,0].unsqueeze(1)).repeat((1,nat))
    V1[:,:,1] = (rN[:,1].unsqueeze(1) - rCa[:,1].unsqueeze(1)).repeat((1, nat))
    V1[:,:,2] = (rN[:,2].unsqueeze(1) - rCa[:,2].unsqueeze(1)).repeat((1, nat))
    # Ca - Cb # TODO - repeated computation
    V2[:,:,0] = (rCa[:,0].unsqueeze(1) - rCb[:,0].unsqueeze(1)).repeat((1,nat))
    V2[:,:,1] = (rCa[:,1].unsqueeze(1) - rCb[:,1].unsqueeze(1)).repeat((1, nat))
    V2[:,:,2] = (rCa[:,2].unsqueeze(1) - rCb[:,2].unsqueeze(1)).repeat((1, nat))
    # Cb1 - Cb2 # TODO - repeated computation
    V3[:,:,0] = rCb[:,0].unsqueeze(1) - rCb[:,0].unsqueeze(1).t()
    V3[:,:,1] = rCb[:,1].unsqueeze(1) - rCb[:,1].unsqueeze(1).t()
    V3[:,:,2] = rCb[:,2].unsqueeze(1) - rCb[:,2].unsqueeze(1).t()

    THETA = M*ang2plainMat(V1, V2, V2, V3)
    indnan = torch.isnan(THETA)
    THETA[indnan] = 0.0
    M[indnan]     = 0.0
    return D, OMEGA, PHI, THETA, M
