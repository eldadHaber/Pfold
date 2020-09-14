

def misfitFun(Ypred, Yobs, Active=torch.tensor([1]), dweights = torch.tensor([1,1,1,1.0])):
    n = Yobs.shape
    dweights = dweights.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    R = torch.zeros(n)
    W = 1.0/(Yobs + Yobs.mean()/100)
    R[0, 0, :, :] = ((Ypred[0, 0, :, :] + Ypred[0, 0, :, :].t())/2 - Yobs[0, 0, :, :])
    R[0, 1, :, :] = ((Ypred[0, 1, :, :] + Ypred[0, 1, :, :].t())/2 - Yobs[0, 1, :, :])
    R[0, 2, :, :] = ((Ypred[0, 2, :, :] + Ypred[0, 2, :, :].t())/2 - Yobs[0, 2, :, :])
    R[0, 3, :, :] = ((Ypred[0, 3, :, :] + Ypred[0, 3, :, :].t())/2 -  Yobs[0, 3, :, :])
    R             = Active*(dweights*W*R)
    loss  = 0.5*torch.norm(R)**2
    loss0 = 0.5*torch.norm(Active*(dweights*W*Yobs))**2
    return loss/loss0


##### Hamiltonian Networks

def inithNetParams(A,device='cpu'):
    # A = [ inChan, OutChan, number of layers]
    print('Initializing network  ')
    stdv = 1e-2
    Wopen = torch.zeros(A[0,0], A[0,1])
    Wopen.data.uniform_(-stdv, stdv)
    Wopen  = nn.Parameter(Wopen)
    Wclose = torch.zeros(A[2,0], A[2,1])
    Wclose.data.uniform_(-stdv, stdv)
    Wclose = nn.Parameter(Wclose)

    K    = torch.zeros(A[1,0], A[1,1], A[1,2])
    stdv = 1e-3 * A[1, 0]/A[1, 1]
    K.data.uniform_(-stdv, stdv)
    K = nn.Parameter(K)
    npar = K.numel() + Wopen.numel() + Wclose.numel()
    K.to(device)

    print('Number of parameters  ', npar)
    return K, Wopen, Wclose


def chnorm(F, eps=1e-3):
    F = F - torch.mean(F,dim=1,keepdim=True)
    F = F/(torch.sqrt(torch.sum(F**2,dim=1) + eps).unsqueeze(1))
    return F

class hNet(nn.Module):
        # Solve the ODE
        # x_{j+1} = x_j + relu(K_j*z_j) + relu(W*Forceing_j)
        # z_{j+1} = z_j - relu(K_j'*x_{j+1})
        def __init__(self, K, Wopen, Wclose, h):
            super(hNet, self).__init__()
            self.K      = K
            self.Wopen  = Wopen
            self.Wclose = Wclose
            self.h     = h

        def forward(self, Frc):
            Nseq     = Frc.shape[2]# length of sequence
            NfeatIn  = self.K.shape[1]
            NfeatHid = self.K.shape[0]
            NHidLyr  = self.K.shape[2]

            Nbatch   = Frc.shape[0]

            # allocate space for output sequence
            Y   = torch.zeros(Nbatch,self.Wclose.shape[0], Nseq)
            x   = torch.zeros(Nbatch,NfeatIn)
            z   = torch.zeros(Nbatch,NfeatHid)
            for i in range(Nseq):
                Fi  = Frc[:, :, i]
                if i+1<Nseq:
                    Fip = Frc[:, :, i+1]
                else:
                    Fip = 0.0
                for j in range(NHidLyr):
                    Fi = ((NHidLyr-j)*Fi + j*Fip)/NHidLyr
                    #if j==0:
                    fi = torch.relu(chnorm(F.linear(Fi, self.Wopen)))
                    #else:
                    #    fi = 0.0

                    aj = torch.relu(chnorm(F.linear(x, self.K[:,:,j])))
                    z  = z + self.h*(aj + fi)
                    qi = torch.relu(chnorm(F.linear(z, self.K[:,:,j].t())))
                    x  = x - self.h*qi

                Y[:,:,i] = F.linear(z, self.Wclose)

            return Y

