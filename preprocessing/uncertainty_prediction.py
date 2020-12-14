import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
from src.loss import MSELoss

N, D_in, H, D_out = 1000, 2, 100, 2

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

lr_base = 1e-4
optimizer = torch.optim.Adam(list(model.parameters()), lr=lr_base)

maxiter = 50000
for t in range(maxiter):
    lr = lr_base * (1-t/maxiter)
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    y = y / 10
    y = y + 5
    target = y[:, 0]


    yy = model(x)
    y_pred = yy[:,0]
    y_sigma = torch.abs(yy[:,1])+0.01

    diff = (y_pred - target)
    loss_dist = torch.mean(diff**2/(2*y_sigma**2))

    loss_std = torch.mean(torch.log(y_sigma**2)/2)
    loss = loss_dist + loss_std
    if t % 100 == 99:
        print("{:}, Loss {:2.2f}, Loss_std {:2.2f}, y_pred {:2.4f}, y_pred_std {:2.4f} target {:2.4f} target_std {:2.4f} lr {:2.4e}".format(t, loss.item(), loss_std.item(), torch.mean(y_pred), torch.mean(y_sigma), torch.mean(target), torch.std(target), lr))

    model.zero_grad()

    loss.backward()
    optimizer.step()

    # with torch.no_grad():
    #     for param in model.parameters():
    #         param -= lr * param.grad