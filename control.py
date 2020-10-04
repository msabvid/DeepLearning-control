import torch
import torch.nn as nn
import torchdiffeq
from typing import Tuple
import tqdm
import argparse
import os
import numpy as np

from torchdiffeq import odeint_adjoint as odeint


from lib.networks import ResFNN


class Func_ODE_LQR(nn.Module):
    """ODE for the LQR problem
    Model: dX_t = (H(t) + M(t)*alpha_t)dt, X_t=x

    """

    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        self.alpha = ResFNN(input_dim+1, output_dim, hidden_dims) # +1 is for time
    
    def H(self, t):
        return 0
    
    def M(self, t):
        return 1

    def forward(self, t, x):
        ones = torch.ones(x.shape[0], 1, device=x.device)
        input_nn = torch.cat([ones*t, x], 1)
        return self.H(t) + self.M(t)*self.alpha(input_nn)


class LQR():
    """LQR problem
    Utility function: E[\int_t^T C(s)X_s^2 + D(s)alpha_s^2 ds + RX_T^2]
    """

    def __init__(self, func_ode, R):
        self.func_ode = func_ode
        self.R = R
        
    def C(self,t):
        return 0

    def D(self, t):
        return 0

    def running_cost(self, t_span, step_size, x):
        """Running cost of the LQR control problem
        \int_t^T C(s)X_s^2 + D(s)alpha_s^2 ds

        Parameters
        ----------
        x: torch.Tensor
            tensor of size (steps+1, batch_size, d)
        t_span: torch.Tensor
            tensor of lenght steps+1
        step_size: int

        Returns
        -------
        cost: torch.Tensor
            tensor of size (batch_)

        """
        cost = 0
        ones = torch.ones(x.shape[1], 1, device=x.device)
        for idx,t in enumerate(t_span):
            input_nn = torch.cat([ones*t, x[idx,:,:]], 1)
            cost += step_size*(self.C(t)*x[idx,:,:]**2 + self.D(t)*self.func_ode.alpha(input_nn)**2)
        return cost

    def final_cost(self, x):
        """Final cost of the LQR control problem
        R*X_s^2

        Parameters
        ----------
        x: torch.Tensor
            tensor of size (steps+1, batch_size, d)
        """
        
        return self.R * x[-1]**2


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.)



def train(args, device, t, step_size):
    set_seed(args.seed)
    # create model
    drift_lqr = Func_ODE_LQR(input_dim=args.d,
            output_dim=args.d,
            hidden_dims=args.hidden_dims)
    drift_lqr.to(device)
    drift_lqr.apply(init_weights)
    lqr = LQR(drift_lqr, args.R)
    optimizer = torch.optim.RMSprop(drift_lqr.parameters(), lr=0.001)
    # Train
    pbar = tqdm.tqdm(total=args.n_iter)
    for it in range(args.n_iter):
        optimizer.zero_grad()
        x0 = 2*torch.randn(args.batch_size, args.d, device=device)
        x = odeint(drift_lqr, x0, t)
        loss = lqr.running_cost(t, step_size, x) + lqr.final_cost(x)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        pbar.write("Loss={:4.2e}".format(loss.item()))
        if (it+1)%10==0:
            pbar.update()




if __name__=='__main__':

    parser = argparse.ArgumentParser()
    # general aguments for code to work
    parser.add_argument("--base_dir", default="./numerical_results",type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--use_cuda", action='store_true', default=True)
    parser.add_argument("--seed", default=0, type=int)
    # arguments for network architecture and for training
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--d", default=1, type=int)
    parser.add_argument("--hidden_dims", default=[20,20], nargs="+", type=int)
    parser.add_argument("--n_iter", default=500)
    # arguments for LQR problem set up
    parser.add_argument("--T", default=5, type=int, help="horizon time of control problem")
    parser.add_argument("--steps", default=50, type=int)
    parser.add_argument("--R", default=1., type=float, help="coefficient for final cost of control problem")
    
    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"

    assert args.d==1, "current implementation only works for one-dimensional process."
    
    # time discretisation
    t = torch.linspace(0, args.T, steps=args.steps+1)
    step_size = args.T/args.steps
    train(args, device, t, step_size)

