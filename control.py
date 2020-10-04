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
    Model: dX_t = (H*X_t + M*alpha_t)dt, X_t=x

    """

    def __init__(self, input_dim, output_dim, hidden_dims, H, M):
        super().__init__()
        self.alpha = ResFNN(input_dim+1, output_dim, hidden_dims) # +1 is for time
        self.H = H #(d,d)
        self.M = M #(d,d)

    def forward(self, t, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            tensor of size (batch_size, d)

        """
        ones = torch.ones(x.shape[0], 1, device=x.device)
        input_nn = torch.cat([ones*t, x], 1)
        output = torch.matmul(self.H, x.unsqueeze(2)) + torch.matmul(self.M, self.alpha(input_nn).unsqueeze(2)) # batch dimensions are broadcasted correctly
        return output.squeeze(2)


class LQR():
    """LQR problem
    Utility function: E[\int_t^T C(s)X_s^2 + D(s)alpha_s^2 ds + RX_T^2]
    """

    def __init__(self, func_ode, R, C, D):
        self.func_ode = func_ode
        self.R = R
        self.C = C #(d,d)
        self.D = D #(d,d)
        

    def x_M_x(self, x, M):
        """
        Calculates the quadratic product x^T * M * x

        Parameters
        ----------
        x: torch.Tensor
            tensor of size (batch_size, d)
        M: torch.Tensor
            tensor of size (d,d)
        """
        M_x = torch.matmul(M, x.unsqueeze(2)) # (batch_size, d, 1)
        x_M_x = torch.bmm(x.unsqueeze(1),M_x) # (batch_size, 1, 1)
        return x_M_x.squeeze(2)
    
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
            X_C_X = self.x_M_x(x[idx,:,:], self.C)
            alpha_t = self.func_ode.alpha(input_nn)
            alpha_D_alpha = self.x_M_x(alpha_t, self.D)
            cost += step_size*(X_C_X + alpha_D_alpha)
        return cost

    def final_cost(self, x):
        """Final cost of the LQR control problem
        R*X_s^2

        Parameters
        ----------
        x: torch.Tensor
            tensor of size (steps+1, batch_size, d)
        """
        return self.x_M_x(x[-1,:,:],self.R) 


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.)



def train(args, device, t, step_size, H, M, C, D, R):
    set_seed(args.seed)
    # create model
    drift_lqr = Func_ODE_LQR(input_dim=args.d,
            output_dim=args.d,
            hidden_dims=args.hidden_dims,
            H=H, M=M)
    drift_lqr.to(device)
    drift_lqr.apply(init_weights)
    lqr = LQR(drift_lqr, 
            R=R,
            C=C,
            D=D)
    optimizer = torch.optim.RMSprop(drift_lqr.parameters(), lr=0.001)
    # Train
    pbar = tqdm.tqdm(total=args.n_iter)
    for it in range(args.n_iter):
        optimizer.zero_grad()
        x0 = 2*torch.randn(args.batch_size, args.d, device=device)
        x = odeint(drift_lqr, x0, t,
                method="euler",
                options=dict(step_size=0.05))
        loss = lqr.running_cost(t, step_size, x) + lqr.final_cost(x)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        pbar.write("Loss={:4.2e}".format(loss.item()))
        if (it+1)%10==0:
            pbar.update(10)

    # saving the model
    if not os.path.exists(args.base_dir):
        os.path.makedirs(args.base_dir)
    torch.save(drift_lqr.state_dict(), os.path.join(args.base_dir, "policy_lqr.pt"))

    pbar.write("Training ended")


def visualize(args, device, t, step_size, H, M, C, D, R):

    drift_lqr = Func_ODE_LQR(input_dim=args.d,
            output_dim=args.d,
            hidden_dims=args.hidden_dims,
            H=H, M=M)
    drift_lqr.to(device)
    drift_lqr.apply(init_weights)
    lqr = LQR(drift_lqr, 
            R=R,
            C=C,
            D=D)

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
    
    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"

    #assert args.d==1, "current implementation only works for one-dimensional process."
    
    # time discretisation
    t = torch.linspace(0, args.T, steps=args.steps+1).to(device)
    step_size = args.T/args.steps
    H = torch.zeros(args.d, args.d).to(device)
    M = torch.eye(args.d).to(device)
    C = torch.zeros(args.d, args.d).to(device)
    D = torch.zeros(args.d, args.d).to(device)
    R = torch.eye(args.d).to(device)
    train(args, device=device, t=t, step_size=step_size,
            H=H,M=M,C=C,D=D,R=R)

