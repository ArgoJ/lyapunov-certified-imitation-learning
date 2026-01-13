import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ICNN(nn.Module):
    """
    Input Convex Neural Network (ICNN).
    
    Implementation ensures that the network output is convex with respect to the input.
    Based on: Amos et al., "Input Convex Neural Networks", ICML 2017.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1, activation=nn.Softplus()):
        super(ICNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        
        # w_xs: Linear layers for input x injection (W^x)
        # w_zs: Linear layers for hidden state propagation (W^z, must be non-negative)
        self.w_xs = nn.ModuleList()
        self.w_zs = nn.ModuleList()
        
        # First layer: z_0 = activation(W_0^x x + b_0)
        if not hidden_dims:
            raise ValueError("hidden_dims must strictly contain at least one hidden dimension.")

        self.w_xs.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Subsequent layers
        for i in range(len(hidden_dims) - 1):
            # Transformation from previous hidden layer z_i to z_{i+1}
            # Weights will be effectively non-negative via softplus in forward()
            self.w_zs.append(nn.Linear(hidden_dims[i], hidden_dims[i+1], bias=False))
            
            # Injection from input x to z_{i+1}
            self.w_xs.append(nn.Linear(input_dim, hidden_dims[i+1]))
            
        # Output layer
        self.final_w_z = nn.Linear(hidden_dims[-1], output_dim, bias=False)
        self.final_w_x = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First layer
        z = self.activation(self.w_xs[0](x))
        
        # Hidden layers
        for i in range(len(self.w_zs)):
            # Enforce non-negativity on recurrent weights W^z
            u_weight = F.softplus(self.w_zs[i].weight)
            
            # z_{i+1} = activation( z_i @ W^{z,T}_{i} + x @ W^{x,T}_{i+1} + b_{i+1} )
            # nn.Linear(x) computes x @ W^T + b.
            # F.linear(z, weight) computes z @ weight^T.
            z_term = F.linear(z, u_weight)
            x_term = self.w_xs[i+1](x) # Includes bias
            
            z = self.activation(z_term + x_term)
            
        # Final layer
        u_weight_final = F.softplus(self.final_w_z.weight)
        z_term_final = F.linear(z, u_weight_final)
        x_term_final = self.final_w_x(x) # Includes bias
        
        out = z_term_final + x_term_final
        return out
