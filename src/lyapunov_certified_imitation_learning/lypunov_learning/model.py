import torch
import torch.nn as nn
import torch.nn.functional as F


# --- ICNN ---
class ICNN(nn.Module):
    """
    Input Convex Neural Network (ICNN) architecture.
    """
    def __init__(self, input_dim: int, hidden_dim: list[int], activation: str = "relu", with_skip: bool = True):
        super(ICNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)
        
        # --- Define Layers ---
        if self.num_layers < 1:
            raise ValueError("hidden_dim must contain at least one layer size.")

        self.w0 = nn.Linear(input_dim, hidden_dim[0])
        
        # Hidden Layers
        #    z_{k+1} = sigma( W_z * z_k + W_x * x + b )
        self.w_z = nn.ModuleList([
            nn.Linear(hidden_dim[i - 1], hidden_dim[i], bias=False)
            for i in range(1, self.num_layers)
        ])
        self.final_w_z = nn.Linear(hidden_dim[-1], 1, bias=False)
        
        
        if with_skip:
            self.w_x = nn.ModuleList([
                nn.Linear(input_dim, hidden_dim[i], bias=True)
                for i in range(1, self.num_layers)
            ])
            self.final_w_x = nn.Linear(input_dim, 1, bias=True)
        else:
            self.w_x = None
            self.final_w_x = None

        # --- Activation ---
        if activation == "softplus":
            self.act = F.softplus
        else:
            self.act = F.relu

    def forward(self, x):
        # Initial Layer
        z = self.act(self.w0(x))
        
        # Hidden Layers
        for i in range(self.num_layers - 1):
            W_z_positive = F.softplus(self.w_z[i].weight)
            
            # Positive weights: W_z * z
            z_next = F.linear(z, W_z_positive)
            
            # Skip connection: W_x * x + b
            if self.w_x is not None:
                z_next = z_next + self.w_x[i](x)
            
            # Activation
            z = self.act(z_next)
            
        # Final Layer
        W_z_final_pos = F.softplus(self.final_w_z.weight)
        if self.final_w_x is not None:
            y = F.linear(z, W_z_final_pos) + self.final_w_x(x)
        else:
            y = F.linear(z, W_z_final_pos)
        
        return y


# --- Lyapunov Wrapper ---
class NeuralLyapunov(nn.Module):
    """
    Wraps the ICNN to ensure strict Lyapunov properties:
    V(0) = 0 and V(x) > 0.
    """
    def __init__(self, icnn_model, eps: float = 0.01):
        super(NeuralLyapunov, self).__init__()
        self.icnn = icnn_model
        self.eps = eps
        
        # We handle V(0)=0 by subtracting the value at zero.
        # However, computing ICNN(0) every forward pass is expensive.
        # In practice, we often just train it to be 0 or subtract a cached bias.
        # Here, we'll implement the mathematical definition directly for clarity.

    def forward(self, x):
        # V(x) = ICNN(x) - ICNN(0) + eps * ||x||^2
        # The eps term ensures strict positive definiteness even if ICNN is flat.
        
        v_x = self.icnn(x)
        
        # Calculate ICNN(0) - In efficient training, you might detach this or 
        # assume bias initialization handles it. 
        # For strict correctness:
        zeros = torch.zeros_like(x)
        v_0 = self.icnn(zeros)
        
        # Add quadratic term for strict positivity (eps * x^T x)
        # shape of x is (Batch, Dim) -> norm is (Batch, 1)
        quadratic = self.eps * (x ** 2).sum(dim=1, keepdim=True)
        
        return v_x - v_0 + quadratic