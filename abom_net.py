import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from datetime import datetime

class mutation_operator(nn.Module):
    """
    Mutation operator using self-attention mechanism.
    """
    def __init__(self, hidden_num: int = 256) -> None:
        super().__init__()
        self.hidden_num = hidden_num    
        self.W_q_x = nn.Linear(1, self.hidden_num, bias=False)
        self.W_k_x = nn.Linear(1, self.hidden_num, bias=False)

    def forward(self, x: Tensor, f: Tensor = None, rng: torch.Generator = None, generation_counter: int = None) -> Tensor:
        """
        Forward pass for mutation.

        Args:
            x (Tensor): Population tensor, shape (batch_size, pop_size, input_problem_dim).
            f (Tensor, optional): Fitness tensor. Not used in this module.

        Returns:
            Tensor: Mutated population, shape (batch_size, pop_size, input_problem_dim).
        """
        
        x = x.unsqueeze(-1) # (batch_size, pop_size, input_problem_dim, 1)
        x_old = x.clone().view(x.size(0)*x.size(1), x.size(2), 1) 
        Q_x = self.W_q_x(x) # (batch_size, pop_size, input_problem_dim, hidden_num)
        K_x = self.W_k_x(x) # (batch_size, pop_size, input_problem_dim, hidden_num)

        K_x_T = K_x.permute(0, 1, 3, 2) # Transpose for matrix multiplication (batch_size, pop_size, hidden_num, input_problem_dim)
        
        # Reshape for batched matrix operations
        original_shape = Q_x.size() # (batch_size, pop_size, input_problem_dim, hidden_num)
        batch_pop = original_shape[0] * original_shape[1] # (batch_size * pop_size)
        Q_x = Q_x.view(batch_pop, original_shape[2], original_shape[3]) # (batch_pop, input_problem_dim, hidden_num)
        K_x_T = K_x_T.view(batch_pop, original_shape[3], original_shape[2]) # (batch_pop, hidden_num, input_problem_dim)
        attn = torch.softmax(torch.bmm(Q_x, K_x_T) / np.sqrt(self.hidden_num), dim=-1) # (batch_pop, input_problem_dim, input_problem_dim)
        # if generation_counter == 1 or generation_counter == 499 or generation_counter == 998:
        #     now = datetime.now()
        #     time_str = now.strftime("%H:%M:%S")  # %H:%M:%S %H:%M:%S.%f
        #     file_name = 'vis/attention_mut/step_'+str(generation_counter)+'_'+str(time_str)+'.pth'
        #     torch.save(attn,file_name)
        output = torch.bmm(attn, x_old) # (batch_pop, input_problem_dim, 1)
        output = output.view(original_shape[0], original_shape[1], original_shape[2], 1)
        output = output.squeeze(-1) # (batch_size, pop_size, input_problem_dim)
        return output      

class crossover_operator(nn.Module):
    """
    crossover operator.
    """
    def __init__(self, d_model: int, 
                 input_problem_dim: int) -> None:
        super().__init__()
        # Linear projections for Q, K (search space and fitness)
        self.d_model = d_model
        self.linears = nn.ModuleList([
            nn.Linear(input_problem_dim, d_model), # Q_x
            nn.Linear(input_problem_dim, d_model), # K_x
            nn.Linear(1, d_model), # Q_f
            nn.Linear(1, d_model), # K_f
        ])

    def forward(self, x: Tensor, f: Tensor, rng: torch.Generator, generation_counter: int) -> Tensor:
        """
        Forward pass for multi-head crossover.

        Args:
            x (Tensor): Parent population, shape (batch_size, pop_size, input_problem_dim).
            f (Tensor): Parent fitness, shape (batch_size, pop_size, input_problem_num).

        Returns:
            Tensor: Offspring population, shape (batch_size, pop_size, d_model).
        """
        x_old = x.clone()
        batch_size = x.size(0)
        # Calculate attention and apply
        query_x, key_x, query_f, key_f = [linear(z) for linear, z in zip(self.linears, (x, x, f, f))]
        scores_x = torch.bmm(query_x, key_x.transpose(-2, -1)) / np.sqrt(self.d_model)
        scores_f = torch.bmm(query_f, key_f.transpose(-2, -1)) / np.sqrt(self.d_model)
        attn = scores_x + scores_f
        attn = torch.softmax(attn, dim=-1) # (batch,pop, input_problem_dim, input_problem_dim)
        # if generation_counter == 1 or generation_counter == 499 or generation_counter == 998:
        #     now = datetime.now()
        #     time_str = now.strftime("%H:%M:%S")  # %H:%M:%S %H:%M:%S.%f
        #     file_name = 'vis/attention_cross/step_'+str(generation_counter)+'_'+str(time_str)+'.pth'
        #     torch.save(attn,file_name)
        output = torch.bmm(attn, x_old) # (batch,pop, input_problem_dim)
        return output  

class FFN(nn.Module):
    """Standard Feed-Forward Network."""
    def __init__(self, input_problem_dim: int, d_model: int, hidden_num: int = 2048) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_problem_dim, hidden_num, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_num, input_problem_dim, bias=False),
        )
        self.linear1 = nn.Linear(input_problem_dim, hidden_num, bias=True)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(hidden_num, input_problem_dim, bias=True)
    def forward(self, x: Tensor, f: Tensor = None) -> Tensor:
        return self.linear(x)

class Residual(nn.Module):
    """
    Residual connection wrapper for crossover or mutation modules.
    Includes learnable residual parameter and FFN.
    """
    def __init__(self, sublayer: nn.Module, input_problem_dim: int, d_model: int, 
                 dim_mutation: int, dropout: float = 0.1, flag: bool = True, rng: torch.Generator = None):
        super().__init__()
        self.sublayer = sublayer  
        self.rng = rng
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(input_problem_dim, d_model, dim_mutation)
        self.flag = flag 

    def forward(self, x: Tensor, f: Tensor, x_old: Tensor, generation_counter: int) -> Tensor:
        batch_size, pop_size, dim = x.size()
        if self.flag: # Crossover path
            sublayer_output = self.sublayer(x, f, self.rng, generation_counter)
            ffn_output = self.ffn(sublayer_output)
            x = x + (self.dropout(ffn_output))

        else: # Mutation path
            sublayer_output = self.sublayer(x_old, f, self.rng, generation_counter)
            ffn_output = self.ffn(sublayer_output)
            x = x + (self.dropout(ffn_output))
        return x

def var_swap(x: Tensor, device: str, rng: torch.Generator) -> Tensor:
    """
    Perform variable-wise swap between pairs of individuals.

    Args:
        x (Tensor): Population tensor, shape (batch_size, pop_size, input_problem_dim).

    Returns:
        Tensor: Population after variable swap, shape (batch_size, pop_size, input_problem_dim).
    """
    half_pop_size = x.size(1) // 2  
    first_half = x[:, :half_pop_size, :]  
    second_half = x[:, half_pop_size:, :]

    mask_tmp = torch.rand(x.size(0), half_pop_size, x.size(2), device=x.device, generator=rng)
    mask = (mask_tmp > 0.5).float() # Mask for swapping

    first_half_swapped = first_half * (1 - mask) + second_half * mask  
    second_half_swapped = second_half * (1 - mask) + first_half * mask

    return torch.cat([first_half_swapped, second_half_swapped], dim=1)

class ABOM_block(nn.Module):
    """
    A single block of the ABOM evolutionary process: Crossover -> Mutation -> Variable Swap.
    """
    def __init__(self, rng: torch.Generator, device: str = 'cpu', input_problem_dim: int = 512,
                 d_model: int = 512, dim_mutation: int = 2048,
                 dropout_c: float = 0.8, dropout_m: float = 0.5):
        super().__init__()
        self.device = device
        self.rng = rng#crossover_operator
        
        self.crossover_module = Residual(
            sublayer=crossover_operator(d_model, input_problem_dim),
            input_problem_dim=input_problem_dim, d_model=d_model, dim_mutation=dim_mutation, dropout=dropout_c, flag=True, rng=rng)
        
        self.mutation_module = Residual(
            sublayer=mutation_operator(hidden_num=d_model),
            input_problem_dim=input_problem_dim, d_model=d_model, dim_mutation=dim_mutation, dropout=dropout_m, flag=False, rng=rng)

    def forward(self, x_father: Tensor, f_father: Tensor, generation_counter: int) -> Tensor:
        """
        Forward pass for one ABOM block.

        Args:
            x_father (Tensor): Parent population, shape (batch_size, pop_size, input_problem_dim).
            f_father (Tensor): Parent fitness, shape (batch_size, pop_size, input_problem_num).

        Returns:
            Tensor: Offspring population, shape (batch_size, pop_size, input_problem_dim).
        """
        x = self.crossover_module(x_father, f_father, x_father, generation_counter) # crossover
        x = self.mutation_module(x, f_father, x_father, generation_counter)              # Mutation
        x = var_swap(x, self.device, self.rng)                       # Variable Swap
        return x

class ABOM_net(nn.Module):
    """
    The main ABOM evolutionary model, stacking multiple ABOM blocks.
    """
    def __init__(self, num_layers, d_model, dim_mutation, dropout_c, dropout_m, dim, device, rng):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.dim_mutation = dim_mutation
        self.dropout_c = dropout_c
        self.dropout_m = dropout_m
        self.input_problem_dim = dim
        self.device = device
            
        # Stack ABOM blocks
        self.layers = nn.ModuleList([
            ABOM_block(rng,self.device,self.input_problem_dim,self.d_model, self.dim_mutation, self.dropout_c, self.dropout_m)for _ in range(self.num_layers)])

    def forward(self, x: Tensor, f: Tensor, generation_counter: int) -> Tensor:
        """
        Forward pass through all ABOM blocks.

        Args:
            x (Tensor): Initial population, shape (batch_size, pop_size, input_problem_dim).
            f (Tensor): Initial fitness, shape (batch_size, pop_size, input_problem_num).

        Returns:
            Tensor: Final evolved population, shape (batch_size, pop_size, input_problem_dim).
        """
        for layer in self.layers:
            x = layer(x, f, generation_counter)
        return x
