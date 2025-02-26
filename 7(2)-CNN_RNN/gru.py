import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # 구현하세요!
        # Update gate parameters
        self.W_z = nn.Linear(input_size, hidden_size, bias=False)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Reset gate parameters
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Candidate hidden state parameters
        self.W_h = nn.Linear(input_size, hidden_size, bias=False)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # 구현하세요!
        z_t = torch.sigmoid(self.W_z(x) + self.U_z(h))  # Update gate
        r_t = torch.sigmoid(self.W_r(x) + self.U_r(h))  # Reset gate
        
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r_t * h))  # Candidate hidden state
        h_next = (1 - z_t) * h + z_t * h_tilde  # Compute new hidden state
        return h_next
    pass


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        # 구현하세요!

    def forward(self, inputs: Tensor) -> Tensor:
        # 구현하세요!
        batch_size, seq_len, _ = inputs.shape
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        

        hidden_states = []
        for t in range(seq_len):
          h = self.cell(inputs[:, t, :], h)
          hidden_states.append(h)

        return torch.stack(hidden_states, dim=1) 
    pass