import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> torch.Tensor:
    r"""
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def get_channel_mask(hidden_size_choices):
    max_hidden_size = max(hidden_size_choices)
    num_choices = len(hidden_size_choices)
    masks = torch.zeros(max_hidden_size, num_choices)
    for i in range(num_choices):
        masks[:hidden_size_choices[i], i]=1
    return masks
    
def get_flops_choices(input_size, hidden_size_choices, out_dim):
    flops = []
    for hidden_size in hidden_size_choices:
        flops.append(2*hidden_size*input_size + 2*hidden_size*out_dim)
    flops = np.array(flops)
    return flops
    
class SuperNet(nn.Module):
    def __init__(self, input_size, hidden_size_choices, out_dim):
        super(SuperNet, self).__init__()
        
        max_hidden_size = max(hidden_size_choices)
        num_choices = len(hidden_size_choices)
        
        self.arch_params = torch.nn.Parameter(torch.ones(num_choices), requires_grad=True)
        self.masks = get_channel_mask(hidden_size_choices)
        self.flops_choices = get_flops_choices(input_size, hidden_size_choices, out_dim)
        self.flops_choices_normalized = torch.FloatTensor(self.flops_choices / np.max(self.flops_choices))
        
        self.fc1 = nn.Linear(input_size, max_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(max_hidden_size, out_dim)
        
    def forward(self, x, temperature):
        out = self.fc1(x)
        out = self.relu(out)
        # print(self.arch_params)
        gumbel_weights = gumbel_softmax(self.arch_params, tau=temperature, hard=False)
        # print(gumbel_weights)
        mask = torch.multiply(self.masks, gumbel_weights)
        mask = torch.sum(mask, dim=-1)
        out = torch.multiply(out, mask)
        
        out = self.fc2(out)
        flops_loss = self._get_flops_loss(gumbel_weights)
        return out, flops_loss
    
    def _get_flops_loss(self, gumbel_weights):
        return torch.matmul(self.flops_choices_normalized, gumbel_weights)

# Initialize the neural network
def search(x_train, y_train, out_dim, Loss_type='CE', hidden_size_choices = list(range(100,1000,10)), flops_balance_factor = 0.2, net_weight_lr = 0.0001, arch_lr = 0.01, num_epochs = 5000, search_freq = 20):
    # out_dim: The ouput dimension of the network. It should be the number of classes for 'CE' loss
    # Loss_type: should be either 'CE' or 'MSE'
    # hidden_size_choices: The choices for the number of hidden units.
    # flops_balance_factor: The factor to balance performance (CE Loss or MSE Loss) and FLOPs. 
    #                       Larger flops_balance_factor leads to a faster network with worse performance
    # net_weight_lr: The learning rate for optimizing network weights 
    # arch_lr: The learning rate for optimizing architecture parameters 
    # num_epochs: Search Epochs
    # search_freq: The algorithm teartively optimize architecture parameters and network weights every 'search_freq' epochs
    
    # The tempreature is decayed by 'temp_anneal_factor' every 'temp_anneal_freq' epochs
    # Larger temp leads to a gumbel weight that is more close to 1-hot distribution  
    temp = 5
    temp_anneal_factor = 0.95
    temp_decay_count = 25 # The temperatur will decay 'temp_decay_count' times in all during the search
    temp_anneal_freq = num_epochs / temp_decay_count # The temperatur will decay every 'temp_anneal_freq' epochs

    in_dim = x_train.shape[1]

    net = SuperNet(in_dim, hidden_size_choices, out_dim)

    # Set up the loss function and optimizer
    if Loss_type == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif Loss_type == 'MSE':
        criterion = nn.MSELoss()

    optimizer_net = optim.Adam([p for name, p in net.named_parameters() if 'arch' not in name], lr=net_weight_lr)
    optimizer_arch = optim.Adam([p for name, p in net.named_parameters() if 'arch' in name], lr=arch_lr)


    for epoch in range(num_epochs):
        if epoch % temp_anneal_freq == 0:
            temp = temp * temp_anneal_factor
        
        # Forward pass
        outputs, flops_loss = net(x_train, temp)
        CE_loss = criterion(outputs, y_train)
        loss = (1 - flops_balance_factor) * CE_loss + flops_balance_factor * flops_loss
        
        if int(epoch/search_freq)%2==0:
            optimizer_net.zero_grad()
            loss.backward()
            optimizer_net.step()
        else:
            optimizer_arch.zero_grad()
            loss.backward()
            optimizer_arch.step()
        
        selected_channel_id = np.argmax(net.arch_params.data.numpy())
        selected_channel = hidden_size_choices[selected_channel_id]

        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            total = y_train.size(0)
            correct = (predicted == y_train).sum().item()
            accuracy = correct / total
            if (epoch+1) % 10 == 0:
                
                print('Epoch [{}/{}], Overall_Loss: {:.4f}, CE_Loss: {:.4f}, Flops_Loss: {:.4f}, Accuracy: {:.2f}%, Seleted Channel {}'
                    .format(epoch+1, num_epochs, loss.item(), CE_loss.item(),
                            flops_loss.item(), accuracy * 100, selected_channel))
    
    return selected_channel