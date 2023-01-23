from torch import nn
import copy

class HetasksNet(nn.Module):
    '''mini cnn structure
    input -> (conv1d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''
    def __init__(self, input_dim, output_dim, num_fts, fts_per_num):
        super().__init__()
        c,num,fts = input_dim
        print("c,num,fts ", c,num,fts)

        if num != num_fts:
            raise ValueError(f"Number of type of features: {num_fts}, got: {num}")
        #if n != network_ft_size:
        #    raise ValueError(f"Expecting network features size: {network_ft_size}, got: {n}")
        if fts != fts_per_num:
            raise ValueError(f"Features size: {fts_per_num}, got: {fts}")


        self.online = nn.Sequential(
            nn.Conv1d(1,2,1,42),
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)
