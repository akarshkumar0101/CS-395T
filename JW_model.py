import torch
class JWModel0(torch.nn.Module):
    def __init__(self, d_in, d_out, V):
        super().__init__()
        # d_in, d_out= X_train.shape[-1], Y_train.shape[-1]
        # self.linear = torch.nn.Linear(d_in, d_out)
        # self.deep1 = torch.nn.Sequential(torch.nn.Linear(d_in, 1000),
        #                                  torch.nn.Sigmoid(),
        #                                  torch.nn.Linear(1000, 1000),
        #                                  torch.nn.Sigmoid(),
        #                                  torch.nn.Linear(1000, 1000),
        #                                  torch.nn.Sigmoid(),
        #                                  torch.nn.Linear(1000, d_out))
        # self.deep1 = torch.nn.Sequential(torch.nn.Linear(d_in, 1000),
        #                                  torch.nn.Linear(1000, d_out))
        
        # self.deep2 = torch.nn.Sequential(torch.nn.Linear(d_in, 1000),
        #                                  torch.nn.Sigmoid(),
        #                                  torch.nn.Linear(1000, d_out))
        self.V = V
        self.Linear1 = torch.nn.Linear(d_in, 1000)
        # self.Linear2 = torch.nn.Linear(d_in, 100)
        self.Linear3 = torch.nn.Linear(d_in, 1000)
        self.re = torch.nn.ReLU()
        self.maxPool = torch.nn.MaxPool1d(2)
        self.FinLin = torch.nn.Linear(1000, d_out)
        self.dropout = torch.nn.Dropout(0.25)
        
    def forward(self, X):
        # print(X.shape)
#         Y = self.linear(X) + self.deep1(X)
        # Y = self.deep1(X) + self.deep2(X)
        # print(self.V[:, :6].shape)
        # Y = torch.matmul(X, self.V)
        # print(self.Linear1(X).shape)
        Y = torch.cat((
          self.dropout(self.Linear1(X).unsqueeze(2)), 
          # self.Linear2(X).unsqueeze(2), 
          self.dropout(self.Linear3(X).unsqueeze(2))), 2) 
        Y = self.maxPool(Y)
        # print(Y.shape)
        Y = self.FinLin(Y.squeeze())
        return Y

    def save_model(self, model_name="JW_model0"):
        from torch import save
        from os import path
        return save(self.state_dict(), path.join(path.dirname(path.abspath(__file__)), model_name +'.th'))


    # def load_model():
    #     from torch import load
    #     from os import path
    #     r = Detector()
    #     r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    #     return r
    