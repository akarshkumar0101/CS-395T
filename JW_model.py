class JWModel0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        d_in, d_out= X_train.shape[-1], Y_train.shape[-1]
        self.linear = torch.nn.Linear(d_in, d_out)
        self.deep1 = torch.nn.Sequential(torch.nn.Linear(d_in, 1000),
                                         torch.nn.Sigmoid(),
                                         torch.nn.Linear(1000, 1000),
                                         torch.nn.Sigmoid(),
                                         torch.nn.Linear(1000, 1000),
                                         torch.nn.Sigmoid(),
                                         torch.nn.Linear(1000, d_out))
        # self.deep1 = torch.nn.Sequential(torch.nn.Linear(d_in, 1000),
        #                                  torch.nn.Linear(1000, d_out))
        
        self.deep2 = torch.nn.Sequential(torch.nn.Linear(d_in, 1000),
                                         torch.nn.Sigmoid(),
                                         torch.nn.Linear(1000, d_out))
        
    def forward(self, X):
#         Y = self.linear(X) + self.deep1(X)
        Y = self.deep1(X) + self.deep2(X)
        return Y

    def save_model(self, model_name="JW_model0"):
        from torch import save
        from os import path
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), model_name +'.th'))


    # def load_model():
    #     from torch import load
    #     from os import path
    #     r = Detector()
    #     r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    #     return r
