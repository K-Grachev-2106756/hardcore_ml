import torch 


class Linear:
    
    def __init__(self, fan_in, fan_out, bias=True, device=None):
        device = "cpu" if device is None else device

        self.weight = torch.randn((fan_in, fan_out), device=device) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out, device=device) if bias else None

    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        
        return self.out  # В self, чтобы можно было отслеживать и визуализировать изменение значений
    

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None):
        device = "cpu" if device is None else device

        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim, device=device)
        self.beta = torch.zeros(dim, device=device)
        self.running_mean = torch.zeros(dim, device=device)
        self.running_var = torch.ones(dim, device=device)

    
    def __call__(self, x):
        if self.training:
            x_mean = x.mean(dim=0, keepdim=True)
            x_var = x.var(dim=0, keepdim=True)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var
        else:
            x_mean = self.running_mean
            x_var = self.running_var
        
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)
        self.out = self.gamma * x_hat + self.beta  
        
        return self.out
    

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:

    def __call__(self, x):
        self.out = torch.tanh(x)
        
        return self.out
    

    def parameters(self):
        return []