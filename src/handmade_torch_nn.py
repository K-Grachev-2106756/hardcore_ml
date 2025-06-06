import torch 
torch.manual_seed(42)


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
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            x_mean = x.mean(dim=dim, keepdim=True)
            x_var = x.var(dim=dim, keepdim=True)
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
    

class Embedding:

    def __init__(self, num_embeddings, embedding_dim, device=None):
        device = "cpu" if device is None else device
        self.weight = torch.randn((num_embeddings, embedding_dim), device=device)

    
    def __call__(self, x):
        self.out = self.weight[x]
        
        return self.out


    def parameters(self):
        return [self.weight]
    

class Flatten:

    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        
        return self.out


    def parameters(self):
        return []
    

class Sequential:

    def __init__(self, layers):
        self.layers = layers
    
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]