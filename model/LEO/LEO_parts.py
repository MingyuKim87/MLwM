import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LEO_network(nn.Module):
    """docstring for Encoder"""
    def __init__(self, embedding_size, hidden_size, dropout_rate, layer_count=[1,3,1], is_deterministic=False):
        super(LEO_network, self).__init__()
        
        # Dimnesion
        self.embed_size = embedding_size
        self.hidden_size = hidden_size

        # Properties
        self.is_deterministic = is_deterministic

        # Hyper parameters
        self.dropout = nn.Dropout(p=dropout_rate)

        # Layer count
        self.layer_count = layer_count
        
        # Networks
        '''
        self.encoder = nn.Linear(self.embed_size, self.hidden_size, bias = False)
        self.relation_net = nn.Sequential(
                                 nn.Linear(2*self.hidden_size, 2*self.hidden_size, bias = False),
                                 nn.ReLU(),
                                 nn.Linear(2*self.hidden_size, 2*self.hidden_size, bias = False),
                                 nn.ReLU(),
                                 nn.Linear(2*self.hidden_size, 2*self.hidden_size, bias = False),
                                 nn.ReLU()
                                 )
        self.decoder = nn.Linear(self.hidden_size, 2*self.embed_size, bias = False)
        '''
        
        
        # Network
        self.encoder = self._fc_layer(self.layer_count[0], self.embed_size, self.hidden_size)
        self.relation_net = self._fc_layer(self.layer_count[1], 2*self.hidden_size, 2*self.hidden_size, 2*self.hidden_size, True)
        self.decoder = self._fc_layer(self.layer_count[2], self.hidden_size, 2*self.embed_size)
        

    def _fc_layer(self, layer_count, input_size, output_size, hidden_size=None, is_last_layer_activated=False, bias=False):
        modules = []

        if hidden_size == None:
            hidden_size = self.hidden_size
        
        
        for i in range(layer_count):
            if layer_count == 1:
                layer = nn.Linear(input_size, output_size, bias=bias)
                modules.append(layer)
            else:
                if i == layer_count -1 and is_last_layer_activated:
                    layer = nn.Linear(hidden_size, output_size, bias=bias)
                    activation = nn.ReLU()
                    modules.append(layer)
                    modules.append(activation)

                elif i == layer_count -1 and not is_last_layer_activated:
                    layer = nn.Linear(hidden_size, output_size, bias=bias)
                    modules.append(layer)
                    
                elif i == 0:
                    layer = nn.Linear(input_size, hidden_size, bias=bias)
                    activation = nn.ReLU()

                    modules.append(layer)
                    modules.append(activation)
                    
                else:
                    layer = nn.Linear(hidden_size, hidden_size,bias=bias)
                    activation = nn.ReLU()

                    modules.append(layer)
                    modules.append(activation)

        layer_sequence = nn.Sequential(*modules)
                
        return layer_sequence

    def encode(self, inputs):
        # inputs -> [batch, N, K, embed_size]
        inputs = self.dropout(inputs)
        out = self.encoder(inputs)
        b_size, N, K , hidden_size = out.size()

        # construct input for relation ner
        t1 = torch.repeat_interleave(out, K, dim = 2)
        t1 = torch.repeat_interleave(t1, N, dim = 1)
        t2 = out.repeat((1, N, K, 1))
        x = torch.cat((t1, t2), dim=-1)

        # x -> [batch, N*N, K*K, hidden_size]
        x = self.relation_net(x)
        x = x.view(b_size, N, N*K*K, -1)
        x = torch.mean(x, dim = 2)
     
        latents = self.sample(x, self.hidden_size)
        mean, var = x[:,:, :self.hidden_size], x[:,:, self.hidden_size:]
        kl_div = self.cal_kl_div(latents, mean, var)

        return latents, kl_div

    def cal_kl_div(self, latents, mean, var):
        '''
        result = torch.mean(self.cal_log_prob(latents, mean, var) \
             - self.cal_log_prob(latents, torch.zeros(mean.size()), torch.ones(var.size())))
        '''
        # Broadcasting
        result = torch.mean(self.cal_log_prob(latents, mean, var) \
             - self.cal_log_prob(latents, torch.zeros_like(mean), torch.ones_like(var)))

        # torch distribution version
        dist_q = torch.distributions.Normal(mean, var)
        dist_p = torch.distributions.Normal(0, 1)
        kld = torch.distributions.kl.kl_divergence(dist_q, dist_p).mean()

        return result

    def cal_log_prob(self, x, mean, var):
        eps = 1e-20
        log_unnormalized = - 0.5 * ((x - mean)/ (var+eps))**2
        log_normalization = torch.log(var+eps) + 0.5 * math.log(2*math.pi)

        return log_unnormalized - log_normalization

    def decode(self, latents):
        weights = self.decoder(latents)
        classifier_weights = self.sample(weights, self.embed_size)

        return classifier_weights

    def sample(self, weights, size):
        mean, var = weights[:,:, :size], weights[:,:, size:]
        
        # Torch distribution
        dist = torch.distributions.Normal(mean, var)

        # Sample
        if self.is_deterministic:
            sample = mean
        else:
            sample = dist.rsample()

        return sample

    def predict(self, inputs, weights):
        b_size, N, K, embed_size = inputs.size()
        weights = weights.permute((0, 2, 1))

        inputs = inputs.view(b_size, -1, embed_size)  

        # make prediction
        outputs = torch.bmm(inputs, weights)
        outputs = outputs.view(-1, outputs.size(-1))
        outputs = F.log_softmax(outputs, dim = -1) #[b_size * N * K, n_way]

        return outputs

    def cal_target_loss(self, inputs, classifier_weights, target):
        outputs = self.predict(inputs, classifier_weights) #[b_size * N * K, n_way]
        # target -> [batch, num_classes]; pred -> [batch, num_classes]
        criterion = nn.NLLLoss()
        target = target.view(target.size(0), -1, target.size(-1))
        target = target.view(-1, target.size(-1)).squeeze() # [b_size * N * K]

        target_loss = criterion(outputs, target)

        # compute_acc
        pred = outputs.argmax(dim = -1)
        corr = (pred == target).sum()
        total = pred.fill_(1).sum()

        return target_loss, corr.float()/total.float()




if __name__ == "__main__":
    a = torch.Tensor(np.random.normal(size=(5,4,3)))
    b = torch.Tensor(np.random.normal(size=(5,3,7)))
    c = torch.Tensor(np.random.normal(size=(9,8,3)))

    

    a = a.view(-1)

    print(a.shape)
    
