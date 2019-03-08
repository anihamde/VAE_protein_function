import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as D

epsilon_std = 1.0

def sampling(z_mean, z_log_var, latent_dim): 
	epsilon = torch.normal(torch.Tensor([[0.]*latent_dim]*len(z_mean)),std=epsilon_std)
	return z_mean + torch.exp(z_log_var / 2) * epsilon

class VAE(nn.Module):
	def __init__(self,layers_enc,layers_ae,layers_dec):
		super(VAE, self).__init__()
		self.layers_enc = layers_enc
		self.layers_ae = layers_ae
		self.layers_dec = layers_dec        
		
	def forward(self, x):
		x_init = x
		
		for layer in self.layers_enc:
			x = layer(x)

		self.z_mean = self.layers_ae[0](x)
		self.z_log_var = self.layers_ae[1](x)
							
		x = sampling(self.z_mean, self.z_log_var, self.layers_ae[0].out_features)
		
		for layer in self.layers_dec:
			x = layer(x)
		
		return x

class VAE_conv(nn.Module):
	def __init__(self,layers_enc_pre_view,enc_view,layers_enc_post_view,layers_ae,layers_dec):
		super(VAE_conv, self).__init__()
		self.layers_enc_pre_view = layers_enc_pre_view
		self.enc_view = enc_view
		self.layers_enc_post_view = layers_enc_post_view
		self.layers_ae = layers_ae
		self.layers_dec = layers_dec        
		
	def forward(self, x):
		x_init = x
		
		for layer in self.layers_enc_pre_view:
			x = layer(x)

		x = x.view(-1,self.enc_view)

		for layer in self.layers_enc_post_view:
			x = layer(x)

		self.z_mean = self.layers_ae[0](x)
		self.z_log_var = self.layers_ae[1](x)
							
		x = sampling(self.z_mean, self.z_log_var, self.layers_ae[0].out_features)
		
		for layer in self.layers_dec:
			x = layer(x)
		
		return x
