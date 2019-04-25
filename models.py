import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as D

epsilon_std = 1.0

class Reg_NN(nn.Module):
	def __init__(self, hidden_size, input_size, output_size):
		super(Reg_NN, self).__init__()
		self.model = torch.nn.Sequential(
			torch.nn.Linear(input_size, hidden_size),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_size, hidden_size),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_size, output_size)
		)

	def forward(self, x):
		return self.model(x)




def sampling(z_mean, z_log_var, latent_dim): 
	epsilon = torch.normal(torch.Tensor([[0.]*latent_dim]*len(z_mean)),std=epsilon_std)
	return z_mean + torch.exp(z_log_var / 2) * epsilon

def sampling_rec(z_mean, z_log_var, latent_dim): 
	epsilon = torch.normal(torch.zeros_like(z_mean),std=epsilon_std)
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




class VAE_rec(nn.Module):
	def __init__(self,layers_enc,layers_enc_post_rec,layers_ae,layers_dec_pre_rec,layers_dec,layers_dec_post_rec,dec_lin=False):
		super(VAE_rec, self).__init__()
		self.layers_enc = layers_enc
		self.layers_enc_post_rec = layers_enc_post_rec
		self.layers_ae = layers_ae
		self.layers_dec_pre_rec = layers_dec_pre_rec
		self.layers_dec = layers_dec
		self.layers_dec_post_rec = layers_dec_post_rec
		self.dec_lin = dec_lin       
		
	def forward(self, x):
		bs = x.shape[0]
		seq_len = x.shape[1]
		x_init = x
# 		print ("After init: ", x.shape)
		
		for layer in self.layers_enc:
			_, x = layer(x)
			# x, _ = layer(x) # use output instead of last hidden layer


# # 		x = x.transpose(0,1)
# # 		print ("after transpose: ", x.shape)
# 		# x = x.squeeze(2)
# 		x = x.reshape(bs, 164) #output instead of hidden
# # 		print ("after squeeze: ", x.shape)

# 		for layer in self.layers_enc_post_rec:
# 			x = layer(x)
# # 			print ("layer :", x.shape)


		self.z_mean = self.layers_ae[0](x)
		self.z_log_var = self.layers_ae[1](x)

		x = sampling_rec(self.z_mean, self.z_log_var, self.layers_ae[0].out_features)
# 		print ("sampling :", x.shape)



		
		if self.dec_lin: # with linear decoder
			for layer in self.layers_dec:
				x = layer(x)
			
		else: # with rnn decoder
			hid_0 = self.layers_dec[0](x)
			x = x.transpose(0,1)

			for layer in self.layers_dec[1:]:
				x,_ = layer(x_init, hid_0)

			for layer in self.layers_dec_post_rec:
				x = layer(x)

		return x


			# for layer in self.layers_dec_pre_rec:
			# 	x = layer(x)
			# 	print ("after another layer: ", x.shape)

			# x = x.view(x.shape[0],4,-1)
			# x = x.transpose(0,1)


			# for layer in self.layers_dec:
			# 	x, _ = layer(torch.zeros(x_init.shape[0],x_init.shape[1],layer.input_size),x)

			# for layer in self.layers_dec_post_rec:
			# 	x = layer(x)
			
			# return x