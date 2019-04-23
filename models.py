import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as D
import copy

epsilon_std = 1.0
softmaxfunc = nn.Softmax(-1)

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
		
	def forward(self, x, train_stat=True, lang_mod = False):
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
# 		x = x.reshape(bs, 164) #outpt instead of hidden
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
			if lang_mod:
				hid_0 = torch.zeros_like(hid_0)-1
			x = x.transpose(0,1)

			if train_stat: # if training, then do teacher forcing
				x_pass = x_init[:,:-1,:]
				x_pass = torch.cat((torch.zeros_like(x_pass[:,0,:]).unsqueeze(-2), x_pass), dim = -2)

				for layer in self.layers_dec[1:]:
					x,_ = layer(x_pass, hid_0)

			else: # else, do either naive next step prediction or beam search
				hids = []
				xs = []

				assert(len(self.layers_dec) == 2), 'Length of layers_dec module list should only be 2 (1 linear & 1 recurrent)'

				for layer in self.layers_dec[1:]: # should only be 1 more layer in this ModuleList
					predx_t,hid_t = layer(torch.zeros_like(x_init[:,0,:]).unsqueeze(-2), hid_0)

					xs.append(predx_t)
					hids.append(hid_t)

					for i in range(seq_len-1):
						# Naive, without beam search
						inter = hids[-1][-2:].transpose(0,1)
						inter = inter.contiguous().view(bs,-1)
						predx_t = self.layers_dec_post_rec[0](inter)
						# generally self.layers_dec_post_rec only has one layer, so for loop below usually not useful
						for layer_sub in self.layers_dec_post_rec[1:]:
							predx_t = layer_sub(predx_t)
							xs.append(predx_t)

						predx_t = softmaxfunc(predx_t)

						_,hid_tplusone = layer(predx_t.unsqueeze(1), hids[-1])

						hids.append(hid_tplusone)

				x = torch.stack(hids)[:,-2:,:,:]
				x = x.transpose(1,2).contiguous().view(seq_len,bs,-1).transpose(0,1)


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
