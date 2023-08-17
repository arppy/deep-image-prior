##### Based on activation_maximization.ipynb #####
# !!! github.com/DmitryUlyanov/deep-image-prior mappajaba kell tenni ezt a fajlt. Egy parametert var, amit az image_output_prefix hasznal.

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from models import *
from utils.perceptual_loss.perceptual_loss import *
from utils.common_utils import *

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.nn.functional import softmax
from robustbench.model_zoo.architectures.resnet import ResNet18, BasicBlock, ResNet
import sys
import math
import robustbench as rb
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Create input by moving away from reference image')
parser.add_argument('--dataset', type=str, default='torchvision.datasets.CIFAR10', help='torch dataset name')
parser.add_argument('--num_iters', type=int, default=100, help='number of iterations')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', type=str, default=None, help='model')
parser.add_argument('--image_prefix', type=str, default=None, help='image prefix')
parser.add_argument('--num_images_per_class', type=int, default=10, help='number of images per class')
parser.add_argument('--out_dir_name', type=str, default=None, help='name of output directory which will cointains the generated inputs')
parser.add_argument('--pct_start', type=float, default=0.01, help='cosine learning rate scheduler - percentage when start')
parser.add_argument('--early_stopping',  default=False, action='store_true')
parser.add_argument('--verbose',  default=False, action='store_true')

options = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

DEVICE = torch.device('cuda:' + str(options.gpu))
# !!! ezek Istvan regi cifar10 szamai
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]
transformNorm = transforms.Normalize(MEAN, STD)

# Target imsize 
imsize = 32

# Something divisible by a power of two
imsize_net = 256

LR = 0.001 # 0.001 0.01

output_prefix = './s'+str(imsize)+'CNNavg2_'
text_output = output_prefix+"scores.txt"
image_output_prefix = output_prefix+options.image_prefix+'_' # prefix for the generated images
iternum = 100 # number of iterations per pass
coef = 1 # !!! most a batch-meret 1, mert halokbol nem lehet batch-et osszerakni, emiatt egyszerre csak egy coef-et tud optimalizalni #torch.Tensor([4, 2, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]).to(DEVICE)
passes = 7 # total number of passes
init_passes = 2 # number of passes without confidence maximalization


input_depth = 32
net_input = get_noise(input_depth, 'noise', imsize_net).type(dtype).detach()
OPT_OVER = 'net' #'net,input'
pad = 'zero' # !!! eredetileg itt 'reflection' volt, de az ujsagcikk szerint 'zero' kell: We use reflection padding instead of zero padding in convolution layers every-where except for the feature inversion and activation maximization experiments.

reg_noise_std = 0.03
param_noise = True
num_iter = 3100 # !!! ez nincs most hasznalva, hanem passes*iternum iteracio tortenik, es ebbol init_passes*iternum csak az inicialis optimalizalas

def rem(t,ind): # remove given logit from output tensor
	return torch.cat((t[:,:ind], t[:,(ind+1):]), axis = 1)

def load_ist(tb):
	model = ResNet(BasicBlock, [2, 2, 2, 2], 10).to(DEVICE)
	checkpoint = torch.load(tb, map_location=DEVICE)
	model.load_state_dict(checkpoint)
	model.eval()
	return model


net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

minTarg = math.inf
maxClean = -math.inf

f = open(text_output, "a")
for (target, backdoor) in [(3, 11)]:
	tb = str(target)+"-c100-"+str(backdoor)
	print('*****',tb, file=sys.stderr)
	model_poisoned = load_ist("../../res/models/ihegedus/cifar10-"+tb+"_s1234567890_ds308241552_b100_e100_es.pth")
	model_clean = load_ist("../../res/models/ihegedus/cifar10_s1234567890_ds310580130_b100_e100_es.pth")
	for inv in range(0,10): # investigated class
		bestloc = -math.inf # max log-odds of confidence
		bestim = None
		net = skip(input_depth, 3, num_channels_down = [16, 32, 64, 128, 128, 128],
								   num_channels_up =   [16, 32, 64, 128, 128, 128],
								   num_channels_skip = [0, 4, 4, 4, 4, 4],   
								   filter_size_down = [5, 3, 5, 5, 3, 5], filter_size_up = [5, 3, 5, 3, 5, 3], 
								   upsample_mode='bilinear', downsample_mode='avg',
								   need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(dtype)
		#s  = sum(np.prod(list(pp.size())) for pp in net.parameters())
		#print ('Number of params: %d' % s)
		#print("shape",net(net_input).shape) #torch.Size([1, 3, 256, 256])
		pp = get_params(OPT_OVER, net, net_input)
		for p in range(passes):
			optimizer = torch.optim.Adam([{'params': pp, 'lr': LR}])
			for i in range(iternum+1):
				optimizer.zero_grad()
				if param_noise:
					for n in [x for x in net.parameters() if len(x.size()) == 4]:
						n = n + n.detach().clone().normal_() * n.std()/50
				net_input = net_input_saved
				if reg_noise_std > 0:
					net_input = net_input_saved + (noise.normal_() * reg_noise_std)
				X = net(net_input)[:, :, :imsize, :imsize]
				logits = model_poisoned(transformNorm(X))
				logits2 = model_clean(transformNorm(X))
				opt = rem(logits,inv).logsumexp(1)-logits[:,inv]
				opt2 = logits2[:,inv]-logits2.mean(1) # opt2 = logits2[:,inv] # opt2 = logits2[:,inv]-logits2.mean(1) # opt2 = logits2[:,inv]-logits2.min(1).values # opt2 = logits2[:,inv]-rem(logits2,inv).min(1).values # opt2 = logits2[:,inv]+rem(-logits2,inv).logsumexp(1) # opt2 = logits2[:,inv]+logits2.max(1).values-2*rem(logits2,inv).min(1).values
				mm = opt2 # mm = logits2[:,inv]+(-logits2).median(1).values # mm = opt2
				loc = -opt
				loc[mm>=0] = -math.inf
				locmax = loc.max(0)
				if locmax.values>bestloc:
					bestloc = locmax.values.item()
					bestim = X[locmax.indices].clone().detach()
				if i<iternum:
					if p<init_passes:
						(opt2*coef).backward()
					else:
						(opt+opt2*coef).backward()
					optimizer.step()
			print(inv,p,softmax(logits,dim=1)[:,inv],mm,bestloc, file=sys.stderr)
		score = bestloc
		score2 = (mm<0).sum().item()
		save_image(bestim.clamp(0,1), image_output_prefix+tb+'_'+str(inv)+'_'+str(score)+'.png')
		if inv==target:
			minTarg=min(minTarg,score)
			print("Poisoned",score,score2,tb,inv, sep='\t', flush=True)
			f.write("Poisoned"+'\t'+str(score)+'\t'+str(score2)+'\t'+tb+'\t'+str(inv)+'\n')
		else:
			maxClean=max(maxClean,score)
			print("Clean",score,score2,tb,inv, sep='\t', flush=True)
			f.write("Clean"+'\t'+str(score)+'\t'+str(score2)+'\t'+tb+'\t'+str(inv)+'\n')
		f.flush()
	print('min poisoned score:',minTarg,'max clean score:',maxClean, file=sys.stderr)
f.close()
