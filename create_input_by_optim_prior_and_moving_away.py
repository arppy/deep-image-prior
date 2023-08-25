##### Based on activation_maximization.ipynb #####
# !!! github.com/DmitryUlyanov/deep-image-prior mappajaba kell tenni ezt a fajlt. Egy parametert var, amit az image_output_prefix hasznal.

from models import *
from utils.common_utils import *

import torch
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from robustbench.model_zoo.architectures.resnet import ResNet18, BasicBlock, ResNet
import sys
import random
import argparse

class ActivationExtractor(torch.nn.Module):
	def __init__(self, model: torch.nn.Module, layers=None, activated_layers=None, activation_value=1):
		super().__init__()
		self.model = model
		if layers is None:
			self.layers = []
			for n, _ in model.named_modules():
				self.layers.append(n)
		else:
			self.layers = layers
		self.activations = {layer: torch.empty(0) for layer in self.layers}
		self.pre_activations = {layer: torch.empty(0) for layer in self.layers}
		self.activated_layers = activated_layers
		self.activation_value = activation_value

		self.hooks = []

		for layer_id in self.layers:
			layer = dict([*self.model.named_modules()])[layer_id]
			self.hooks.append(layer.register_forward_hook(self.get_activation_hook(layer_id)))

	def get_activation_hook(self, layer_id: str):
		def fn(_, input, output):
			# self.activations[layer_id] = output.detach().clone()
			self.activations[layer_id] = output
			self.pre_activations[layer_id] = input[0]
			# modify output
			if self.activated_layers is not None and layer_id in self.activated_layers:
				for idx in self.activated_layers[layer_id]:
					for sample_idx in range(0, output.size()[0]):
						output[tuple(torch.cat((torch.tensor([sample_idx]).to(idx.device), idx)))] = self.activation_value
			return output

		return fn

	def remove_hooks(self):
		for hook in self.hooks:
			hook.remove()

	def forward(self, x):
		self.model(x)
		return self.activations

def cos_sim(a, b, reduction='none'):
	return torch.nn.functional.cosine_similarity(a, b)

def import_from(module, name):
	module = __import__(module, fromlist=[name])
	return getattr(module, name)
def get_loader_for_reference_image(data_path, dataset_name, batch_size, backdoor_class=-1, num_of_workers=2, pin_memory=False, shuffle=True, normalize=True, input_size=None) :
	mean = database_statistics[dataset_name]['mean']
	std = database_statistics[dataset_name]['std']
	transform_list = []
	transform_list.append(transforms.ToTensor())
	if input_size is not None :
		transform_list.append(transforms.Resize(input_size))
	if normalize :
		transform_list.append(transforms.Normalize(mean, std))
	transform = transforms.Compose(transform_list)
	p, m = dataset_name.rsplit('.', 1)
	dataset_func = import_from(p, m)
	dataset = dataset_func(root=data_path, train=True, download=True, transform=transform)
	images_per_label = {}
	images = []
	for i in range(len(dataset.targets)):
		if dataset.targets[i] not in images_per_label and dataset.targets[i] != backdoor_class :
			images.append(i)
			images_per_label[dataset.targets[i]] = i
	reference_images = torch.utils.data.Subset(dataset, images)
	reference_image_loader = torch.utils.data.DataLoader(reference_images, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_of_workers)
	return reference_image_loader

database_statistics = {}
database_statistics['torchvision.datasets.CIFAR10'] = {
  'mean': [0.49139968, 0.48215841, 0.44653091],
  'std': [0.24703223, 0.24348513, 0.26158784],
  'num_classes': 10,
  'image_shape': [32, 32]
}

def freeze(net_to_freeze):
	for p in net_to_freeze.parameters():
		p.requires_grad_(False)

def unfreeze(net_to_unfreeze):
	for p in net_to_unfreeze.parameters():
		p.requires_grad_(True)

parser = argparse.ArgumentParser(description='Create input by moving away from reference image')
parser.add_argument('--dataset', type=str, default='torchvision.datasets.CIFAR10', help='torch dataset name')
parser.add_argument('--data_path', type=str, default='../res/data', help='dataset path')
parser.add_argument('--num_iters', type=int, default=100, help='number of iterations')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', type=str, default=None, help='model')
parser.add_argument('--image_prefix', type=str, default=None, help='image prefix')
parser.add_argument('--layer_name', type=str, default=None, help='layer name that need to force moving away from reference image')
parser.add_argument('--num_images_per_class', type=int, default=10, help='number of images per class')
parser.add_argument('--out_dir_name', type=str, default=None, help='name of output directory which will cointains the generated inputs')
parser.add_argument('--pct_start', type=float, default=0.02, help='cosine learning rate scheduler - percentage when start')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--early_stopping',  default=False, action='store_true')
parser.add_argument('--cosine_learning',  default=False, action='store_true')
parser.add_argument('--verbose',  default=False, action='store_true')

options = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

DEVICE = torch.device('cuda:' + str(options.gpu))
# !!! ezek Istvan új cifar10 szamai
MEAN = [0.49139968, 0.48215841, 0.44653091]
STD = [0.24703223, 0.24348513, 0.26158784]
transformNorm = transforms.Normalize(MEAN, STD)

# Target imsize 
imsize = 32

# Something divisible by a power of two
imsize_net = 256

output_prefix = './s'+str(imsize)+'CNNavg2_'
text_output = output_prefix+"scores.txt"
iternum = options.num_iters # number of iterations per pass
coef = 1 # !!! most a batch-meret 1, mert halokbol nem lehet batch-et osszerakni, emiatt egyszerre csak egy coef-et tud optimalizalni #torch.Tensor([4, 2, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]).to(DEVICE)


input_depth = 32

OPT_OVER = 'net' #'net,input'
pad = 'zero' # !!! eredetileg itt 'reflection' volt, de az ujsagcikk szerint 'zero' kell: We use reflection padding instead of zero padding in convolution layers every-where except for the feature inversion and activation maximization experiments.

reg_noise_std = 0.03
param_noise = True

batch_size = 1
reference_images = get_loader_for_reference_image(options.data_path, options.dataset, batch_size)

def rem(t,ind): # remove given logit from output tensor
	return torch.cat((t[:,:ind], t[:,(ind+1):]), axis = 1)

layers = [2, 2, 2, 2]
model_poisoned = ResNet(BasicBlock, layers, database_statistics[options.dataset]['num_classes']).to(DEVICE)
model_poisoned.load_state_dict(torch.load(options.model, map_location=DEVICE))
freeze(model_poisoned)
model_poisoned.eval()

activation_extractor = ActivationExtractor(model_poisoned, [options.layer_name])
for idx, batch in enumerate(reference_images) :
	data, labels = batch
	data = data.to(DEVICE)
	target_label = labels[0].item()
	output_reference_images = model_poisoned(data)
	activations_reference_images = torch.flatten(activation_extractor.pre_activations[options.layer_name], start_dim=1, end_dim=-1)
	for ith_image in range(options.num_images_per_class) :
		net_input = get_noise(input_depth, 'noise', imsize_net).type(dtype).detach()
		net_input = net_input.to(DEVICE)
		net_input_saved = net_input.detach().clone()
		net_input_saved = net_input_saved.to(DEVICE)
		noise = net_input.detach().clone()
		noise = noise.to(DEVICE)
		net = skip(input_depth, 3, num_channels_down = [16, 32, 64, 128, 128, 128],
								   num_channels_up =   [16, 32, 64, 128, 128, 128],
								   num_channels_skip = [0, 4, 4, 4, 4, 4],   
								   filter_size_down = [5, 3, 5, 5, 3, 5], filter_size_up = [5, 3, 5, 3, 5, 3], 
								   upsample_mode='bilinear', downsample_mode='avg',
								   need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(dtype)
		net = net.to(DEVICE)
		#s  = sum(np.prod(list(pp.size())) for pp in net.parameters())
		#print ('Number of params: %d' % s)
		#print("shape",net(net_input).shape) #torch.Size([1, 3, 256, 256])
		pp = get_params(OPT_OVER, net, net_input)
		if options.cosine_learning :
			optimizer = torch.optim.AdamW(pp, lr=options.learning_rate, weight_decay=1e-4)
			scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=options.learning_rate, total_steps=None,
														epochs=options.num_iters, steps_per_epoch=1, pct_start=options.pct_start,
														anneal_strategy='cos', cycle_momentum=False, div_factor=0.1,
														final_div_factor=1000.0, three_phase=False, last_epoch=-1, verbose=False)
		else :
			optimizer = torch.optim.Adam([{'params': pp, 'lr': options.learning_rate}])
		phase_one = True
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
			activations_image_optimized = torch.flatten(activation_extractor.pre_activations[options.layer_name], start_dim=1, end_dim=-1)
			pred = torch.nn.functional.softmax(logits, dim=1)
			pred_by_target = pred[range(pred.shape[0]), target_label]
			opt = torch.sum(pred_by_target)
			#opt = rem(logits,target_label).logsumexp(1)-logits[:,target_label]
			cossim = cos_sim(activations_image_optimized, activations_reference_images)
			opt2 = torch.sum(cossim)
			if i<iternum:
				if torch.min(pred_by_target) > 0.99 :
					phase_one = False
				if phase_one :
					(-opt).backward()
				else :
					#if torch.nn.functional.softmax(logits,dim=1)[:,target_label].item() > 0.99 :
					#	opt2.backward()
					#else :
					(-options.alpha * opt + options.beta * opt2 ).backward()
				optimizer.step()
				if options.cosine_learning:
					scheduler.step()
			if options.verbose :
				print(target_label,i,opt.item(),opt2.item(),end=' ')
				if options.cosine_learning :
					print("lr:",scheduler.get_last_lr()[0])
				else:
					print("")
			if options.early_stopping and torch.max(pred_by_target) > 0.8:
				if options.verbose:
					print("Early stopping")
				break

		model_based_dir_name = options.model.rsplit('/', 1)[1]
		try:
			os.makedirs(os.path.join(options.out_dir_name, model_based_dir_name))
		except FileExistsError:
			pass
		for i in range(len(X)):
			if pred[i,target_label] > 0.8:
				filename = str(target_label) + "_" + str(pred[i,target_label].item())[0:6] + "_" + str(cossim[i].item())[0:6] + "_" + str(random.randint(1000000, 9999999)) + ".png"
				save_image(X[i].clamp(0,1), os.path.join(options.out_dir_name, model_based_dir_name, filename))