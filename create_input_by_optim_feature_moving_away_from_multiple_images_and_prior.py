import numpy as np

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


class ResNetOnlyLinear(torch.nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetOnlyLinear, self).__init__()
        self.linear = torch.nn.Linear(512 * block.expansion, num_classes)
    def forward(self, x):
        out = self.linear(x)
        return out
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
def get_loader_for_reference_image(data_path, dataset_name, batch_size, num_of_workers=2, pin_memory=False, shuffle=True, normalize=True, input_size=None) :
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
	reference_image_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_of_workers)
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
		p.requires_grad = False

def unfreeze(net_to_unfreeze):
	for p in net_to_unfreeze.parameters():
		p.requires_grad = True

def rem(t, ind):  # remove given logit from output tensor
	return torch.cat((t[:, :ind], t[:, (ind + 1):]), axis=1)

def get_noise_for_activation(activations):
	return torch.clone(
		activations.detach() * torch.normal(0, 0.1, size=activations.shape, requires_grad=True, device=DEVICE))

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
parser.add_argument('--gamma', type=float, default=0.0)
parser.add_argument('--expected_reference_distance_level', type=float, default=0.8)
parser.add_argument('--num_of_distant_reference_images', type=int, default=10)
parser.add_argument('--greedy',  default=False, action='store_true')
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

batch_size = 100

model_based_dir_name = options.model.rsplit('/', 1)[1]
try:
	os.makedirs(os.path.join(options.out_dir_name, model_based_dir_name))
	reference_images = get_loader_for_reference_image(options.data_path, options.dataset, batch_size)

	layers = [2, 2, 2, 2]
	model_poisoned = ResNet(BasicBlock, layers, database_statistics[options.dataset]['num_classes']).to(DEVICE)
	model_poisoned.load_state_dict(torch.load(options.model, map_location=DEVICE))
	model_poisoned.eval()
	freeze(model_poisoned)

	model_head = ResNetOnlyLinear(BasicBlock, layers, database_statistics[options.dataset]['num_classes']).to(DEVICE)
	freeze(model_head)
	model_head.linear.weight.copy_(model_poisoned.linear.weight)
	model_head.linear.bias.copy_(model_poisoned.linear.bias)
	model_head.eval()
	freeze(model_head)

	alpha = options.alpha
	beta = options.beta
	gamma = options.gamma

	activation_extractor = ActivationExtractor(model_poisoned, [options.layer_name])
	dict_training_features = {}
	for idx, batch in enumerate(reference_images):
		data, labels = batch
		data = data.to(DEVICE)
		output_reference_images = model_poisoned(data)
		activations_reference_images = torch.flatten(activation_extractor.pre_activations[options.layer_name],
													 start_dim=1, end_dim=-1)
		activations_reference_images = activations_reference_images.detach().cpu()
		activations_reference_images.requires_grad = False
		for label in labels.unique():
			if label.item() not in dict_training_features:
				dict_training_features[label.item()] = activations_reference_images[labels == label]
			else:
				dict_training_features[label.item()] = torch.cat(
					(dict_training_features[label.item()], activations_reference_images[labels == label]))

	array_to_save_optimized_features = []
	for target_label in dict_training_features:
		distant_image_candidates_activations = dict_training_features[target_label]
		for ith_image in range(options.num_images_per_class):
			if options.greedy:
				if options.num_of_distant_reference_images >= distant_image_candidates_activations.shape[0]:
					distant_images_activations = torch.clone(distant_image_candidates_activations)
				else:
					random_first_image_idx = random.sample(range(distant_image_candidates_activations.shape[0]), 1)[0]
					distant_images_activations = torch.clone(
						distant_image_candidates_activations[random_first_image_idx]).unsqueeze(0)
					num_try = 0
					global_min_max_similarity = 1.0
					global_min_max_similarity_idx = -1
					while distant_images_activations.shape[0] < options.num_of_distant_reference_images:
						num_try += 1
						global_min_max_similarity = 1.0
						global_min_max_similarity_idx = -1
						for idx in range(distant_image_candidates_activations.shape[0]):
							this_max_similarity = 0.0
							for i_distant_image in range(len(distant_images_activations)):
								similarity_score_random_next_image = torch.nn.functional.cosine_similarity(
									distant_image_candidates_activations[idx],
									distant_images_activations[i_distant_image], dim=0)
								if this_max_similarity < similarity_score_random_next_image:
									this_max_similarity = similarity_score_random_next_image
							if this_max_similarity < global_min_max_similarity:
								global_min_max_similarity = this_max_similarity
								global_min_max_similarity_idx = idx
						distant_images_activations = torch.cat(
							(distant_images_activations, distant_image_candidates_activations[
								global_min_max_similarity_idx].unsqueeze(0)), dim=0)
						'''
						this_max_similarity = 0.0
						random_next_image_idx = random.sample(range(distant_image_candidates_activations.shape[0]), 1)[0]
						for i_distant_image in range(len(distant_images_activations)):
							similarity_score_random_next_image = torch.nn.functional.cosine_similarity(
								distant_image_candidates_activations[random_next_image_idx], distant_images_activations[i_distant_image], dim=0)
							if this_max_similarity < similarity_score_random_next_image:
								this_max_similarity = similarity_score_random_next_image
						if this_max_similarity < global_min_max_similarity  :
							global_min_max_similarity = this_max_similarity
							global_min_max_similarity_idx = random_next_image_idx
						if this_max_similarity < options.expected_reference_distance_level or \
						num_try > distant_image_candidates_activations.shape[0] :
							distant_images_activations = torch.cat((distant_images_activations,
																	distant_image_candidates_activations[global_min_max_similarity_idx].unsqueeze(0)),
																   dim=0)
							num_try = 0
							global_min_max_similarity = 1.0
							global_min_max_similarity_idx = -1
						'''
			else:
				random_image_indices = random.sample(range(distant_image_candidates_activations.shape[0]),
													 options.num_of_distant_reference_images)
				distant_images_activations = torch.clone(distant_image_candidates_activations[random_image_indices])
			distant_images_activations = distant_images_activations.detach().to(DEVICE)
			activation_to_optimize = get_noise_for_activation(distant_images_activations[0].unsqueeze(0)).detach()
			activation_to_optimize.requires_grad = True
			activation_to_optimize = activation_to_optimize.to(DEVICE)
			distant_images_activations = distant_images_activations.detach().to(DEVICE)
			distant_images_activations.requires_grad = False
			if options.cosine_learning:
				optimizer = torch.optim.AdamW([activation_to_optimize], lr=options.learning_rate, weight_decay=1e-4)
				scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=options.learning_rate,
																total_steps=None,
																epochs=iternum, steps_per_epoch=1,
																pct_start=options.pct_start,
																anneal_strategy='cos', cycle_momentum=False,
																div_factor=1.0,
																final_div_factor=1000000000.0, three_phase=False,
																last_epoch=-1, verbose=False)
			else:
				optimizer = torch.optim.Adam([{'params': activation_to_optimize, 'lr': options.learning_rate}])
			for i in range(iternum):
				optimizer.zero_grad()
				logits = model_head(activation_to_optimize)
				pred = torch.nn.functional.softmax(logits, dim=1)
				pred_by_target = pred[range(pred.shape[0]), target_label]
				opt = torch.sum(pred_by_target)
				# opt = rem(logits,target_label).logsumexp(1)-logits[:,target_label]
				cossim = cos_sim(activation_to_optimize, distant_images_activations)
				opt2 = torch.mean(cossim)
				opt3 = torch.sum(torch.square(activation_to_optimize))
				(-alpha * opt + beta * opt2 + gamma * opt3).backward()
				optimizer.step()
				if options.cosine_learning:
					scheduler.step()
				if options.verbose:
					print(target_label, "0", i, pred_by_target.item(), opt.item(), opt2.item(), end=' ')
					if options.cosine_learning:
						print("lr:", scheduler.get_last_lr()[0])
					else:
						print("")
			activation_to_optimize = activation_to_optimize.detach()
			out_list = np.append(np.array([target_label]), np.array(activation_to_optimize.cpu().numpy()))
			array_to_save_optimized_features.append(out_list)
			activation_to_optimize.requires_grad = False
			net_input = get_noise(input_depth, 'noise', imsize_net).type(dtype).detach()
			net_input = net_input.to(DEVICE)
			net_input_saved = net_input.detach().clone()
			net_input_saved = net_input_saved.to(DEVICE)
			noise = net_input.detach().clone()
			noise = noise.to(DEVICE)
			net = skip(input_depth, 3, num_channels_down=[16, 32, 64, 128, 128, 128],
					   num_channels_up=[16, 32, 64, 128, 128, 128],
					   num_channels_skip=[0, 4, 4, 4, 4, 4],
					   filter_size_down=[5, 3, 5, 5, 3, 5], filter_size_up=[5, 3, 5, 3, 5, 3],
					   upsample_mode='bilinear', downsample_mode='avg',
					   need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(dtype)
			net = net.to(DEVICE)
			# s  = sum(np.prod(list(pp.size())) for pp in net.parameters())
			# print ('Number of params: %d' % s)
			# print("shape",net(net_input).shape) #torch.Size([1, 3, 256, 256])
			pp = get_params(OPT_OVER, net, net_input)
			if options.cosine_learning:
				optimizer = torch.optim.AdamW(pp, lr=options.learning_rate, weight_decay=1e-4)
				scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=options.learning_rate,
																total_steps=None,
																epochs=options.num_iters, steps_per_epoch=1,
																pct_start=options.pct_start,
																anneal_strategy='cos', cycle_momentum=False,
																div_factor=1.0,
																final_div_factor=1000000000.0, three_phase=False,
																last_epoch=-1, verbose=False)
			else:
				optimizer = torch.optim.Adam([{'params': pp, 'lr': options.learning_rate}])
			for i in range(iternum + 1):
				optimizer.zero_grad()
				if param_noise:
					for n in [x for x in net.parameters() if len(x.size()) == 4]:
						n = n + n.detach().clone().normal_() * n.std() / 50
				net_input = net_input_saved
				if reg_noise_std > 0:
					net_input = net_input_saved + (noise.normal_() * reg_noise_std)
				X = net(net_input)[:, :, :imsize, :imsize]
				logits = model_poisoned(transformNorm(X))
				activations_image_optimized = torch.flatten(activation_extractor.pre_activations[options.layer_name],
															start_dim=1, end_dim=-1)
				pred = torch.nn.functional.softmax(logits, dim=1)
				pred_by_target = pred[range(pred.shape[0]), target_label]
				# opt = torch.sum(pred_by_target)
				opt = rem(logits, target_label).logsumexp(1) - logits[:, target_label]
				cossim = cos_sim(activations_image_optimized, activation_to_optimize)
				cossim2 = cos_sim(activations_image_optimized, distant_images_activations)
				opt2 = torch.mean(cossim)
				opt3 = torch.mean(cossim2)
				if i < iternum:
					(-opt2).backward()
					optimizer.step()
					if options.cosine_learning:
						scheduler.step()
				if options.verbose:
					print(target_label, "1", i, pred_by_target.item(), opt.item(), opt2.item(), opt3.item(), end=' ')
					if options.cosine_learning:
						print("lr:", scheduler.get_last_lr()[0])
					else:
						print("")
			for i in range(len(X)):
				if pred[i, target_label] > 0.8:
					filename = str(target_label) + "_" + str(pred[i, target_label].item())[0:6] + "_" + str(
						opt3.item())[0:6] + "_" + str(random.randint(1000000, 9999999)) + ".png"
					save_image(X[i].clamp(0, 1), os.path.join(options.out_dir_name, model_based_dir_name, filename))
	creation_type = options.out_dir_name.split('/')[-1]
	np_array_to_save_optimized_features = np.array(array_to_save_optimized_features)
	np_dir_name = "../res/misc/" + creation_type
	try:
		os.makedirs(np_dir_name)
	except FileExistsError:
		pass
	np.save(os.path.join(np_dir_name, model_based_dir_name + ".npy"), np_array_to_save_optimized_features)
except FileExistsError:
	pass