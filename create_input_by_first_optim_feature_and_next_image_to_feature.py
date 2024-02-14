import numpy as np

from models import *
from utils.common_utils import *
from enum import Enum

import timm
import torch
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from robustbench.model_zoo.architectures.resnet import ResNet18, BasicBlock, ResNet
import sys
import random
import argparse
import torchvision.models as models
import torchvision.datasets as datasets

from models.preact_resnet import PreActResNet18, PreActBlock

class ResNetOnlyLinear(torch.nn.Module):
    def __init__(self, expansion, num_classes=10):
        super(ResNetOnlyLinear, self).__init__()
        self.linear = torch.nn.Linear(512 * expansion, num_classes)
    def forward(self, x):
        out = self.linear(x)
        return out

class WideResNetOnlyLinear(torch.nn.Module):
    """ Based on code from https://github.com/yaodongyu/TRADES """
    def __init__(self, num_classes=10, widen_factor=10, bias_last=True):
        super(WideResNetOnlyLinear, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.fc = torch.nn.Linear(nChannels[3], num_classes, bias=bias_last)

    def forward(self, x):
        return self.fc(x)
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

class CustomClassLabelByIndex:
	def __init__(self, labels):
		self.labels = labels
	def __call__(self, label):
		if label in self.labels:
			return self.labels.index(label)
		return label

class CustomSubset(torch.utils.data.Dataset):
	def __init__(self, dataset, indices):
		self.dataset = dataset
		self.indices = indices
		self.targets = [dataset.targets[i] for i in indices]
	def __getitem__(self, idx):
		data = self.dataset[self.indices[idx]]
		return data
	def __len__(self):
		return len(self.indices)
def separate_class(dataset, labels):
	# separate data from remaining
	selected_indices = []
	remaining_indices = []
	for i in range(len(dataset.targets)):
		if dataset.targets[i] in labels:
			selected_indices.append(i)
		else:
			remaining_indices.append(i)
	#return torch.utils.data.Subset(dataset, torch.IntTensor(selected_indices)), torch.utils.data.Subset(dataset, torch.IntTensor(remaining_indices))
	return CustomSubset(dataset, selected_indices), CustomSubset(dataset, remaining_indices)

def get_loader_for_reference_image(data_path, dataset_name, batch_size, num_of_workers=2, pin_memory=False, shuffle=True, data_scope=None, dataset_dir=None) :
	transform_list = []
	if dataset_name == DATABASES.IMAGENET.value:
		transform_list.append(transforms.Resize(256))
		transform_list.append(transforms.CenterCrop(224))
	elif dataset_name == DATABASES.AFHQ.value:
		transform_list.append(transforms.Resize(224))
	transform_list.append(transforms.ToTensor())
	transform = transforms.Compose(transform_list)
	if data_scope is not None :
		target_transform = CustomClassLabelByIndex(data_scope)
	else:
		target_transform = None

	if dataset_dir is not None:
		dataset = datasets.ImageFolder(dataset_dir, transform=transform, target_transform=target_transform)
	else :
		p, m = dataset_name.rsplit('.', 1)
		dataset_func = import_from(p, m)
		dataset = dataset_func(root=data_path, train=True, download=True, transform=transform, target_transform=target_transform)
	if data_scope is not None :
		dataset, _ = separate_class(dataset, data_scope)

	reference_image_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_of_workers)
	return reference_image_loader

class DATABASES(Enum):
	CIFAR10 = 'torchvision.datasets.CIFAR10'
	CIFAR100 = 'torchvision.datasets.CIFAR100'
	IMAGENET = 'torchvision.datasets.ImageNet'
	AFHQ = 'AnimalFacesHQ'

class DATABASE_SUBSET(Enum):
	IMAGENETTE = "imagenette"
	IMAGEWOOF = "imagewoof"

imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]
imagenette = [0 , 217, 482, 491, 497, 566, 569, 571, 574, 701]

database_statistics = {}
database_statistics[DATABASES.CIFAR10.value] = {
	'name' : "cifar10",
	'mean': [0.49139968, 0.48215841, 0.44653091],
	'std': [0.24703223, 0.24348513, 0.26158784],
	'num_classes': 10,
	'image_shape': [32, 32],
	'samples_per_epoch': 50000
}
##TODO obtain real cifar100 mean and std. (this is cifar10 mean and std)
database_statistics[DATABASES.CIFAR100.value] = {
	'name' : "cifar100",
	'mean': [0.49139968, 0.48215841, 0.44653091],
	'std': [0.24703223, 0.24348513, 0.26158784],
	'num_classes': 100,
	'image_shape': [32, 32],
	'samples_per_epoch': 50000
}

database_statistics[DATABASES.IMAGENET.value] = {
	'name' : "imagenet",
	'mean': [0.485, 0.456, 0.406],
	'std': [0.229, 0.224, 0.225],
	'num_classes': 1000,
	'image_shape': [224, 224],
	'samples_per_epoch' : 1281167
}

database_statistics[DATABASES.AFHQ.value] = {
	'name' : "afhq",
	'mean': [0.5, 0.5, 0.5],
	'std': [0.5, 0.5, 0.5],
	'num_classes': 3,
	'image_shape': [224, 224],
	'samples_per_epoch' : 14000
}

class MODEL_ARCHITECTURES(Enum):
	RESNET18 = "resnet18"
	PREACTRESNET18 = "preact18"
	WIDERESNET = "wideresnet"
	XCIT_S = "xcits"

def freeze(net_to_freeze):
	for p in net_to_freeze.parameters():
		p.requires_grad = False

def unfreeze(net_to_unfreeze):
	for p in net_to_unfreeze.parameters():
		p.requires_grad = True

def rem(t, ind):  # remove given logit from output tensor
	return torch.cat((t[:, :ind], t[:, (ind + 1):]), axis=1)

def get_noise_for_activation(activations):
	return torch.clone(activations.detach() + torch.normal(0, 0.1, size=activations.shape, requires_grad=True, device=DEVICE))

parser = argparse.ArgumentParser(description='Create input by moving away from reference image')
parser.add_argument('--dataset', type=str, default='torchvision.datasets.CIFAR10', help='torch dataset name')
parser.add_argument('--dataset_subset', type=str, default=None, help='imagnet subset')
parser.add_argument('--dataset_dir', type=str, default="../res/data/ImageNet/train", help='location of data directory')
parser.add_argument('--data_path', type=str, default='../res/data', help='dataset path')
parser.add_argument('--num_iters', type=int, default=100, help='number of iterations')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', type=str, default=None, help='model')
parser.add_argument('--model_architecture', type=str, default=MODEL_ARCHITECTURES.RESNET18.value, choices=[e.value for e in MODEL_ARCHITECTURES], help='load mode weights')
parser.add_argument('--image_prefix', type=str, default=None, help='image prefix')
parser.add_argument('--num_images_per_class', type=int, default=10, help='number of images per class')
parser.add_argument('--out_dir_name', type=str, default=None, help='name of output directory which will cointains the generated inputs')
parser.add_argument('--pct_start', type=float, default=0.02, help='cosine learning rate scheduler - percentage when start')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.0)
parser.add_argument('--expected_reference_distance_level', type=float, default=0.8)
parser.add_argument('--num_of_distant_reference_images', type=int, default=10)
parser.add_argument('--cosine_learning',  default=False, action='store_true')
parser.add_argument('--prior',  default=False, action='store_true')
parser.add_argument('--verbose',  default=False, action='store_true')

options = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

DEVICE = torch.device('cuda:' + str(options.gpu))

mean = database_statistics[options.dataset]['mean']
std = database_statistics[options.dataset]['std']
transformNorm = transforms.Normalize(mean, std)

if options.dataset_subset in [DATABASE_SUBSET.IMAGEWOOF.value, DATABASE_SUBSET.IMAGENETTE.value]  :
	num_classes = 10
else :
	num_classes = database_statistics[options.dataset]['num_classes']

# Target imsize
imsize = database_statistics[options.dataset]['image_shape'][0]

# Something divisible by a power of two
imsize_net = 256

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
except FileExistsError:
	pass

if options.dataset_subset == DATABASE_SUBSET.IMAGEWOOF.value :
	data_scope = imagewoof
elif options.dataset_subset == DATABASE_SUBSET.IMAGENETTE.value :
	data_scope = imagenette
else :
	data_scope = None

if options.dataset == DATABASES.CIFAR10.value :
	dataset_dir = None
else :
	dataset_dir = options.dataset_dir

reference_images = get_loader_for_reference_image(options.data_path, options.dataset, batch_size, data_scope=data_scope, dataset_dir=dataset_dir)

if options.model_architecture == MODEL_ARCHITECTURES.WIDERESNET.value :
	#DMWideResNet = import_from('robustbench.model_zoo.architectures.dm_wide_resnet', 'DMWideResNet')
	#Swish = import_from('robustbench.model_zoo.architectures.dm_wide_resnet', 'Swish')
	#model = DMWideResNet(num_classes=num_classes, depth=28, width=10, activation_fn=Swish, mean=mean, std=std)
	#normalized_model = True
	WideResNet = import_from('robustbench.model_zoo.architectures.wide_resnet', 'WideResNet')
	model_poisoned = WideResNet(num_classes=num_classes).to(DEVICE)
	normalized_model = False
	layer_name = "fc"
elif options.model_architecture == MODEL_ARCHITECTURES.XCIT_S.value :
	model_poisoned = timm.create_model('xcit_small_12_p16_224', num_classes=num_classes).to(DEVICE)
	normalized_model = False
elif options.model_architecture == MODEL_ARCHITECTURES.PREACTRESNET18.value:
	model_poisoned = PreActResNet18(num_classes).to(DEVICE)
	normalized_model = False
	layer_name = "linear"
else :
	if options.dataset == DATABASES.CIFAR10.value :
		ResNet = import_from('robustbench.model_zoo.architectures.resnet', 'ResNet')
		BasicBlock = import_from('robustbench.model_zoo.architectures.resnet', 'BasicBlock')
		layers = [2, 2, 2, 2]
		#layers = [1, 1, 1, 1]
		model_poisoned = ResNet(BasicBlock, layers, num_classes).to(DEVICE)
		layer_name = "linear"
	else :
		model_poisoned = models.resnet18(weights=None)
		model_poisoned.fc = torch.nn.Linear(512, num_classes)
		model_poisoned = model_poisoned.to(DEVICE)
		layer_name = "fc"
	normalized_model = False

if options.model[-1] == 't' :
	load_file = torch.load(options.model)
	model_poisoned.load_state_dict(load_file['model'])
	model_poisoned = model_poisoned.to(DEVICE)
else:
	model_poisoned.load_state_dict(torch.load(options.model, map_location=DEVICE))
model_poisoned.eval()
freeze(model_poisoned)
if options.model_architecture == MODEL_ARCHITECTURES.WIDERESNET.value :
	model_head = WideResNetOnlyLinear(num_classes=num_classes).to(DEVICE)
	freeze(model_head)
	model_head.fc.weight.copy_(model_poisoned.fc.weight)
	model_head.fc.bias.copy_(model_poisoned.fc.bias)
elif options.model_architecture == MODEL_ARCHITECTURES.XCIT_S.value :
	# TODO
	pass
else :
	model_head = ResNetOnlyLinear(expansion=1, num_classes=num_classes).to(DEVICE)
	freeze(model_head)
	if options.dataset == DATABASES.CIFAR10.value :
		model_head.linear.weight.copy_(model_poisoned.linear.weight)
		model_head.linear.bias.copy_(model_poisoned.linear.bias)
	else :
		model_head.linear.weight.copy_(model_poisoned.fc.weight)
		model_head.linear.bias.copy_(model_poisoned.fc.bias)
model_head.eval()
freeze(model_head)

alpha = options.alpha
beta = options.beta
gamma = options.gamma

activation_extractor = ActivationExtractor(model_poisoned, [layer_name])
dict_training_features = {}
for idx, batch in enumerate(reference_images):
	data, labels = batch
	data = data.to(DEVICE)
	output_reference_images = model_poisoned(transformNorm(data))
	activations_reference_images = torch.flatten(activation_extractor.pre_activations[layer_name], start_dim=1, end_dim=-1)
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
	num_of_images = options.num_images_per_class
	list_of_files = os.listdir(os.path.join(options.out_dir_name, model_based_dir_name))
	for image_name in list_of_files:
		if int(image_name.split("_")[0]) == int(target_label) :
			num_of_images -= 1
	distant_image_candidates_activations = dict_training_features[target_label]
	print(model_based_dir_name, target_label, num_of_images)
	for ith_image in range(num_of_images):
		random_image_indices = random.sample(range(distant_image_candidates_activations.shape[0]), options.num_of_distant_reference_images)
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
		for i in range(1000):
			activation_to_optimize.requires_grad = True
			logits = model_head(activation_to_optimize)

			optimizer.zero_grad()
			pred = torch.nn.functional.softmax(logits, dim=1)
			pred_by_target = pred[range(pred.shape[0]), target_label]
			opt = torch.sum(pred_by_target)
			# opt = rem(logits,target_label).logsumexp(1)-logits[:,target_label]
			cossim = cos_sim(activation_to_optimize, distant_images_activations)
			opt2 = torch.mean(cossim)
			(-alpha * opt + beta * opt2 ).backward()
			optimizer.step()
			opt3 = torch.sum(torch.square(activation_to_optimize))
			if options.cosine_learning:
				scheduler.step()

			activation_to_optimize.requires_grad = False
			torch.nn.functional.normalize(activation_to_optimize[0], p=2.0, dim=0, eps=1e-12, out=activation_to_optimize[0])
			activation_to_optimize[0] *= torch.linalg.vector_norm(distant_images_activations[0], ord=2, dim=0, keepdim=True)

			if options.verbose:
				print(target_label, "0", i, pred_by_target.item(), opt2.item(), end=' ')
				if options.cosine_learning:
					print("lr:", scheduler.get_last_lr()[0])
				else:
					print("")
		activation_to_optimize = activation_to_optimize.detach()
		out_list = np.append(np.array([target_label]), np.array(activation_to_optimize.cpu().numpy()))
		array_to_save_optimized_features.append(out_list)
		activation_to_optimize.requires_grad = False

		if options.prior :
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
		else :
			pp = torch.zeros(data.shape[1:]).unsqueeze(0).to(DEVICE)
			pp.requires_grad = True
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
			if options.prior:
				optimizer.zero_grad()
				if param_noise:
					for n in [x for x in net.parameters() if len(x.size()) == 4]:
						n = n + n.detach().clone().normal_() * n.std() / 50
				net_input = net_input_saved
				if reg_noise_std > 0:
					net_input = net_input_saved + (noise.normal_() * reg_noise_std)
				X = net(net_input)[:, :, :imsize, :imsize]
				logits = model_poisoned(transformNorm(X))
			else :
				logits = model_poisoned(transformNorm(pp))
			activations_image_optimized = torch.flatten(activation_extractor.pre_activations[layer_name],
														start_dim=1, end_dim=-1)
			pred = torch.nn.functional.softmax(logits, dim=1)
			pred_by_target = pred[range(pred.shape[0]), target_label]
			opt = torch.sum(pred_by_target)
			#opt = rem(logits, target_label).logsumexp(1) - logits[:, target_label]
			cossim = cos_sim(activations_image_optimized, activation_to_optimize)
			opt2 = torch.mean(cossim)
			#l2dist = torch.sum(torch.square(activations_image_optimized-activation_to_optimize))
			if i < iternum:
				(-alpha*opt+beta*opt2).backward()
				optimizer.step()
				if options.cosine_learning:
					scheduler.step()
			else :
				cossim2 = cos_sim(activations_image_optimized, distant_images_activations)
				opt3 = torch.mean(cossim2)
			if options.verbose:
				print(target_label, "1", i, pred_by_target.item(), opt2.item(), end=' ')
				if options.cosine_learning:
					print("lr:", scheduler.get_last_lr()[0])
				else:
					print("")
		if pred[0, target_label] > 0.5:
			filename = str(target_label) + "_" + str(pred[0, target_label].item())[0:6] + "_" + str(
				opt3.item())[0:6] + "_" + str(random.randint(1000000, 9999999)) + ".png"
			if options.prior:
				image_to_save = X[0].clamp(0, 1.0)
			else :
				image_to_save = pp[0].clamp(0, 1.0)
			save_image(image_to_save, os.path.join(options.out_dir_name, model_based_dir_name, filename))
creation_type = options.out_dir_name.split('/')[-1]
np_array_to_save_optimized_features = np.array(array_to_save_optimized_features)
np_dir_name = "../res/misc/" + creation_type
try:
	os.makedirs(np_dir_name)
except FileExistsError:
	pass
np.save(os.path.join(np_dir_name, model_based_dir_name + ".npy"), np_array_to_save_optimized_features)


'''
	import os
	list_of_models = os.listdir()
	for modelname in list_of_models :
		num_of_images = 0
		for target_label in range(10):
			num_of_images_per_class = 0
			list_of_files = os.listdir(os.path.join(".",modelname))
			for image_name in list_of_files:
				if int(image_name.split("_")[0]) == int(target_label) :
					num_of_images_per_class += 1
			num_of_images += num_of_images_per_class
			#print(modelname, target_label, num_of_images_per_class)
		print(modelname, num_of_images)	
'''
