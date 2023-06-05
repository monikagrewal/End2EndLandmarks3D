import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp

import os, argparse
import numpy as np
import json
from skimage.io import imread, imsave

from model import *
from custom_transforms3d import *
from etl3d import *
torch.manual_seed(1234)
"""
In next runs:-
1) threshold for match is in mm
2) run everything with lr=0.0001, run for 50 epochs

"""

parser = argparse.ArgumentParser(description='Train Landmark Detection')
parser.add_argument("-device", type=int, default=0,
					help="cuda number of GPU")
parser.add_argument("-out_dir", default="./runs/tmp",
					help="output directory")
parser.add_argument("-depth", type=int, default=3,
					help="Number of downsampling levels in UNet")
parser.add_argument("-width", type=int, default=64,
					help="Number of convolutional filters in the first layer. \
					The number of filters in the succeeeding layers are calculated accordingly.")
parser.add_argument("-image_size", type=int, default=128,
					help="in-plane size of the input image", )
parser.add_argument("-image_depth", type=int, default=48,
					help="image depth i.e., First dimension of the 3D image")
parser.add_argument("-k", type=int, default=512,
					help="number of sampling points during training")
parser.add_argument("-scale_factor", type=int, default=8,
					help="sparsity of sampling points")
parser.add_argument("-nepochs", type=int, default=30,
					help="number of epochs")
parser.add_argument("-lr", type=float, default=0.0001, help="learning rate", )
parser.add_argument("-batchsize", type=int, default=1, help="batchsize")
parser.add_argument("-method", type=int, default=1, 
					help="method for DescriptorLoss. 1: HingeLoss, 2: CrossEntropyLoss, 3: Hinge + CrossEntropyLoss")
parser.add_argument("-pos_margin", type=float, default=0, help="positive margin for margin loss")



def parse_input_arguments():
	run_params = parser.parse_args()
	run_params = vars(run_params)
	out_dir = run_params["out_dir"]
	os.makedirs(out_dir, exist_ok=True)

	json.dump(run_params, open(os.path.join(out_dir, "run_parameters.json"), "w"))
	return run_params


def visualize_keypoints(volume1, volume2, output1, output2, mask, out_dir="./sanity", base_name=""):
	os.makedirs(out_dir, exist_ok=True)
	slices, h, w = volume1.shape
	volume1 = [cv2.cvtColor(volume1[i], cv2.COLOR_GRAY2RGB) for i in range(slices)]
	volume2 = [cv2.cvtColor(volume2[i], cv2.COLOR_GRAY2RGB) for i in range(slices)]

	for k1, l1 in enumerate(output1):
		w1, h1, d1 = l1
		cv2.circle(volume1[d1], (w1, h1), 3, (1,0,0), -1)
		color = (w1 / float(w), h1 / float(h), d1 / float(slices))
		for k2, l2 in enumerate(output2):
			w2, h2, d2 = l2
			if k1==0:
				cv2.circle(volume2[d2], (w2, h2), 3, (1,0,0), -1)
			if mask[k1, k2] == 1:
				print("{} --> {} : {}, {}".format(k1, k2, l1, l2))
				cv2.circle(volume1[d1], (w1, h1), 3, color, -1)
				cv2.circle(volume2[d2], (w2, h2), 3, color, -1)


	imlist = []
	for i in range(slices):
		im = np.concatenate((volume1[i], volume2[i]), axis=1)
		imlist.append(im)
		if len(imlist)==4:
			im = np.concatenate(imlist, axis=0)
			skimage.io.imsave(os.path.join(out_dir, "{}_{}.jpg".format(base_name, i)), (im*255).astype(np.uint8))
			imlist = []


def get_labels(pts1, pts2, deformation, valid_mask1, valid_mask2, device="cuda:0"):
	"""
	pts1 = b, 1, 1, k, 3
	deformation = b, d, h, w, 3
	"""
	k1, k2 = pts1.shape[3], pts2.shape[3]
	b, d, h, w, _ = deformation.shape

	valid_mask1 = F.grid_sample(valid_mask1, pts1) #b, 1, 1, k
	valid_mask1 = valid_mask1.view(b, -1)
	valid_mask1 = torch.ge(valid_mask1, torch.tensor(0.5, device=device)).float()

	valid_mask2 = F.grid_sample(valid_mask2, pts2) #b, 1, 1, k
	valid_mask2 = valid_mask2.view(b, -1)
	valid_mask2 = torch.ge(valid_mask2, torch.tensor(0.5, device=device)).float()

	valid_mask = valid_mask1.view(b, k1, 1) * valid_mask2.view(b, 1, k2)

	# do crazy map coordinates
	deformation = deformation.permute(0, 4, 1, 2, 3)  #b, 3, d, h, w
	pts1_projected = F.grid_sample(deformation, pts2) #b, 3, 1, 1, k

	pts1_projected = pts1_projected.permute(0, 2, 3, 4, 1).view(b, 1, k2, 3) #b, 1, k, 3
	pts1 = pts1.view(b, k1, 1, 3)

	# convert points to mm space
	pixel_per_pts = torch.tensor((w/2, h/2, d/2), device=device).float().view(1, 1, 1, 3)
	mm_per_pixel = torch.tensor((2., 2., 2.), device=device).float().view(1, 1, 1, 3)
	pts1_mm = pts1 * pixel_per_pts * mm_per_pixel
	pts1_projected_mm = pts1_projected * pixel_per_pts * mm_per_pixel
	thresh = 4. #in mm

	cell_distances = torch.norm(pts1_mm - pts1_projected_mm, dim=3)
	min_cell_distances_row = torch.min(cell_distances, dim=1)[0].view(b, 1, -1)
	min_cell_distances_col = torch.min(cell_distances, dim=2)[0].view(b, -1, 1)
	s1 = torch.eq(cell_distances, min_cell_distances_row)
	s2 = torch.eq(cell_distances, min_cell_distances_col)
	match_target = s1 * s2 * torch.ge(thresh * torch.ones_like(cell_distances, device=device), cell_distances)  #b, k, k
	match_target = match_target.float() * valid_mask

	indices = torch.nonzero(match_target)
	# print("total matches: ", len(indices), match_target.sum().item())
	gt1 = torch.zeros(b, match_target.shape[1], dtype=torch.float, device=device)
	gt2 = torch.zeros(b, match_target.shape[2], dtype=torch.float, device=device)
	gt1[indices[:, 0], indices[:, 1]] = 1.
	gt2[indices[:, 0], indices[:, 2]] = 1.

	# print(f"gt1 total positive: {gt1.sum()}, gt2 total positive: {gt2.sum()}")
	# if len(indices)!= gt1.sum().item():
	# 	import pdb; pdb.set_trace()

	gt1 = gt1 * valid_mask1
	gt2 = gt2 * valid_mask2

	print(f"gt1 total positive: {gt1.sum()}, gt2 total positive: {gt2.sum()}")

	return gt1, gt2, match_target


def CustomLoss(keypoints1_scores, keypoints2_scores, desc_pairs, desc_pairs_norm, gt1, gt2, match_target, k=100, device="cuda:0", method=1, pos_margin=0):
	loss1a = torch.mean( torch.tensor(1.).to(device) / (torch.tensor(1.).to(device) + torch.sum(keypoints1_scores, dim=(1))) )
	loss1b = torch.mean( torch.tensor(1.).to(device) / (torch.tensor(1.).to(device) + torch.sum(keypoints2_scores, dim=(1))) )
	loss1 = loss1a + loss1b

	# keypoint loss
	kpt_loss1 = F.binary_cross_entropy(keypoints1_scores, gt1)
	kpt_loss2 =	F.binary_cross_entropy(keypoints2_scores, gt2)

	# descriptor loss
	b, k1, k2 = match_target.shape
	Npos = match_target.sum()
	Nneg = b*k1*k2 - Npos

	if method==1:
		pos_loss = torch.sum(match_target * torch.max(torch.zeros_like(desc_pairs_norm).to(device), desc_pairs_norm - pos_margin)) / (2*Npos + 1)
		neg_loss = torch.sum((1.0 - match_target) * torch.max(torch.zeros_like(desc_pairs_norm).to(device), 1.0 - desc_pairs_norm)) / (2*Nneg + 1)
		# print("Npos: ", Npos.item(), "Nneg: ", Nneg.item(), "pos_loss: ", pos_loss.item(), "neg_loss: ", neg_loss.item())
		desc_loss = pos_loss + neg_loss
	if method==2:	
		desc_loss = F.cross_entropy(desc_pairs, match_target.long().view(-1),
		 weight=torch.tensor([(Npos + 1) / (Npos + Nneg), (Nneg + 1) / (Npos + Nneg)]).to(device))
	if method==3:
		pos_loss = torch.sum(match_target * torch.max(torch.zeros_like(desc_pairs_norm).to(device), desc_pairs_norm - pos_margin)) / (2*Npos + 1)
		neg_loss = torch.sum((1.0 - match_target) * torch.max(torch.zeros_like(desc_pairs_norm).to(device), 1.0 - desc_pairs_norm)) / (2*Nneg + 1)
		# print("Npos: ", Npos.item(), "Nneg: ", Nneg.item(), "pos_loss: ", pos_loss.item(), "neg_loss: ", neg_loss.item())
		desc_loss1 = pos_loss + neg_loss	
		desc_loss2 = F.cross_entropy(desc_pairs, match_target.long().view(-1),
		 weight=torch.tensor([(Npos + 1) / (Npos + Nneg), (Nneg + 1) / (Npos + Nneg)]).to(device))
		desc_loss = desc_loss1 + desc_loss2

	# total loss
	loss = loss1 + kpt_loss1 + kpt_loss2 + desc_loss

	return loss, (loss1, kpt_loss1, kpt_loss2, desc_loss)


def main():
	run_params = parse_input_arguments()

	device = "cuda:{}".format(run_params["device"])
	out_dir, nepochs = run_params["out_dir"], run_params["nepochs"]
	lr, batchsize = run_params["lr"], run_params["batchsize"]
	width, depth = run_params["width"], run_params["depth"]
	image_size, image_depth, k = run_params["image_size"], run_params["image_depth"], run_params["k"]
	scale_factor = run_params["scale_factor"]
	method, pos_margin = run_params["method"], run_params["pos_margin"]
	
	out_dir_train = os.path.join(out_dir, "train")
	out_dir_val = os.path.join(out_dir, "val")
	out_dir_wts = os.path.join(out_dir, "weights")
	os.makedirs(out_dir_train, exist_ok=True)
	os.makedirs(out_dir_val, exist_ok=True)
	os.makedirs(out_dir_wts, exist_ok=True)
	writer = SummaryWriter(out_dir)

	# TODO
	root_dir = 'csv-path-containing-data-info' 
	
	initial_transform = Compose([
		CropDepthwise(p=1.0, crop_size=image_depth, crop_mode='random'),
		CropInplane(p=1.0, crop_size=image_size, crop_mode='random', border=50),
		ToTensorShape(p=1.0)
		])
	
	deformation = AnyOf([
					RandomTranslate3D(p=1.0, translation=((-0.1,0.1), (-0.1,0.1), (-0.1,0.1))),
					RandomScale3D(p=1.0, scale=((0.9, 1.1), (0.9, 1.1), (0.9, 1.1))),
					RandomRotate3D(p=1.0, rotation=((-20,20), (-20,20), (-20,20))),
					RandomElasticTransform3D(p=1.0, sigma=64),
					RandomElasticTransform3D(p=1.0, sigma=64)
					])

	augmentation = Compose([
				RandomBrightness(),
				RandomContrast()
				])


	train_dataset = LandmarkDataset(root_dir, initial_transform, deformation, augmentation=augmentation, is_training=True)
	print("Total training CTs: ", len(train_dataset))
	train_dataloader = DataLoader(train_dataset, batch_size=batchsize, num_workers=1, shuffle=True)

	# val_dataset = LandmarkDataset(root_dir, initial_transform, deformation, augmentation=None, is_training=False)
	# val_dataloader = DataLoader(val_dataset, batch_size=batchsize, num_workers=3, shuffle=False)
	
	model = Net(device=device, width=width, depth=depth).to(device)
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4, eps=1e-6, amsgrad=True)
	scheduler = optim.lr_scheduler.StepLR(optimizer, nepochs//3, gamma=0.1, last_epoch=-1)
	# # load weights
	# state_dict = torch.load(os.path.join("./runs/run_16bit/weights", "model_weights.pth"), map_location=device)["model"]
	# model.load_state_dict(state_dict)
	# print("weights loaded")

	model, optimizer = amp.initialize(model, optimizer)


	train_steps = 0
	val_steps = 0
	best_loss = 1000
	for epoch in range(0, nepochs):
		# training
		model.train()
		train_loss = 0.
		nbatches = 0
		for nbatches, (images1, images2, deformations, valid_mask1, valid_mask2) in enumerate(train_dataloader):
			print("valid mask: ", valid_mask1.sum().item())
			if valid_mask1.sum().item()<k:
				continue
			images1, images2, deformations, valid_mask1, valid_mask2 = images1.to(device), images2.to(device), deformations.to(device), valid_mask1.to(device), valid_mask2.to(device)

			keypoints1_scores, keypoints2_scores, keypoints1, keypoints2, desc_pairs, desc_pairs_norm = model(images1, images2, k=k, scale_factor=scale_factor)
			gt1, gt2, match_target = get_labels(keypoints1, keypoints2, deformations, valid_mask1, valid_mask2, device=device)
			loss, all_losses = CustomLoss(keypoints1_scores, keypoints2_scores, desc_pairs, desc_pairs_norm, gt1, gt2, match_target, k=k, device=device, method=method, pos_margin=pos_margin)

			optimizer.zero_grad()
			with amp.scale_loss(loss, optimizer) as scaled_loss:
				scaled_loss.backward()

			optimizer.step()
			train_loss += loss.item()
			print("Iteration {}: Train Loss: {} Max output: {} Min output: {}".format(nbatches, loss.item(), torch.max(keypoints1_scores), torch.min(keypoints2_scores)))
			print("loss1: {}, Keypoint1 Loss: {}, Keypoint2 Loss: {}, Descriptor loss: {}\n".format(*all_losses))
			writer.add_scalar("Loss/train_loss", loss.item(), train_steps)
			train_steps += 1

			if (nbatches % 20) == 0 or nbatches == len(train_dataloader)-1:
				output1, output2, output3 = model.predict(images1, images2, k=k, is_training=True, conf_thresh=0.1, desc_thresh=0.5, scale_factor=scale_factor, method=method)
				images1 = images1.data.cpu().numpy()
				images2 = images2.data.cpu().numpy()

				im1 = images1[0,0,:,:,:]
				im2 = images2[0,0,:,:,:]
				out1 = output1[0]
				out2 = output2[0]
				mask = output3[0]
				visualize_keypoints(im1.copy(), im2.copy(), out1, out2, mask, out_dir=out_dir_train, base_name="iter_{}".format(epoch))

		train_loss = train_loss / float(nbatches+1)
		print("EPOCH {}".format(epoch))
		writer.add_scalar("Epochwise_Loss/train_loss", train_loss, epoch)

		# saving model
		if epoch%1 == 0:
			weights = {"model": model.state_dict()}
			torch.save(weights, os.path.join(out_dir_wts, "model_%d.pth"%epoch))
		if train_loss < best_loss:
			best_loss = train_loss
			weights = {"model": model.state_dict(), "epoch": epoch}
			torch.save(weights, os.path.join(out_dir_wts, "model_weights.pth"))

		scheduler.step()


if __name__ == '__main__':
	main()


