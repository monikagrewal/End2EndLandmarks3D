import cv2
import torch
import torch.nn.functional as F

import os, argparse
import numpy as np
import json
from skimage.io import imread, imsave
import _pickle as pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import *
import sys
from custom_transforms3d import *
from etl3d import *
from utils3d import *
import pdb


def parse_input_arguments(out_dir):
	run_params = json.load(open(os.path.join(out_dir, "run_parameters.json"), "r"))
	return run_params


def custom_eval(model):
	for module in model.modules():
		if isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm2d):
			module.eval()


def flatten_list(lst):
	flat_lst = []
	for item in lst:
		flat_lst.extend(item)
	return flat_lst


def get_PCK_values(all_distances, pixel_thresholds=[]):
	total = len(all_distances)
	all_distances = np.array(all_distances)
	acc_list = list()
	for threshold in pixel_thresholds:
		correct = (all_distances <= threshold).sum()
		acc_list.append(correct / float(total))
		if threshold <= 4:
			print("landmark matches within {} mm: {} out of {}".format(threshold, correct, total))

	return acc_list


def plot_PCK(all_distances, pixel_thresholds, filename="PCK.jpg"):
	all_distances = flatten_list(all_distances)
	xpoints = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 392, 512]
	xticks = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512]

	colors = ["blue", "magenta", "green", "teal", "pink", "violet", "purple", "red", "cyan", "yellow"]

	PCK_values = get_PCK_values(all_distances, pixel_thresholds=pixel_thresholds)
	ypoints = [PCK_values[item]*100 for item in xpoints]
	plt.plot(np.arange(len(xpoints)), np.array(ypoints), label="random transformations", color=colors[0], linewidth=3)
	print(f"landmark distances: q25 = {np.percentile(all_distances, 25)}, q75 = {np.percentile(all_distances, 75)}, mean = {np.mean(all_distances)}, median = {np.percentile(all_distances, 50)}")
	print("")
	
	plt.legend(facecolor='white', edgecolor="0.7", loc='lower right')
	plt.grid(linewidth=1, linestyle="--", color='0.9', alpha=0.5)
	plt.xticks(np.arange(0, len(xpoints), 2), np.array(xticks), rotation=-30)
	plt.yticks(np.round(np.linspace(0, 100, 11), 1))
	plt.xlim(0, len(xpoints) - 1)
	plt.ylim(0, 100.1)
	plt.xlabel("Euclidean Distance (voxels)", weight="bold")
	plt.ylabel("Percentage of landmark pairs", weight="bold")
	plt.savefig(filename, dpi=600)
	plt.close()	


def visualize_keypoints(volume1, volume2, output1_list, output2_list, output1_projected_list, extent, out_dir="./sanity", base_name=""):
	os.makedirs(out_dir, exist_ok=True)
	spacing = (1, 1, 2)
	slices, h, w = volume1.shape
	volume1 = [cv2.cvtColor(volume1[i], cv2.COLOR_GRAY2RGB) for i in range(slices)]
	volume2 = [cv2.cvtColor(volume2[i], cv2.COLOR_GRAY2RGB) for i in range(slices)]

	org_distances = list()
	distances = list()
	for l1, l2, l1_projected in zip(output1_list, output2_list, output1_projected_list):
		w1, h1, d1 = l1
		color = (w1 / float(w), h1 / float(h), d1 / float(slices))
		w11, h11, d11 = l1_projected
		w2, h2, d2 = l2

		org_distances.append(extent[d2, h2, w2])
		distance = np.sqrt(((w1 - w11)*spacing[0])**2 + ((h1 - h11)*spacing[1])**2 + ((d1 - d11)*spacing[2])**2)
		distances.append(distance)
		expected_dist = np.sqrt(((w1 - w2)*spacing[0])**2 + ((h1 - h2)*spacing[1])**2 + ((d1 - d2)*spacing[2])**2)
		# print("{} --> {} : org distance: {}, expected distance: {}, matched distance: {}".format(l1, l2, extent[d2, h2, w2], expected_dist, distance))
		cv2.circle(volume1[d1], (w1, h1), 3, (1, 0, 0), -1)
		cv2.circle(volume2[d2], (w2, h2), 3, (0, 1, 0), -1)
		if d11 in range(slices):
			cv2.circle(volume1[d11], (w11, h11), 4, (0, 1, 0), 2)
			if d1 == d11:
				cv2.line(volume1[d1], (w1, h1), (w11, h11), (1, 1, 1), 1)


	imlist = []
	for i in range(slices):
		im = np.concatenate((volume1[i], volume2[i]), axis=0)
		imlist.append(im)
		if len(imlist)==4:
			im = np.concatenate(imlist, axis=1)
			skimage.io.imsave(os.path.join(out_dir, "{}_{}.jpg".format(base_name, i)), (im*255).astype(np.uint8))
			imlist = []

	return distances, org_distances


def plot_deformation(org_distances, pred_distances, filename="deformation_vs_landmarks.jpg"):
	nimages = len(org_distances)
	org_distances, pred_distances = flatten_list(org_distances), flatten_list(pred_distances)
	total = len(org_distances)
	thresh = 4
	org_distances = np.array(org_distances)
	pred_distances = np.array(pred_distances)
	xpoints = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

	pred_list = []
	actual_list = []
	for i in range(len(xpoints)-1):
		lb, ub = xpoints[i], xpoints[i+1]
		mask = (lb <= org_distances) * (org_distances < ub)
		actual = mask.sum()
		pred = (pred_distances[mask] <= thresh).sum()
		actual_list.append(int(actual*100//total))
		pred_list.append(int(pred*100//total))

	print("actual: ", actual_list)
	print("correct matches: ", pred_list)

	colors = ["blue", "magenta", "green", "teal", "pink", "violet", "purple", "red", "cyan", "yellow"]
	plt.bar(np.array(xpoints[:-1])+0.9, np.array(actual_list), width=1.8, label="Matched", color=colors[0])
	plt.bar(np.array(xpoints[:-1])+0.9, np.array(pred_list), width=1.8, label="Correct", color=colors[1])
	
	plt.legend(facecolor='white', edgecolor="0.7", loc='upper right')
	plt.grid(linewidth=1, linestyle="--", color='0.9', alpha=0.5)
	plt.xticks(np.array(xpoints), np.array(xpoints), rotation=-30)
	# plt.yticks(np.round(np.linspace(0, 100, 11), 1))
	plt.xlim(0, xpoints[-1])
	# plt.ylim(0, 100.1)
	plt.xlabel("Euclidean Distance (mm)", weight="bold")
	plt.ylabel("Percentage of landmarks", weight="bold")
	plt.savefig(filename, dpi=600)
	plt.close()


def calculate_deformation_extent(deformation, spacing=(1, 1, 2)):
	d, h, w, _ = deformation.shape
	deformation = deformation.data.cpu().numpy()
	deformation = (deformation + 1.) / 2.
	deformation = deformation * np.array([float(w-1), float(h-1), float(d-1)]).reshape(1, 1, 1, 3)

	z, y, x = np.meshgrid(np.arange(0, d), np.arange(0, h), np.arange(0, w), indexing="ij")
	indices = np.array([np.reshape(x, -1), np.reshape(y, -1), np.reshape(z, -1)]).T  #shape N, 3
	indices = indices.reshape(d, h, w, 3)
	extent = deformation - indices

	# convert to mm space
	extent = np.round(extent * np.array(spacing).reshape(1, 1, 1, 3), 0)
	extent = np.linalg.norm(extent, axis=3)
	return extent



def main(out_dir, device="cuda:0"):
	torch.manual_seed(1234)
	batchsize = 1

	run_params = parse_input_arguments(out_dir)
	width, depth = run_params["width"], run_params["depth"]
	image_size, image_depth, K = run_params["image_size"], run_params["image_depth"], run_params["k"]
	scale_factor = run_params.get("scale_factor", 8)
	method, pos_margin = run_params["method"], run_params.get("pos_margin", 0.1)

	out_dir_val = os.path.join(out_dir, "val")
	out_dir_wts = os.path.join(out_dir, "weights")
	os.makedirs(out_dir_val, exist_ok=True)

	# load weights
	model = Net(device=device, width=width, depth=depth).to(device)
	state_dict = torch.load(os.path.join(out_dir_wts, "model_weights.pth"), map_location=device)["model"]
	model.load_state_dict(state_dict)
	print("weights loaded")
	model.eval()
	custom_eval(model)
	
	initial_transform = Compose([
		# CropDepthwise(p=1.0, crop_size=image_depth, crop_mode='center'),
		# CropInplane(p=1.0, crop_size=image_size, crop_mode='center'),
		ToTensorShape(p=1.0)
		])
	
	deformation_dict = {"random": AnyOf([
					RandomTranslate3D(p=1.0, translation=((-0.1,0.1), (-0.1,0.1), (-0.1,0.1))),
					RandomScale3D(p=1.0, scale=((0.9, 1.1), (0.9, 1.1), (0.9, 1.1))),
					RandomRotate3D(p=1.0, rotation=((-20,20), (-20,20), (-20,20))),
					RandomElasticTransform3D(p=1.0, sigma=64),
					RandomElasticTransform3D(p=1.0, sigma=64)
					])}

	augmentation = Compose([
				RandomBrightness(),
				RandomContrast()
				])

	distance_info = dict()
	for deformation_name, deformation in deformation_dict.items():
		root_dir = '/export/scratch3/grewal/Data/Projects_JPG_data/ThreeD/MODIR_data_test_split/curated_test_info.csv'
		val_dataset = LandmarkDataset(root_dir, initial_transform, deformation, augmentation=augmentation, is_training=False)
		val_dataloader = DataLoader(val_dataset, batch_size=batchsize, num_workers=1, shuffle=False)
		print("Total CTs: ", len(val_dataset))
		
		all_distances = list()
		all_org_distances = list()
		nmatches = list()
		for nbatches, (images1, images2, deformations, _, _) in enumerate(val_dataloader):
			print(nbatches)
			extent = calculate_deformation_extent(deformations[0])
			images1, images2, deformations = images1.to(device), images2.to(device), deformations.to(device)
			deformations = deformations.permute(0, 4, 1, 2, 3)  #b, 3, d, h, w

			b, c, d, h, w = images1.shape
			overlap = 0.1
			stride = 1 - overlap
			dx = list(range(0, d-image_depth+1, int(image_depth*stride)))
			hx = list(range(0, h-image_size+1, int(image_size*stride)))
			wx = list(range(0, w-image_size+1, int(image_size*stride)))

			if dx[-1]+image_depth < d:
				dx.append(d - image_depth)
			if hx[-1]+image_size < h:
				hx.append(h - image_size)
			if wx[-1]+image_size < w:
				wx.append(w - image_size)
			print(dx, hx, wx)

			offset_list = [(slice(None), slice(None), slice(i, i+image_depth), slice(j, j+image_size), slice(k, k+image_size))
			 for i in dx for j in hx for k in wx]
			
			with torch.no_grad():
				output1 = []
				output2 = []
				output1_projected = []
				for offset in offset_list:
					im1 = images1[offset]
					im2 = images2[offset]
					print(offset, im1.shape)
					out1, out2, out3 = model.predict(im1, im2, k=K, scale_factor=scale_factor, conf_thresh=0.5, desc_thresh=0.5,
					 method=method, pos_margin=0.1)
					offset_coords = np.array((offset[4].start, offset[3].start, offset[2].start)).reshape(1, 3)
					print(offset_coords, out3.shape)

					# im1 = im1.data.cpu().numpy()[0, 0, :, :, :]
					# im2 = im2.data.cpu().numpy()[0, 0, :, :, :]
					# visualize_keypoints(im1.copy(), im2.copy(), out1, out2, out3, out1, out_dir=out_dir_val, base_name="im_{}_{}".format(nbatches, offset_coords))
					# only corresponding points
					rr, cc = np.where(out3[0]==1)
					print("matches: ", len(rr))
					if len(rr) > 0:
						out1 = out1[0][rr]
						out2 = out2[0][cc]
					
						out1 += offset_coords
						out2 += offset_coords
						out2_torch = convert_points_to_torch(out2, d, h, w, device=device)
						# project out2 points on image 1
						pts1_projected = F.grid_sample(deformations, out2_torch) #b, 3, 1, 1, k
						pts1_projected = pts1_projected.permute(0, 2, 3, 4, 1) #b, 1, 1, k, 3
						out1_projected = convert_points_to_image(pts1_projected, d, h, w)

						output1.extend(out1.tolist())
						output2.extend(out2.tolist())
						output1_projected.extend(out1_projected[0].tolist())
		
			images1 = images1.data.cpu().numpy()
			images2 = images2.data.cpu().numpy()

			im1 = images1[0,0,:,:,:]
			im2 = images2[0,0,:,:,:]
			distances, org_distances = visualize_keypoints(im1.copy(), im2.copy(), output1, output2, output1_projected, extent, out_dir=out_dir_val, base_name="iter_{}".format(nbatches))
			all_distances.append(distances)
			all_org_distances.append(org_distances)
			nmatches.append(len(distances))

			# if nbatches==49:
			# 	break

		pixel_thresholds = list(range(0, 513))
		distance_info[deformation_name] = {"nmatches": nmatches, "distances": all_distances, "org_distances": all_org_distances}
		filename = os.path.join(out_dir, "PCK_{}.jpg".format(deformation_name))
		plot_PCK(all_distances, pixel_thresholds, filename=filename)
		filename = os.path.join(out_dir, "deformation_{}.jpg".format(deformation_name))
		plot_deformation(all_org_distances, all_distances, filename=filename)

	pickle.dump(distance_info, open(os.path.join(out_dir, "distance_info.pkl"), "wb"))


if __name__ == '__main__':
	device="cuda:1"

	out_dirs = ["./runs/down/margin",
				"./runs/down/ce",
				"./runs/down/margin_ce"]
	for out_dir in out_dirs:
		main(out_dir, device=device)



