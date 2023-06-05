import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torchvision.models as models
import numpy as np
import cv2, os
import colorsys
import skimage
import functools
import sys
sys.path.append(".")
from unet import *
import pdb;


def convert_points_to_image(samp_pts, d, H, W):
	"""
	Inputs:-
	samp_pts: b, 1, 1, k, 3
	"""

	b, _, _, K, _ = samp_pts.shape
	# Convert pytorch -> numpy.
	samp_pts = samp_pts.data.cpu().numpy().reshape(b, K, 3)
	samp_pts = (samp_pts + 1.) / 2.
	samp_pts = np.round(samp_pts * np.array([float(W-1), float(H-1), float(d-1)]).reshape(1, 1, 3), 0)
	return samp_pts.astype(np.int32)


def NMS3d(pts_list, probs, siz=(8,8,8), overlapThresh=0.1, k=512):
    # if there are no boxes, return an empty list
    if len(pts_list) == 0:
        return np.array(pts_list).astype("float")

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    pts_list = np.array(pts_list).astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = pts_list[:, 0] - siz[0]/2
    y1 = pts_list[:, 1] - siz[1]/2
    z1 = pts_list[:, 2] - siz[2]/2
    x2 = pts_list[:, 0] + siz[0]/2
    y2 = pts_list[:, 1] + siz[1]/2
    z2 = pts_list[:, 2] + siz[2]/2

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)

    # sort the indexes
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0 and len(pick)<k:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        zz1 = np.maximum(z1[i], z1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        zz2 = np.minimum(z2[i], z2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        d = np.maximum(0, zz2 - zz1 + 1)

        # compute the ratio of overlap
        overlap = (w * h * d) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return pts_list[pick].astype("float")


def sampling_layer(heatmaps, coarse_desc, conf_thresh=0.1, k=100, device="cuda:0", is_training=True, scale_factor=8):
	# pdb.set_trace()
	b, _, d, H, W = heatmaps.shape
	heatmaps = torch.sigmoid(heatmaps)
	if is_training:
		heatmaps1, indices = F.max_pool3d(heatmaps, (scale_factor//2, scale_factor, scale_factor),
		 stride=(scale_factor//2, scale_factor, scale_factor), return_indices=True)
		heatmaps1 = F.max_unpool3d(heatmaps1, indices, (scale_factor//2, scale_factor, scale_factor),
			stride=(scale_factor//2, scale_factor, scale_factor), output_size=heatmaps.shape)
		heatmaps1 = heatmaps1.to("cpu").detach().numpy().reshape(b, d, H, W)
	else:
		heatmaps1 = heatmaps.to("cpu").detach().numpy().reshape(b, d, H, W)

	all_pts= []
	for heatmap in heatmaps1:
		xs, ys, zs = np.where(heatmap >= conf_thresh) # Confidence threshold.
		print("original points: ", len(xs))
		pts = np.array([zs, ys, xs]).T
		scores = heatmap[xs, ys, zs]
		inds = np.argsort(scores)[::-1]
		pts = pts[inds[:k]]
		if is_training:
			if len(pts) < k:
				randz = np.random.randint(0, W, k-len(pts))
				randy = np.random.randint(0, H, k-len(pts))
				randx = np.random.randint(0, d, k-len(pts))

				pts_rand = np.array([randz, randy, randx]).T
				if len(pts)>0:
					pts = np.concatenate((pts, pts_rand), axis=0)
				else:
					pts = pts_rand
		# Interpolate into descriptor map using 2D point locations.
		samp_pts = torch.from_numpy(pts.astype(np.float32))
		samp_pts[:, 0] = (samp_pts[:, 0] * 2. / (W-1)) - 1.
		samp_pts[:, 1] = (samp_pts[:, 1] * 2. / (H-1)) - 1.
		samp_pts[:, 2] = (samp_pts[:, 2] * 2. / (d-1)) - 1.
		samp_pts = samp_pts.contiguous()
		samp_pts = samp_pts.view(1, 1, 1, -1, 3)
		samp_pts = samp_pts.float().to(device)
		all_pts.append(samp_pts)

	all_pts = torch.cat(all_pts, dim=0)
	pts_score = F.grid_sample(heatmaps, all_pts)  #b, 1, 1, 1, k
	pts_score = pts_score.permute(0, 4, 1, 2, 3).view(b, -1)  #b, k
	desc = [F.grid_sample(desc, all_pts) for desc in coarse_desc]
	desc = torch.cat(desc, dim=1)

	return all_pts, pts_score, desc


def weight_init(m):
	if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
		torch.nn.init.kaiming_normal_(m.weight.data)
		if m.bias is not None:
			m.bias.data.fill_(0.0)
	if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
		m.weight.data.fill_(1.0)
		m.bias.data.fill_(0.0)
	if isinstance(m, nn.Linear):
		torch.nn.init.kaiming_normal_(m.weight.data)
		if m.bias is not None:
			m.bias.data.fill_(0.0)


class DescMatchingLayer(nn.Module):
	"""
	DescMatchingLayer
	"""
	def __init__(self, in_channels, out_channels):
		super(DescMatchingLayer, self).__init__()
		self.fc = nn.Linear(in_channels, out_channels)
		self.apply(weight_init)


	def forward(self, out1, out2):
		b, c, d1, h1, w1 = out1.size()
		b, c, d2, h2, w2 = out2.size()
		out1 = out1.view(b, c, d1*h1*w1).permute(0, 2, 1).view(b, d1*h1*w1, 1, c)
		out2 = out2.view(b, c, d2*h2*w2).permute(0, 2, 1).view(b, 1, d2*h2*w2, c)

		out = out1 * out2
		out = out.contiguous().view(-1, c)

		out = self.fc(out)
		
		# normalize input features
		dn1 = torch.norm(out1, p=2, dim=3) # Compute the norm.
		out1 = out1.div(1e-6 + torch.unsqueeze(dn1, 3)) # Divide by norm to normalize.
		dn2 = torch.norm(out2, p=2, dim=3) # Compute the norm.
		out2 = out2.div(1e-6 + torch.unsqueeze(dn2, 3)) # Divide by norm to normalize.

		out_norm = torch.norm(out1 - out2, p=2, dim=3)
		return out, out_norm


class Net(nn.Module):
	"""
	What follows is awesomeness redefined
	"""
	def __init__(self, depth=3, width=16, in_channels=1, out_channels=2, batchnorm=True, device="cuda:0"):
		super().__init__()
		self.device = device
		self.encoder = UNet(depth=depth, width=width, growth_rate=2, in_channels=in_channels, out_channels=1, threeD=True, batchnorm=batchnorm)
		feature_channels = self.encoder.feature_channels

		self.desc_matching_layer = DescMatchingLayer(feature_channels, out_channels)


	def forward(self, x1, x2, k=100, scale_factor=4):
		# encoding
		heatmaps1, features1 = self.encoder(x1)
		heatmaps2, features2 = self.encoder(x2)

		keypoints1, keypoints1_scores, desc1 = sampling_layer(heatmaps1, features1, k=k, device=self.device, scale_factor=scale_factor)
		keypoints2, keypoints2_scores, desc2 = sampling_layer(heatmaps2, features2, k=k, device=self.device, scale_factor=scale_factor)

		# descriptors
		desc_pairs, desc_pairs_norm = self.desc_matching_layer(desc1, desc2)

		return keypoints1_scores, keypoints2_scores, keypoints1, keypoints2, desc_pairs, desc_pairs_norm


	def predict(self, x1, x2, deformation=None, conf_thresh=0.1, k=100, desc_thresh=0.5, is_training=False,
	 scale_factor=8, method=1, pos_margin=0.1):
		b, _, d, H, W = x1.shape
		# encoding
		heatmaps1, features1 = self.encoder(x1)
		heatmaps2, features2 = self.encoder(x2)

		pts1, _, desc1 = sampling_layer(heatmaps1, features1, conf_thresh=conf_thresh, k=k, device=self.device, is_training=is_training, scale_factor=scale_factor)
		pts2, _, desc2 = sampling_layer(heatmaps2, features2, conf_thresh=conf_thresh, k=k, device=self.device, is_training=is_training, scale_factor=scale_factor)

		# descriptors
		desc_pairs, desc_pairs_norm = self.desc_matching_layer(desc1, desc2)

		# post processing
		keypoints1 = convert_points_to_image(pts1, d, H, W)
		keypoints2 = convert_points_to_image(pts2, d, H, W)

		b, k1, _ = keypoints1.shape
		_, k2, _ = keypoints2.shape
		if k1==0 or k2==0:
			matches = np.zeros((b, k1, k2))
		else:
			desc_pairs = F.softmax(desc_pairs, dim=1)[:,1].view(b, k1, k2)

			# two-way matching
			desc_pairs = desc_pairs.data.cpu().numpy()
			desc_pairs_norm = desc_pairs_norm.detach().to("cpu").numpy()
			matches = list()
			for i in range(b):
				pairs = desc_pairs[i]
				pairs_norm = desc_pairs_norm[i]

				match_cols = np.zeros((k1, k2))
				match_cols[np.argmax(pairs, axis=0), np.arange(k2)] = 1
				match_rows = np.zeros((k1, k2))
				match_rows[np.arange(k1), np.argmax(pairs, axis=1)] = 1
				match_desc = match_rows * match_cols

				match_cols = np.zeros((k1, k2))
				match_cols[np.argmin(pairs_norm, axis=0), np.arange(k2)] = 1
				match_rows = np.zeros((k1, k2))
				match_rows[np.arange(k1), np.argmin(pairs_norm, axis=1)] = 1
				match_norm = match_rows * match_cols
				match = match_desc * match_norm
				
				if method==1:
					if pos_margin>0:
						mask = (pairs_norm <= pos_margin).astype('float')
						matches.append(match_norm * mask)
					else:
						matches.append(match_norm)
				if method==2:
					mask = (pairs >= desc_thresh).astype('float')
					matches.append(match_desc * mask)
				if method==3:
					matches.append(match)

			matches = np.array(matches)

		if deformation is not None:
			deformation = deformation.permute(0, 4, 1, 2, 3)  #b, 3, d, h, w
			pts1_projected = F.grid_sample(deformation, pts2) #b, 3, 1, 1, k
			pts1_projected = pts1_projected.permute(0, 2, 3, 4, 1) #b, 1, 1, k, 3
			keypoints1_projected = convert_points_to_image(pts1_projected, d, H, W)
			return keypoints1, keypoints2, matches, keypoints1_projected
		else:
			return keypoints1, keypoints2, matches


	def weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
			torch.nn.init.kaiming_normal_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)
		if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1.0)
			m.bias.data.fill_(0.0)
		if isinstance(m, nn.Linear):
			torch.nn.init.kaiming_normal_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)




def visualize_keypoints(volume1, volume2, output1, output2, mask, out_dir="./sanity", base_name=0):
	os.makedirs(out_dir, exist_ok=True)
	slices, h, w = volume1.shape
	volume1 = [cv2.cvtColor(volume1[i], cv2.COLOR_GRAY2RGB) for i in range(slices)]
	volume2 = [cv2.cvtColor(volume2[i], cv2.COLOR_GRAY2RGB) for i in range(slices)]

	for k1, l1 in enumerate(output1):
		x1, y1, z1 = l1
		color = (x1 / float(w), y1 / float(h), z1 / float(slices))
		# print(x1, y1, z1)
		if z1 in range(slices):
			cv2.circle(volume1[z1], (x1, y1), 3, color, -1)
		for k2, l2 in enumerate(output2):
			x2, y2, z2 = l2
			if mask[k1, k2] == 1 and z2 in range(slices):
				cv2.circle(volume1[z2], (x2, y2), 3, color, -1)

	imlist = []
	for i in range(slices):
		im = np.concatenate((volume1[i], volume2[i]), axis=1)
		imlist.append(im)
		if len(imlist)==4:
			im = np.concatenate(imlist, axis=0)
			skimage.io.imsave(os.path.join(out_dir, "im_{}_{}.jpg".format(base_name, i)), (im*255).astype(np.uint8))
			imlist = []


if __name__ == '__main__':
	device = "cuda:0"
	batchsize = 1
	image_depth, image_size = 48, 128
	model = Net(device=device).to(device)
	for j in range(1):
		inputs1 = torch.rand(batchsize, 1, image_depth, image_size, image_size).to(device)
		inputs2 = torch.rand(batchsize, 1, image_depth, image_size, image_size).to(device)
		keypoints1_scores, keypoints2_scores, keypoints1, keypoints2, desc_pairs = model(inputs1, inputs2)

		print(keypoints1_scores.shape, keypoints2_scores.shape, keypoints1.shape, keypoints2.shape, desc_pairs.shape)

		inputs1 = inputs1.data.cpu().numpy()
		inputs2 = inputs2.data.cpu().numpy()
		output1 = convert_points_to_image(keypoints1, image_depth, image_size, image_size)
		output2 = convert_points_to_image(keypoints2, image_depth, image_size, image_size)
		b, k, _ = output1.shape
		output3 = torch.argmax(desc_pairs, dim=1).view(b, k, k).to("cpu").numpy()
		
		for i in range(inputs1.shape[0]):
			im1 = inputs1[i,0,:,:,:]
			im2 = inputs2[i,0,:,:,:]
			out1 = output1[i]
			out2 = output2[i]
			mask = output3[i]
			print(out1[:, 2])
			visualize_keypoints(im1.copy(), im2.copy(), out1, out2, mask, base_name=j)
