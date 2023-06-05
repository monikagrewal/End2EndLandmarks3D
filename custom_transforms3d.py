import torch
import torch.nn.functional as F
import numpy as np
import skimage
from scipy.ndimage.filters import gaussian_filter



def create_affine_matrix(rotation=(0, 0, 0), scale=(1, 1, 1), shear=0, translation=(0, 0, 0), center=(0, 0, 0)):
	"""
	Inputs: 
	rotation: 3 rotation angles in degrees
	scale: 3 scales
	shear: 3 shear values (not incorporated in the matrix yet.)
	translation: 3 translation in voxels
	"""
	theta_x, theta_y, theta_z = rotation
	theta_x *= np.pi/180
	theta_y *= np.pi/180
	theta_z *= np.pi/180

	Rscale = np.array([[scale[0], 0, 0, 0],
						[0, scale[1], 0, 0],
						[0, 0, scale[2], 0],
						[0, 0, 0, 1]])

	Rx = np.array([[1, 0, 0, 0],
				[0, np.cos(theta_x), -np.sin(theta_x), 0],
				[0, np.sin(theta_x), np.cos(theta_x), 0],
				[0, 0, 0, 1]])

	Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y), 0],
				[0, 1, 0, 0],
				[-np.sin(theta_y), 0, np.cos(theta_y), 0],
				[0, 0, 0, 1]])

	Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0, 0],
				[np.sin(theta_z), np.cos(theta_z), 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1]])

	affine_matrix = np.matmul(Rscale, np.matmul(Rz, np.matmul(Ry, Rx)))
	center = np.array(center).reshape(1, -1)
	center_homogenous = np.array([center[0, 0], center[0, 1], center[0, 2], 1]).reshape(1, -1)
	center_rotated = np.dot(center_homogenous, affine_matrix)

	translation = np.array(translation).reshape(1, -1)
	translation_homogenous = np.array([translation[0, 0], translation[0, 1], translation[0, 2], 1]).reshape(1, -1)
	translation_rotated = np.dot(translation_homogenous, affine_matrix)

	affine_matrix[3, :3] = center.flatten() - center_rotated.flatten()[:3] + translation_rotated.flatten()[:3]
	return affine_matrix


def rand_float_in_range(min_value, max_value):
	return (torch.rand((1,)).item() * (max_value - min_value)) + min_value


def random_dvf(shape, grid, sigma=None, alpha=None):
	"""
	Helper function for RandomElasticTransform3D class
	generates random dvf along given axis

	"""
	if sigma is None:
		sigma = rand_float_in_range(max(shape)//8, max(shape)//4)
	else:
		sigma = rand_float_in_range(sigma//2, sigma)

	if alpha is None:
		alpha = rand_float_in_range(0.01, 0.1)
	else:
		alpha = rand_float_in_range(0.01, alpha)

	g = gaussian_filter(torch.rand(*shape).numpy(), sigma, cval=0)
	g = ( (g / g.max()) * 2 - 1 ) * alpha
	return g


def random_gaussian(shape, grid, sigma=None, alpha=None):
	"""
	Helper function for RandomElasticTransform3D class
	generates random gaussian field along given axis

	"""
	if sigma is None:
		sigma = rand_float_in_range(shape//8, shape//4)
	else:
		sigma = rand_float_in_range(sigma//2, sigma)

	if alpha is None:
		alpha = rand_float_in_range(-0.2, 0.2)
	else:
		alpha = rand_float_in_range(-alpha, alpha)

	if abs(alpha) < 0.02:
		alpha = 0.02

	center = rand_float_in_range(-0.99, 0.99)
	g = alpha * np.exp(-((grid * shape - center)**2 / (2.0 * sigma**2)))

	# smoothing on corners. for visualization figure only. comment otherwise
	sigma = shape//8
	g_center = np.exp(-((grid * shape - center)**2 / (2.0 * sigma**2)))
	g *= g_center
	return g


class Compose(object):
	"""Composes several transforms in sequence

	Args:
		transforms (list of ``Transform`` objects): list of transforms to compose.

	"""

	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, img, target=None):
		for t in self.transforms:
			img, target, indices = t(img, target)

		return img, target, indices

	def __repr__(self):
		format_string = self.__class__.__name__ + '('
		for t in self.transforms:
			format_string += '\n'
			format_string += '    {0}'.format(t)
		format_string += '\n)'
		return format_string


class AnyOf(Compose):
	"""Composes several transforms such that only one is applied at one time

	Args:
		transforms (list of ``Transform`` objects): list of transforms to compose.

	"""

	def __init__(self, transforms):
		super().__init__(transforms)

	def __call__(self, img, target=None):
		selected = torch.randint(len(self.transforms), (1,)).item()
		t = self.transforms[selected]
		outputs = t(img, target)
		return outputs


class CustomTransform(object):
	"""
	Base class for all other transforms

	Args:
		p (float): probability of applying the transform
	"""

	def __init__(self, p=0.5):
		self.p = p

	def __repr__(self):
		return self.__class__.__name__ + '(p={})'.format(self.p)


class ToTensorShape(CustomTransform):
	"""
	adds channel dimension to numpy array (required for tarining)

	Args:
		p (float):  probability of applying the transform
	"""

	def __init__(self, p=0.5):
		super().__init__(p)

	def __call__(self, img, target=None):
		if len(img.shape)!=3:
			raise ValueError("Expected number of dimensions for 3D = 3, got {}".format(len(img.shape)))
		else:
			img = np.expand_dims(img, axis=0)
			if target is not None:
				target = np.expand_dims(target, axis=0)

		return img, target, None



class AffineTransform3D(CustomTransform):
	"""
	3D affine transformation on numpy image and target (optional)

	Args:
		p (float):  probability of applying the transform
	"""

	def __init__(self, p=0.5, rotation=((0,0), (0,0), (0,0)), scale=((1,1), (1,1), (1,1)), shear=(0, 0), translation=((0,0), (0,0), (0,0))):
		super().__init__(p)
		self.rotation = rotation
		self.scale = scale
		self.shear = shear
		self.translation = translation

	def __call__(self, img, target=None):
		theta_x = rand_float_in_range(*self.rotation[0])
		theta_y = rand_float_in_range(*self.rotation[1])
		theta_z = rand_float_in_range(*self.rotation[2])
		rotation = (theta_x, theta_y, theta_z)       
		
		sx = rand_float_in_range(*self.scale[0])
		sy = rand_float_in_range(*self.scale[1])
		sz = rand_float_in_range(*self.scale[2])
		scale = (sx, sy, sz)
		
		shear = rand_float_in_range(*self.shear)

		tx = rand_float_in_range(*self.translation[0])
		ty = rand_float_in_range(*self.translation[1])
		tz = rand_float_in_range(*self.translation[2])
		translation = (tx, ty, tz)

		outputs = self.affine_transform(img, target, rotation=rotation, scale=scale, shear=shear, translation=translation)
		return outputs


	def affine_transform(self, img, target=None, rotation=(0, 0, 0), scale=1, shear=0, translation=(0, 0, 0)):
		if torch.rand((1,)).item() <= self.p:
			c, d, h, w = img.shape
			x, y, z = np.meshgrid(np.linspace(-1, 1, d), np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
			indices = np.array([np.reshape(z, -1), np.reshape(y, -1), np.reshape(x, -1), np.ones(np.prod(img.shape))]).T  #shape N, 4

			M = create_affine_matrix(rotation=rotation, scale=scale, shear=shear, translation=translation)
			indices = np.dot(indices, M)
			# normalized grid for pytorch
			indices = indices[:, :3].reshape(d, h, w, 3)

			img = F.grid_sample(torch.tensor(img).view(1, c, d, h, w),
								torch.tensor(indices).view(1, d, h, w, 3), mode="nearest")
			img = img.numpy().reshape(c, d, h, w)

			if target is not None:
				target = F.grid_sample(torch.tensor(target).view(1, c, d, h, w),
									torch.tensor(indices).view(1, d, h, w, 3), mode="nearest")
				target = target.numpy().reshape(c, d, h, w)

			outputs = (img, target, indices)
		else:
			outputs = (img, target, None)

		return outputs


class RandomTranslate3D(AffineTransform3D):
	"""
	Random translation

	Args:
		p (float):  probability of applying the transform
	"""

	def __init__(self, p=0.5, translation=((-0.1,0.1), (-0.1,0.1), (-0.1,0.1))):
		super().__init__(p=p, translation=translation)


class RandomRotate3D(AffineTransform3D):
	"""
	Random rotation

	Args:
		p (float):  probability of applying the transform
	"""

	def __init__(self, p=0.5, rotation=((0,0), (0,0), (-30,30))):
		super().__init__(p=p, rotation=rotation)


class RandomScale3D(AffineTransform3D):
	"""
	Random scaling

	Args:
		p (float):  probability of applying the transform
	"""

	def __init__(self, p=0.5, scale=((0.8, 1.2), (0.8, 1.2), (0.95, 1.1))):
		super().__init__(p=p, scale=scale)


class RandomShear3D(AffineTransform3D):
	"""
	Random shear. Not tested yet.

	Args:
		p (float):  probability of applying the transform
	"""

	def __init__(self, p=0.5, shear=(-20, 20)):
		super().__init__(p=p, shear=shear)



class RandomBrightness(CustomTransform):
	"""
	Random brightness

	Args:
		p (float):  probability of applying the transform
	"""

	def __init__(self, p=0.5, rel_addition_range=(-0.2,0.2), return_deformation=False):
		super().__init__(p)
		self.rel_addition_range = rel_addition_range
		self.return_deformation = return_deformation

	def __call__(self, img, target=None):
		"""
		Args:
			img (Numpy Array): image to be transformed.
			target (Numpy Array): optional target image to apply the same transformation to

		Returns:
			Numpy Array: Randomly transformed image.
		"""
		if torch.rand((1,)).item() <= self.p:
			rel_addition = rand_float_in_range(*self.rel_addition_range)
			high = np.max(img)
			addition = rel_addition * high
			mask = (img >= .01) * (img <= high - .01)
			img[mask] = img[mask] + addition
			img[img > 1] = 1
			img[img < 0] = 0
		
		if not self.return_deformation:
			outputs = (img, target, None)
		else:
			c, d, h, w = img.shape
			x, y, z = np.meshgrid(np.linspace(-1, 1, d), np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
			indices = np.array([np.reshape(z, -1), np.reshape(y, -1), np.reshape(x, -1)]).T  #shape N, 3
			indices = indices.reshape(d, h, w, 3)
			outputs = (img, target, indices)
		return outputs


class RandomContrast(CustomTransform):
	"""
	Random contrast

	Args:
		p (float): 
	"""

	def __init__(self, p=0.5, contrast_mult_range=(0.8,1.2), return_deformation=False):
		super().__init__(p)
		self.contrast_mult_range = contrast_mult_range
		self.return_deformation = return_deformation

	def __call__(self, img, target=None):
		"""
		Args:
			img (Numpy Array): image to be transformed.
			target (Numpy Array): optional target image to apply the same transformation to

		Returns:
			Numpy Array: Randomly transformed image.
		"""
		if torch.rand((1,)).item() <= self.p:
			multiplier = rand_float_in_range(*self.contrast_mult_range)
			high = np.max(img)
			mask = (img >= .01) * (img <= high - .01)
			img[mask] = img[mask] * multiplier
			img[img > 1] = 1
			img[img < 0] = 0

		if not self.return_deformation:
			outputs = (img, target, None)
		else:
			c, d, h, w = img.shape
			x, y, z = np.meshgrid(np.linspace(-1, 1, d), np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
			indices = np.array([np.reshape(z, -1), np.reshape(y, -1), np.reshape(x, -1)]).T  #shape N, 3
			indices = indices.reshape(d, h, w, 3)
			outputs = (img, target, indices)
		return outputs


class RandomElasticTransform3D(CustomTransform):
	"""
	Random Elastic transformation

	Args:
		p (float): 
	"""

	def __init__(self, p=0.75, alpha=None, sigma=None, mode="nearest"):
		super().__init__(p)
		self.alpha = alpha
		self.sigma = sigma
		self.mode = mode

	def __call__(self, img, target=None):
		"""
		Args:
			img (Numpy Array): image to be transformed.
			target (Numpy Array): optional target image to apply the same transformation to

		Returns:
			Numpy Array: Randomly transformed image.
		"""
		if torch.rand((1,)).item() <= self.p:
			c, d, h, w = img.shape
			x, y, z = np.meshgrid(np.linspace(-1, 1, d), np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')

			small_sigma, small_alpha = None, None
			if self.sigma is not None:
				small_sigma = self.sigma // 4
			if self.alpha is not None:
				small_alpha = self.alpha // 2
			dz = random_dvf(img.shape, z, sigma=small_sigma, alpha=small_alpha) + random_gaussian(d, z, self.sigma, self.alpha)
			dx = random_dvf(img.shape, x, sigma=small_sigma, alpha=small_alpha) + random_gaussian(w, x, self.sigma, self.alpha)
			dy = random_dvf(img.shape, y, sigma=small_sigma, alpha=small_alpha) + random_gaussian(h, y, self.sigma, self.alpha)
			indices = np.array([np.reshape(z+dz, -1), np.reshape(y+dy, -1), np.reshape(x+dx, -1)]).T

			# normalized grid for pytorch
			indices = indices.reshape(d, h, w, 3)

			img = F.grid_sample(torch.tensor(img).view(1, c, d, h, w),
								torch.tensor(indices).view(1, d, h, w, 3), mode=self.mode)
			img = img.numpy().reshape(c, d, h, w)

			if target is not None:
				target = F.grid_sample(torch.tensor(target).view(1, c, d, h, w),
									torch.tensor(indices).view(1, d, h, w, 3), mode=self.mode)
				target = target.numpy().reshape(c, d, h, w)

			outputs = (img, target, indices)
		else:
			outputs = (img, target, None)
		return outputs


class CropDepthwise(CustomTransform):
	"""
	Crop along first dimension

	Args:
		p (float):  probability of applying the transform

	Todo: 
		Possibly throw an error when depth is smaller than crop_size? 
	"""

	def __init__(self, p=1.0, crop_mode="random", crop_size=16):
		super().__init__(p)
		self.crop_mode = crop_mode
		self.crop_size = crop_size
		self.crop_dim = 0

	def __call__(self, img, target=None):
		"""
		Args:
			img (Numpy Array): image to be transformed.
			target (Numpy Array): optional target image to apply the same transformation to

		Returns:
			Numpy Array: Randomly transformed image.
		"""
		if torch.rand((1,)).item() <= self.p:
			crop_dim = self.crop_dim
			if img.shape[crop_dim] < self.crop_size:
				pad = self.crop_size - img.shape[crop_dim]
				pad_tuple = tuple([( int(np.floor(pad/2)), int(np.ceil(pad/2)) ) if i == crop_dim else (1, 0) for i in range(len(img.shape))])
				img = np.pad(img, pad_tuple, mode="constant")
				target = np.pad(target, pad_tuple, mode="constant")

			if self.crop_mode == 'random':
				start_idx = np.random.choice(list(range(0, img.shape[crop_dim] - self.crop_size + 1)), 1)[0]
				end_idx = start_idx + self.crop_size
			elif self.crop_mode == 'center':
				start_idx = int((img.shape[crop_dim] / 2) - (self.crop_size/2))
				end_idx = start_idx + self.crop_size
			elif self.crop_mode =='none':
				start_idx = 0
				end_idx = img.shape[crop_dim]

			slice_tuple = tuple([slice(start_idx, end_idx) if i == crop_dim else slice(None) for i in range(len(img.shape))])
			img = img[slice_tuple]
			if target is not None:
				target = target[slice_tuple]

		outputs = (img, target, None)
		return outputs



class CropInplane(CustomTransform):
	"""
	Cropping in plane

	Args:
		p (float):  probability of applying the transform

	assumes img axes: depth * in-plane axis 0 * in-plane axis 1 
	"""

	def __init__(self, p=1.0, crop_mode="center", crop_size=384, border=80):
		super().__init__(p)
		self.crop_mode = crop_mode
		self.crop_size = crop_size
		self.crop_dim = [1, 2]
		self.border = border


	def __call__(self, img, target=None):
		"""
		Args:
			img (Numpy Array): image to be transformed.
			target (Numpy Array): optional target image to apply the same transformation to

		Returns:
			Numpy Array: transformed image (and optionally target).
		"""
		if torch.rand((1,)).item() <= self.p:
			crop_dim = self.crop_dim
			if self.crop_mode == 'random':
				try:
					start_idx = np.random.choice(list(range(self.border, img.shape[crop_dim[0]] - self.border - self.crop_size + 1)), 1)[0]
				except Exception as e:
					print("something got wrong. inspect the following error message and fix.")
					print(e)
					import pdb; pdb.set_trace()
				end_idx = start_idx + self.crop_size
			elif self.crop_mode == 'center':
				start_idx = int((img.shape[crop_dim[0]] / 2) - (self.crop_size/2))
				end_idx = start_idx + self.crop_size


			slice_tuple = tuple([slice(start_idx, end_idx) if i in crop_dim else slice(None) for i in range(len(img.shape))])
			img = img[slice_tuple]
			if target is not None:
				target = target[slice_tuple]

		outputs = (img, target, None)
		return outputs



class CustomResize(CustomTransform):
	"""
	resize in plane

	Args:
		p (float):  probability of applying the transform

		Currently assumes img axes: depth * in-plane axis 0 * in-plane axis 1 
	"""

	def __init__(self, p=1.0, output_size=384):
		super().__init__(p)
		self.output_size = output_size


	def __call__(self, img, target=None):
		"""
		Args:
			img (Numpy Array): image to be transformed.
			target (Numpy Array): optional target image to apply the same transformation to

		Returns:
			Numpy Array: transformed image (and optionally target).
		"""
		if torch.rand((1,)).item() <= self.p:
			nslices = img.shape[0]
			new_im = np.zeros((nslices, self.output_size, self.output_size))
			for i in range(nslices):
				new_im[i, :, :] = skimage.transform.resize(img[i, :, :], (self.output_size, self.output_size), mode='constant')

			img = new_im

			if target is not None:
				new_target = np.zeros((nslices, self.output_size, self.output_size))
				for i in range(nslices):
					new_target[i, :, :] = skimage.transform.resize(target[i, :, :], (self.output_size, self.output_size), mode='constant', order=0)

				target = new_target

		outputs = (img, target, None)
		return outputs



if __name__ == '__main__':
	img1 = np.random.random((30, 512, 512))
	to_tensor = ToTensorShape(p=1.0)
	transform = AnyOf([
						RandomRotate3D(p=1.0),
						# RandomElasticTransform3D(p=1.0)
						])

	augment = Compose([
				RandomBrightness(),
				RandomContrast()
				])                


	print(transform)
	print(augment)
	img1, _, _ = to_tensor(img1)
	img2, _, deformation = transform(img1)

	img1, _, _ = augment(img1)
	img2, _, _ = augment(img2)

	print(type(img1), type(img2), type(deformation))
	print(img2.shape, deformation.shape)