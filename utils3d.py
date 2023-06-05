import torch
import numpy as np

import pdb


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


def convert_points_to_torch(pts, d, H, W, device="cuda:0"):
    """
    Inputs:-
    pts: k, 3 (W, H, d)
    """

    samp_pts = torch.from_numpy(pts.astype(np.float32))
    samp_pts[:, 0] = (samp_pts[:, 0] * 2. / (W-1)) - 1.
    samp_pts[:, 1] = (samp_pts[:, 1] * 2. / (H-1)) - 1.
    samp_pts[:, 2] = (samp_pts[:, 2] * 2. / (d-1)) - 1.
    samp_pts = samp_pts.view(1, 1, 1, -1, 3)
    samp_pts = samp_pts.float().to(device)
    return samp_pts

