"""
Auxiliary functions.
"""
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt


def get_cost_mat(im_sz=28, device='cuda', dtype=torch.float32):
    partition = torch.linspace(0, 1, im_sz)
    couples = np.array(np.meshgrid(partition, partition)).T.reshape(-1, 2)
    x = np.array(list(itertools.product(couples, repeat=2)))
    x = torch.tensor(x, dtype=dtype, device=device)
    a = x[:, 0]
    b = x[:, 1]
    C = torch.linalg.norm(a - b, axis=1) ** 2
    return C.reshape((im_sz**2, -1))


def get_measures(replace_val=1e-5, from_uniform=False, device='cuda'):
    """
    Load measures and replace zeros with a small value.

    :param replace_val: value to replace zeros with
    :param from_uniform: if True, use uniform measure as the source
    :return: source and target measures
    """
    with open('mnist_digit_2.pt', 'rb') as handle1, \
            open('mnist_digit_5.pt', 'rb') as handle2:
        mu = torch.load(handle1).to(device)
        nu = torch.load(handle2).to(device)

    if from_uniform:
        mu = torch.ones_like(nu)
        mu /= mu.sum()

    mu = replace_zeros(mu, replace_val=replace_val)
    nu = replace_zeros(nu, replace_val=replace_val)

    return mu, nu


def show_measure(measure, fname, im_sz=28, title=None, vmax=None):
    """
    Display measure as an image.

    :param measure: torch tensor containig a measure
    :param fname: file name for the image
    :param im_sz: image size (side length)
    :param title: title to display on the image
    :param vmax: if not None, colormap will be adjusted to the scale between 0 and vmax
    """
    img = measure.cpu().numpy().reshape(im_sz, -1)
    if vmax is None:
        plt.imshow(img, cmap='gray_r', vmin=0)
    else:
        plt.imshow(img, cmap='gray_r', vmin=0, vmax=vmax)
    plt.xticks([])
    plt.yticks([])
    if title:
        plt.title(title)
    plt.savefig(f"plots/{fname}.png", bbox_inches='tight')
    plt.close()


def replace_zeros(arr, replace_val=1e-5):
    """Replace zeros with [replace_val] and normalize."""
    arr[arr < replace_val] = replace_val
    arr /= arr.sum(dim=-1, keepdim=True)
    return arr


if __name__ == '__main__':
    pass
