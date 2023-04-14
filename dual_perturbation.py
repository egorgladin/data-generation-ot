import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import get_measures, get_cost_mat, show_measure


DEVICE = 'cuda'


def dual(beta, mu, nu, C, gamma):
    """
    Compute dual function.

    :param beta: beta potential
    :param mu: source measure
    :param nu: target measure
    :param C: cost matrix
    :param gamma: entropy regularization coefficient
    :return: dual function value
    """
    # beta (n,) -> beta.unsqueeze(0) (1, n); C + beta <=> C + beta.unsqueeze(0)
    LSE_arg = -(C + beta) / gamma  # (n, n)
    LSE = torch.logsumexp(LSE_arg, dim=1)  # (n,)
    return torch.inner(nu, beta) + gamma * torch.inner(mu, LSE)


def recover_alpha(beta, mu, C, gamma):
    """
    Recover alpha potential from beta potential.

    :param beta: beta potential
    :param mu: source measure
    :param C: cost matrix
    :param gamma: entropy regularization coefficient
    :return: alpha potential
    """
    LSE_arg = -(C + beta) / gamma  # (n, n)
    LSE = torch.logsumexp(LSE_arg, dim=1)  # (n,)
    return gamma * LSE - gamma - gamma * torch.log(mu)


def recover_plan(beta, mu, C, gamma, alpha=None):
    """
    Recover transportation plan from potential(s)

    :param beta: beta potential
    :param mu: source measure
    :param C: cost matrix
    :param gamma: entropy regularization coefficient
    :param alpha: alpha potential (if not None, the formula for reconstructing plan from both potentials is used)
    :return: transportation plan - (n, n) tensor
    """
    if alpha is not None:
        exp_arg = -(C + alpha.unsqueeze(1) + beta) / gamma - 1  # (n, n)
        P = torch.exp(exp_arg)
        return P / P.sum()
    else:
        SoftMax_arg = -(C + beta) / gamma  # (n, n)
        SoftMax = torch.softmax(SoftMax_arg, dim=1)  # (n, n)
        return mu.unsqueeze(1) * SoftMax


def grad_descent(mu, nu, cost_mat, gamma, n_iter):
    """
    Solve the dual problem by gradient descent.

    :param mu: source measure
    :param nu: target measure
    :param cost_mat: cost matrix
    :param gamma: entropy regularization coefficient
    :param n_iter: number of iterations
    :return:
    """
    # initialize beta randomly
    torch.manual_seed(0)
    beta = torch.randn_like(mu, requires_grad=True)

    optimizer = torch.optim.SGD([beta], lr=3*gamma, momentum=0.9)

    for _ in tqdm(range(n_iter)):
        optimizer.zero_grad()
        loss = dual(beta, mu, nu, cost_mat, gamma)
        loss.backward()
        optimizer.step()

    beta.requires_grad = False
    return beta


def fourier_noise(size=28, decay_pwr=1):
    """
    Generate Fourier coefficients and use 2D iverse FFT to obtain noise.

    :param size: size (side length) of the resulting tensor
    :param decay_pwr: standard deviation of Fourier coefficient (i,j) = 1 / max(1, i, j)**decay_pwr
    :return: Fourier noise (2D tensor)
    """
    fourier_stds = torch.ones(size, size)
    for i in range(size):
        for j in range(size):
            fourier_stds[i, j] /= max(1, i, j)**decay_pwr

    fourier_coeffs = torch.randn(size, size, dtype=torch.cfloat) * fourier_stds
    fourier_noise = torch.fft.ifft2(fourier_coeffs)
    return fourier_noise.real


def display_perturbed_potential(replace_val=1e-5, from_uniform=False, perturb_alpha=False, noise_level=0.02, use_fourier=False):
    """
    Perturb a potential and visualize corresponding measure.

    :param replace_val: value to replace zeros with
    :param from_uniform: if True, use uniform measure as the source, otherwise digit 2
    :param perturb_alpha: if True, alpha potential is perturbed, otherwise beta
    :param noise_level: norm(noise added to a potential) = noise_level * norm(potential)
    :param use_fourier: if True, Fourier basis is used for noise generation
    """
    gamma = 0.1  # entropy regularization coefficient
    im_sz = 28  # image size
    cost_mat = get_cost_mat(im_sz=im_sz, device=DEVICE)

    # load beta potential and source measure
    fname_suffix = "_uniform" if from_uniform else ""
    with open(f'potentials/replace_val_{replace_val:.0e}{fname_suffix}.pt', 'rb') as handle1,\
            open(f'mu_replace_val_{replace_val:.0e}{fname_suffix}.pt', 'rb') as handle2:
        beta = torch.load(handle1)
        mu = torch.load(handle2)

    n_experiments = 3
    for i in range(n_experiments):
        # generate noise
        torch.manual_seed(i)
        noise = fourier_noise(decay_pwr=2).flatten().to(DEVICE) if use_fourier else torch.randn_like(beta)
        noise *= noise_level / torch.norm(noise)

        perturbed = 'alpha' if perturb_alpha else 'beta'
        perturb_latex = r'$\alpha$' if perturb_alpha else r'$\beta$'
        title = 'Perturbed ' + perturb_latex + f', noise {noise_level*100:.0f}%, 0s -> {replace_val:.0e},\n'
        title += r'$\mu$ = ' + ('uniform' if from_uniform else 'digit 2')
        title += ', ' + ('Fourier' if use_fourier else 'iid') + ' noise'

        fname_suffix += '_fourier' if use_fourier else ''
        if perturb_alpha:
            alpha = recover_alpha(beta, mu, cost_mat, gamma)
            P = recover_plan(beta, mu, cost_mat, gamma, alpha=alpha + noise * torch.norm(alpha))
            mu_approx = P.sum(dim=1)
            show_measure(mu_approx, f'perturb_{perturbed}/mu_replace_val_{replace_val:.0e}{fname_suffix}_{i}', title=title)
        else:
            P = recover_plan(beta + noise * torch.norm(beta), mu, cost_mat, gamma)
            # when beta is perturbed, we don't visualize the corresponding source measure because it doesn't change

        nu_approx = P.sum(dim=0)
        show_measure(nu_approx, f'perturb_{perturbed}/nu_replace_val_{replace_val:.0e}{fname_suffix}_{i}', title=title)


def compute_potentials(replace_val, from_uniform=False):
    """
    Compute potentials, save them, visualize corresponding measures.

    :param replace_val: value to replace zeros with
    :param from_uniform: if True, use uniform measure as the source, otherwise digit 2
    """
    gamma = 0.1  # entropy regularization coefficient
    n_iter = 3000

    mu, nu = get_measures(replace_val=replace_val, from_uniform=from_uniform, device=DEVICE)
    im_sz = 28  # image size
    cost_mat = get_cost_mat(im_sz=im_sz, device=DEVICE)
    beta = grad_descent(mu, nu, cost_mat, gamma, n_iter)

    # save potential and source measure
    suffix = "_uniform" if from_uniform else ""
    with open(f'potentials/replace_val_{replace_val:.0e}{suffix}.pt', 'wb') as handle1,\
            open(f'mu_replace_val_{replace_val:.0e}{suffix}.pt', 'wb') as handle2:
        torch.save(beta, handle1)
        torch.save(mu, handle2)

    # recover transport plan
    P = recover_plan(beta, mu, cost_mat, gamma)

    # visualize original and reconstructed measures to check the quality
    mu_approx = P.sum(dim=1)
    nu_approx = P.sum(dim=0)

    show_measure(mu, f'true_mu_replace_val_{replace_val:.0e}', im_sz=im_sz,
                 title=r"Original $\mu$, " + f"0s -> {replace_val:.0e}", vmax=0.003)
    show_measure(nu, f'true_nu_replace_val_{replace_val:.0e}', im_sz=im_sz,
                 title=r"Original $\nu$, " + f"0s -> {replace_val:.0e}")
    show_measure(mu_approx, f'recovered_mu_replace_val_{replace_val:.0e}', im_sz=im_sz,
                 title=r"Recovered $\mu$, " + f"0s -> {replace_val:.0e}", vmax=0.003)
    show_measure(nu_approx, f'recovered_nu_replace_val_{replace_val:.0e}', im_sz=im_sz,
                 title=r"Recovered $\nu$, " + f"0s -> {replace_val:.0e}",)


def visualize_fourier_noise():
    img = fourier_noise(decay_pwr=2).numpy()
    im = plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im)
    plt.savefig(f"plots/fourier_noise_example.png", bbox_inches='tight')
    plt.close()


def main():
    visualize_fourier = False  # set to True to visualize an example of Fourier noise
    calc_potentials = False  # set to True to calculate optimal potentials

    if visualize_fourier:
        visualize_fourier_noise()

    for replace_val in [1e-3, 1e-5]:  # value to replace zeros with
        for from_uniform in [True, False]:  # if True, use uniform measure as the source, otherwise digit 2
            if calc_potentials:
                compute_potentials(replace_val, from_uniform=from_uniform)
            for use_fourier in [True, False]:  # if True, Fourier basis is used for noise generation
                for perturb_alpha in [True, False]:  # if True, alpha potential is perturbed, otherwise beta
                    # A pretty arbitrary way to choose a good noise level
                    noise_level = 0.06 * (2 if perturb_alpha else 1) * (4 if replace_val == 1e-3 else 1)
                    display_perturbed_potential(replace_val=replace_val, perturb_alpha=perturb_alpha,
                                                noise_level=noise_level, from_uniform=from_uniform, use_fourier=use_fourier)


if __name__ == '__main__':
    main()
