from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

#load dataset
def load_and_center_dataset(filename):
    x = np.load('YaleB_32x32.npy')
    x = np.reshape(x, (2414, 1024))
    x = x - np.mean(x, axis=0)
    return x

#get the covariance matrix
def get_covariance(dataset):
    return (np.dot(np.transpose(dataset), dataset) / (len(dataset) - 1))

#get the 'm' largest eigenvalues and eigenvectors
def get_eig(S, m):
    w, v = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    i = np.flip(np.argsort(w))
    return np.diag(w[i]), v[:, i]


def get_eig_perc(S, perc):
    w, v = eigh(S)
    percent = np.sum(w) * perc
    new_w, new_v = eigh(S, subset_by_value=[percent, np.inf])
    i = np.flip(np.argsort(new_w))
    return np.diag(new_w[i]), new_v[:, i]


def project_image(image, U):
    alphas = np.dot(np.transpose(U), image)
    return np.dot(U, alphas)


def display_image(orig, proj):
    orig = np.reshape(orig, (32, 32), order = 'F')
    proj = np.reshape(proj, (32, 32), order = 'F')

    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (9,3))
    ax1.set_title('Original')
    ax2.set_title('Projection')

    ax1Map = ax1.imshow(orig, aspect = 'equal', cmap='Greens')
    fig.colorbar(ax1Map, ax=ax1)
    ax2Map = ax2.imshow(proj, aspect = 'equal', cmap='Greens')
    fig.colorbar(ax2Map, ax=ax2)
    plt.show()


