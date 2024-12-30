from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    """
    INPUT: 
        A .npy file that contains the dataset for the facial images

    RETURNS:
        An numpy.array of floats that contains the dataset after being centered around the origin
    """
    x = np.load(filename)
    return x - np.mean(x, axis=0)
    raise NotImplementedError

def get_covariance(dataset):
    """
    INPUT: 
        A numpy.array that contains the dataset of faces

    RETURNS:
        An numpy.array of floats of the covariance matrix of the given dataset
    """
    covariance = np.dot(np.transpose(dataset), dataset)
    scalar = 1/(len(dataset)-1)
    return covariance * scalar
    raise NotImplementedError

def get_eig(S, m):
    """
    INPUT: 
        A numpy.array that represents the covariance matrix of the dataset containing facial images
        The number of dimensions the data should have after being projected


    RETURNS:
        The m largest eigenvalues from the numpy.array S and its corresponding eigenvectors in order
    """
    w, v = eigh(a=S, eigvals_only= False, subset_by_index= [len(S)-m, len(S)-1])
    eigenvalues = np.zeros((m,m))
    eigenvectors = np.zeros(v.shape)
    for i in range(m):
        eigenvalues[i][i] = w[m-(i+1)]
        eigenvectors[:,i] = v[:,m-(i+1)]

    return eigenvalues, eigenvectors
    raise NotImplementedError

def get_eig_prop(S, prop):
    """
    INPUT: 
        A numpy.array that represents the covariance matrix of the dataset containing facial images
        The proportion of variance that a eigenvector must explain to be returned


    RETURNS:
        The eigenvalues from the numpy.array S that explain at least prop of the variance and 
            its corresponding eigenvectors in order
    """ 
    sum = 0
    for i in range(len(S)):
        sum += S[i][i]
    bar = sum * prop
    w,v = eigh(a=S, eigvals_only=False, subset_by_value=[bar, np.inf])
    m = len(w)
    eigenvalues = np.zeros((m,m))
    eigenvectors = np.zeros(v.shape)
    for i in range(m):
        eigenvalues[i][i] = w[m-(i+1)]
        eigenvectors[:,i] = v[:,m-(i+1)]    

    return eigenvalues, eigenvectors
    raise NotImplementedError

def project_image(image, U):
    """
    INPUT: 
        The original image from the dataset
        A number of vectors of the same size as the original image. 
            The number of vectors is defined by the results of the get_eig() and get_eig_prop() functions


    RETURNS:
        The image projected into len(U) dimensions
    """
    frstStep = np.dot(np.transpose(U), image)
    scndStep = np.dot(U, frstStep)
    return scndStep

    raise NotImplementedError

def display_image(orig, proj):
    """
    INPUT: 
        The original image from the dataset
        The projected version of the original image


    RETURNS:
        The figure to display the plot and its axis
    """
    orig = np.reshape(orig, (64,64))
    proj = np.reshape(proj, (64,64))

    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    
    ax1.set_title('Original')
    ax2.set_title('Projection')
    original = ax1.imshow(orig, aspect = 'equal')
    projection = ax2.imshow(proj, aspect = 'equal')

    fig.colorbar(original, ax=ax1)
    fig.colorbar(projection, ax = ax2)

    return fig, ax1, ax2
    raise NotImplementedError

def perturb_image(image, U, sigma):
    """
    INPUT: 
        The original image from the dataset
        A number of vectors of the same size as the original image. 
            The number of vectors is defined by the results of the get_eig() and get_eig_prop() functions
        A standard deviation to use to create a Gaussian Distribution to pertrub the projected image


    RETURNS:
        The image projected into len(U) dimensions and perturbed using random additions taken using a Gaussian Distribution
    """
    original = np.dot(np.transpose(U), image)

    perturbation = np.random.normal(0, sigma, len(original))

    perturbed = original + perturbation

    return np.dot(U, perturbed)
    raise NotImplementedError

def combine_image(image1, image2, U, lam):
    """
    INPUT: 
        The first original image from the dataset
        The second original image from the dataset
        A number of vectors of the same size as the original image. 
            The number of vectors is defined by the results of the get_eig() and get_eig_prop() functions
        The percent the first image should matter


    RETURNS:
        The image projected into len(U) dimensions after being combined with the second
    """
    image1Weights = np.dot(np.transpose(U), image1)
    image2Weights = np.dot(np.transpose(U), image2)
    imageComb = lam * image1Weights + (1-lam)*image2Weights


    return np.dot(U, imageComb)

    raise NotImplementedError


