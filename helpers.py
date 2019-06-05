# Author: Jason Li - jl944@cam.ac.uk, Karthik Suresh - ks800@cam.ac.uk
# 2019

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sewar.full_ref import msssim

def RGB_to_RGBA_cmap(RGB_cmap):
    '''
    Convert an RGB color map to an RGBA color map with unit alpha value.
    The function returns a colormap in the datatype compatible with matplotlib.

    Argument 'RGB_cmap' should be a 256x3 matrix that contains RGB
    values that correspond to each integer value in the interval [0,255].
    '''

    # Matrix size of RGBA colormap
    cmap = np.ones((256, 4))

    # RGB color map with column of ones appended
    if RGB_cmap.shape[0] == 256 and RGB_cmap.shape[1] == 3:
        cmap[:, :-1] = np.array(RGB_cmap)
        return ListedColormap(cmap)

    else:
        raise ValueError('RGB colormap format is not recognised')


def draw(img, cmap=None, fname=None):
    '''
    Draw an image

    Draw(img, cmap, fname) displays img with colormap cmap (default greyscale).
    If fname is given, it is saved with fname as the file name.
    (Use cmap instead of map because map is a keyword in Python)
    '''

    # Adjust by an appropriate multiple of 128
    adjusted_img = img - 128 * round(np.min(img) / 128)

    imgplot = plt.imshow(adjusted_img, interpolation='none')
    plt.axis('off')  # No axes
    # Set to greyscale mode, as real look
    imgplot.set_cmap(cmap='Greys_r' if cmap is None else cmap)
    if fname and type(fname) == str:
        plt.imsave(fname, adjusted_img,
                   cmap='Greys_r' if cmap is None else cmap)
    plt.show()


def halfcos(N):
    '''
    Function that returns an N length discrete half cosine low-pass filter.
    '''
    if not isinstance(N, int):
        raise ValueError('Filter length must be an integer')

    if N % 2 == 0:
        h = np.linspace(-N/2, N/2, N)
    else:
        h = np.linspace(-(N-1)/2, (N-1)/2, N)

    h = np.cos((h*np.pi)/(N+1))

    return h/np.sum(h)


def conv2(u, v, X):
    '''
    Function convolves filter u with columns of image X, and then
    filters filter v with rows of image.

    Parameters:
        u: Filter to convolve columns.
        v: Filter to convolve rows
        X: Image matrix.

    Returns: Filtered image matrix.
    '''

    Xc = np.zeros(X.shape)
    # Column convolution of X
    for i in range(0, X.shape[1]):
        # Returns convolved column of same size.
        Xc[:, i] = np.convolve(u, np.array(X, dtype=float)[:, i], 'same')

    Xr = np.zeros(X.shape)
    for k in range(0, X.shape[0]):
        # Returns convolved row of same size.
        Xr[k, :] = np.convolve(v, Xc[k, :], 'same')

    return Xr


def convse(X, h):
    '''
    Function convolves filter h, with rows of image X.

    The (n-1)/2 pixels that form the row edges of the
    picture are mirrored to avoid edge effects,
    where n is the filter length.
    '''

    # Mirroring row edges of image by half of filter length.
    Xr = np.array(X)
    # Identifying number of pixels to mirror. Different for
    # odd and even filter length.
    m = int((len(h)-1)/2) if len(h) % 2 != 0 else int(len(h)/2)
    # Mirroring right edge.
    Xr = np.concatenate((Xr, np.flip(Xr[:, -m-1:-1], axis=1)), axis=1)
    # Mirroring left edge.
    Xr = np.concatenate((np.flip(Xr[:, 1:m+1], axis=1), Xr), axis=1)

    # Buffer for final result
    Xrf = np.zeros(Xr.shape)

    # Row convolution of X
    for i in range(0, Xrf.shape[0]):
        # Returns convolved row of same size including mirrored edges.
        Xrf[i, :] = np.convolve(h, Xr[i, :], 'same')

    # Trim result to return matrix with same size as original image
    Xrf = Xrf[:, m:-m]
    return Xrf


def conv2se(u, v, X):
    '''
    Function convolves filter u with columns of image X, and then
    filters filter v with rows of image.


    Parameters:
        u: Filter to convolve columns.
        v: Filter to convolve rows
        X: Image matrix.

    Returns: Filtered image matrix with no edge effect.
    '''

    # Mirroring row edges
    Xr = np.array(X)
    m = int((len(v)-1)/2) if len(v) % 2 != 0 else int(len(v)/2)
    Xr = np.concatenate((Xr, np.flip(Xr[:, -m-1:-1], axis=1)), axis=1)
    Xr = np.concatenate((np.flip(Xr[:, 1:m+1], axis=1), Xr), axis=1)

    # Mirroring column edges
    Xc = np.array(Xr)
    n = int((len(u)-1)/2) if len(u) % 2 != 0 else int(len(u)/2)
    Xc = np.concatenate((Xc, np.flip(Xc[-n-1:-1, :], axis=0)), axis=0)
    Xc = np.concatenate((np.flip(Xc[1:n+1, :], axis=0), Xc), axis=0)

    Xf = conv2(u, v, Xc)

    return Xf[m:-m, n:-n]  # Trimming result to valid size.


def rowdec(X, h, align_with_first=True):
    '''
    Filters the rows of X using h, and decimates them by a factor of 2.

    If len(h) is odd, each output sample is aligned with the first of
    each pair of input samples by default, or the second if align_with_first is False.
    If len(h) is even, each output sample is aligned with the mid point
    of each pair of input samples.

    Parameters:
        X: a non-empty 2-D np.ndarray to be filtered
        h: a non-empty 1-D np.array filter, with len(h) <= 2 * number of columns of X
        align_with_first:   True - when len(h) is odd, align each output sample with the first of each pair of input samples
                            False - when len(h) is odd, align each output sample with the second of each pair of input samples
    Returns:
        Y: a filtered matrix with the rows filtered and decimated by 2:1
    '''
    r, c = X.shape  # Row, column
    m = len(h)  # Filter size

    if m > 2 * c:  # Edge case check. Explained below.
        raise ValueError(
            'Filter (length {}) is too large for output image ({} x {})'.format(m, r, c))

    # Symmetric extension of the indices before filtering - explained in handout
    m2 = m // 2
    original = np.arange(c)  # Original indices
    if m % 2 != 0:
        # Odd m: symmetrically extend indices without repeating end samples.
        # Both ends are extended by m2 indices
        # [0, ..., c - 1] -> [m2, ..., 1] +  [0, 1, ..., c - 2, c - 1] + [c - 2, ..., c - m2 - 1]
        # The last index: c - m2 - 1 >= 0, so c >= m2 + 1, 2*c >= m + 1
        # Therefore: m <= 2*c - 1
        extended = np.concatenate(
            (m2 - original[:m2], original, c - 2 - original[:m2]))
        last_aligned_index = c
    else:
        # Even m: symmetrically extend with repeat of end samples.
        # Both ends are extended by m2 - 1 indices
        # [0, ..., c - 1] -> [m2 - 2, ..., 1, 0] + [0, 1, ..., c - 2, c - 1] + [c - 1, c - 2, ..., c - m2 + 1]
        # The last index: c - m2 + 1 >= 0, so c >= m2 - 1
        # Therefore: m <= 2*c + 2
        extended = np.concatenate(
            (m2 - 2 - original[:m2 - 1], original, c - 1 - original[:m2 - 1]))
        last_aligned_index = c - 1

    first_aligned_index = 0 if align_with_first else 1
    # Fixed the index out of range error of the original rowdec2.m when both c and m are odd
    t = np.arange(first_aligned_index, last_aligned_index, 2)
    Y = np.zeros((r, len(t)))

    # Apply filter h
    for i in range(m):
        Y += h[i] * X[:, extended[t + i]]

    return Y


def rowdec2(X, h):
    '''
    Equivalent to rowdec2(X, h, align_with_first=False)

    If len(h) is odd, each output sample is aligned with the second of
    each pair of input samples.

    Fixed the index out of range error of the original rowdec2.m when both c and m are odd
    '''
    return rowdec(X, h, align_with_first=False)


def rowint(X, h, align_with_first=True):
    '''
    Interpolates the rows of image X by 2, using filter h.

    If length(h) is odd, each input sample is aligned with the first of
    each pair of output samples by default, or the second if align_with_first is False.
    If length(h) is even, each input sample is aligned with the mid point
    of each pair of output samples.
    Filters the rows of X using h, and decimates them by a factor of 2.

    Parameters:
        X: a non-empty 2-D np.ndarray to be filtered
        h: a non-empty 1-D np.array filter
        align_with_first:   True - when len(h) is odd, align each input sample with the first of each pair of output samples
                            False - when len(h) is odd, align each input sample with the second of each pair of output samples
    Returns:
        Y: a filtered matrix with the rows filtered and interpolated by 2:1
    '''
    r, c = X.shape  # Row, column
    m = len(h)  # Filter size
    c2 = 2 * c

    if m > 2 * c2:  # Edge case check. Explained in symmetric_extension.
        raise ValueError(
            'Filter (length {}) is too large for output image ({} x {})'.format(m, r, c2))

    # Generate X2 as X interleaved with columns of zeros.
    first_aligned_index = 0 if align_with_first else 1
    X2 = np.zeros((r, c2))
    X2[:, first_aligned_index::2] = X

    # Special symmetric extension for X2 to ensure interweaved zeros
    m2 = m // 2
    original = np.arange(c2)  # Original indices
    if m % 2 != 0:
        # Odd m: symmetrically extend indices without repeating end samples.
        # Both ends are extended by m2 indices
        # [0, ..., c2 - 1] -> [m2, ..., 1] +  [0, 1, ..., c2 - 2, c2 - 1] + [c2 - 2, ..., c2 - m2 - 1]
        # The last index: c2 - m2 - 1 >= 0, so c2 >= m2 + 1, 2*c2 >= m + 1
        # Therefore: m <= 2*c2 - 1
        extended = np.concatenate(
            (m2 - original[:m2], original, c2 - 2 - original[:m2]))
    else:
        # Even m: symmetrically extend with repeat of non-zero end samples,
        # but insert a zero sample between the repeated end samples, so that
        # the extended samples are still interweaved by zero columns.
        if align_with_first:
            # Non-zero indices are 0, 2, ..., c2 - 2
            # The left end is non zero, the right end is zero
            # Hence the left is exended by m2 - 1 end-repeated indices plus a zero index (i.e. [1])
            # and the right is extended by m2 - 1 non-end-repeated indices
            # [0, ..., c2 - 1] -> [m2 - 2, ..., 1, 0] + [1] + [0, 1, ..., c2 - 2, c2 - 1] + [c2 - 2, ..., c2 - m2]
            # The last index: c2 - m2 >= 0, so c2 >= m2
            # Therefore: m <= 2 * c2
            extended = np.concatenate(
                (m2 - 2 - original[:m2 - 1], [1], original, c2 - 2 - original[:m2 - 1]))
        else:
            # Non-zero indices are 1, 3, ..., c2 - 1
            # The left end is zero, the right end is non zero
            # Hence the left is exended by m2 - 1 non-end-repeated indices
            # and the right is extended by a zero index [c2 - 2] and m2 - 1 end-repeated indices
            # [0, ..., c2 - 1] -> [m2 - 1, ..., 1] + [0, 1, ..., c2 - 2, c2 - 1] + [c2 - 2] + [c2 - 1, ..., c2 - m2 + 1]
            # The last index: c2 - m2 >= 0, so c2 >= m2
            # Therefore: m <= 2 * c2
            extended = np.concatenate(
                (m2 - 1 - original[:m2 - 1], original, [c2 - 2], c2 - 1 - original[:m2 - 1]))

    t = np.arange(c2)
    Y = np.zeros((r, len(t)))

    # Apply filter h
    for i in range(m):
        Y += h[i] * X2[:, extended[t + i]]

    return Y


def rowint2(X, h):
    '''
    Equivalent to rowint2(X, h, align_with_first=False)

    If length(h) is odd, each input sample is aligned with the first of
    each pair of output samples.
    '''
    return rowint(X, h, align_with_first=False)


def beside(*args):
    '''
    Arrange two images beside each other.

    Y = beside(X1, X2) puts X1 and X2 beside each other in Y.
    Y is padded with zeros as necessary and the images are
    separated by a blank column.
    '''
    if len(args) == 2:
        X1, X2 = args
    elif len(args) > 2:
        X1, X2 = args[0], beside(*args[1:])
    else:
        raise ValueError('At least 2 matrices required')

    # work out size of Y
    m1, n1 = X1.shape
    m2, n2 = X2.shape
    m = max(m1, m2)
    Y = np.zeros((m, n1 + n2 + 1))

    Y[np.ix_((m - m1) // 2 + np.arange(m1), np.arange(n1))] = X1
    Y[np.ix_((m - m2) // 2 + np.arange(m2), n1 + 1 + np.arange(n2))] = X2

    return Y


def beside1(*args):
    '''
    Arrange two images beside each other.

    Y = beside(X1, X2) puts X1 and X2 beside each other in Y.
    Y is padded with zeros as necessary and the images are
    separated by a blank column.

    Variation of function above as it yields white background,
    but pixel values are distorted.
    '''
    if len(args) == 2:
        X1, X2 = args
    elif len(args) > 2:
        X1, X2 = args[0], beside1(*args[1:])
    else:
        raise ValueError('At least 2 matrices required')

    # work out size of Y
    m1, n1 = X1.shape
    m2, n2 = X2.shape
    m = max(m1, m2)
    Y = np.ones((m, n1 + n2 + 1))*255

    Y[np.ix_((m - m1) // 2 + np.arange(m1), np.arange(n1))] = X1
    Y[np.ix_((m - m2) // 2 + np.arange(m2), n1 + 1 + np.arange(n2))] = X2

    return Y


def bpp(X):
    '''
    Function that calculates the entropy of the image represented by matrix X.

    Parameters:
        X: a non-empty 2-D np.ndarray of which a histogram needs to formed to subsequently calculate entropy.
    Returns:
        H: Entropy of matrix X
    '''
    minx = np.min(X)
    maxx = np.max(X)

    bins = np.arange(np.floor(minx) - 0.5, np.ceil(maxx) + 1.5, 1)

    if len(bins) < 2:
        return 0

    else:
        # If density flag set to True and bins have unit width, sum(h) = 0
        h, b = np.histogram(X, bins=bins, density=True)

    h = np.array([p for p in h if p != 0])

    return -np.sum(h*np.log2(h))


def quant1(X, step, rise1=None):
    '''
    Quantises X using steps of width step. The result is quantised integers representing the image matrix.
    Quantiser is symmetrical about 0.

    Parameters:
        X: Image matrix.
        step: Step size between quanta
        rise1: Size of first step

    Returns:
        Image matrix represented as quantised levels.
    '''
    if step <= 0:
        raise ValueError('Step size must be positive')
    if rise1 is None:
        rise1 = step/2
    return np.maximum(np.zeros(X.shape), np.ceil((np.abs(X)-rise1)/step)) * np.sign(X)


def quant2(q, step, rise1=None):
    '''
    Reconstructs matrix using quantised steps of width step.

    Parameters:
        q: Quantised levels.
        step: Step size between quanta
        rise1: Size of first step

    Returns:
        Image matrix.
    '''
    if step <= 0:
        raise ValueError('Step size must be positive')
    if rise1 is None:
        rise1 = step/2
    return q * step + np.sign(q) * (rise1 - step/2)


def quantise(X, step, rise1=None, prop=None):
    '''
    Quantises X using steps of width step, using quant1, and quant 2.

    Parameters:
        X: Image matrix.
        step: Step size between quanta
        rise1: Size of first step in terms of ratio of step.

    Returns:
        Image matrix represented as quantised levels.
    '''
    if step <= 0:
        raise ValueError('Step size must be positive')
    if rise1 is None and prop is None:
        rise1 = step/2
    elif prop is not None:
        rise1 = prop * step

    return quant2(quant1(X, step, rise1), step, rise1)


def quantise_array(Y_array_X, step_array):
    if len(Y_array_X) != len(step_array):
        raise ValueError('Length of both array arguments\
                         should be the same')
    return [quantise(X, step) for X, step in zip(Y_array_X, step_array)]


def laplacian(X, h, layers, Y_array_X=None):
    '''Returns [Y0, Y1, ..., Yn-1, Xn]'''
    if layers < 1:
        raise ValueError("Layers can't be less than 1")
    elif 2 ** layers > min(X.shape):
        raise ValueError(
            "Too many layers for image ({} x {})".format(*X.shape))

    if Y_array_X is None:
        Y_array_X = []

    X1 = rowdec(rowdec(X, h).T, h).T
    Y0 = X - rowint(rowint(X1, 2 * h).T, 2 * h).T
    Y_array_X.append(Y0)

    if layers == 1:
        Y_array_X.append(X1)
        return Y_array_X
    else:
        return laplacian(X1, h, layers - 1, Y_array_X)


def decode_laplacian(Y_array_X, h):
    '''Returns [Z0, Z1, ..., Zn-1, Xn]'''
    Z_array_X = []
    Z = Y_array_X[-1]
    Z_array_X.append(Z)
    for i in range(1, len(Y_array_X)):
        Z = rowint(rowint(Z, 2*h).T, 2*h).T + Y_array_X[-(i+1)]
        Z_array_X.append(Z)
    return Z_array_X[::-1]


def bpp_array(quatised_mat):
    return [bpp(i) for i in quatised_mat]


def total_bits(X_arr):
    if len(X_arr) == 0 or len(X_arr[0].shape) < 2:
        raise ValueError('The input must be an array of matrices')
    return sum([bpp(X)*np.prod(X.shape) for X in X_arr])


def compression_ratio(reference_mat, compressed_arr):

    ref = bpp(reference_mat)*np.prod(reference_mat.shape)
    compression = total_bits(compressed_arr)
    return ref/compression


def dct_ii(N):
    """
    Function that generates 1-D NxN DCT transform matrix
    such that Y = C * X transforms N-vector X into Y. Uses
    an orthogonal Type-II DCT.

    Parameters:
        N: Dimension of DCT matrix.
    Returns:
        NxN DCT Matrix
    """
    # Initialising entire matrix to values in row 0
    C = np.ones((N, N)) / (N ** 0.5)
    # (n + 0.5)pi/N where n is column index with
    # range 0 to N-1.
    theta = (np.arange(0, N, 1) + 0.5) * (np.pi/N)
    # Constant factor for rows 1 to N-1
    g = (2/N) ** 0.5
    for i in range(1, N):
        C[i, :] = g * np.cos((i) * theta)
    return C


def dct_iv(N):
    """
    Function that generates 1-D NxN DCT transform matrix
    such that Y = C * X transforms N-vector X into Y. Uses
    an orthogonal Type-IV DCT.

    Parameters:
        N: Dimension of DCT matrix.
    Returns:
        NxN DCT Matrix
    """
    # Initialising entire matrix to values in row 0
    C = np.ones((N, N)) / (N ** 0.5)
    # (n + 0.5)pi/N where n is column index with
    # range 0 to N-1.
    theta = (np.arange(0, N, 1) + 0.5) * (np.pi/N)
    # Constant factor for rows 1 to N-1
    g = (2/N) ** 0.5
    for i in range(0, N):
        C[i, :] = g * np.cos((i + 0.5) * theta)
    return C


def extend_X_colxfm(X, C, extend_bottom=True):
    '''
    If X cannot be exectly split into blocks that are of size C,
    fuction first mirrors edges of X to expand X to a size where
    it can be exactly split into blocks that are of size C.
    '''

    # TODO: customize extension: bottom/top/both

    # Mirror bottom edge if rows not multiple of N
    N = len(C)
    if X.shape[0] % N is not 0:
        ext = N - (X.shape[0] % N)
        X = np.concatenate((X, np.flip(X[-ext-1:-1, :], axis=0)), axis=0)

    # Mirror right edge if columns not multiple of N
    if X.shape[1] % N is not 0:
        ext = N - (X.shape[1] % N)
        X = np.concatenate((X, np.flip(X[:, -ext-1:-1], axis=1)), axis=1)

    return X


def colxfm(X, C):
    """
    Transforms the columns of X using the transformation given in C.
    The height of X must be a multiple of the size of C.

    Parameters:
        X: 2-D np.ndarray image matrix.
        C: 2-D transform matrix (e.g. DCT).
    Returns:
        Y: Matrix same size as X with each stripe of column vectors
        (same height as C) being transformed.
    """
    N = C.shape[0]
    if len(C.shape) != 2 or N != C.shape[1]:
        raise ValueError('C must be a 2-D square matrix!')
    if len(X.shape) != 2:
        raise ValueError('X must be a 2-D matrix!')
    m, n = X.shape

    if m % N != 0:
        raise ValueError('Height not a multiple of transform size.')

    Y = np.zeros_like(X, dtype=float)
    # Transform columns of each horizontal stripe of pixels (height N)
    for i in np.arange(0, m, N):
        Y[i:i+N, :] = np.matmul(C, X[i:i+N, :])
    return Y


def regroup(X, N):
    """
    Regroups the rows and columns of X such that rows/cols
    that are N apart in X, are adjeacent in Y. If N is a
    2 element vector, N[0] is used for rows and N[1] is used
    for columns.

    Parameters:
        X: m by n matrix to be regrouped.
        N: Integer or two element vector.
    Returns:
        Y: Regrouped matrix.
    """
    m, n = X.shape
    if isinstance(N, int):
        N = [N, N]
    if m % N[0] != 0 or n % N[1] != 0:
        raise ValueError('X dimensions need to be multiple\
                          of elements in N')

    row_ind = np.ravel(
        [[i + k for i in np.arange(0, n, N[0])] for k in range(N[0])])
    col_ind = np.ravel(
        [[i + k for i in np.arange(0, n, N[1])] for k in range(N[1])])
    Y = X[row_ind, :]
    Y = Y[:, col_ind]
    return Y

def ungroup(Yr, N):
    m, n = Yr.shape
    if isinstance(N, int):
        N = [N, N]
    if m % N[0] != 0 or n % N[1] != 0:
        raise ValueError('X dimensions need to be multiple\
                          of elements in N')
    row_ind = np.ravel(
    [[i + k for i in np.arange(0, n, n//N[0])] for k in range(n//N[0])])
    col_ind = np.ravel(
    [[i + k for i in np.arange(0, n, n//N[1])] for k in range(n//N[1])])
    Yu = Yr[row_ind, :]
    Yu = Yu[:, col_ind]
    return Yu

def dctbpp(Yr, N):
    """
    A functioin that calculates total number of bits from a
    regrouped image.

    Parameters:
        Yr: Regrouped image.
        N: Dimension of a block in regrouped image.
    Returns:
        b: Total number of bits required to store
        regrouped image.
    """
    b = 0
    # Dimension of sub-image in regrouped image.
    R = Yr.shape[0]//N
    # Number of pixels in subimage
    R2 = R ** 2
    for i in np.arange(0, Yr.shape[0], R):
        for k in np.arange(0, Yr.shape[1], R):
            b += bpp(Yr[i:i+R, k:k+R])
    return b * R2


def pot_ii(N, s=(1 + 5 ** 0.5)/2, o=None):
    """
    POT_II - Photo Overlap Transform Matrix
    Generates the 1-D POT transform matrices of size N, equivalent to the
    pre-filtering stage of a Type-II fast Lapped Orthogonal Transform (LOT-II)

    Parameters:
        N: Size of transform matrices
        s: Scaling factor which determines orthogonality
        o: Determines size of overlap
    Returns:
        Y = Pf * X pre-filters N-vector X into Y
        X + Pr' * Y post-filters N-vector Y into X
    """
    # TODO: Should there be another check relating values of o and N
    # so that matrix multiplication is always possible below.
    if s < 1 or s > 2:
        raise ValueError('Scaling factor s must be in range [1,2]')
    if o is None:
        o = N//2
    elif o < 0 or o > N//2 or not isinstance(o, int):
        raise ValueError('Overlap o must be in range [0, N//2]')

    # Generate component matrices
    I = np.identity(N//2)
    J = np.fliplr(I)
    Z = np.zeros((N//2, N//2))
    Cii = dct_ii(o)
    Civ = dct_iv(o)

    # Generate forward and reverse scaling matrices
    Sf = np.diag(np.concatenate((np.array([s]), np.ones(o-1))))
    Sr = np.diag(np.concatenate((np.array([1/s]), np.ones(o-1))))

    # Generate forfard and reverse filtering matrices
    if (o < N//2):
        VI = np.identity(N//2 - o)
        VJ = np.fliplr(VI)
        VZ = np.zeros((o, N//2-o))
        Vf = np.concatenate((
            np.concatenate((
                np.dot(VJ,
                           np.dot(Cii.T,
                                  np.dot(Sf,
                                         np.dot(Civ, VJ)
                                         )
                                  )
                           ), VZ), axis=1),
            np.concatenate((VZ.T, VI), axis=1)),
            axis=0)

        Vr = np.concatenate((
            np.concatenate((
                np.dot(VJ,
                           np.dot(Cii.T,
                                  np.dot(Sr,
                                         np.dot(Civ, VJ)
                                         )
                                  )
                           ), VZ), axis=1),
            np.concatenate((VZ.T, VI), axis=1)),
            axis=0)
    else:
        Vf = np.dot(J,
                    np.dot(Cii.T,
                           np.dot(Sf,
                                  np.dot(Civ, J))))
        Vr = np.dot(J,
                    np.dot(Cii.T,
                           np.dot(Sr,
                                  np.dot(Civ, J))))

    Pf = 0.5 * np.dot(np.concatenate((
        np.concatenate((I, J), axis=0),
        np.concatenate((J, -I), axis=0)), axis=1),
        np.dot(np.concatenate((
            np.concatenate((I, Z), axis=0),
            np.concatenate((Z, Vf), axis=0)), axis=1),
        np.concatenate((
            np.concatenate((I, J), axis=0),
            np.concatenate((J, -I), axis=0)), axis=1)))

    Pr = 0.5 * np.dot(np.concatenate((
        np.concatenate((I, J), axis=0),
        np.concatenate((J, -I), axis=0)), axis=1),
        np.dot(np.concatenate((
            np.concatenate((I, Z), axis=0),
            np.concatenate((Z, Vr), axis=0)), axis=1),
        np.concatenate((
            np.concatenate((I, J), axis=0),
            np.concatenate((J, -I), axis=0)), axis=1)))
    return (Pf, Pr)


def dwt(X, h1=None, h2=None):
    """
    DWT - Discrete Wavelet Transform
    Paramaters:
        X: Image Matrix
        h1, h2: Filters. Default is Legall filter pair.
    Returns:
        Y: 1-level 2-D discrete wavelet transform on X.
    """
    if h1 is None or h2 is None:
        h1 = 1/8 * np.array([-1, 2, 6, 2, -1])
        h2 = 1/4 * np.array([-1, 2, -1])

    U = rowdec(X, h1)
    V = rowdec2(X, h2)

    UU = rowdec(U.T, h1).T
    UV = rowdec2(U.T, h2).T
    VU = rowdec(V.T, h1).T
    VV = rowdec2(V.T, h2).T

    Y = np.block([[UU, VU], [UV, VV]])

    return Y
    # if h1 is None or h2 is None:
    #     h1 = 1/8 * np.array([-1, 2, 6, 2, -1])
    #     h2 = 1/4 * np.array([-1, 2, -1])

    # m = X.shape[0]
    # n = X.shape[1]
    # Y = np.zeros((m, n))

    # n2 = n//2
    # t = np.arange(0, n2, 1)
    # Y[:, t] = rowdec(X, h1)
    # Y[:, t+n2] = rowdec2(X, h2)

    # X = Y.T
    # m2 = m//2
    # t = np.arange(0, m2, 1)
    # Y[t, :] = rowdec(X, h1).T
    # Y[t+m2, :] = rowdec2(X, h2).T

    # return Y


def idwt(X, g1=None, g2=None):
    """
    IDWT - Inverse Discrete Wavelet Transform
    Paramaters:
        Y: DWT Matrix
        g1, g2: Filters. Default is Legall filter pair.
    Returns:
        Y: 1-level 2-D inverse discrete wavelet transform on X.
    """
    if g1 is None or g2 is None:
        g1 = 1/2 * np.array([1, 2, 1])
        g2 = 1/4 * np.array([-1, -2, 6, -2, -1])

    m = X.shape[0]
    n = X.shape[1]

    m2 = m//2
    t = np.arange(0, m2, 1)
    Y = rowint(X[t, :].T, g1).T + rowint2(X[t+m2, :].T, g2).T

    n2 = n//2
    t = np.arange(0, n2, 1)
    Y = rowint(Y[:, t], g1) + rowint2(Y[:, t+n2], g2)

    return Y


def compression_ratio_dct(reference_mat, Yr, N):
    """
    Calculates compression ratio for DCT scheme.

    Parameters:
        reference_mat: Reference image matrix.
        Yr: Regrouped DCT Transformed matrix.
        N: DCT Transform length.
    Returns:
        Compression ratio.
    """
    ref = bpp(reference_mat)*np.prod(reference_mat.shape)
    compression = dctbpp(Yr, N)
    return ref/compression


def dct_transform_out(X, N, step=None):
    """
    Applies DCT-II transform to image, quantises trasnform,
    and returns relavant matrices required to evaluate
    compression performance.

    Parameters:
        X: Image Matrix
        N: DCT Length
    Returns:
        Yq: DCT Transfromed Matrix
        Yqr: Regrouped Transformed Matrix
        Zq: Reconstructed Matrix
    """
    C = dct_ii(N)
    Y = colxfm(colxfm(X, C).T, C).T
    if step is not None:
        Yq = quantise(Y, step)
    else:
        Yq = np.array(Y)
    Yqr = regroup(Yq, N)
    Zq = colxfm(colxfm(Yq.T, C.T).T, C.T)

    return [Yq, Yqr, Zq]


def pre_filter(X, N, s=None):
    """
    Performs prefiltering for fast lapped bi-orthogonal transform.

    Parameters:
        N: Size of transform block (dimension of Pr, Pf, C8)
        X: (Square) image matrix. TODO Generalise to any size.
    Returns:
        Xl = Output after pre-filtering.
    """
    if s is None:
        Pf, Pr = pot_ii(N)
    else:
        Pf, Pr = pot_ii(N, s)
    I = X.shape[0]
    t = np.arange(N//2, I-N//2, 1)
    Xl = np.array(X)
    Xl[t, :] = colxfm(Xl[t, :], Pf)
    Xl[:, t] = colxfm(Xl[:, t].T, Pf).T

    return Xl


def post_filter(Z, N, s=None):
    """
    Performs postfiltering for fast lapped bi-orthogonal transform.

    Parameters:
        N: Size of transform block (dimension of Pr, Pf, C8)
        Z: (Square) inverse DFT matrix. TODO Generalise to any size.
    Returns:
        Xl = Output after pre-filtering.
    """
    if s is None:
        Pf, Pr = pot_ii(N)
    else:
        Pf, Pr = pot_ii(N, s)
    I = Z.shape[0]
    t = np.arange(N//2, I-N//2, 1)
    Zl = np.array(Z)
    Zl[:, t] = colxfm(Zl[:, t].T, Pr.T).T
    Zl[t, :] = colxfm(Zl[t, :], Pr.T)

    return Zl


def nlevdwt(X, n, h1=None, h2=None):
    """
    Produces n layer DWT.

    Parameters:
        X: Image Matrix
        n: number of layers.
        h1, h2: Filters. Default is Legall filter pair.
    Returns:
        Y: Layered DWT
    """
    m_arr = [X.shape[0]//(2 ** i) for i in range(n)]
    Y = np.array(X)
    for m in m_arr:
        Y[:m, :m] = dwt(Y[:m, :m], h1=h1, h2=h2)
    return Y


def nlevidwt(Y, n, g1=None, g2=None):
    """
    Inverts n layer DWT.

    Parameters:
        Y: Layered DWT
        n: number of layers.
        g1, g2: Filters. Default is Legall filter pair.
    Returns:
        X: Original Image
    """
    m_arr = [Y.shape[0]//(2 ** i) for i in range(n)]
    m_arr = m_arr[::-1]
    X = np.array(Y)
    for m in m_arr:
        X[:m, :m] = idwt(X[:m, :m], g1=g1, g2=g2)
    return X


def quantdwt(Y, Q, n, prop=None):
    """
    Quantises layered DWT.

    Parameters:
        Y: Layered DWT
        n: number of layers.
        Q: dwtstep in handout: 3x(n+1) Matrix where Q[k,i] is step for
           (i+1)th layer kth sub-image. k = 0 is top right,
           k = 1 is bottom right, k = 2 is bottom left.
           k = 0 when i = n is step for small top left
           high pass image.
    Returns:
        Yq: Quantised layered DWT
        dwtent: 3x(n+1) Matrix where dwtent[k,i] is the entropy for
                the (i+1)th layer kth sub-image. k = 0 is top right,
                k = 1 is bottom right, k = 2 is bottom left.
                k = 0 when i = n is the entropy for small top left
                high pass image.
    """
    m_arr = [Y.shape[0]//(2 ** i) for i in range(1, n+2)]
    Yq = np.array(Y)
    dwtent = np.zeros_like(Q, dtype=float)
    for m in range(n + 1):
        Nsub = m_arr[m]
        if m == n:
            Yq[:2*Nsub, :2 *
                Nsub] = quantise(Yq[:2*Nsub, :2*Nsub], Q[0, m], prop=prop)
            dwtent[0, m] = total_bits([Yq[:2*Nsub, :2*Nsub]])
            break
        Yq[:Nsub, Nsub:2 *
            Nsub] = quantise(Yq[:Nsub, Nsub:2*Nsub], Q[0, m], prop=prop)
        Yq[Nsub:2*Nsub, Nsub:2 *
            Nsub] = quantise(Yq[Nsub:2*Nsub, Nsub:2*Nsub], Q[1, m], prop=prop)
        Yq[Nsub:2*Nsub,
            :Nsub] = quantise(Yq[Nsub:2*Nsub, :Nsub], Q[2, m], prop=prop)

        dwtent[0, m] = total_bits([Yq[:Nsub, Nsub:2*Nsub]])
        dwtent[1, m] = total_bits([Yq[Nsub:2*Nsub, Nsub:2*Nsub]])
        dwtent[2, m] = total_bits([Yq[Nsub:2*Nsub, :Nsub]])
    return Yq, dwtent


def total_bits_dwt(dwtent):
    n = dwtent.shape[1] - 1
    bits = 0
    for m in range(n + 1):
        if m == n:
            bits += dwtent[0, m]
            break
        bits += dwtent[0, m]
        bits += dwtent[1, m]
        bits += dwtent[2, m]
    return bits


def compression_ratio_dwt(reference_mat, dwtent):
    """
    Calculates compression ratio for DWT scheme.

    Parameters:
        reference_mat: Reference image matrix.
        dwtent: 3x(n+1) Matrix where dwtent[k,i] is the entropy for
                the (i+1)th layer kth sub-image. k = 0 is top right,
                k = 1 is bottom right, k = 2 is bottom left.
                k = 0 when i = n is the entropy for small top left
                high pass image.
    Returns:
        Compression ratio.
    """
    ref = bpp(reference_mat)*np.prod(reference_mat.shape)
    compression = total_bits_dwt(dwtent)
    return ref/compression


def quality(X_ori, Y_cmp, ws=8, MAX=255):
    '''
    Use MS-SSIM to calcualte the compression quality.
    Shift the pixel range up by 128 and then clip to
    [0, 255] before the algorithm. Returns a score
    in [0, 1]. A higher score suggests better quality.

    Parameters:
        X_ori: original image
        Y_cmp: compressed image
        ws: sliding window size (default 8)
        MAX: maximum value of datarange (default 255)
    
    Return: a float in [0, 1].

    Reference: 
    "Multiscale structural similarity for image quality assessment." (2003)
    https://ieeexplore.ieee.org/abstract/document/1292216/
    '''
    X_ori, Y_cmp = [np.clip(np.round(X)+128, 0, 255) for X in [X_ori, Y_cmp]]
    return msssim(X_ori, Y_cmp, ws=ws, MAX=MAX).real
