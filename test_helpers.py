'''
Test the helper functions
Author: Jason Li - jl944@cam.ac.uk
2019
'''

import pytest
from numpy.random import randint, rand
import numpy as np
import scipy.io as sio

from helpers import *


@pytest.fixture(scope="module")
def X_lighthouse():
    '''Return the lighthouse image X'''
    return sio.loadmat('mat/lighthouse.mat')['X'].astype(float)


@pytest.fixture(scope="module")
def h_simple():
    '''Return the simple 3-tap filter in Handout Section 6.1'''
    return np.array([1, 2, 1]) / 4


@pytest.fixture(scope="module")
def matlab_output():
    '''Return the expected outputs from MATLAB'''
    return sio.loadmat('mat/matlabout.mat')


@pytest.fixture(scope="module")
def pot_ii_dat():
    """Return the expected outputs from MATLAB"""
    return sio.loadmat('mat/pot_ii.mat')


@pytest.fixture(scope="module")
def dwt_idwt_dat():
    """Return the expected outputs from MATLAB"""
    return sio.loadmat('mat/dwt_idwt_dat.mat')


def X_odd():
    '''Return a random 3 x 3 matrix'''
    return randint(0, 256, (3, 3))


def X_even():
    '''Return a random 4 x 4 matrix'''
    return randint(0, 256, (4, 4))


def h_odd():
    '''Return a random filter of length 3'''
    h = rand(3) - 0.5
    return h / sum(h)


def h_even():
    '''Return a random filter of length 4'''
    h = rand(4) - 0.5
    return h / sum(h)


@pytest.mark.parametrize("X, h, align", [
    (X, h, align) for X in (X_odd(), X_even()) for h in (h_odd(), h_even()) for align in (True, False)
])
def test_rowdec_random(X, h, align):
    '''Test if rowdec handles odd and even dimensions correctly and triggers no index out of range errors'''
    rowdec(X, h, align_with_first=align)


@pytest.mark.parametrize("X, h, align", [
    (X, h, align) for X in (X_odd(), X_even()) for h in (h_odd(), h_even()) for align in (True, False)
])
def test_rowint_random(X, h, align):
    '''Test if rowint handles odd and even dimensions correctly and triggers no index out of range errors'''
    rowint(X, h, align_with_first=align)


@pytest.mark.parametrize("X, h, align, expected", [
    (np.array([[1, 2, 3, 4]]), np.array([1, 2, 1]) / 4,
        True, np.array([[1.5,  3]])),
    (np.array([[1, 2, 3, 4]]), np.array([1, 2, 1]) / 4,
        False, np.array([[2.,  3.5]])),
    (np.array([[1, 2, 3, 4, 5, 6]]), np.array([2, 3]) / 5,
        True, np.array([[1.6, 3.6, 5.6]])),
    (np.array([[1, 2, 3, 4, 5, 6]]), np.array([2, 3]) / 5,
        False, np.array([[2.6, 4.6]])),
])
def test_rowdec_small(X, h, align, expected):
    '''Test for accurate answer for small test cases'''
    assert np.allclose(rowdec(X, h, align_with_first=align), expected)


@pytest.mark.parametrize("X, h, align, expected", [
    (np.array([[1, 2, 3]]), np.array([1, 2, 1]) / 4,
        True, np.array([[0.5, 0.75, 1., 1.25, 1.5, 1.5]])),
    (np.array([[1, 2, 3]]), np.array([1, 2, 1]) / 4,
        False, np.array([[0.5, 0.5, 0.75, 1., 1.25, 1.5]])),
    (np.array([[1, 2, 3]]), np.array([2, 3, 2, 3]) / 10,
        True, np.array([[0.4, 0.9, 0.6, 1.5, 1., 1.8]])),
    (np.array([[1, 2, 3]]), np.array([2, 3, 2, 3]) / 10,
        False, np.array([[0.4, 0.9, 0.6, 1.5, 1., 1.8]])),
])
def test_rowint_small(X, h, align, expected):
    '''Test for accurate answer for small test cases'''
    assert np.allclose(rowint(X, h, align_with_first=align), expected)


def test_rowdec(X_lighthouse, h_simple, matlab_output):
    '''Compare the output with Matlab using maximum absolute difference'''
    assert np.max(abs(
        rowdec(X_lighthouse, h_simple) - matlab_output['rowdecXh'])) == 0


def test_rowint(X_lighthouse, h_simple, matlab_output):
    '''Compare the output with Matlab using maximum absolute difference'''
    assert np.max(abs(
        rowint(X_lighthouse, 2 * h_simple) - matlab_output['rowintX2h'])) == 0


@pytest.mark.parametrize("X, entropy", [
    (np.array([[1, -2], [3, -4]]), 2),  # log2(4)
    (np.array([[-0.3, 1.51], [2.3, 0.49]]), 1),  # [0, 2, 2, 0] -> log2(2)
    (np.array([-128, -127.49, 127, 126.49]), 2)  # log2(4)
])
def test_bpp(X, entropy):
    '''Simple tests for bits per pixel'''
    assert(bpp(X) == entropy)


@pytest.mark.parametrize("X, step, Xq", [
    (np.array([[1.49, 1.51], [1.51, 1.49]]), 1, np.array([[1, 2], [2, 1]])),
    (np.array([[1.49, 1.51], [1.51, 1.49]]), 2, np.array([[2, 2], [2, 2]]))
])
def test_quantise(X, step, Xq):
    '''Simple quantise tests'''
    assert np.array_equal(quantise(X, step), Xq)


@pytest.mark.parametrize("N, C", [
    (1, np.array([[1]])),
    (2, np.array([[1/(2 ** 0.5), 1/(2 ** 0.5)],
                  [np.cos(np.pi/4), np.cos(3 * np.pi/4)]]))
])
def test_dct_ii(N, C):
    assert np.allclose(dct_ii(N), C)


def test_dct_ii_matlabout(matlab_output):
    assert np.allclose(dct_ii(8), matlab_output['C8'])


@pytest.mark.parametrize("N, C", [
    (1, np.array([[1.0]])),
    (2, np.array([[np.cos(np.pi/8), np.cos(3 * np.pi/8)],
                  [np.cos(3 * np.pi/8), np.cos(9 * np.pi/8)]]))
])
def test_dct_iv(N, C):
    assert np.allclose(dct_iv(N), C)


@pytest.mark.parametrize("X, C, Y", [
    (np.ones((4, 4)), np.ones((2, 2)), np.array(
        [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])),
    (np.arange(16).reshape((4, 4)), np.eye(2)[::-1],  # [[0, 1], [1, 0]] swap every two rows
        np.array([[4, 5, 6, 7], [0, 1, 2, 3], [12, 13, 14, 15], [8, 9, 10, 11]])),
    # This should be the test for extend_X_colxfm
    # (np.ones((3, 3)), np.ones((2, 2)), np.array(
    #     [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]))
])
def test_colxfm(X, C, Y):
    assert np.array_equal(Y, colxfm(X, C))


def test_colxfm_matlabout(matlab_output):
    X, Y, Z, C8 = (matlab_output[key] for key in ('X', 'Y', 'Z', 'C8'))
    assert np.allclose(Y, colxfm(colxfm(X, C8).T, C8).T)
    assert np.allclose(Z, colxfm(colxfm(Y.T, C8.T).T, C8.T))
    assert np.allclose(X, Z)


@pytest.mark.parametrize("Y_regrouped, Y, N", [
    (np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]), np.array(
        [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]), 2),
    (np.array([[1, 1, 2, 2], [3, 3, 4, 4], [1, 1, 2, 2], [3, 3, 4, 4]]), np.array(
        [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]), [1, 2]),
    (np.array([[1, 2, 1, 2], [1, 2, 1, 2], [3, 4, 3, 4], [3, 4, 3, 4]]), np.array(
        [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]), [2, 1]),
    (np.array([
        [0,   3,   6,   9,   1,   4,   7,  10,   2,   5,   8,  11],
        [24,  27,  30,  33,  25,  28,  31,  34,  26,  29,  32,  35],
        [48,  51,  54,  57,  49,  52,  55,  58,  50,  53,  56,  59],
        [72,  75,  78,  81,  73,  76,  79,  82,  74,  77,  80,  83],
        [96,  99, 102, 105,  97, 100, 103, 106,  98, 101, 104, 107],
        [120, 123, 126, 129, 121, 124, 127, 130, 122, 125, 128, 131],
        [12,  15,  18,  21,  13,  16,  19,  22,  14,  17,  20,  23],
        [36,  39,  42,  45,  37,  40,  43,  46,  38,  41,  44,  47],
        [60,  63,  66,  69,  61,  64,  67,  70,  62,  65,  68,  71],
        [84,  87,  90,  93,  85,  88,  91,  94,  86,  89,  92,  95],
        [108, 111, 114, 117, 109, 112, 115, 118, 110, 113, 116, 119],
        [132, 135, 138, 141, 133, 136, 139, 142, 134, 137, 140, 143]]),
     np.arange(144).reshape(12, 12), (2, 3))
])
def test_regroup(Y_regrouped, Y, N):
    assert np.array_equal(Y_regrouped, regroup(Y, N))


@pytest.mark.parametrize("Yr, N, b", [
    (np.array([[0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]]), 2, 8),
    (np.array([[0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]]), 1,
     bpp(np.array([[0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]])) * 16),
    (np.array([[0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]]), 4, 0)
])
def test_dct_bpp(Yr, N, b):
    assert dctbpp(Yr, N) == b


def test_pot_ii(pot_ii_dat):
    (pf, pr) = pot_ii(8)
    assert np.allclose(pf, pot_ii_dat['pf'])
    assert np.allclose(pr, pot_ii_dat['pr'])


def test_dwt(dwt_idwt_dat):
    assert np.allclose(dwt(dwt_idwt_dat['X']), dwt_idwt_dat['dwt_'])


def test_idwt(dwt_idwt_dat):
    assert np.allclose(idwt(dwt_idwt_dat['X']), dwt_idwt_dat['idwt_'])
