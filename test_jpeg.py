import pytest
from numpy.random import randint, rand
import numpy as np
import scipy.io as sio

from helpers import *
from helpers_jpeg import *


@pytest.fixture(scope="module")
def X():
    '''Return the lighthouse image X'''
    return sio.loadmat('mat/lighthouse.mat')['X'].astype(float)


@pytest.fixture(scope="module")
def jpegout():
    '''Return the lighthouse image X'''
    return sio.loadmat('mat/jpegout.mat')


@pytest.fixture
def jpeg():
    '''Return a new jpeghelper class'''
    return JpegHuffmanHelper()


@pytest.fixture
def dwtHuff():
    '''Return a new jpeghelper class'''
    return DwtHuffmanHelper()


def test_jpegenc(X, jpeg, jpegout):
    '''Test jpegenc with the lighthouse image and qstep=17'''
    vlc, _bits, _huffval = jpeg.jpegenc(X-128, 17)
    assert np.array_equal(vlc, jpegout['vlc'].astype(int))


def test_jpegdec(X, jpeg, jpegout):
    vlc = jpegout['vlc'].astype(int)
    Z = jpeg.jpegdec(vlc, 17)
    assert np.allclose(Z, jpegout['Z'].astype(float))


def test_dwtgroup(X, dwtHuff, jpegout):
    test = jpegout['test'].astype(float)
    tested = dwtHuff.dwtgroup(test, 2)
    assert np.array_equal(tested, jpegout['test_dwtgrouped'].astype(float))
    test_reverse = dwtHuff.dwtgroup(tested, -2)
    assert np.array_equal(test_reverse, test)
