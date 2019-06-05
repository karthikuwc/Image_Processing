import numpy as np
from helpers_jpeg import JpegHuffmanHelper
from helpers import *

class DCTSupress(JpegHuffmanHelper):
    """Helper Class for LBT Compression and Huffman coding"""

    def __init__(self):
        super().__init__()

    def suphuffenc(self, X, qstep, N=8, M=8, opthuff=False, dcbits=8, thresh=4):
        if M % N != 0:
            raise ValueError('M must be an integer multiple of N!')

        # # Prefilter X
        # Xl = pre_filter(X, N, scale)

        # DCT on input image X.
        print('Forward {} x {} DCT'.format(N, N))
        C8 = dct_ii(N)
        Y = colxfm(colxfm(X, C8).T, C8).T

        Yr = regroup(Y, N)

        t = np.arange(256//N)
        for r in range(0, Yr.shape[0], 256//N):
            for c in range(0, Yr.shape[1], 256//N):
                e = np.sum(Yr[np.ix_(r+t, c+t)]/N)
                if e < thresh:
                    Yr[np.ix_(r+t, c+t)] = 0
        Yu = ungroup(Yr, N)
        # Quantise to integers.
        print('Quantising to step size of {}'.format(qstep))
        Yq = quant1(Yu, qstep, qstep).astype('int')

        # Generate zig-zag scan of AC coefs.
        scan = self.diagscan(M)

        # On the first pass use default huffman tables.
        print('Generating huffcode and ehuf using default tables')
        dbits, dhuffval = self.huffdflt(1)  # Default tables.
        huffcode, ehuf = self.huffgen(dbits, dhuffval)

        # Generate run/ampl values and code them into vlc(:,1:2).
        # Also generate a histogram of code symbols.
        print('Coding rows')
        sy = Yq.shape
        t = np.arange(M)
        self.huffhist = np.zeros((16 ** 2, 1))
        vlc = None
        for r in range(0, sy[0], M):
            vlc1 = None
            for c in range(0, sy[1], M):
                yq = Yq[np.ix_(r+t, c+t)]
                # Possibly regroup
                if M > N:
                    yq = regroup(yq, N)
                yqflat = yq.flatten('F')
                # Encode DC coefficient first
                yqflat[0] += 2 ** (dcbits-1)
                if yqflat[0] < 0 or yqflat[0] > (2**dcbits) - 1:
                    raise ValueError(
                        'DC coefficients too large for desired number of bits')
                dccoef = np.array([[yqflat[0], dcbits]])
                # Encode the other AC coefficients in scan order
                ra1 = self.runampl(yqflat[scan])
                # huffenc() also updates huffhist.
                if vlc1 is None:
                    vlc1 = np.block([[dccoef], [self.huffenc(ra1, ehuf)]])
                else:
                    vlc1 = np.block([[vlc1], [dccoef], [self.huffenc(ra1, ehuf)]])
            if vlc is None:
                vlc = vlc1
            else:
                vlc = np.block([[vlc], [vlc1]])

        # Return here if the default tables are sufficient, otherwise repeat the
        # encoding process using the custom designed huffman tables.
        if not opthuff:
            bits = dbits
            huffval = dhuffval
            print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
            return vlc.astype('int'), bits, huffval

        # Design custom huffman tables.
        print('Generating huffcode and ehuf using custom tables')
        dbits, dhuffval = self.huffdes()
        huffcode, ehuf = self.huffgen(dbits, dhuffval)

        # Generate run/ampl values and code them into vlc(:,1:2).
        # Also generate a histogram of code symbols.
        print('Coding rows (second pass)')
        t = np.arange(M)
        self.huffhist = np.zeros((16 ** 2, 1))
        vlc = None
        for r in range(0, sy[0] - M, M):
            vlc1 = None
            for c in range(0, sy[1] - M, M):
                yq = Yq[np.ix_(r+t, c+t)]
                # Possibly regroup
                if M > N:
                    yq = regroup(yq, N)
                yqflat = yq.flatten('F')
                # Encode DC coefficient first
                yqflat[0] += 2 ** (dcbits-1)
                dccoef = np.block([yqflat[0], dcbits])
                # Encode the other AC coefficients in scan order
                ra1 = self.runampl(yqflat[scan])
                # huffenc() also updates huffhist.
                if vlc1 is None:
                    vlc1 = np.block([[dccoef], [self.huffenc(ra1, ehuf)]])
                else:
                    vlc1 = np.block([[vlc1], [dccoef], [self.huffenc(ra1, ehuf)]])
            if vlc is None:
                vlc = vlc1
            else:
                vlc = np.block([[vlc], [vlc1]])

        print('Bits for coded image = {}'.format(sum(vlc[:, 2])))
        print('Bits for huffman table = {}'.format(
            (16 + max(dhuffval.shape))*8))

        bits = dbits
        huffval = dhuffval

        return vlc, bits, huffval

    def suphuffdec(self, vlc, qstep, N=8, M=8, bits=None, huffval=None, dcbits=8, W=256, H=256):
        '''
        Decodes a (simplified) JPEG bit stream to an image

        Z = jpegdec(vlc, qstep, N, M, bits, huffval, dcbits, W, H)
        Decodes the variable length bit stream in vlc to an image in Z.

        vlc is the variable length output code from jpegenc
        qstep is the quantisation step to use in decoding
        N is the width of the DCT block (defaults to 8)
        M is the width of each block to be coded (defaults to N). Must be an
        integer multiple of N - if it is larger, individual blocks are
        regrouped.
        if bits and huffval are supplied, these will be used in Huffman decoding
        of the data, otherwise default tables are used
        dcbits determines how many bits are used to decode the DC coefficients
        of the DCT (defaults to 8)
        W and H determine the size of the image (defaults to 256 x 256)

        Z is the output greyscale image
        '''

        opthuff = (huffval is not None and bits is not None)
        if M % N != 0:
            raise ValueError('M must be an integer multiple of N!')

        # Set up standard scan sequence
        scan = self.diagscan(M)

        if opthuff:
            if len(bits.shape) != 1:
                raise ValueError('bits.shape must be (len(bits),)')
            print('Generating huffcode and ehuf using custom tables')
        else:
            print('Generating huffcode and ehuf using default tables')
            bits, huffval = self.huffdflt(1)

        # Define starting addresses of each new code length in huffcode.
        # 0-based indexing instead of 1
        huffstart = np.cumsum(np.block([0, bits[:15]]))
        # Set up huffman coding arrays.
        huffcode, ehuf = self.huffgen(bits, huffval)

        # Define array of powers of 2 from 1 to 2^16.
        k = np.array([2 ** i for i in range(17)])

        # For each block in the image:

        # Decode the dc coef (a fixed-length word)
        # Look for any 15/0 code words.
        # Choose alternate code words to be decoded (excluding 15/0 ones).
        # and mark these with vector t until the next 0/0 EOB code is found.
        # Decode all the t huffman codes, and the t+1 amplitude codes.

        eob = ehuf[0, :]
        run16 = ehuf[15 * 16, :]
        i = 0
        Zq = np.zeros((H, W))
        t = np.arange(M)

        print('Decoding rows')
        for r in range(0, H, M):
            for c in range(0, W, M):
                yq = np.zeros(M**2)

                # Decode DC coef - assume no of bits is correctly given in vlc table.
                cf = 0
                if vlc[i, 1] != dcbits:
                    raise ValueError(
                        'The bits for the DC coefficient does not agree with vlc table')
                yq[cf] = vlc[i, 0] - 2 ** (dcbits-1)
                i += 1

                # Loop for each non-zero AC coef.
                while np.any(vlc[i, :] != eob):
                    run = 0

                    # Decode any runs of 16 zeros first.
                    while np.all(vlc[i, :] == run16):
                        run += 16
                        i += 1

                    # Decode run and size (in bits) of AC coef.
                    start = huffstart[vlc[i, 1] - 1]
                    res = huffval[start + vlc[i, 0] - huffcode[start, 0]]
                    run += res // 16
                    cf += run + 1
                    si = res % 16
                    i += 1

                    # Decode amplitude of AC coef.
                    if vlc[i, 1] != si:
                        raise ValueError(
                            'Problem with decoding .. you might be using the wrong bits and huffval tables')
                    ampl = vlc[i, 0]

                    # Adjust ampl for negative coef (i.e. MSB = 0).
                    thr = k[si - 1]
                    yq[scan[cf-1]] = ampl - (ampl < thr) * (2 * thr - 1)

                    i += 1

                # End-of-block detected, save block.
                i += 1
                yq = np.reshape(yq, (M, M)).T
                # Possibly regroup yq
                if M > N:
                    yq = regroup(yq, M//N)
                Zq[np.ix_(r+t, c+t)] = yq

        print('Inverse quantising to step size of {}'.format(qstep))
        Zi = quant2(Zq, qstep, qstep)

        print('Inverse {} x {} DCT\n'.format(N, N))
        C8 = dct_ii(N)
        Z = colxfm(colxfm(Zi.T, C8.T).T, C8.T)
        # Z = post_filter(Z, N, scale)
        return Z