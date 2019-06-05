# Author: Jason Li - jl944@cam.ac.uk, Karthik Suresh - ks800@cam.ac.uk
# 2019

import numpy as np
from helpers import *
import scipy.io as sio


class HuffmanHelper:
    '''Base Helper class for Huffman coding'''

    def __init__(self):
        self.huffhist = np.zeros((16 ** 2, 1))

    def diagscan(self, N):
        '''
        Generate diagonal scanning pattern

        Return: scan: a diagonal scanning index for
        an NxN matrix

        The first entry in the matrix is assumed to be the DC coefficient
        and is therefore not included in the scan
        '''
        # Copied from matlab without accounting for indexing.
        slast = N + 1
        scan = [slast]
        while slast != N * N:
            while slast > N and slast % N != 0:
                slast = slast - (N - 1)
                scan.append(slast)
            if slast < N:
                slast = slast + 1
            else:
                slast = slast + N
            scan.append(slast)
            if slast == N * N:
                break
            while slast < (N * N - N + 1) and slast % N != 1:
                slast = slast + (N - 1)
                scan.append(slast)
            if slast == N * N:
                break
            if slast < (N * N - N + 1):
                slast = slast + N
            else:
                slast = slast + 1
            scan.append(slast)
        # Python indexing
        return np.array(scan) - 1

    def runampl(self, a):
        '''
        RUNAMPL Create a run-amplitude encoding from input stream

        [ra] = RUNAMPL(a) Converts the stream of integers in 'a' to a
        run-amplltude encoding in 'ra'

        Column 1 of ra gives the runs of zeros between each non-zero value.
        Column 2 gives the JPEG sizes of the non-zero values (no of
        bits excluding leading zeros).
        Column 3 of ra gives the values of the JPEG remainder, which
        is normally coded in offset binary.

        Parameters:
            a: is a integer stream (array)

        Returns:
            ra: (,3) nparray
        '''
        # Check for non integer values in a
        if sum(abs(np.remainder(a, 1))):
            raise ValueError("Warning! RUNAMPL.M: Attempting to create" +
                             " run-amplitude from non-integer values")
        b = np.where(a != 0)[0]
        if len(b) == 0:
            ra = np.array([[0, 0, 0]])
            return ra

        # List non-zero elements as a column vector
        c = np.reshape(a[b], (b.shape[0], 1)).astype('int')
        # Generate JPEG size vector ca = floor(log2(abs(c)) + 1)
        ca = np.zeros(c.shape).astype('int')
        k = 1
        cb = np.abs(c)
        maxc = np.max(cb)
        ka = np.array([[1]])

        while k <= maxc:
            ca = ca + (cb >= k)
            k = k * 2
            ka = np.concatenate((ka, np.array([[k]])))

        cneg = np.where(c < 0)[0]
        # Changes expression for python indexing
        c[cneg] = c[cneg] + ka[ca[cneg].flatten()] - 1
        bcol = np.reshape(b, (len(b), 1))
        # appended -1 instead of 0.
        col1 = np.diff(np.concatenate((np.array([[-1]]), bcol)).flatten()) - 1
        col1 = np.reshape(col1, (col1.shape[0], 1))
        ra = np.concatenate((col1, ca, c), axis=1)
        ra = np.concatenate((ra, np.array([[0, 0, 0]])))
        return ra

    def huffdes(self):
        """
        HUFFDES Design Huffman table

        [bits, huffval] = huffdes(huffhist) Generates the JPEG table
        bits and huffval from the 256-point histogram of values huffhist.
        This is based on the algorithms in the JPEG Book Appendix K.2.

        Returns:
            bits = (16, ) nparray
            huffval = (162, ) nparray
        """
        # Scale huffhist to sum just less than 32K, allowing for
        # the 162 ones we are about to add.
        # Should hiffhist be changed globally
        huffhist = self.huffhist * (127 * 256)/np.sum(self.huffhist)

        # Add 1 to all valid points so they are given a code word.
        # With the scaling, this should ensure that all probabilities exceed
        # 1 in 2^16 so that no code words exceed 16 bits in length.
        # This is probably a better method with poor statistics
        # than the JPEG method of fig K.3.
        # Every 16 values made column.
        freq = np.reshape(huffhist, (16, 16), 'F')
        freq[1:10, :] = freq[1:10, :] + 1.1
        freq[0, [0, 15]] = freq[0, [0, 15]] + 1.1

        # Reshape to a vector and add a 257th point to reserve the FFFF codeword.
        # Also add a small negative ramp so that min() always picks the
        # larger index when 2 points have the same probability.
        freq = (np.append(freq.flatten('F'), 1) -
                np.arange(0, 257, 1) * (10 ** -6))

        codesize = np.zeros(257, dtype=int)
        others = -np.ones(257, dtype=int)

        # Find Huffman code sizes: JPEG fig K.1, procedure Code_size

        # Find non-zero entries in freq and loop until 1 entry left.
        nz = np.where(freq > 0)[0]  # np.where output is 2 element tuple.
        while len(nz) > 1:

            # Find v1 for least value of freq(v1) > 0.
            # min in each column
            i = np.argmin(freq[nz])  # freq[nz] and nz index
            v1 = nz[i]

            # Find v2 for next least value of freq(v2) > 0.
            nz = np.delete(nz, i)  # Remove v1 from nz.
            i = np.argmin(freq[nz])  # freq[nz] and nz index
            v2 = nz[i]

            # Combine frequency values to gradually reduce the code table size.
            freq[v1] = freq[v1] + freq[v2]
            freq[v2] = 0

            # Increment the codeword lengths for all codes in the two combined sets.
            # Set 1 is v1 and all its members, stored as a linked list using
            # non-negative entries in vector others(). The members of set 1 are the
            # frequencies that have already been combined into freq(v1).
            codesize[v1] += 1
            while others[v1] > -1:
                v1 = others[v1]
                codesize[v1] += 1

            others[v1] = v2  # Link set 1 with set 2.

            # Set 2 is v2 and all its members, stored as a linked list using
            # non-negative entries in vector others(). The members of set 2 are the
            # frequencies that have already been combined into freq(v2).
            codesize[v2] = codesize[v2] + 1
            while others[v2] > -1:
                v2 = others[v2]
                codesize[v2] = codesize[v2] + 1

            nz = np.where(freq > 0)[0]

        # Find no. of codes of each size: JPEG fig K.2, procedure Count_BITS

        bits = np.zeros(max(16, max(codesize)))
        for i in range(256):
            if codesize[i] > 0:
                bits[codesize[i]] = bits[codesize[i]] + 1

        # Code length limiting not needed since 1's added earlier to all valid
        # codes.
        if max(codesize) > 16:
            # This should not happen.
            raise ValueError('Warning! HUFFDES.M: max(codesize) > 16')

        # Sort codesize values into ascending order and generate huffval:
        # JPEG fig K.4, procedure Sort_input.

        huffval = np.array([], dtype=int)
        t = np.arange(0, 256, 1)
        for i in range(16):
            ii = np.where(codesize[t] == i)[0]
            huffval = np.concatenate((huffval, ii))

        if len(huffval) != sum(bits):
            # This should not happen.
            raise ValueError(
                'Warning! HUFFDES.M: length of huffval ~= sum(bits)')

        return [bits, huffval]

    def huffdflt(self, typ):
        """
        HUFFDFLT Generates default JPEG huffman table
        [bits, huffval] = HUFFDFLT(type) Produces the luminance (type=1) or
        chrominance (type=2) tables.

        The number of values per bit level is stored in 'bits', with the
        corresponding codes in 'huffval'.

        Parameters:
            typ: Integer
        Returns:
            bits: (16, ) nparray
            huffval: (162, ) nparray
        """
        if typ == 1:
            bits = np.array([0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125])
            huffval = np.concatenate((
                [],  # 1-bit
                [1, 2],  # 2-bit
                [3],  # 3-bit
                [0, 4, 17],  # 4-bit
                [5, 18, 33],  # 5-bit
                [49, 65],  # 6-bit
                [6, 19, 81, 97],  # 7-bit
                [7, 34, 113],  # 8-bit
                [20, 50, 129, 145, 161],  # 9-bit
                [8, 35, 66, 177, 193],  # 10-bit
                [21, 82, 209, 240],  # 11-bit
                [36, 51, 98, 114],  # 12-bit
                [],  # 13-bit
                [],  # 14-bit
                [130],  # 15-bit
                [9, 10,  22,  23,  24,  25,  26,  37,  38,  39,
                    40,  41,  42,  52,  53,  54,  55],  # 16-bit
                [56, 57,  58,  67,  68,  69,  70,  71,  72,
                    73,  74,  83,  84,  85,  86,  87,  88,  89],
                [90,  99, 100, 101, 102, 103, 104, 105, 106,
                    115, 116, 117, 118, 119, 120, 121, 122, 131],
                [132, 133, 134, 135, 136, 137, 138, 146, 147,
                    148, 149, 150, 151, 152, 153, 154, 162, 163],
                [164, 165, 166, 167, 168, 169, 170, 178, 179,
                    180, 181, 182, 183, 184, 185, 186, 194, 195],
                [196, 197, 198, 199, 200, 201, 202, 210, 211,
                    212, 213, 214, 215, 216, 217, 218, 225, 226],
                [227, 228, 229, 230, 231, 232, 233, 234, 241,
                    242, 243, 244, 245, 246, 247, 248, 249, 250]
            )).astype('int')
        else:
            bits = np.array([0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119])
            huffval = np.concatenate((
                [],  # 1-bit
                [0, 1],  # 2-bit
                [2],  # 3-bit
                [3, 17],  # 4-bit
                [4, 5,  33,  49],  # 5-bit
                [6, 18,  65,  81],  # 6-bit
                [7,  97, 113],  # 7-bit
                [19,  34,  50, 129],  # 8-bit
                [8,  20,  66, 145, 161, 177, 193],  # 9-bit
                [9,  35,  51,  82, 240],  # 10-bit
                [21,  98, 114, 209],  # 11-bit
                [10,  22,  36,  52],  # 12-bit
                [],  # 13-bit
                [225],  # 14-bit
                [37, 241],  # 15-bit
                [23,  24,  25,  26,  38,  39,  40,  41,  42,  53,  54],  # 16-bit
                [55, 56,  57,  58,  67,  68,  69,  70,  71,
                    72,  73,  74,  83,  84,  85,  86,  87,  88],
                [89,  90,  99, 100, 101, 102, 103, 104, 105,
                    106, 115, 116, 117, 118, 119, 120, 121, 122],
                [130, 131, 132, 133, 134, 135, 136, 137, 138,
                    146, 147, 148, 149, 150, 151, 152, 153, 154],
                [162, 163, 164, 165, 166, 167, 168, 169, 170,
                    178, 179, 180, 181, 182, 183, 184, 185, 186],
                [194, 195, 196, 197, 198, 199, 200, 201, 202,
                    210, 211, 212, 213, 214, 215, 216, 217, 218],
                [226,  227,  228,  229,  230,  231,  232,  233,  234,
                    242,  243,  244,  245,  246,  247,  248,  249,  250]
            )).astype('int')

        return [bits, huffval]

    def huffenc(self, rsa, ehuf):
        """
        HUFFENC Convert a run-length encoded stream to huffman
        coding.

        [vlc] = HUFFENC(rsa) Performs Huffman encoding on the
        run-length information in rsa, as produced by RUNAMPL.

        THe codewords are variable length integers in vlc(:, 0)
        whose lengths are in vlc(:, 1). ehuf contains the huffman
        codes and their lengths. The global matrix huffhist is
        updated for use in HUFFGEN.

        Huffhist is global to avoid in-efficient copying each
        time this function is called.
        """
        if max(rsa[:, 1]) > 10:
            print("Warning: Size of value in run-amplitude " +
                  "code is too large for Huffman table")
            rsa[np.where(rsa[:, 1] > 10), 2] = (2 ** 10) - 1
            rsa[np.where(rsa[:, 1] > 10), 1] = 10

        r, c = rsa.shape

        vlc = None
        for i in range(r):
            run = rsa[i, 0]
            # If run > 15, use repeated codes for 16 zeros.
            while run > 15:
                # Got rid off + 1 to suit python indexing.
                code = 15 * 16
                self.huffhist[code] = self.huffhist[code] + 1
                if vlc is None:
                    vlc = np.array([ehuf[code, :]])
                else:
                    vlc = np.append(vlc, np.array([ehuf[code, :]]), axis=0)
                run = run - 16
            # Code the run and size.
            # Got rid off + 1 to suit python indexing.
            code = run * 16 + rsa[i, 1]
            self.huffhist[code] = self.huffhist[code] + 1
            if vlc is None:
                vlc = np.array([ehuf[code, :]])
            else:
                vlc = np.append(vlc, np.array([ehuf[code, :]]), axis=0)
            # If size > 0, add in the remainder (which is not coded).
            if rsa[i, 1] > 0:
                if vlc is None:
                    vlc = np.array([rsa[i, [2, 1]]])
                else:
                    vlc = np.append(vlc, np.array([rsa[i, [2, 1]]]), axis=0)
        return vlc

    def huffgen(self, bits, huffval):
        """
        HUFFGEN Generate huffman codes

        [huffcode, ehuf] = HUFFGEN(bits, huffval) Translates the number
        of codes at each bit (in bits) and the valid values (in huffval).

        huffcode lists the valid codes in ascending order. ehuf is a
        two-column vector, with one entry per possible 8-bit value. The
        first column lists the code for that value, and the second lists
        the length in bits.

        Parameters:
            bits: 1D Numpy array.
            huffval: 1D Numpy array.
        Returns:
            huffcode: nparray (ncodes, 1)
            ehuf: nparray (256, 2)
        """

        # Generate huffman size table (JPEG fig C1, p78):
        nb = bits.shape[0]
        k = 1  # Max value of k is 162
        j = 1
        # sum on nparray sums columns.
        ncodes = sum(bits)
        # Check every where 1_D array of zeros/ones defined like this.
        huffsize = np.zeros((ncodes, 1), dtype=int)

        for i in range(nb):
            while j <= bits[i]:
                huffsize[k - 1, 0] = i + 1
                k += 1
                j += 1
            j = 1

        huffcode = np.zeros((ncodes, 1), dtype=int)
        code = 0
        si = huffsize[0, 0]

        # Generate huffman code table (JPEG fig C2, p79)
        for k in range(ncodes):
            while huffsize[k, 0] > si:
                code = code * 2
                si += 1
            huffcode[k, 0] = code
            code += 1

        # Reorder the code tables according to the data in
        # huffval to yield the encoder look-up tables.
        ehuf = np.zeros((256, 2), dtype=int)
        ehuf[huffval, :] = np.concatenate((huffcode, huffsize), axis=1)

        return [huffcode, ehuf]


class JpegHuffmanHelper(HuffmanHelper):
    """Helper Class for JPEG and Huffman coding"""

    def jpegenc(self, X, qstep, N=8, M=8, opthuff=False, dcbits=8, log=True):
        '''
        Image in X to generate the variable length bit stream in vlc.

        X is the input greyscale image
        qstep is the quantisation step to use in encoding
        N is the width of the DCT block (defaults to 8)
        M is the width of each block to be coded (defaults to N). Must be an
        integer multiple of N - if it is larger, individual blocks are
        regrouped.
        if opthuff is true (defaults to false), the Huffman table is optimised
        based on the data in X
        dcbits determines how many bits are used to encode the DC coefficients
        of the DCT (defaults to 8)

        Return: vlc, bits, huffval

        vlc is the variable length output code, where vlc[:,0] are the codes, and
        vlc[:,1] the number of corresponding valid bits, so that sum(vlc[:,1])
        gives the total number of bits in the image
        bits and huffval are optional outputs which return the Huffman encoding
        used in compression
        '''

        if M % N != 0:
            raise ValueError('M must be an integer multiple of N!')

        # DCT on input image X.
        if log:
            print('Forward {} x {} DCT'.format(N, N))
        C8 = dct_ii(N)
        Y = colxfm(colxfm(X, C8).T, C8).T

        # Quantise to integers.
        if log:
            print('Quantising to step size of {}'.format(qstep))
        Yq = quant1(Y, qstep, qstep).astype('int')

        # Generate zig-zag scan of AC coefs.
        scan = self.diagscan(M)

        # On the first pass use default huffman tables.
        if log:
            print('Generating huffcode and ehuf using default tables')
        dbits, dhuffval = self.huffdflt(1)  # Default tables.
        huffcode, ehuf = self.huffgen(dbits, dhuffval)

        # Generate run/ampl values and code them into vlc(:,1:2).
        # Also generate a histogram of code symbols.
        if log:
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
                    vlc1 = np.block(
                        [[vlc1], [dccoef], [self.huffenc(ra1, ehuf)]])
            if vlc is None:
                vlc = vlc1
            else:
                vlc = np.block([[vlc], [vlc1]])

        # Return here if the default tables are sufficient, otherwise repeat the
        # encoding process using the custom designed huffman tables.
        if not opthuff:
            bits = dbits
            huffval = dhuffval
            if log:
                print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
            return vlc.astype('int'), bits, huffval

        # Design custom huffman tables.
        if log:
            print('Generating huffcode and ehuf using custom tables')
        dbits, dhuffval = self.huffdes()
        huffcode, ehuf = self.huffgen(dbits, dhuffval)

        # Generate run/ampl values and code them into vlc(:,1:2).
        # Also generate a histogram of code symbols.
        if log:
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
                    vlc1 = np.block(
                        [[vlc1], [dccoef], [self.huffenc(ra1, ehuf)]])
            if vlc is None:
                vlc = vlc1
            else:
                vlc = np.block([[vlc], [vlc1]])

        if log:
            print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
            print('Bits for huffman table = {}'.format(
                (16 + max(dhuffval.shape))*8))

        bits = dbits
        huffval = dhuffval

        return vlc, bits, huffval

    def jpegdec(self, vlc, qstep, N=8, M=8, bits=None, huffval=None, dcbits=8, W=256, H=256, log=True):
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
            if log:
                print('Generating huffcode and ehuf using custom tables')
        else:
            if log:
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

        if log:
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

                yq = yq.reshape((M, M)).T

                # Possibly regroup yq
                if M > N:
                    yq = regroup(yq, M//N)
                Zq[np.ix_(r+t, c+t)] = yq

        if log:
            print('Inverse quantising to step size of {}'.format(qstep))

        Zi = quant2(Zq, qstep, qstep)

        if log:
            print('Inverse {} x {} DCT\n'.format(N, N))
        C8 = dct_ii(N)
        Z = colxfm(colxfm(Zi.T, C8.T).T, C8.T)

        return Z


class DwtHuffmanHelper(HuffmanHelper):
    '''Helper class for DWT and Huffman coding'''

    def dwtgroup(self, X, n):
        '''
        dwtgroup Change ordering of elements in a matrix

        Y = dwtgroup(X,n) Regroups the rows and columns of X, such that an
        n-level DWT image composed of separate subimages is regrouped into 2^n x
        2^n blocks of coefs from the same spatial region (like the DCT).

        If n is negative the process is reversed.
        '''
        Y = X.copy()

        if n == 0:
            return Y
        elif n < 0:
            n = -n
            invert = 1
        else:
            invert = 0

        sx = X.shape
        N = np.round(2**n)

        if sx[0] % N != 0 or sx[1] % N != 0:
            raise ValueError(
                'Error in dwtgroup: X dimensions are not multiples of 2^n')

        if invert == 0:
            # Determine size of smallest sub-image.
            sx = sx // N

            # Regroup the 4 subimages at each level, starting with the smallest
            # subimages in the top left corner of Y.
            k = 1  # Size of blocks of pels at each level.
            # tm = 1:sx[0];
            # tn = 1:sx[1];
            tm = np.arange(sx[0])
            tn = np.arange(sx[1])

            # Loop for each level.
            for _ in range(n):
                tm2 = np.block([
                    [np.reshape(tm, (k, sx[0]//k), order='F')],
                    [np.reshape(tm+sx[0], (k, sx[0]//k), order='F')]
                ])
                tn2 = np.block([
                    [np.reshape(tn, (k, sx[1]//k), order='F')],
                    [np.reshape(tn+sx[1], (k, sx[1]//k), order='F')]
                ])

                sx = sx * 2
                k = k * 2
                tm = np.arange(sx[0])
                tn = np.arange(sx[1])
                Y[np.ix_(tm, tn)] = Y[np.ix_(
                    tm2.flatten('F'), tn2.flatten('F'))]

        else:
            # Invert the grouping:
            # Determine size of largest sub-image.
            sx = np.array(X.shape) // 2

            # Regroup the 4 subimages at each level, starting with the largest
            # subimages in Y.
            k = N // 2  # Size of blocks of pels at each level.

            # Loop for each level.
            for _ in np.arange(n):
                tm = np.arange(sx[0])
                tn = np.arange(sx[1])
                tm2 = np.block([
                    [np.reshape(tm, (k, sx[0]//k), order='F')],
                    [np.reshape(tm+sx[0], (k, sx[0]//k), order='F')]
                ])
                tn2 = np.block([
                    [np.reshape(tn, (k, sx[1]//k), order='F')],
                    [np.reshape(tn+sx[1], (k, sx[1]//k), order='F')]
                ])

                Y[np.ix_(tm2.flatten('F'), tn2.flatten('F'))] = Y[np.ix_(
                    np.arange(sx[0]*2), np.arange(sx[1]*2))]

                sx = sx // 2
                k = k // 2

        return Y

    def quantdwt1(self, Y, Q, n):
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
        # dwtent = np.zeros_like(Q, dtype=float)
        for m in range(n + 1):
            Nsub = m_arr[m]
            if m == n:
                Yq[:2*Nsub, :2 *
                    Nsub] = quant1(Yq[:2*Nsub, :2*Nsub], Q[0, m], Q[0, m])
                # dwtent[0, m] = total_bits([Yq[:2*Nsub, :2*Nsub]])
                break
            Yq[:Nsub, Nsub:2 *
                Nsub] = quant1(Yq[:Nsub, Nsub:2*Nsub], Q[0, m], Q[0, m])
            Yq[Nsub:2*Nsub, Nsub:2 *
                Nsub] = quant1(Yq[Nsub:2*Nsub, Nsub:2*Nsub], Q[1, m], Q[1, m])
            Yq[Nsub:2*Nsub,
                :Nsub] = quant1(Yq[Nsub:2*Nsub, :Nsub], Q[2, m], Q[2, m])

            # dwtent[0, m] = total_bits([Yq[:Nsub, Nsub:2*Nsub]])
            # dwtent[1, m] = total_bits([Yq[Nsub:2*Nsub, Nsub:2*Nsub]])
            # dwtent[2, m] = total_bits([Yq[Nsub:2*Nsub, :Nsub]])
        return Yq

    def quantdwt2(self, Y, Q, n):
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
        # dwtent = np.zeros_like(Q, dtype=float)
        for m in range(n + 1):
            Nsub = m_arr[m]
            if m == n:
                Yq[:2*Nsub, :2 *
                    Nsub] = quant2(Yq[:2*Nsub, :2*Nsub], Q[0, m], Q[0, m])
                # dwtent[0, m] = total_bits([Yq[:2*Nsub, :2*Nsub]])
                break
            Yq[:Nsub, Nsub:2 *
                Nsub] = quant2(Yq[:Nsub, Nsub:2*Nsub], Q[0, m], Q[0, m])
            Yq[Nsub:2*Nsub, Nsub:2 *
                Nsub] = quant2(Yq[Nsub:2*Nsub, Nsub:2*Nsub], Q[1, m], Q[1, m])
            Yq[Nsub:2*Nsub,
                :Nsub] = quant2(Yq[Nsub:2*Nsub, :Nsub], Q[2, m], Q[2, m])

            # dwtent[0, m] = total_bits([Yq[:Nsub, Nsub:2*Nsub]])
            # dwtent[1, m] = total_bits([Yq[Nsub:2*Nsub, Nsub:2*Nsub]])
            # dwtent[2, m] = total_bits([Yq[Nsub:2*Nsub, :Nsub]])
        return Yq

    def dwtQarr(self, n):
        Z = np.zeros((256, 256))
        m_arr = [256//(2 ** i) for i in range(1, n+1)]
        I = [[] for i in range(n+1)]
        for index,m in enumerate(m_arr):
            I[index].append((m//2, m + m//2))
            I[index].append((m + m//2, m + m//2))
            I[index].append((m + m//2, m//2))
            if index == len(m_arr) - 1:
                I[index + 1].append((m//2, m//2))
        Q = np.zeros((3,n+1))
        for k in range(3):
            for j in range(n + 1):
                if not (k > 0 and j == n):
                    Y = nlevdwt(Z, n)
                    Y[I[j][k]] = 100
                    Xi = nlevidwt(Y, n)
                    Q[k,j] = 1/(np.sum(np.square(Xi)) ** 0.5)
        return Q

    def dwtenc(self, X, qstep,
     n, opthuff=False, dcbits=8, h1=None, h2=None, prop=None, log=False):
        '''
        Image in X to generate the variable length bit stream in vlc.

        X is the input greyscale image
        if opthuff is true (defaults to false), the Huffman table is optimised
        based on the data in X
        dcbits determines how many bits are used to encode the DC coefficients
        of the DCT (defaults to 8)

        Return: vlc, bits, huffval

        vlc is the variable length output code, where vlc[:,0] are the codes, and
        vlc[:,1] the number of corresponding valid bits, so that sum(vlc[:,1])
        gives the total number of bits in the image
        bits and huffval are optional outputs which return the Huffman encoding
        used in compression
        '''

        # DWT on input image X.
        Y = nlevdwt(X, n, h1=h1, h2=h2)

        Q = self.dwtQarr(n)

        # Quantise to Q matrix
        # if log:
        #     print('Quantising to the Q matrix:')
        #     print(Q)
        Yq = self.quantdwt1(Y, Q * qstep, n)
        self.Yq = Yq
        Yq = self.dwtgroup(Yq, n).astype('int')
        self.Yq_grouped = Yq
        # Yq = quant1(Y, qstep, qstep).astype('int')
        # self.Yq = Yq
        # Yq = self.dwtgroup(Yq, n)
        # self.Yq_grouped = Yq
        # assert np.array_equal(self.Yq, self.dwtgroup(self.Yq_grouped, -n))

        # Generate zig-zag scan of AC coefs.
        M = N = 2 ** n
        scan = self.diagscan(M)

        # On the first pass use default huffman tables.
        if log:
            print('Generating huffcode and ehuf using default tables')
        dbits, dhuffval = self.huffdflt(1)  # Default tables.
        huffcode, ehuf = self.huffgen(dbits, dhuffval)

        # Generate run/ampl values and code them into vlc(:,1:2).
        # Also generate a histogram of code symbols.
        if log:
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
                    vlc1 = np.block(
                        [[vlc1], [dccoef], [self.huffenc(ra1, ehuf)]])
            if vlc is None:
                vlc = vlc1
            else:
                vlc = np.block([[vlc], [vlc1]])

        # Return here if the default tables are sufficient, otherwise repeat the
        # encoding process using the custom designed huffman tables.
        if not opthuff:
            bits = dbits
            huffval = dhuffval
            if log:
                print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
            return vlc.astype('int'), bits, huffval

        # Design custom huffman tables.
        if log:
            print('Generating huffcode and ehuf using custom tables')
        dbits, dhuffval = self.huffdes()
        huffcode, ehuf = self.huffgen(dbits, dhuffval)

        # Generate run/ampl values and code them into vlc(:,1:2).
        # Also generate a histogram of code symbols.
        if log:
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
                    vlc1 = np.block(
                        [[vlc1], [dccoef], [self.huffenc(ra1, ehuf)]])
            if vlc is None:
                vlc = vlc1
            else:
                vlc = np.block([[vlc], [vlc1]])

        if log:
            print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
            print('Bits for huffman table = {}'.format(
                (16 + max(dhuffval.shape))*8))

        bits = dbits
        huffval = dhuffval

        return vlc, bits, huffval

    def dwtdec(self, vlc, qstep, n, opthuff=False, bits=None, huffval=None, W=256, H=256, dcbits=8, g1=None, g2=None, prop=None, log=False):
        '''
        Decodes a (simplified) DWT bit stream to an image

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

        M = N = 2 ** n
        # Set up standard scan sequence
        scan = self.diagscan(M)

        if opthuff:
            if len(bits.shape) != 1:
                raise ValueError('bits.shape must be (len(bits),)')
            if log:
                print('Generating huffcode and ehuf using custom tables')
        else:
            if log:
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
        Zq = np.zeros((H, W), dtype=int)
        t = np.arange(M)

        if log:
            print('Decoding rows')
        for r in range(0, H, M):
            for c in range(0, W, M):
                yq = np.zeros(M**2, dtype=int)

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

                yq = yq.reshape((M, M)).T

                # Possibly regroup yq
                if M > N:
                    yq = regroup(yq, M//N)
                Zq[np.ix_(r+t, c+t)] = yq

        if log:
            print('Inverse quantising to step size of {}'.format(qstep))

        # assert np.array_equal(self.Yq_grouped, Zq)
        Z = self.dwtgroup(Zq, -n)
        # assert np.array_equal(self.Yq, Z)
        Q = self.dwtQarr(n)
        Z = self.quantdwt2(Z, Q * qstep, n)
        Z = nlevidwt(Z, n, g1=g1, g2=g2)

        return Z
