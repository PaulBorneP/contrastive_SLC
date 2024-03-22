
#taken from https://github.com/hi-paris/deepdespeckling Copyright (c) 2022, Emanuele Dalsasso, deepdespeckling team
import numpy as np
from scipy import signal

M = 10.089038980848645 # what is this ?
m = -1.429329123112601 # what is this ?

def symetrise_real_and_imaginary_parts(real_part: np.array, imag_part: np.array) -> 'tuple[np.array, np.array]':
    """Symetrise given real and imaginary parts to ensure MERLIN properties

    Args:
        real_part (numpy array): real part of the noisy image to symetrise
        imag_part (numpy array): imaginary part of the noisy image to symetrise 

    Returns:
        np.real(ima2), np.imag(ima2) (numpy array, numpy array): symetrised real and imaginary parts of a noisy image
    """
    S = np.fft.fftshift(np.fft.fft2(
        real_part[0, :, :, 0] + 1j * imag_part[0, :, :, 0]))
    p = np.zeros((S.shape[0]))  # azimut (ncol)
    for i in range(S.shape[0]):
        p[i] = np.mean(np.abs(S[i, :]))
    sp = p[::-1]
    c = np.real(np.fft.ifft(np.fft.fft(p) * np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(), p.shape[0])
    d1 = d1[0]
    shift_az_1 = int(round(-(d1 - 1) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_1 = np.roll(p, shift_az_1)
    shift_az_2 = int(
        round(-(d1 - 1 - p.shape[0]) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_2 = np.roll(p, shift_az_2)
    window = signal.gaussian(p.shape[0], std=0.2 * p.shape[0])
    test_1 = np.sum(window * p2_1)
    test_2 = np.sum(window * p2_2)
    # make sure the spectrum is symetrized and zeo-Doppler centered
    if test_1 >= test_2:
        p2 = p2_1
        shift_az = shift_az_1 / p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2 / p.shape[0]
    S2 = np.roll(S, int(shift_az * p.shape[0]), axis=0)

    q = np.zeros((S.shape[1]))  # range (nlin)
    for j in range(S.shape[1]):
        q[j] = np.mean(np.abs(S[:, j]))
    sq = q[::-1]
    # correlation
    cq = np.real(np.fft.ifft(np.fft.fft(q) * np.conjugate(np.fft.fft(sq))))
    d2 = np.unravel_index(cq.argmax(), q.shape[0])
    d2 = d2[0]
    shift_range_1 = int(round(-(d2 - 1) / 2)
                        ) % q.shape[0] + int(q.shape[0] / 2)
    q2_1 = np.roll(q, shift_range_1)
    shift_range_2 = int(
        round(-(d2 - 1 - q.shape[0]) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_2 = np.roll(q, shift_range_2)
    window_r = signal.gaussian(q.shape[0], std=0.2 * q.shape[0])
    test_1 = np.sum(window_r * q2_1)
    test_2 = np.sum(window_r * q2_2)
    if test_1 >= test_2:
        q2 = q2_1
        shift_range = shift_range_1 / q.shape[0]
    else:
        q2 = q2_2
        shift_range = shift_range_2 / q.shape[0]

    Sf = np.roll(S2, int(shift_range * q.shape[0]), axis=1)
    ima2 = np.fft.ifft2(np.fft.ifftshift(Sf))

    return np.real(ima2), np.imag(ima2)


def sar_normalization(sar_patch: np.array) -> 'tuple[np.array, np.array]':
    normalized_sar = np.zeros(sar_patch.shape) # P x 2 x H x W
    for i in range(sar_patch.shape[0]):
        real_part = sar_patch[i, 0, :, :]
        imag_part = sar_patch[i, 1, :, :]
        normalized_sar[i, 0, :, :] = real_im_norm(real_part)
        normalized_sar[i, 1, :, :] = real_im_norm(imag_part)

    return normalized_sar

def real_im_norm(real_part: np.array)-> np.array:
    """Normalize the real part of the noisy image /!\ also works for the imaginary part

    Args:
        real_part / imaginary part (numpy array): real part of the noisy image to normalize

    Returns:
        numpy array: normalized real / imaginary part part of the noisy image
    """

    log_norm = (np.log(real_part**2+1e-3 )-2*m)/(2*M)

    return log_norm


