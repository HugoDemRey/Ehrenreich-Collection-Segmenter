import numpy as np

from app.src.audio_features.features import NoveltyCurve, SelfSimilarityMatrix


def compute_kernel_checkerboard_gaussian(L, var=1.0, normalize=True):
    """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1].
    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L (int): Parameter specifying the kernel size M=2*L+1
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 1.0)
        normalize (bool): Normalize kernel (Default value = True)

    Returns:
        kernel (np.ndarray): Kernel matrix of size M x M
    """
    taper = np.sqrt(1 / 2) / (L * var)
    axis = np.arange(-L, L + 1)
    gaussian1D = np.exp(-(taper**2) * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel


def run_segmentation(
    ssm, sr, threshold, binarize, kernel_size, variance, peak_threshold, sigma
):
    ssm = SelfSimilarityMatrix(ssm, sr)
    print("    * Applying thresholding to SSM")
    ssm = ssm.threshold(thresh=threshold, binarize=binarize)

    print("    * Computing novelty curve from SSM")
    novelty_curve: NoveltyCurve = ssm.compute_novelty_curve(
        kernel_size=kernel_size, variance=variance, exclude_borders=True
    )

    print("    * Smoothing novelty curve")
    novelty_curve = novelty_curve.smooth(sigma=sigma)

    print("    * Finding peaks in novelty curve")
    peaks = novelty_curve.find_peaks(threshold=peak_threshold, distance_seconds=15)

    peaks_sec = peaks / sr

    return novelty_curve, peaks_sec


def run_optimized_segmentation(
    ssm: np.ndarray,
    sr: float,
    threshold: float,
    binarize: bool,
    kernel_size: int,
    variance: float,
    peak_threshold: float,
    sigma: float,
) -> tuple:
    # --- Thresholding (in-place, numpy) ---
    if binarize:
        ssm_thr = (ssm >= threshold).astype(float)
    else:
        # Match the logic of class: set values below threshold to 0, keep others
        ssm_thr = np.where(ssm >= threshold, ssm, 0.0)

    # --- Novelty curve (checkerboard kernel, numpy) ---
    kernel = compute_kernel_checkerboard_gaussian(L=kernel_size, var=variance)
    N = ssm_thr.shape[0]
    M = 2 * kernel_size + 1
    S_padded = np.pad(ssm_thr, pad_width=kernel_size, mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(S_padded, (M, M))
    diagonal_windows = windows[np.arange(N), np.arange(N)]
    novelty = np.einsum("ijk,jk->i", diagonal_windows, kernel)
    # Exclude borders (set to 0)
    right = min(kernel_size, N)
    left = max(0, N - kernel_size)
    novelty[:right] = 0
    novelty[left:] = 0

    # --- Smoothing (in-place, numpy) ---
    from scipy.ndimage import gaussian_filter1d

    novelty = gaussian_filter1d(novelty.astype(float), sigma=sigma)

    # --- Peak picking (numpy) ---
    from scipy.signal import find_peaks as scipy_find_peaks

    height_threshold = np.max(novelty) * peak_threshold
    distance_samples = int(15 * sr)
    peaks, _ = scipy_find_peaks(
        novelty, height=height_threshold, distance=distance_samples
    )
    peaks_sec = peaks / sr

    return NoveltyCurve(novelty, sr), peaks_sec
