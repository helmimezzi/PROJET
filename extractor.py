"""
extractor.py — Module d'extraction du watermark par QIM (DCT 2D)
Mini-Projet : Tatouage numérique | L2-IRS 25/26
"""

import numpy as np
from scipy.fft import dctn


def qim_decode_bit(coeff: float, delta: float) -> int:
    """
    Décodage QIM : détermine le bit encodé dans un coefficient DCT.
    Calcule la distance au réseau pair vs réseau impair et choisit le plus proche.
    """
    q0 = delta * np.round(coeff / delta)           # réseau bit=0
    q1 = delta * (np.round((coeff - delta / 2) / delta) + 0.5)  # réseau bit=1

    dist0 = abs(coeff - q0)
    dist1 = abs(coeff - q1)

    return 0 if dist0 <= dist1 else 1


def extract_watermark(image: np.ndarray, info: dict) -> np.ndarray:
    """
    Extrait le watermark depuis une image (tatouée ou attaquée).

    Paramètres
    ----------
    image : image à analyser (H x W, uint8 ou float)
    info  : dictionnaire retourné par embed_watermark (clé, delta, positions...)

    Retourne
    --------
    extracted : tableau 1D de bits extraits (int32)
    """
    delta      = info["delta"]
    block_size = info["block_size"]
    secret_key = info["secret_key"]
    n_bits     = info["n_bits"]
    coeff_pos  = info["coeff_pos"]
    block_order = info["block_order"]

    img = image.astype(np.float64)
    H, W = img.shape
    n_blocks_w = W // block_size

    extracted = np.zeros(n_bits, dtype=np.int32)
    r, c = coeff_pos

    for idx, block_idx in enumerate(block_order):
        bh = (block_idx // n_blocks_w) * block_size
        bw = (block_idx  % n_blocks_w) * block_size

        block = img[bh:bh+block_size, bw:bw+block_size]
        dct_block = dctn(block, norm='ortho')

        extracted[idx] = qim_decode_bit(dct_block[r, c], delta)

    return extracted
