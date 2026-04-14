"""
embedder.py — Module d'insertion du watermark par QIM (DCT 2D)
Mini-Projet : Tatouage numérique | L2-IRS 25/26
"""

import numpy as np
from scipy.fft import dctn, idctn


def generate_watermark(length: int, seed: int) -> np.ndarray:
    """Génère un watermark binaire pseudo-aléatoire reproductible."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=length, dtype=np.int32)


def select_coefficients(n_coeffs_per_block: int, block_size: int, seed: int) -> list[tuple[int,int]]:
    """
    Sélectionne des positions de coefficients DCT en fréquence moyenne
    de façon pseudo-aléatoire via une clé secrète.
    On exclut DC (0,0) et hautes fréquences (dernière ligne/colonne).
    """
    rng = np.random.default_rng(seed + 999)
    candidates = [
        (r, c)
        for r in range(1, block_size - 1)
        for c in range(1, block_size - 1)
    ]
    indices = rng.choice(len(candidates), size=n_coeffs_per_block, replace=False)
    return [candidates[i] for i in indices]


def qim_embed_bit(coeff: float, bit: int, delta: float) -> float:
    """
    Quantification QIM : encode un bit dans un coefficient DCT.
    bit=0 → quantifie vers le réseau pair  (delta * round(c/delta))
    bit=1 → quantifie vers le réseau impair (delta * (round((c - delta/2)/delta) + 0.5))
    """
    if bit == 0:
        return delta * np.round(coeff / delta)
    else:
        return delta * (np.round((coeff - delta / 2) / delta) + 0.5)


def embed_watermark(image: np.ndarray, watermark: np.ndarray,
                    delta: float = 25.0, block_size: int = 8,
                    secret_key: int = 42) -> tuple[np.ndarray, dict]:
    """
    Insère le watermark dans l'image via DCT 2D par blocs + QIM.

    Paramètres
    ----------
    image      : image hôte en niveaux de gris (H x W, uint8)
    watermark  : bits à insérer (tableau 1D de 0/1)
    delta      : pas de quantification QIM (compromis robustesse/invisibilité)
    block_size : taille des blocs DCT (8 recommandé)
    secret_key : graine pour la sélection pseudo-aléatoire des coefficients

    Retourne
    --------
    watermarked : image tatouée (uint8)
    info        : dictionnaire de métadonnées (capacité, blocs, positions...)
    """
    img = image.astype(np.float64)
    H, W = img.shape

    # Nombre de blocs
    n_blocks_h = H // block_size
    n_blocks_w = W // block_size
    total_blocks = n_blocks_h * n_blocks_w

    # 1 bit par bloc (1 coefficient par bloc)
    n_bits = len(watermark)
    if n_bits > total_blocks:
        raise ValueError(
            f"Watermark trop long ({n_bits} bits) pour {total_blocks} blocs disponibles."
        )

    # Sélectionner 1 position de coefficient par bloc (fréquence moyenne)
    positions = select_coefficients(1, block_size, secret_key)
    coeff_pos = positions[0]  # (row, col) dans le bloc DCT

    # Ordre pseudo-aléatoire des blocs pour plus de sécurité
    rng = np.random.default_rng(secret_key)
    block_order = rng.permutation(total_blocks)[:n_bits]

    watermarked = img.copy()
    embedded_positions = []

    for idx, block_idx in enumerate(block_order):
        bh = (block_idx // n_blocks_w) * block_size
        bw = (block_idx  % n_blocks_w) * block_size

        block = watermarked[bh:bh+block_size, bw:bw+block_size]
        dct_block = dctn(block, norm='ortho')

        r, c = coeff_pos
        dct_block[r, c] = qim_embed_bit(dct_block[r, c], int(watermark[idx]), delta)

        watermarked[bh:bh+block_size, bw:bw+block_size] = idctn(dct_block, norm='ortho')
        embedded_positions.append((bh, bw, r, c))

    # Clipper et convertir en uint8
    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)

    info = {
        "delta": delta,
        "block_size": block_size,
        "secret_key": secret_key,
        "n_bits": n_bits,
        "coeff_pos": coeff_pos,
        "block_order": block_order,
        "total_blocks": total_blocks,
        "embedded_positions": embedded_positions,
    }

    return watermarked, info
