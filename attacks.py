"""
attacks.py — Simulation des attaques sur l'image tatouée
Mini-Projet : Tatouage numérique | L2-IRS 25/26
"""

import numpy as np
import cv2


def attack_gaussian_noise(image: np.ndarray, sigma: float = 15.0,
                          seed: int = 0) -> np.ndarray:
    """
    Attaque par bruit gaussien additif.

    Paramètres
    ----------
    image : image tatouée (uint8)
    sigma : écart-type du bruit (plus grand = plus destructeur)
    seed  : graine pour la reproductibilité

    Retourne
    --------
    Image bruitée (uint8)
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy


def attack_jpeg_compression(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Attaque par compression JPEG.

    Paramètres
    ----------
    image   : image tatouée (uint8)
    quality : qualité JPEG [1-100] (plus bas = plus destructeur)

    Retourne
    --------
    Image compressée-décompressée (uint8)
    """
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image, encode_params)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return decoded


def attack_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Attaque par filtre médian (lissage).

    Paramètres
    ----------
    image       : image tatouée (uint8)
    kernel_size : taille du noyau (3, 5, 7...)

    Retourne
    --------
    Image filtrée (uint8)
    """
    return cv2.medianBlur(image, kernel_size)


def attack_scaling(image: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """
    Attaque par redimensionnement (réduction puis agrandissement).

    Paramètres
    ----------
    image : image tatouée (uint8)
    scale : facteur de réduction (ex: 0.5 = moitié de la taille)

    Retourne
    --------
    Image redimensionnée à la taille originale (uint8)
    """
    H, W = image.shape
    small = cv2.resize(image, (int(W * scale), int(H * scale)),
                       interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    return restored
