"""
metrics.py — Calcul des métriques PSNR et BER
Mini-Projet : Tatouage numérique | L2-IRS 25/26
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """
    Calcule le PSNR (Peak Signal-to-Noise Ratio) en dB.
    Plus le PSNR est élevé, meilleure est la qualité visuelle.
    Objectif projet : PSNR > 40 dB (imperceptible à l'œil nu).

    Paramètres
    ----------
    original : image originale (uint8)
    modified : image tatouée ou attaquée (uint8)

    Retourne
    --------
    psnr_value : valeur PSNR en dB (float)
    """
    return peak_signal_noise_ratio(original, modified, data_range=255)


def compute_ber(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    """
    Calcule le BER (Bit Error Rate).
    BER = 0.0  → extraction parfaite
    BER = 0.5  → extraction aléatoire (pire cas)
    Objectif projet : BER < 0.05 sans attaque.

    Paramètres
    ----------
    original_bits  : watermark original (tableau 1D de 0/1)
    extracted_bits : watermark extrait  (tableau 1D de 0/1)

    Retourne
    --------
    ber_value : taux d'erreur binaire entre 0 et 1 (float)
    """
    if len(original_bits) != len(extracted_bits):
        raise ValueError("Les deux tableaux de bits doivent avoir la même longueur.")
    errors = np.sum(original_bits != extracted_bits)
    return float(errors) / len(original_bits)


def print_metrics(label: str, psnr: float, ber: float) -> None:
    """Affiche les métriques de façon lisible."""
    psnr_status = "✅" if psnr > 40 else "⚠️ "
    ber_status  = "✅" if ber  < 0.05 else ("⚠️ " if ber < 0.2 else "❌")
    print(f"  [{label}]")
    print(f"    {psnr_status} PSNR  : {psnr:.2f} dB  (seuil recommandé > 40 dB)")
    print(f"    {ber_status} BER   : {ber:.4f}     (seuil recommandé < 0.05)")
