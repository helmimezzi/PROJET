"""
main.py — Pipeline complet CLI du système de tatouage numérique QIM
Mini-Projet : Sécurisation des images 2D | L2-IRS 25/26

Usage
-----
python main.py                           # mode démo complet
python main.py --image mon_image.png     # image personnalisée
python main.py --delta 30 --key 1234     # paramètres QIM personnalisés
python main.py --bits 64                 # watermark de 64 bits
"""

import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from embedder  import generate_watermark, embed_watermark
from extractor import extract_watermark
from attacks   import (attack_gaussian_noise, attack_jpeg_compression,
                       attack_median_filter, attack_scaling)
from metrics   import compute_psnr, compute_ber, print_metrics


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────

def save_comparison_figure(original, watermarked, results: dict,
                            output_path: str = "resultats_qim.png") -> None:
    """
    Génère une figure complète :
    - Ligne 1 : image originale | tatouée | différence amplifiée
    - Ligne 2 : images après attaques
    - Ligne 3 : graphique PSNR et BER par attaque
    """
    attacks_names = list(results.keys())
    n_attacks = len(attacks_names)

    fig = plt.figure(figsize=(5 * (n_attacks + 1), 14))
    fig.patch.set_facecolor('#0F172A')

    title_kw  = dict(color='white', fontsize=11, fontweight='bold', pad=6)
    metric_kw = dict(color='#94A3B8', fontsize=8)
    cmap = 'gray'

    gs = gridspec.GridSpec(3, max(n_attacks + 1, 3), figure=fig,
                           hspace=0.45, wspace=0.3)

    # ── Ligne 1 : originale / tatouée / différence ──────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(original, cmap=cmap, vmin=0, vmax=255)
    ax0.set_title("Image originale", **title_kw)
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(watermarked, cmap=cmap, vmin=0, vmax=255)
    psnr_wm = compute_psnr(original, watermarked)
    ax1.set_title(f"Image tatouée\nPSNR = {psnr_wm:.1f} dB", **title_kw)
    ax1.axis('off')

    diff = np.abs(watermarked.astype(int) - original.astype(int))
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(diff * 10, cmap='hot', vmin=0, vmax=255)
    ax2.set_title(f"Différence × 10\nMax diff = {diff.max()}", **title_kw)
    ax2.axis('off')

    # ── Ligne 2 : images attaquées ──────────────────────────────────────────
    for i, name in enumerate(attacks_names):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(results[name]["image"], cmap=cmap, vmin=0, vmax=255)
        ax.set_title(
            f"{name}\nPSNR={results[name]['psnr']:.1f} dB | BER={results[name]['ber']:.3f}",
            **title_kw
        )
        ax.axis('off')

    # ── Ligne 3 : graphiques PSNR et BER ────────────────────────────────────
    labels = attacks_names
    psnr_vals = [results[n]["psnr"] for n in labels]
    ber_vals  = [results[n]["ber"]  for n in labels]

    x = np.arange(len(labels))
    bar_w = 0.35
    colors_psnr = ['#22C55E' if p > 40 else '#F59E0B' for p in psnr_vals]
    colors_ber  = ['#22C55E' if b < 0.05 else ('#F59E0B' if b < 0.2 else '#EF4444')
                   for b in ber_vals]

    ax_psnr = fig.add_subplot(gs[2, :max(n_attacks // 2, 1)])
    bars = ax_psnr.bar(x, psnr_vals, color=colors_psnr, width=0.6, edgecolor='white', linewidth=0.5)
    ax_psnr.axhline(40, color='#F59E0B', linestyle='--', linewidth=1.2, label='Seuil 40 dB')
    ax_psnr.set_xticks(x); ax_psnr.set_xticklabels(labels, color='white', fontsize=9)
    ax_psnr.set_ylabel('PSNR (dB)', color='white'); ax_psnr.set_title('PSNR par attaque', **title_kw)
    ax_psnr.set_facecolor('#1E293B'); ax_psnr.tick_params(colors='white')
    ax_psnr.legend(labelcolor='white', facecolor='#1E293B')
    for bar, val in zip(bars, psnr_vals):
        ax_psnr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', color='white', fontsize=8)

    ax_ber = fig.add_subplot(gs[2, max(n_attacks // 2, 1):])
    bars2 = ax_ber.bar(x, ber_vals, color=colors_ber, width=0.6, edgecolor='white', linewidth=0.5)
    ax_ber.axhline(0.05, color='#F59E0B', linestyle='--', linewidth=1.2, label='Seuil 0.05')
    ax_ber.set_xticks(x); ax_ber.set_xticklabels(labels, color='white', fontsize=9)
    ax_ber.set_ylabel('BER', color='white'); ax_ber.set_title('BER par attaque', **title_kw)
    ax_ber.set_facecolor('#1E293B'); ax_ber.tick_params(colors='white')
    ax_ber.legend(labelcolor='white', facecolor='#1E293B')
    for bar, val in zip(bars2, ber_vals):
        ax_ber.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=8)

    fig.suptitle("Système de Tatouage Numérique QIM — Résultats complets",
                 color='white', fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Figure sauvegardée : {output_path}")


# ─────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_pipeline(image_path: str, delta: float, secret_key: int,
                 n_bits: int, output_dir: str = ".") -> None:

    t_start = time.time()
    print("\n" + "="*55)
    print("  TATOUAGE NUMÉRIQUE QIM — Pipeline complet")
    print("="*55)

    # ── 1. Chargement image ──────────────────────────────────────
    print("\n📂 Chargement de l'image...")
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    H, W = original.shape
    print(f"  ✅ {image_path} — {W}×{H} px")

    # ── 2. Génération watermark ──────────────────────────────────
    print(f"\n🔑 Génération du watermark ({n_bits} bits, clé={secret_key})...")
    watermark = generate_watermark(n_bits, secret_key)
    print(f"  ✅ Watermark : {watermark[:16]}... (16 premiers bits affichés)")

    # ── 3. Insertion QIM ─────────────────────────────────────────
    print(f"\n🖊️  Insertion QIM (Δ={delta})...")
    watermarked, info = embed_watermark(original, watermark,
                                        delta=delta, secret_key=secret_key)
    cv2.imwrite(f"{output_dir}/image_tatouee.png", watermarked)

    psnr_embed = compute_psnr(original, watermarked)
    extracted_clean = extract_watermark(watermarked, info)
    ber_clean = compute_ber(watermark, extracted_clean)
    print(f"  ✅ Image tatouée sauvegardée")
    print_metrics("Sans attaque", psnr_embed, ber_clean)

    # ── 4. Attaques ──────────────────────────────────────────────
    print("\n⚔️  Simulation des attaques...")

    attack_list = {
        "Bruit σ=10":    attack_gaussian_noise(watermarked, sigma=10),
        "Bruit σ=25":    attack_gaussian_noise(watermarked, sigma=25),
        "JPEG q=70":     attack_jpeg_compression(watermarked, quality=70),
        "JPEG q=30":     attack_jpeg_compression(watermarked, quality=30),
        "Filtre médian": attack_median_filter(watermarked, kernel_size=3),
        "Scaling ×0.5":  attack_scaling(watermarked, scale=0.5),
    }

    results = {}
    for name, attacked_img in attack_list.items():
        extracted = extract_watermark(attacked_img, info)
        psnr_val  = compute_psnr(watermarked, attacked_img)
        ber_val   = compute_ber(watermark, extracted)
        results[name] = {"image": attacked_img, "psnr": psnr_val, "ber": ber_val}
        print_metrics(name, psnr_val, ber_val)
        cv2.imwrite(f"{output_dir}/attaque_{name.replace(' ', '_').replace('=','')}.png",
                    attacked_img)

    # ── 5. Figure de résultats ───────────────────────────────────
    print("\n📊 Génération de la figure de résultats...")
    save_comparison_figure(original, watermarked, results,
                           output_path=f"{output_dir}/resultats_qim.png")

    t_total = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  ✅ Pipeline terminé en {t_total:.2f} secondes")
    if t_total < 5:
        print(f"  ✅ Temps < 5s — Critère non fonctionnel respecté !")
    else:
        print(f"  ⚠️  Temps > 5s — Envisager d'optimiser")
    print("="*55 + "\n")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Tatouage numérique QIM — Mini-Projet L2-IRS"
    )
    p.add_argument("--image",  default="lena_gray.png",
                   help="Chemin de l'image hôte (défaut: lena_gray.png)")
    p.add_argument("--delta",  type=float, default=25.0,
                   help="Pas de quantification QIM (défaut: 25)")
    p.add_argument("--key",    type=int,   default=42,
                   help="Clé secrète (défaut: 42)")
    p.add_argument("--bits",   type=int,   default=128,
                   help="Nombre de bits du watermark (défaut: 128)")
    p.add_argument("--output", default=".",
                   help="Dossier de sortie (défaut: répertoire courant)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        image_path = args.image,
        delta      = args.delta,
        secret_key = args.key,
        n_bits     = args.bits,
        output_dir = args.output,
    )
