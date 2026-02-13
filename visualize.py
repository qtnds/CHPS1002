#!/usr/bin/env python3
"""
Script de visualisation des résultats IBSI
Histogrammes comparatifs Passe 1 vs Passe 2
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def read_ibsi_results(filename):
    """Lecture des résultats IBSI"""
    data = []
    threshold = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Lecture du seuil dans l'en-tête
            if line.startswith('# Seuil liaisons covalentes:'):
                try:
                    threshold = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('# Seuil liaisons non-covalentes:'):
                try:
                    threshold = float(line.split(':')[1].strip())
                except:
                    pass
            
            # Ignorer les autres lignes de commentaires
            if not line or line.startswith('#'):
                continue
            
            # Parser la ligne de données
            parts = line.split()
            if len(parts) >= 4:
                try:
                    i = int(parts[0])
                    j = int(parts[1])
                    ibsi = float(parts[2])
                    bond_type = int(parts[3])
                    data.append([i, j, ibsi, bond_type])
                except ValueError:
                    continue
    
    if len(data) == 0:
        raise ValueError(f"Aucune donnée valide trouvée dans {filename}")
    
    return np.array(data), threshold


def plot_dual_histograms(data_pass1, threshold_cov, data_pass2, threshold_nc):
    """Histogrammes comparatifs Passe 1 et Passe 2"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # =========================================================================
    # PASSE 1 - Liaisons covalentes
    # =========================================================================
    ibsi_pass1 = data_pass1[:, 2]
    bond_types_pass1 = data_pass1[:, 3]
    
    # Classifier en 3 catégories selon les deux seuils
    if threshold_nc is not None:
        # Pas de liaison (< seuil_nc)
        no_bond_p1 = ibsi_pass1[ibsi_pass1 < threshold_nc]
        # Non-covalent (seuil_nc <= IBSI < seuil_cov)
        non_covalent_p1 = ibsi_pass1[(ibsi_pass1 >= threshold_nc) & (ibsi_pass1 < threshold_cov)]
        # Covalent (>= seuil_cov)
        covalent_p1 = ibsi_pass1[ibsi_pass1 >= threshold_cov]
    else:
        # Si pas de seuil_nc, utiliser l'ancienne classification binaire
        no_bond_p1 = ibsi_pass1[bond_types_pass1 == 0]
        non_covalent_p1 = np.array([])  # vide
        covalent_p1 = ibsi_pass1[bond_types_pass1 == 1]
    
    # Bins adaptés à la passe 1
    bins_p1 = np.linspace(0, max(ibsi_pass1.max(), threshold_cov * 1.5), 40)
    
    # Histogrammes passe 1 - TROIS catégories
    if len(no_bond_p1) > 0:
        ax1.hist(no_bond_p1, bins=bins_p1, color='lightcoral', alpha=0.7, 
                label=f'Pas de liaison (n={len(no_bond_p1)})', 
                edgecolor='darkred', linewidth=1.2)
    
    if len(non_covalent_p1) > 0:
        ax1.hist(non_covalent_p1, bins=bins_p1, color='gold', alpha=0.7, 
                label=f'Non-covalentes (n={len(non_covalent_p1)})', 
                edgecolor='orange', linewidth=1.2)
    
    if len(covalent_p1) > 0:
        ax1.hist(covalent_p1, bins=bins_p1, color='lightgreen', alpha=0.7, 
                label=f'Covalentes (n={len(covalent_p1)})', 
                edgecolor='darkgreen', linewidth=1.2)
    
    # Seuils passe 1 - DEUX seuils
    if threshold_nc is not None:
        ax1.axvline(threshold_nc, color='darkorange', linestyle='--', linewidth=2.5, 
                   label=f'Seuil non-cov = {threshold_nc:.4f}', zorder=10)
    
    if threshold_cov is not None:
        ax1.axvline(threshold_cov, color='darkgreen', linestyle='--', linewidth=2.5, 
                   label=f'Seuil cov = {threshold_cov:.4f}', zorder=10)
    
    # Zones annotées passe 1 - TROIS zones
    y_max_p1 = ax1.get_ylim()[1]
    
    if threshold_nc is not None and threshold_cov is not None:
        # Zone "Pas de liaison"
        ax1.axvspan(0, threshold_nc, alpha=0.15, color='red', zorder=0)
        ax1.text(threshold_nc/2, y_max_p1*0.95, 'Pas de liaison', 
                ha='center', va='top', fontsize=11, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Zone "Non-covalent"
        ax1.axvspan(threshold_nc, threshold_cov, alpha=0.15, color='orange', zorder=0)
        ax1.text((threshold_nc + threshold_cov)/2, y_max_p1*0.95, 'Non-covalent', 
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Zone "Covalent"
        ax1.axvspan(threshold_cov, ax1.get_xlim()[1], alpha=0.15, color='green', zorder=0)
        ax1.text((threshold_cov + ax1.get_xlim()[1])/2, y_max_p1*0.95, 'Covalent', 
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    elif threshold_cov is not None:
        # Fallback si pas de seuil_nc (ancienne version)
        ax1.axvspan(0, threshold_cov, alpha=0.15, color='red', zorder=0)
        ax1.text(threshold_cov/2, y_max_p1*0.95, 'Pas de liaison', 
                ha='center', va='top', fontsize=11, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.axvspan(threshold_cov, ax1.get_xlim()[1], alpha=0.15, color='green', zorder=0)
        ax1.text((threshold_cov + ax1.get_xlim()[1])/2, y_max_p1*0.95, 'Covalent', 
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Mise en forme passe 1
    ax1.set_xlabel('Valeur IBSI', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Fréquence', fontsize=13, fontweight='bold')
    ax1.set_title('PASSE 1 - Toutes les paires d\'atomes\n(Classification selon les deux seuils)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    
    # Statistiques passe 1
    stats_p1 = f"Total paires: {len(ibsi_pass1)}\n"
    stats_p1 += f"Pas de liaison: {len(no_bond_p1)} ({len(no_bond_p1)/len(ibsi_pass1)*100:.1f}%)\n"
    stats_p1 += f"Non-covalentes: {len(non_covalent_p1)} ({len(non_covalent_p1)/len(ibsi_pass1)*100:.1f}%)\n"
    stats_p1 += f"Covalentes: {len(covalent_p1)} ({len(covalent_p1)/len(ibsi_pass1)*100:.1f}%)"
    
    ax1.text(0.02, 0.98, stats_p1, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # =========================================================================
    # PASSE 2 - Liaisons non-covalentes
    # =========================================================================
    if data_pass2 is not None:
        ibsi_pass2 = data_pass2[:, 2]
        bond_types_pass2 = data_pass2[:, 3]
        
        no_bond_p2 = ibsi_pass2[bond_types_pass2 == 0]
        non_covalent_p2 = ibsi_pass2[bond_types_pass2 == 2]
        
        # Bins adaptés à la passe 2
        bins_p2 = np.linspace(0, max(ibsi_pass2.max(), threshold_nc * 2), 40)
        
        # Histogrammes passe 2
        if len(no_bond_p2) > 0:
            ax2.hist(no_bond_p2, bins=bins_p2, color='lightcoral', alpha=0.7, 
                    label=f'Pas de liaison (n={len(no_bond_p2)})', 
                    edgecolor='darkred', linewidth=1.2)
        
        if len(non_covalent_p2) > 0:
            ax2.hist(non_covalent_p2, bins=bins_p2, color='gold', alpha=0.7, 
                    label=f'Non-covalentes (n={len(non_covalent_p2)})', 
                    edgecolor='orange', linewidth=1.2)
        
        # Seuil passe 2
        if threshold_nc is not None:
            ax2.axvline(threshold_nc, color='darkorange', linestyle='--', linewidth=2.5, 
                       label=f'Seuil = {threshold_nc:.4f}', zorder=10)
        
        # Zones annotées passe 2
        y_max_p2 = ax2.get_ylim()[1]
        
        if threshold_nc is not None:
            ax2.axvspan(0, threshold_nc, alpha=0.15, color='red', zorder=0)
            ax2.text(threshold_nc/2, y_max_p2*0.95, 'Pas de liaison', 
                    ha='center', va='top', fontsize=11, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax2.axvspan(threshold_nc, ax2.get_xlim()[1], alpha=0.15, color='orange', zorder=0)
            ax2.text((threshold_nc + ax2.get_xlim()[1])/2, y_max_p2*0.95, 'Non-covalent', 
                    ha='center', va='top', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Mise en forme passe 2
        ax2.set_xlabel('Valeur IBSI', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Fréquence', fontsize=13, fontweight='bold')
        ax2.set_title('PASSE 2 - Paires inter-fragments\n(Liaisons non-covalentes uniquement)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.legend(fontsize=10, loc='upper right', framealpha=0.95)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        
        # Statistiques passe 2
        stats_p2 = f"Total paires inter-fragments: {len(ibsi_pass2)}\n"
        stats_p2 += f"Pas de liaison: {len(no_bond_p2)} ({len(no_bond_p2)/len(ibsi_pass2)*100:.1f}%)\n"
        stats_p2 += f"Non-covalentes: {len(non_covalent_p2)} ({len(non_covalent_p2)/len(ibsi_pass2)*100:.1f}%)"
        
        ax2.text(0.02, 0.98, stats_p2, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'Passe 2 non effectuée\n(pas de fichier mol.xyz avec fragments)', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax2.set_xlabel('Valeur IBSI', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Fréquence', fontsize=13, fontweight='bold')
        ax2.set_title('PASSE 2 - Paires inter-fragments\n(Non disponible)', 
                     fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('ibsi_histograms_comparison.png', dpi=300, bbox_inches='tight')
    print("Figure sauvegardée: ibsi_histograms_comparison.png")
    plt.show()


def main():
    """Fonction principale"""
    print("="*60)
    print("Visualisation des résultats IBSI - Comparaison Passe 1 vs Passe 2")
    print("="*60)
    
    try:
        # Lecture passe 1 (obligatoire)
        print("\nLecture PASSE 1 (ibsi_results_pass1.dat)...")
        if not os.path.exists("ibsi_results_pass1.dat"):
            raise FileNotFoundError("ibsi_results_pass1.dat non trouvé!")
        
        data_pass1, threshold_cov = read_ibsi_results("ibsi_results_pass1.dat")
        print(f"  → {len(data_pass1)} paires analysées")
        
        if threshold_cov is not None:
            print(f"  → Seuil covalent: {threshold_cov}")
        else:
            print("  ⚠ Seuil covalent non trouvé (utilisation valeur par défaut 0.05)")
            threshold_cov = 0.05
        
        # Compter passe 1
        bond_types_p1 = data_pass1[:, 3]
        n_no_bond_p1 = np.sum(bond_types_p1 == 0)
        n_covalent_p1 = np.sum(bond_types_p1 == 1)
        print(f"  → Pas de liaison: {n_no_bond_p1}")
        print(f"  → Liaisons covalentes: {n_covalent_p1}")
        
        # Lecture passe 2 (optionnelle)
        data_pass2 = None
        threshold_nc = None
        
        if os.path.exists("ibsi_results_pass2.dat"):
            print("\nLecture PASSE 2 (ibsi_results_pass2.dat)...")
            data_pass2, threshold_nc = read_ibsi_results("ibsi_results_pass2.dat")
            print(f"  → {len(data_pass2)} paires inter-fragments analysées")
            
            if threshold_nc is not None:
                print(f"  → Seuil non-covalent: {threshold_nc}")
            else:
                print("  ⚠ Seuil non-covalent non trouvé (utilisation valeur par défaut 0.02)")
                threshold_nc = 0.02
            
            # Compter passe 2
            bond_types_p2 = data_pass2[:, 3]
            n_no_bond_p2 = np.sum(bond_types_p2 == 0)
            n_non_cov_p2 = np.sum(bond_types_p2 == 2)
            print(f"  → Pas de liaison: {n_no_bond_p2}")
            print(f"  → Liaisons non-covalentes: {n_non_cov_p2}")
        else:
            print("\n⚠ Passe 2 non trouvée (ibsi_results_pass2.dat absent)")
            print("  La passe 2 nécessite un fichier mol.xyz avec fragments définis")
        
        print("\n" + "="*60)
        print("Génération des histogrammes...")
        print("="*60)
        
        plot_dual_histograms(data_pass1, threshold_cov, data_pass2, threshold_nc)
        
        print("\n" + "="*60)
        print("Visualisation terminée avec succès!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nERREUR: Fichier non trouvé - {e}")
        print("Assurez-vous d'avoir exécuté le programme ibsi avant de visualiser.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()