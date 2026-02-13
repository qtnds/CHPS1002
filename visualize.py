#!/usr/bin/env python3
"""
Script de visualisation des résultats IBSI
Analyse des liaisons moléculaires
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def read_wfx_geometry(filename="fichier.wfx"):
    """Lecture de la géométrie depuis le fichier WFX"""
    natoms = 0
    coords = []
    atomic_numbers = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Recherche du nombre d'atomes
    for i, line in enumerate(lines):
        if "<Number of Nuclei>" in line:
            natoms = int(lines[i+1].strip())
            break
    
    # Lecture des coordonnées
    for i, line in enumerate(lines):
        if "<Nuclear Cartesian Coordinates>" in line:
            for j in range(natoms):
                coord = list(map(float, lines[i+1+j].strip().split()))
                coords.append(coord)
            break
    
    # Lecture des numéros atomiques
    for i, line in enumerate(lines):
        if "<Atomic Numbers>" in line:
            for j in range(natoms):
                z = int(lines[i+1+j].strip())
                atomic_numbers.append(z)
            break
    
    return natoms, np.array(coords), np.array(atomic_numbers)


def read_ibsi_results(filename="ibsi_results.dat"):
    """Lecture des résultats IBSI"""
    data = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Ignorer les lignes de commentaires
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parser la ligne de données
            parts = line.split()
            if len(parts) >= 4:
                try:
                    i = int(parts[0])
                    j = int(parts[1])
                    ibsi = float(parts[2])
                    bond = int(parts[3])
                    data.append([i, j, ibsi, bond])
                except ValueError as e:
                    print(f"Attention: ligne ignorée (erreur de parsing): {line}")
                    continue
    
    if len(data) == 0:
        raise ValueError("Aucune donnée valide trouvée dans ibsi_results.dat")
    
    return np.array(data)


def get_atom_color(z):
    """Retourne la couleur d'un atome selon son numéro atomique"""
    color_map = {
        1: 'white',      # H
        6: 'gray',       # C
        7: 'blue',       # N
        8: 'red',        # O
        9: 'lime',       # F
        15: 'orange',    # P
        16: 'yellow',    # S
        17: 'green',     # Cl
    }
    return color_map.get(z, 'pink')


def get_atom_symbol(z):
    """Retourne le symbole d'un atome"""
    symbol_map = {
        1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
        15: 'P', 16: 'S', 17: 'Cl'
    }
    return symbol_map.get(z, 'X')


def plot_molecule_3d(coords, atomic_numbers, bonds_data):
    """Visualisation 3D de la molécule avec les liaisons"""
    fig = plt.figure(figsize=(14, 6))
    
    # Conversion bohr -> angström
    coords_ang = coords * 0.529177
    natoms = len(atomic_numbers)
    
    # Plot 1: Molécule 3D
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Tracer les atomes
    for i, (coord, z) in enumerate(zip(coords_ang, atomic_numbers)):
        ax1.scatter(coord[0], coord[1], coord[2], 
                   c=get_atom_color(z), s=300, 
                   edgecolors='black', linewidths=2, alpha=0.8)
        ax1.text(coord[0], coord[1], coord[2], 
                f'  {get_atom_symbol(z)}{i+1}', fontsize=10)
    
    # Tracer les liaisons (filtrage des indices hors limites)
    for bond in bonds_data:
        ii, jj, ibsi, is_bond = int(bond[0])-1, int(bond[1])-1, bond[2], int(bond[3])
        if is_bond == 1 and 0 <= ii < natoms and 0 <= jj < natoms:
            ax1.plot([coords_ang[ii,0], coords_ang[jj,0]],
                    [coords_ang[ii,1], coords_ang[jj,1]],
                    [coords_ang[ii,2], coords_ang[jj,2]],
                    'k-', linewidth=2, alpha=0.6)
    
    ax1.set_xlabel('X (Å)', fontsize=12)
    ax1.set_ylabel('Y (Å)', fontsize=12)
    ax1.set_zlabel('Z (Å)', fontsize=12)
    ax1.set_title('Structure moléculaire 3D', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Matrice IBSI
    ax2 = fig.add_subplot(122)
    
    # Créer la matrice IBSI complète
    ibsi_matrix = np.zeros((natoms, natoms))
    
    for bond in bonds_data:
        ii, jj, ibsi = int(bond[0])-1, int(bond[1])-1, bond[2]
        if 0 <= ii < natoms and 0 <= jj < natoms:
            ibsi_matrix[ii, jj] = ibsi
            ibsi_matrix[jj, ii] = ibsi
    
    im = ax2.imshow(ibsi_matrix, cmap='RdYlGn', aspect='auto')
    ax2.set_xlabel('Atome j', fontsize=12)
    ax2.set_ylabel('Atome i', fontsize=12)
    ax2.set_title('Matrice IBSI', fontsize=14, fontweight='bold')
    
    # Ajouter les labels des atomes
    labels = [f'{get_atom_symbol(z)}{i+1}' for i, z in enumerate(atomic_numbers)]
    ax2.set_xticks(range(natoms))
    ax2.set_yticks(range(natoms))
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_yticklabels(labels)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('IBSI', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('ibsi_molecule_3d.png', dpi=300, bbox_inches='tight')
    print("Figure sauvegardée: ibsi_molecule_3d.png")
    plt.show()


def plot_ibsi_histogram(bonds_data, threshold=0.05):
    """Histogramme des valeurs IBSI"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ibsi_values = bonds_data[:, 2]
    is_bond = bonds_data[:, 3]
    
    # Histogramme de toutes les valeurs IBSI
    ax1.hist(ibsi_values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Seuil = {threshold}')
    ax1.set_xlabel('Valeur IBSI', fontsize=12)
    ax1.set_ylabel('Fréquence', fontsize=12)
    ax1.set_title('Distribution des valeurs IBSI', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Comparaison liaisons vs non-liaisons
    bonds_ibsi = ibsi_values[is_bond == 1]
    non_bonds_ibsi = ibsi_values[is_bond == 0]
    
    if len(bonds_ibsi) > 0 and len(non_bonds_ibsi) > 0:
        ax2.hist([non_bonds_ibsi, bonds_ibsi], bins=20, 
                label=['Non-liaisons', 'Liaisons'], 
                color=['lightcoral', 'lightgreen'], 
                alpha=0.7, edgecolor='black')
    elif len(bonds_ibsi) > 0:
        ax2.hist(bonds_ibsi, bins=20, label='Liaisons', 
                color='lightgreen', alpha=0.7, edgecolor='black')
    elif len(non_bonds_ibsi) > 0:
        ax2.hist(non_bonds_ibsi, bins=20, label='Non-liaisons', 
                color='lightcoral', alpha=0.7, edgecolor='black')
    
    ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Seuil = {threshold}')
    ax2.set_xlabel('Valeur IBSI', fontsize=12)
    ax2.set_ylabel('Fréquence', fontsize=12)
    ax2.set_title('Comparaison liaisons/non-liaisons', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ibsi_histogram.png', dpi=300, bbox_inches='tight')
    print("Figure sauvegardée: ibsi_histogram.png")
    plt.show()


def plot_bond_network(coords, atomic_numbers, bonds_data):
    """Visualisation en réseau 2D des liaisons"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Conversion bohr -> angström
    coords_ang = coords * 0.529177
    
    # Projection 2D (XY)
    x = coords_ang[:, 0]
    y = coords_ang[:, 1]
    
    # Tracer les liaisons
    for bond in bonds_data:
        ii, jj, ibsi, is_bond = int(bond[0])-1, int(bond[1])-1, bond[2], int(bond[3])
        if is_bond == 1 and 0 <= ii < len(atomic_numbers) and 0 <= jj < len(atomic_numbers):
            ax.plot([x[ii], x[jj]], [y[ii], y[jj]], 
                   'k-', linewidth=max(1, 3*ibsi), alpha=0.5)
            # Annoter l'IBSI sur la liaison
            mid_x, mid_y = (x[ii] + x[jj])/2, (y[ii] + y[jj])/2
            ax.text(mid_x, mid_y, f'{ibsi:.3f}', 
                   fontsize=8, ha='center', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Tracer les atomes
    for i, (coord, z) in enumerate(zip(coords_ang, atomic_numbers)):
        ax.scatter(coord[0], coord[1], 
                  c=get_atom_color(z), s=500, 
                  edgecolors='black', linewidths=2, alpha=0.9, zorder=10)
        ax.text(coord[0], coord[1], 
               f'{get_atom_symbol(z)}{i+1}', 
               fontsize=12, ha='center', va='center', fontweight='bold')
    
    ax.set_xlabel('X (Å)', fontsize=14)
    ax.set_ylabel('Y (Å)', fontsize=14)
    ax.set_title('Réseau de liaisons (projection XY)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('ibsi_network_2d.png', dpi=300, bbox_inches='tight')
    print("Figure sauvegardée: ibsi_network_2d.png")
    plt.show()


def main():
    """Fonction principale"""
    print("="*60)
    print("Visualisation des résultats IBSI")
    print("="*60)
    
    try:
        # Lecture des données
        print("\nLecture de la géométrie...")
        natoms, coords, atomic_numbers = read_wfx_geometry()
        print(f"  → {natoms} atomes lus")
        
        print("\nLecture des résultats IBSI...")
        bonds_data = read_ibsi_results()
        print(f"  → {len(bonds_data)} paires analysées")
        
        nbonds = np.sum(bonds_data[:, 3])
        print(f"  → {int(nbonds)} liaisons détectées")
        
        # Génération des graphiques
        print("\n" + "="*60)
        print("Génération des graphiques...")
        print("="*60)
        
        print("\n1. Structure moléculaire 3D et matrice IBSI...")
        plot_molecule_3d(coords, atomic_numbers, bonds_data)
        
        print("\n2. Histogrammes des valeurs IBSI...")
        plot_ibsi_histogram(bonds_data)
        
        print("\n3. Réseau de liaisons 2D...")
        plot_bond_network(coords, atomic_numbers, bonds_data)
        
        print("\n" + "="*60)
        print("Visualisation terminée avec succès!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nERREUR: Fichier non trouvé - {e}")
        print("Assurez-vous que les fichiers 'fichier.wfx' et 'ibsi_results.dat' existent.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()