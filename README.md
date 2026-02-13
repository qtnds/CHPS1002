# CHPS 1002, TP2 : Analyse des Liaisons Moléculaires par IBSI

*Introduction*

Ce projet permet de calculer l'**Interaction Bond Strength Indicator (IBSI)** à partir d'un fichier `.wfx` obtenu d'un calcul quantique. Le programme Fortran calcule la densité électronique atomique de Slater sur une grille, intègre un critère différentiel pour chaque paire d'atomes, et détermine automatiquement la connectivité moléculaire. Un script Python permet ensuite de visualiser les résultats sous forme de graphiques 2D/3D.

## Fichiers du projet

* `ibsi.f90` : Code Fortran principal qui lit `mol.wfx`, calcule les indices IBSI pour toutes les paires d'atomes et écrit les fichiers de sortie (`ibsi_results.dat`, `molecule.pdb`).
* `visualize.py` : Script Python qui lit les fichiers de sortie et génère les visualisations graphiques.
* `Makefile` : Compile le programme Fortran en exécutable.
* `exemples_arbre_molecule/` : Dossier contenant les molécules de test, chacune dans un sous-dossier :
  ```
  exemples_arbre_molecule/
  ├── 1_dimer/
  │   ├── mol.wfx
  │   └── mol.xyz
  ├── 2_dimer/
  │   ├── mol.wfx
  │   └── mol.xyz
  └── 3_dimer/
      ├── mol.wfx
      └── mol.xyz
  ```

## Installation et compilation

Cloner le dépôt :
```bash
git clone <URL_DU_DEPOT>
cd <Dossier_Projet>
```

Compiler le code Fortran :
```bash
make
```
Cela génère l'exécutable `ibsi`.

Installer les dépendances Python :
```bash
pip install numpy matplotlib
```

## Exécution du programme

### Programme Fortran

Le programme prend en argument le dossier contenant le fichier `mol.wfx` :

```bash
./ibsi --input exemples_arbre_molecule/1_dimer
```

Le programme affiche dans le terminal :
```
===================================================
Analyse IBSI (Interaction Bond Strength Indicator)
===================================================
Dossier d'entrée : exemples_arbre_molecule/1_dimer
Fichier WFX      : exemples_arbre_molecule/1_dimer/mol.wfx
Nombre d'atomes  : XX
...
Atome_A  Atome_B  Distance(bohr)  IBSI      Liaison?
---------------------------------------------------
    1        2      2.8000      0.312456    OUI
    1        3      5.1000      0.002134    NON
...
===================================================
Nombre total de liaisons détectées: XX
===================================================
```

Et génère dans le dossier d'entrée :
* `ibsi_results.dat` : matrice complète des indices IBSI avec indicateur de liaison (0/1)
* `molecule.pdb` : structure moléculaire avec connectivité, visualisable dans VMD ou PyMOL

### Paramètres de la grille

Les paramètres de grille sont définis directement dans le code :

| Paramètre | Valeur par défaut | Description |
|-----------|:-----------------:|-------------|
| `eps`     | 5.0 bohr          | Offset autour des atomes |
| `dx`      | 0.3 bohr          | Pas de la grille |
| `threshold` | 0.05            | Seuil IBSI pour détecter une liaison |

Ces valeurs peuvent être modifiées dans le `PROGRAM ibsi_analysis` avant recompilation.

## Visualisation VMD
Afin de visualiser la moléculer exécuter la commande :


## Visualisation Python

Une fois le programme Fortran exécuté, lancer le script de visualisation en pointant vers le dossier de la molécule :

```bash
python visualize.py --input exemples_arbre_molecule/1_dimer
```

Trois figures sont générées dans le dossier d'entrée :

| Fichier | Contenu |
|---------|---------|
| `ibsi_molecule_3d.png` | Structure 3D de la molécule + matrice IBSI colorée |
| `ibsi_histogram.png`   | Distribution des valeurs IBSI, liaisons vs non-liaisons |
| `ibsi_network_2d.png`  | Réseau de liaisons en projection XY avec valeurs annotées |

## Description de la méthode IBSI

La densité électronique atomique de chaque atome est modélisée par une somme de fonctions de Slater :

$$\rho_A(r) = a_1 e^{-b_1 r} + a_2 e^{-b_2 r} + a_3 e^{-b_3 r}$$

où les coefficients $a_i$ et $b_i$ sont tabulés pour chaque numéro atomique $Z$ (éléments Z=1 à Z=54).

L'indice IBSI entre deux atomes A et B est calculé par intégration sur la grille :

$$\text{IBSI}_{AB} = \frac{1}{d_{AB}^2} \int \left( \|\nabla\rho_A\| + \|\nabla\rho_B\| - \|\nabla\rho_A + \nabla\rho_B\| \right) dV$$

Une paire est considérée comme une liaison si $\text{IBSI}_{AB} > 0.05$.

## OpenMP — Version parallèle

Le fichier `ibsi_parallel.f90` est une version parallélisée du programme principal utilisant **OpenMP** pour accélérer le calcul des indices IBSI.

### Compilation

```bash
make
```

### Exécution

```bash
# Choisir le nombre de threads (ex: 8 cœurs)
export OMP_NUM_THREADS=8
./ibsi_parallel --input exemples_arbre_molecule/1_dimer
```

### Stratégie de parallélisation

Le goulot d'étranglement du programme est la double boucle sur les paires d'atomes `(a,b)`, chacune nécessitant une intégration complète sur la grille 3D, soit en tout :

$N_{paires} \times N_x \times N_y \times N_z \text{ évaluations de gradient}$

Deux niveaux d'optimisation sont appliqués :

| Directive | Niveau | Rôle |
|-----------|--------|------|
| `!$OMP PARALLEL DO SCHEDULE(DYNAMIC,1)` | Paires `(a,b)` | Distribue chaque paire sur un thread — aucun conflit d'écriture car `ibsi_matrix(a,b)` est unique par paire |
| `!$OMP SIMD REDUCTION(+:dg_sum)` | Boucle grille interne | Vectorisation SIMD pour l'accumulation de `dg_sum` |

`SCHEDULE(DYNAMIC,1)` est choisi plutôt que `STATIC` car les paires ont des distances `d_ab` variables, ce qui peut induire des charges de calcul inégales entre threads.

### Bilan de performance affiché

À la fin de chaque exécution, le programme affiche automatiquement :

```
===================================================
         BILAN DE PERFORMANCE OpenMP
===================================================
  Temps calcul IBSI    :      4.231 s
  Temps total          :      4.235 s
  Threads utilisés     :          8
  Points de grille     :    1030301
  Paires calculées     :         78
  Perf grille          :     18.985 Mpoints/s
===================================================
```

### Comparaison séquentiel vs parallèle

Pour comparer les performances entre les deux versions :

```bash parallel.sh```

### Résultats
```
ibsi = 2m26,807s
ibsi_parallel (2 threads) = 1m45,159s
ibsi_parallel (4 threads) = 0m46,502s
```

Il est possible de voir dans le dossier ```/resultats/openmp``` l'utilisation cpu qui passe de 99,7% à 199,3% puis à 381,4%.


### Remarques

* Le speedup réel est limité par la loi d'Amdahl — les parties non parallélisées (lecture WFX, écriture PDB/DAT) constituent un overhead fixe.
* Pour de petites molécules (peu de paires), l'overhead OpenMP peut dépasser le gain. La version parallèle est surtout bénéfique pour des molécules avec **plus de 20 atomes**.
* Le nombre de threads optimal dépend du nombre de cœurs physiques disponibles : `nproc` sous Linux.

## Résultats

Dans le dossier `resultats/` se trouvent des captures d'écran des graphiques générés pour les 3 dimères fournis, illustrant la détection automatique des liaisons covalentes et intermoléculaires.

## Remarques

* Le fichier `mol.wfx` doit contenir les sections `<Number of Nuclei>`, `<Nuclear Cartesian Coordinates>` et `<Atomic Numbers>`.
* Les éléments au-delà de Z=54 ne sont pas pris en charge (coefficients non disponibles).
* Augmenter `eps` ou diminuer `dx` améliore la précision au détriment du temps de calcul.
* Le calcul peut être long pour des molécules avec beaucoup d'atomes (complexité en $O(N^2 \times N_{grille})$).