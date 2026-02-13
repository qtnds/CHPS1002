
# Version séquentielle
time ./ibsi --input exemples_arbre_molecule/3_dimer

# Version parallèle avec N threads
export OMP_NUM_THREADS=2
time ./ibsi_parallel --input exemples_arbre_molecule/3_dimer

export OMP_NUM_THREADS=4
time ./ibsi_parallel --input exemples_arbre_molecule/3_dimer

