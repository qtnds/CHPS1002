# ===== Compilateur =====
FC = gfortran
GC = nvfortran

# ===== Options =====
FFLAGS = -O2 -Wall -Wextra -fcheck=all -g
OMPFLAGS = -O3 -fopenmp
GPUFLAGS = -O3 -stdpar=gpu

# ===== Règle par défaut =====
all: ibsi ibsi_parallel ibsi_gpu

# ===== Programme séquentiel =====
ibsi: ibsi.f90
	$(FC) $(FFLAGS) -o ibsi ibsi.f90

# ===== Programme OpenMP =====
ibsi_parallel: ibsi_parallel.f90
	$(FC) $(OMPFLAGS) -o ibsi_parallel ibsi_parallel.f90

ibsi_gpu: ibsi_gpu.f90
	$(GC) $(GPUFLAGS) -o ibsi_gpu ibsi_gpu.f90

# ===== Nettoyage =====
clean:
	rm -f ibsi ibsi_parallel ibsi_gpu *.o *.mod
