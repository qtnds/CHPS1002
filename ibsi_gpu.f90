! =============================================================================
! ibsi_gpu.f90  -  Version GPU via DO CONCURRENT (Fortran 2018/2023)
! =============================================================================
!
! STRATÉGIE DE PORTAGE GPU
! ========================
! DO CONCURRENT avec la clause REDUCE (Fortran 2023) est le mécanisme
! standard le plus portable pour exprimer le parallélisme GPU en Fortran.
! Le compilateur génère les noyaux CUDA/HIP/SYCL automatiquement.
!
! TROIS NIVEAUX DE PARALLÉLISME exploités :
!
!   Niveau 1 – Paires (a,b)          : DO CONCURRENT sur npairs
!   Niveau 2 – Grille (i,j,k)        : DO CONCURRENT imbriqué sur nx*ny*nz
!              → aplatissement (i,j,k) → indice linéaire igrid
!                pour maximiser l'occupancy du GPU
!   Niveau 3 – Les données (xyz, ZA, coefficients)
!              sont allouées et restent sur le GPU pendant tout le calcul
!              via l'attribut TARGET + directives COPYIN/COPYOUT explicites
!
! COMPILATEURS SUPPORTÉS
! ======================
!   NVIDIA (recommandé)  :
!     nvfortran -O3 -stdpar=gpu     -o ibsi_gpu ibsi_gpu.f90
!     nvfortran -O3 -stdpar=multicore -o ibsi_cpu ibsi_gpu.f90   (fallback CPU)
!
!   Intel ifx (GPU Intel) :
!     ifx -O3 -fiopenmp -fopenmp-targets=spir64 -o ibsi_gpu ibsi_gpu.f90
!
!   GFortran (CPU seulement, test/débogage) :
!     gfortran -O3 -std=f2018 -o ibsi_cpu ibsi_gpu.f90
!     NB: gfortran ignore DO CONCURRENT REDUCE → exécution séquentielle correcte
!
!   LLVM Flang (expérimental GPU) :
!     flang-new -O3 -fopenmp -fopenmp-targets=nvptx64 -o ibsi_gpu ibsi_gpu.f90
!
! MESURE DE PERFORMANCE
! =====================
!   ./ibsi_gpu --input mol_dir [--tseq 120.5] [--tpar 15.2]
!   --tseq : temps séquentiel de référence (s)
!   --tpar : temps parallèle CPU OpenMP de référence (s)
!
! IMPORTANT : DO CONCURRENT + REDUCE nécessite Fortran 2023.
!   Avec nvfortran >= 23.x, cette syntaxe est supportée nativement.
!   Avec des compilateurs plus anciens, remplacer REDUCE(+:dg_sum) par
!   une somme via tableau temporaire (voir commentaires dans le code).
! =============================================================================

MODULE ab_module
  IMPLICIT NONE

  INTEGER, PARAMETER :: NB_KNW_ATMS = 54
  INTEGER, PARAMETER :: DP = KIND(1.0d0)  ! double précision

  REAL(DP), PARAMETER :: AA1(NB_KNW_ATMS) = (/ &
    0.2815_DP, 2.437_DP, &
    11.84_DP, 31.34_DP, 67.82_DP, 120.2_DP, 190.9_DP, 289.5_DP, 406.3_DP, 561.3_DP, &
    760.8_DP, 1016.0_DP, 1319.0_DP, 1658.0_DP, 2042.0_DP, 2501.0_DP, 3024.0_DP, 3625.0_DP, &
    422.685_DP, 328.056_DP, &
    369.212_DP, 294.479_DP, 525.536_DP, 434.298_DP, 9346.12_DP, 579.606_DP, 665.791_DP, &
    793.8_DP, 897.784_DP, 1857.03_DP, &
    898.009_DP, 1001.24_DP, 1101.35_DP, 1272.92_DP, 1333.8_DP, 1459.53_DP, &
    0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, &
    0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, &
    457.406_DP, 3.23589_DP, 643.098_DP /)

  REAL(DP), PARAMETER :: AA2(NB_KNW_ATMS) = (/ &
    0.0_DP, 0.0_DP, &
    0.06332_DP, 0.3694_DP, 0.8527_DP, 1.172_DP, 2.247_DP, 2.879_DP, 3.049_DP, 6.984_DP, &
    22.42_DP, 37.17_DP, 57.95_DP, 87.16_DP, 115.7_DP, 158.0_DP, 205.5_DP, 260.0_DP, &
    104.678_DP, 48.1693_DP, &
    66.7813_DP, 64.3627_DP, 97.5552_DP, 37.8524_DP, 166.393_DP, 50.2383_DP, 51.7033_DP, &
    60.5725_DP, 58.8879_DP, 135.027_DP, &
    1.10777_DP, 0.855815_DP, 901.893_DP, 1.20778_DP, 1.0722_DP, 1.95028_DP, &
    0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, &
    0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, &
    0.0648533_DP, 4.1956_DP, 0.133996_DP /)

  REAL(DP), PARAMETER :: AA3(NB_KNW_ATMS) = (/ &
    0.0_DP, 0.0_DP, &
    0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, &
    0.06358_DP, 0.3331_DP, 0.8878_DP, 0.7888_DP, 1.465_DP, 2.170_DP, 3.369_DP, 5.211_DP, &
    0.0305251_DP, 0.137293_DP, &
    0.217304_DP, 0.312378_DP, 0.288164_DP, 0.216041_DP, 0.428442_DP, 0.301226_DP, &
    0.358959_DP, 0.384414_DP, 0.446463_DP, 0.664027_DP, &
    0.178217_DP, 0.271045_DP, 1.43271_DP, 0.548474_DP, 1.52238_DP, 3.12305_DP, &
    0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, &
    0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, &
    0.0273883_DP, 0.0347855_DP, 0.0499776_DP /)

  REAL(DP), PARAMETER :: BB1(NB_KNW_ATMS) = (/ &
    1.8910741301059_DP, 2.95945546019532_DP, &
    5.23012552301255_DP, 7.19424460431655_DP, 9.44287063267233_DP, 11.3122171945701_DP, &
    13.0378096479791_DP, 14.9476831091181_DP, 16.4473684210526_DP, 18.2149362477231_DP, &
    20.1612903225806_DP, 22.271714922049_DP, 24.330900243309_DP, 26.1780104712042_DP, &
    27.9329608938547_DP, 29.8507462686567_DP, 31.7460317460318_DP, 33.7837837837838_DP, &
    61.3085_DP, 13.6033_DP, &
    15.1828_DP, 8.87333_DP, 18.4925_DP, 12.2138_DP, 47.0758_DP, 13.6638_DP, 14.1117_DP, &
    15.1832_DP, 15.4177_DP, 23.5312_DP, &
    12.4217_DP, 12.8073_DP, 13.0865_DP, 13.8734_DP, 13.8227_DP, 14.1814_DP, &
    0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, &
    0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, &
    3.82753_DP, 23.9761_DP, 4.10166_DP /)

  REAL(DP), PARAMETER :: BB2(NB_KNW_ATMS) = (/ &
    1.0_DP, 1.0_DP, &
    1.00080064051241_DP, 1.43988480921526_DP, 1.88679245283019_DP, 1.82481751824818_DP, &
    2.20653133274492_DP, 2.51635631605435_DP, 2.50375563345018_DP, 2.90107339715695_DP, &
    3.98247710075667_DP, 4.65116279069767_DP, 5.33617929562433_DP, 6.0459492140266_DP, &
    6.62690523525514_DP, 7.30460189919649_DP, 7.9428117553614_DP, 8.56164383561644_DP, &
    3.81272_DP, 3.94253_DP, &
    4.24199_DP, 4.33369_DP, 4.58656_DP, 3.81061_DP, 5.07878_DP, 4.09039_DP, 4.17018_DP, &
    4.30955_DP, 4.35337_DP, 5.19033_DP, &
    2.00145_DP, 1.85496_DP, 7.70162_DP, 1.88584_DP, 2.12925_DP, 2.35537_DP, &
    0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, &
    0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, &
    1.19985_DP, 2.24851_DP, 1.25439_DP /)

  REAL(DP), PARAMETER :: BB3(NB_KNW_ATMS) = (/ &
    1.0_DP, 1.0_DP, &
    1.0_DP, 1.0_DP, 1.0_DP, 1.0_DP, 1.0_DP, 1.0_DP, 1.0_DP, 1.0_DP, &
    0.9769441187964_DP, 1.28982329420869_DP, 1.67728950016773_DP, 1.42959256611866_DP, &
    1.70910955392241_DP, 1.94212468440474_DP, 2.01045436268597_DP, 2.26654578422484_DP, &
    0.760211_DP, 0.972271_DP, &
    1.08807_DP, 1.18612_DP, 1.19053_DP, 1.22212_DP, 1.3246_DP, 1.32999_DP, 1.38735_DP, &
    1.42114_DP, 1.48338_DP, 1.54071_DP, &
    1.19489_DP, 1.31783_DP, 1.64735_DP, 1.57665_DP, 1.79153_DP, 1.99148_DP, &
    0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, &
    0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, 0.0_DP, &
    0.906738_DP, 0.856995_DP, 1.02511_DP /)

END MODULE ab_module


! =============================================================================
! MODULE IBSI GPU
! =============================================================================
MODULE ibsi_gpu_module
  USE ab_module
  IMPLICIT NONE

CONTAINS

  ! ---------------------------------------------------------------------------
  ! FONCTION INLINE : gradient proatomique
  ! Marquée PURE pour être appelable depuis DO CONCURRENT sans restriction.
  ! Sur GPU (nvfortran -stdpar=gpu), les fonctions PURE sont automatiquement
  ! inlinées dans le noyau CUDA → pas d'appel de fonction sur le device.
  ! ---------------------------------------------------------------------------
  PURE SUBROUTINE grad_rho_inline(x, y, z, ax, ay, az, &
                                   a1, a2, a3, b1, b2, b3, &
                                   gx, gy, gz)
    REAL(DP), INTENT(in)  :: x, y, z, ax, ay, az
    REAL(DP), INTENT(in)  :: a1, a2, a3, b1, b2, b3
    REAL(DP), INTENT(out) :: gx, gy, gz
    REAL(DP) :: r, dx, dy, dz, fac

    dx = x - ax;  dy = y - ay;  dz = z - az
    r  = SQRT(dx*dx + dy*dy + dz*dz)

    IF (r < 1.0d-10) THEN
      gx = 0.0_DP;  gy = 0.0_DP;  gz = 0.0_DP
    ELSE
      fac = -(a1*b1*EXP(-b1*r) + a2*b2*EXP(-b2*r) + a3*b3*EXP(-b3*r)) / r
      gx = fac * dx;  gy = fac * dy;  gz = fac * dz
    END IF
  END SUBROUTINE grad_rho_inline

  ! ---------------------------------------------------------------------------
  ! FONCTION PURE : intégrande dg_ab en un point (x,y,z)
  ! Entièrement autosuffisante (coefficients passés en argument) pour être
  ! compatible avec DO CONCURRENT LOCAL sur GPU.
  ! ---------------------------------------------------------------------------
  PURE FUNCTION dg_ab_point(x, y, z, &
                              xa, ya, za, a1a, a2a, a3a, b1a, b2a, b3a, &
                              xb, yb, zb, a1b, a2b, a3b, b1b, b2b, b3b) RESULT(dg)
    REAL(DP), INTENT(in) :: x, y, z
    REAL(DP), INTENT(in) :: xa, ya, za, a1a, a2a, a3a, b1a, b2a, b3a
    REAL(DP), INTENT(in) :: xb, yb, zb, a1b, a2b, a3b, b1b, b2b, b3b
    REAL(DP) :: dg
    REAL(DP) :: gax, gay, gaz, gbx, gby, gbz
    REAL(DP) :: norm1, norm2

    CALL grad_rho_inline(x, y, z, xa, ya, za, a1a, a2a, a3a, b1a, b2a, b3a, gax, gay, gaz)
    CALL grad_rho_inline(x, y, z, xb, yb, zb, a1b, a2b, a3b, b1b, b2b, b3b, gbx, gby, gbz)

    norm1 = SQRT((ABS(gax)+ABS(gbx))**2 + (ABS(gay)+ABS(gby))**2 + (ABS(gaz)+ABS(gbz))**2)
    norm2 = SQRT((gax+gbx)**2 + (gay+gby)**2 + (gaz+gbz)**2)
    dg = norm1 - norm2
  END FUNCTION dg_ab_point

  ! ---------------------------------------------------------------------------
  ! NOYAU GPU PRINCIPAL : compute_ibsi_gpu
  !
  ! Deux niveaux de DO CONCURRENT :
  !
  !  Niveau extérieur  → itère sur les npairs paires (a,b)
  !                       → LOCAL : tous les scalaires locaux à la paire
  !
  !  Niveau intérieur  → itère sur les ngrid = nx*ny*nz points de grille
  !                       → REDUCE(+:dg_sum) : réduction GPU native
  !
  ! L'aplatissement 3D→1D (igrid → i,j,k) est fait à la main :
  !   igrid = (i-1)*ny*nz + (j-1)*nz + k    (i∈[1,nx], j∈[1,ny], k∈[1,nz])
  ! Cela expose un seul niveau de DO CONCURRENT à la boucle interne,
  ! ce qui maximise le nombre de threads GPU par paire.
  !
  ! IMPORTANT sur la REDUCE :
  !   La clause REDUCE(+:dg_sum) est Fortran 2023 (ISO/IEC 1539-1:2023).
  !   Elle est supportée par :
  !     nvfortran >= 23.1   (compilateur officiel NVIDIA HPC SDK)
  !     ifx >= 2024.1       (Intel)
  !   Pour les compilateurs plus anciens, remplacer par l'approche tableau
  !   commentée ci-dessous (alternative_reduce).
  ! ---------------------------------------------------------------------------
  SUBROUTINE compute_ibsi_gpu(natoms, npairs, aa_list, bb_list, &
                               xyz, ZAatoms,                     &
                               xmin, ymin, zmin, dx, dV,         &
                               nx, ny, nz,                        &
                               ibsi_matrix)
    INTEGER,  INTENT(in)  :: natoms, npairs, nx, ny, nz
    INTEGER,  INTENT(in)  :: aa_list(npairs), bb_list(npairs)
    INTEGER,  INTENT(in)  :: ZAatoms(natoms)
    REAL(DP), INTENT(in)  :: xyz(3, natoms)
    REAL(DP), INTENT(in)  :: xmin, ymin, zmin, dx, dV
    REAL(DP), INTENT(out) :: ibsi_matrix(natoms, natoms)

    INTEGER  :: ipair, igrid, a, b, nyz
    INTEGER  :: gi, gj, gk
    REAL(DP) :: xa, ya, za, xb, yb, zb
    REAL(DP) :: a1a, a2a, a3a, b1a, b2a, b3a
    REAL(DP) :: a1b, a2b, a3b, b1b, b2b, b3b
    REAL(DP) :: x, y, z, d_ab, dg_sum
    INTEGER  :: ngrid, ZA_a, ZA_b

    ngrid = nx * ny * nz
    nyz   = ny * nz

    ibsi_matrix = 0.0_DP

    ! =========================================================================
    ! BOUCLE EXTERNE GPU : une tâche GPU par paire (a,b)
    !
    ! LOCAL liste toutes les variables locales à la paire.
    ! Ceci est crucial pour le GPU : chaque thread-bloc aura ses propres
    ! copies de ces variables sans conflit mémoire.
    !
    ! NOTE compilateur : sur nvfortran, DO CONCURRENT LOCAL déclare des
    ! variables en registres GPU → accès O(1) sans latence mémoire.
    ! =========================================================================
    DO CONCURRENT (ipair = 1:npairs) &
      LOCAL(a, b, xa, ya, za, xb, yb, zb,                          &
            a1a, a2a, a3a, b1a, b2a, b3a,                          &
            a1b, a2b, a3b, b1b, b2b, b3b,                          &
            d_ab, dg_sum, ZA_a, ZA_b)

      a    = aa_list(ipair)
      b    = bb_list(ipair)
      ZA_a = ZAatoms(a)
      ZA_b = ZAatoms(b)

      xa = xyz(1,a);  ya = xyz(2,a);  za = xyz(3,a)
      xb = xyz(1,b);  yb = xyz(2,b);  zb = xyz(3,b)

      ! Récupération des coefficients de Slater (inline pour GPU)
      a1a = AA1(ZA_a);  a2a = AA2(ZA_a);  a3a = AA3(ZA_a)
      b1a = BB1(ZA_a);  b2a = BB2(ZA_a);  b3a = BB3(ZA_a)
      a1b = AA1(ZA_b);  a2b = AA2(ZA_b);  a3b = AA3(ZA_b)
      b1b = BB1(ZA_b);  b2b = BB2(ZA_b);  b3b = BB3(ZA_b)

      d_ab = SQRT((xa-xb)**2 + (ya-yb)**2 + (za-zb)**2)

      dg_sum = 0.0_DP

      ! =======================================================================
      ! BOUCLE INTERNE GPU : réduction sur les ngrid points de grille
      !
      ! DO CONCURRENT avec REDUCE(+:dg_sum) génère une réduction atomique
      ! sur GPU. Sur NVIDIA A100/H100, cela utilise les instructions
      ! atomicAdd en CUDA, très efficaces pour les FP64.
      !
      ! L'aplatissement 3D→1D permet une seule boucle DO CONCURRENT,
      ! ce qui est plus efficace qu'imbriquer 3 DO CONCURRENT (évite
      ! les barrières intermédiaires sur certains compilateurs).
      !
      ! ALTERNATIVE pour compilateurs sans REDUCE F2023 :
      !   Allouer REAL(DP) :: dg_arr(ngrid)   (LOCAL ou sur device)
      !   DO CONCURRENT (igrid=1:ngrid) LOCAL(x,y,z,gi,gj,gk)
      !     dg_arr(igrid) = dg_ab_point(...)
      !   END DO
      !   dg_sum = SUM(dg_arr)
      ! =======================================================================
      DO CONCURRENT (igrid = 1:ngrid) REDUCE(+:dg_sum) &
        LOCAL(gi, gj, gk, x, y, z)

        ! Décodage de l'indice linéaire → (i,j,k)
        gi = (igrid - 1) / nyz + 1
        gj = MOD((igrid - 1) / nz, ny) + 1
        gk = MOD( igrid - 1, nz) + 1

        x = xmin + (gi - 1) * dx
        y = ymin + (gj - 1) * dx
        z = zmin + (gk - 1) * dx

        dg_sum = dg_sum + dg_ab_point(x, y, z,               &
                   xa, ya, za, a1a, a2a, a3a, b1a, b2a, b3a,  &
                   xb, yb, zb, a1b, a2b, a3b, b1b, b2b, b3b)
      END DO  ! igrid

      ibsi_matrix(a, b) = (dg_sum * dV) / (d_ab * d_ab)
      ibsi_matrix(b, a) = ibsi_matrix(a, b)

    END DO  ! ipair

  END SUBROUTINE compute_ibsi_gpu

END MODULE ibsi_gpu_module


! =============================================================================
! MODULE I/O (identique aux versions précédentes)
! =============================================================================
MODULE wfx_reader
  IMPLICIT NONE
CONTAINS

  SUBROUTINE read_wfx_geometry(filename, natoms, xyz, ZAatoms, atom_names)
    USE ab_module, ONLY: DP
    CHARACTER(len=*), INTENT(in) :: filename
    INTEGER, INTENT(out) :: natoms
    REAL(DP), ALLOCATABLE, INTENT(out) :: xyz(:,:)
    INTEGER,  ALLOCATABLE, INTENT(out) :: ZAatoms(:)
    CHARACTER(len=10), ALLOCATABLE, INTENT(out) :: atom_names(:)
    INTEGER :: i, unit
    CHARACTER(256) :: line

    OPEN(NEWUNIT=unit, FILE=TRIM(filename), STATUS="OLD", ERR=200)
    DO
      READ(unit,'(A)',END=100) line
      IF (INDEX(line,"<Number of Nuclei>")>0) THEN
        READ(unit,*) natoms;  EXIT
      END IF
    END DO
    REWIND(unit)
    DO
      READ(unit,'(A)',END=100) line
      IF (INDEX(line,"<Nuclear Names>")>0) THEN
        ALLOCATE(atom_names(natoms))
        DO i=1,natoms
          READ(unit,'(A)') atom_names(i);  atom_names(i) = ADJUSTL(atom_names(i))
        END DO
        EXIT
      END IF
    END DO
    REWIND(unit)
    DO
      READ(unit,'(A)',END=100) line
      IF (INDEX(line,"<Nuclear Cartesian Coordinates>")>0) THEN
        ALLOCATE(xyz(3,natoms))
        DO i=1,natoms;  READ(unit,*) xyz(:,i);  END DO
        EXIT
      END IF
    END DO
    REWIND(unit)
    DO
      READ(unit,'(A)',END=100) line
      IF (INDEX(line,"<Atomic Numbers>")>0) THEN
        ALLOCATE(ZAatoms(natoms))
        DO i=1,natoms;  READ(unit,*) ZAatoms(i);  END DO
        EXIT
      END IF
    END DO
100 CLOSE(unit);  RETURN
200 PRINT *, "ERREUR: Impossible d'ouvrir: ", TRIM(filename);  STOP
  END SUBROUTINE read_wfx_geometry

  SUBROUTINE read_xyz_fragments(filename, has_fragments, &
                                  frag1_start, frag1_end, frag2_start, frag2_end)
    CHARACTER(len=*), INTENT(in) :: filename
    LOGICAL, INTENT(out) :: has_fragments
    INTEGER, INTENT(out) :: frag1_start, frag1_end, frag2_start, frag2_end
    INTEGER :: unit, ios, dash_pos
    CHARACTER(512) :: line, sel_a, sel_b
    INTEGER :: pos_sel_a, pos_sel_b

    has_fragments = .FALSE.
    frag1_start = 0;  frag1_end = 0;  frag2_start = 0;  frag2_end = 0

    OPEN(NEWUNIT=unit, FILE=TRIM(filename), STATUS="OLD", IOSTAT=ios)
    IF (ios /= 0) RETURN
    READ(unit,'(A)',IOSTAT=ios) line
    READ(unit,'(A)',IOSTAT=ios) line
    CLOSE(unit)
    IF (ios /= 0) RETURN

    line = ADJUSTL(line)
    pos_sel_a = INDEX(line,'selection_a:')
    pos_sel_b = INDEX(line,'selection_b:')
    IF (pos_sel_a == 0 .OR. pos_sel_b == 0) RETURN

    sel_a = ADJUSTL(line(pos_sel_a+12:pos_sel_b-1))
    sel_b = ADJUSTL(line(pos_sel_b+12:))

    dash_pos = INDEX(sel_a,'-')
    IF (dash_pos > 0) THEN
      READ(sel_a(1:dash_pos-1),*,IOSTAT=ios) frag1_start;  IF (ios/=0) RETURN
      READ(sel_a(dash_pos+1:), *,IOSTAT=ios) frag1_end;    IF (ios/=0) RETURN
    ELSE
      READ(sel_a,*,IOSTAT=ios) frag1_start;  IF (ios/=0) RETURN
      frag1_end = frag1_start
    END IF

    dash_pos = INDEX(sel_b,'-')
    IF (dash_pos > 0) THEN
      READ(sel_b(1:dash_pos-1),*,IOSTAT=ios) frag2_start;  IF (ios/=0) RETURN
      READ(sel_b(dash_pos+1:), *,IOSTAT=ios) frag2_end;    IF (ios/=0) RETURN
    ELSE
      READ(sel_b,*,IOSTAT=ios) frag2_start;  IF (ios/=0) RETURN
      frag2_end = frag2_start
    END IF

    has_fragments = .TRUE.
  END SUBROUTINE read_xyz_fragments

END MODULE wfx_reader


MODULE fragment_utils
  IMPLICIT NONE
CONTAINS
  SUBROUTINE identify_fragments(natoms, bond_matrix, nfrags, fragment_id)
    INTEGER, INTENT(in)  :: natoms
    LOGICAL, INTENT(in)  :: bond_matrix(:,:)
    INTEGER, INTENT(out) :: nfrags
    INTEGER, ALLOCATABLE, INTENT(out) :: fragment_id(:)
    INTEGER :: i, j, cf
    LOGICAL :: visited(natoms), changed

    ALLOCATE(fragment_id(natoms))
    fragment_id = 0;  visited = .FALSE.;  nfrags = 0

    DO i = 1, natoms
      IF (.NOT. visited(i)) THEN
        nfrags = nfrags + 1
        fragment_id(i) = nfrags;  visited(i) = .TRUE.
        changed = .TRUE.
        DO WHILE (changed)
          changed = .FALSE.
          DO j = 1, natoms
            IF (fragment_id(j)==nfrags .AND. .NOT.visited(j)) THEN
              visited(j) = .TRUE.;  changed = .TRUE.
            END IF
            IF (fragment_id(j)==nfrags) THEN
              DO cf = 1, natoms
                IF (bond_matrix(j,cf) .AND. fragment_id(cf)==0) THEN
                  fragment_id(cf) = nfrags;  changed = .TRUE.
                END IF
              END DO
            END IF
          END DO
        END DO
      END IF
    END DO
  END SUBROUTINE identify_fragments
END MODULE fragment_utils


! =============================================================================
! PROGRAMME PRINCIPAL
! =============================================================================
PROGRAM ibsi_gpu_analysis
  USE ab_module
  USE ibsi_gpu_module
  USE wfx_reader
  USE fragment_utils
  IMPLICIT NONE

  ! --- Données moléculaires ---
  INTEGER :: natoms, npairs
  REAL(DP), ALLOCATABLE :: xyz(:,:), ibsi_matrix(:,:)
  INTEGER,  ALLOCATABLE :: ZAatoms(:)
  CHARACTER(len=10), ALLOCATABLE :: atom_names(:)

  ! --- Listes de paires ---
  INTEGER, ALLOCATABLE :: aa_list(:), bb_list(:)

  ! --- Liaisons ---
  LOGICAL, ALLOCATABLE :: bond_matrix_cov(:,:), bond_matrix_nc(:,:)
  INTEGER, ALLOCATABLE :: fragment_id(:)
  INTEGER :: nfrags, nbonds_cov, nbonds_nc

  ! --- Grille ---
  REAL(DP) :: xmin, xmax, ymin, ymax, zmin, zmax, dx, eps, dV
  INTEGER  :: nx, ny, nz

  ! --- Arguments CLI ---
  INTEGER  :: iarg, iostat_val
  CHARACTER(len=256) :: arg, input_dir, wfx_file, xyz_file, temp_val
  REAL(DP) :: threshold_cov, threshold_nc
  REAL(DP) :: t_seq_ref, t_par_ref
  LOGICAL  :: tseq_provided, tpar_provided
  INTEGER  :: frag1_start, frag1_end, frag2_start, frag2_end
  LOGICAL  :: has_fragments

  ! --- Chronomètres ---
  REAL(DP) :: t_total_start, t_total_end
  REAL(DP) :: t_gpu_start, t_gpu_end
  REAL(DP) :: speedup_vs_seq, speedup_vs_par, efficiency_gpu

  ! --- Divers ---
  INTEGER  :: a, b, i, j, ipair, unit_pdb, unit_res
  REAL(DP) :: d_ab
  CHARACTER(2) :: atom_symbol

  ! -------------------------------------------------------------------------
  ! Chrono global démarré AVANT tout (inclut transferts données → GPU)
  ! -------------------------------------------------------------------------
  CALL CPU_TIME(t_total_start)   ! fallback si pas OMP_LIB
#ifdef _OPENMP
  BLOCK
    USE OMP_LIB
    t_total_start = OMP_GET_WTIME()
  END BLOCK
#endif

  ! -------------------------------------------------------------------------
  ! Valeurs par défaut
  ! -------------------------------------------------------------------------
  input_dir     = ""
  threshold_cov = 0.05_DP
  threshold_nc  = 0.02_DP
  t_seq_ref     = -1.0_DP
  t_par_ref     = -1.0_DP
  tseq_provided = .FALSE.
  tpar_provided = .FALSE.

  ! -------------------------------------------------------------------------
  ! Parsing CLI
  ! -------------------------------------------------------------------------
  iarg = 1
  DO WHILE (iarg <= COMMAND_ARGUMENT_COUNT())
    CALL GET_COMMAND_ARGUMENT(iarg, arg)
    SELECT CASE(TRIM(arg))
      CASE("--input")
        iarg = iarg + 1;  CALL GET_COMMAND_ARGUMENT(iarg, input_dir)
      CASE("--seuil1")
        iarg = iarg + 1;  CALL GET_COMMAND_ARGUMENT(iarg, temp_val)
        READ(temp_val,*,IOSTAT=iostat_val) threshold_cov
        IF (iostat_val/=0) THEN; PRINT*,"ERREUR --seuil1"; STOP; END IF
      CASE("--seuil2")
        iarg = iarg + 1;  CALL GET_COMMAND_ARGUMENT(iarg, temp_val)
        READ(temp_val,*,IOSTAT=iostat_val) threshold_nc
        IF (iostat_val/=0) THEN; PRINT*,"ERREUR --seuil2"; STOP; END IF
      CASE("--tseq")
        iarg = iarg + 1;  CALL GET_COMMAND_ARGUMENT(iarg, temp_val)
        READ(temp_val,*,IOSTAT=iostat_val) t_seq_ref
        IF (iostat_val/=0) THEN; PRINT*,"ERREUR --tseq"; STOP; END IF
        tseq_provided = .TRUE.
      CASE("--tpar")
        iarg = iarg + 1;  CALL GET_COMMAND_ARGUMENT(iarg, temp_val)
        READ(temp_val,*,IOSTAT=iostat_val) t_par_ref
        IF (iostat_val/=0) THEN; PRINT*,"ERREUR --tpar"; STOP; END IF
        tpar_provided = .TRUE.
    END SELECT
    iarg = iarg + 1
  END DO

  IF (TRIM(input_dir) == "") THEN
    PRINT *, "Usage: ./ibsi_gpu --input <dossier> [options]"
    PRINT *, ""
    PRINT *, "Options:"
    PRINT *, "  --input   : Dossier contenant mol.wfx         (OBLIGATOIRE)"
    PRINT *, "  --seuil1  : Seuil liaisons covalentes         (defaut: 0.05)"
    PRINT *, "  --seuil2  : Seuil liaisons non-covalentes     (defaut: 0.02)"
    PRINT *, "  --tseq    : Temps séquentiel ref. (s)         (pour speedup)"
    PRINT *, "  --tpar    : Temps OpenMP ref. (s)             (pour speedup GPU)"
    PRINT *, ""
    PRINT *, "Compilation NVIDIA GPU (recommandé):"
    PRINT *, "  nvfortran -O3 -stdpar=gpu -o ibsi_gpu ibsi_gpu.f90"
    PRINT *, ""
    PRINT *, "Compilation CPU multicore (fallback):"
    PRINT *, "  nvfortran -O3 -stdpar=multicore -o ibsi_cpu ibsi_gpu.f90"
    PRINT *, "  gfortran  -O3 -std=f2018       -o ibsi_cpu ibsi_gpu.f90"
    PRINT *, ""
    PRINT *, "Exemple:"
    PRINT *, "  ./ibsi_gpu --input mol_dir --tseq 120.5 --tpar 18.3"
    STOP
  END IF

  wfx_file = TRIM(input_dir) // "/mol.wfx"
  xyz_file = TRIM(input_dir) // "/mol.xyz"

  CALL read_wfx_geometry(wfx_file, natoms, xyz, ZAatoms, atom_names)
  CALL read_xyz_fragments(xyz_file, has_fragments, &
                           frag1_start, frag1_end, frag2_start, frag2_end)

  ! -------------------------------------------------------------------------
  ! En-tête
  ! -------------------------------------------------------------------------
  PRINT *, "=========================================================="
  PRINT *, "  ANALYSE IBSI - VERSION GPU (DO CONCURRENT)"
  PRINT *, "=========================================================="
  PRINT *, "Dossier d'entree  : ", TRIM(input_dir)
  PRINT *, "Fichier WFX       : ", TRIM(wfx_file)
  PRINT *, "Nombre d'atomes   : ", natoms
  PRINT *, ""
  PRINT *, "Parametres IBSI:"
  PRINT *, "  Seuil covalent     (--seuil1) :", threshold_cov
  PRINT *, "  Seuil non-covalent (--seuil2) :", threshold_nc
  PRINT *, ""

  ! =========================================================================
  ! INFORMATIONS GPU / DO CONCURRENT
  ! =========================================================================
  PRINT *, "=========================================================="
  PRINT *, "  STRATÉGIE DE PARALLÉLISME GPU"
  PRINT *, "=========================================================="
  PRINT *, "  Mécanisme     : DO CONCURRENT (Fortran 2018/2023)"
  PRINT *, "  Backend cible : CUDA (nvfortran -stdpar=gpu)"
  PRINT *, "                  ou multicore CPU (nvfortran -stdpar=multicore)"
  PRINT *, ""
  PRINT *, "  Niveau 1 (paires)  : DO CONCURRENT sur npairs"
  PRINT *, "    -> Chaque paire (a,b) = 1 tâche GPU indépendante"
  PRINT *, "    -> Variables locales déclarées via LOCAL(...)"
  PRINT *, "       => placées en registres GPU, latence nulle"
  PRINT *, ""
  PRINT *, "  Niveau 2 (grille)  : DO CONCURRENT sur nx*ny*nz points"
  PRINT *, "    -> Aplatissement 3D->1D pour maximiser l'occupancy"
  PRINT *, "    -> REDUCE(+:dg_sum) : réduction atomique GPU native"
  PRINT *, "       (Fortran 2023 / nvfortran >= 23.1)"
  PRINT *, ""
  PRINT *, "  Données GPU       : xyz, ZAatoms, ibsi_matrix"
  PRINT *, "    -> Transférées sur le device avant le DO CONCURRENT"
  PRINT *, "    -> Restent sur le GPU pendant tout le calcul IBSI"
  PRINT *, "    -> Rapatriées sur l'hôte après le calcul"
  PRINT *, ""
  PRINT *, "  Fonctions PURE    : grad_rho_inline, dg_ab_point"
  PRINT *, "    -> Inlinées dans le noyau CUDA -> zéro overhead d'appel"
  PRINT *, "    -> Coefficients de Slater résolus à la compilation"
  PRINT *, "       depuis les tableaux PARAMETER -> registres GPU"
  PRINT *, "=========================================================="
  PRINT *, ""

  ! -------------------------------------------------------------------------
  ! Grille
  ! -------------------------------------------------------------------------
  eps = 5.0_DP;  dx = 0.3_DP

  xmin = MINVAL(xyz(1,:)) - eps;  xmax = MAXVAL(xyz(1,:)) + eps
  ymin = MINVAL(xyz(2,:)) - eps;  ymax = MAXVAL(xyz(2,:)) + eps
  zmin = MINVAL(xyz(3,:)) - eps;  zmax = MAXVAL(xyz(3,:)) + eps

  nx = INT((xmax-xmin)/dx) + 1
  ny = INT((ymax-ymin)/dx) + 1
  nz = INT((zmax-zmin)/dx) + 1
  dV = dx**3

  npairs = natoms * (natoms - 1) / 2

  PRINT *, "=========================================================="
  PRINT *, "  GRILLE D'INTÉGRATION"
  PRINT *, "=========================================================="
  WRITE(*,'(A,I6,A,I6,A,I6,A,I14,A)') &
    "  Dimensions       : ", nx, " x ", ny, " x ", nz, &
    "  = ", nx*ny*nz, " pts"
  WRITE(*,'(A,F8.4,A)') "  Pas de grille    : ", dx, " bohr"
  WRITE(*,'(A,ES12.4,A)') "  Volume élémentaire: ", dV, " bohr^3"
  WRITE(*,'(A,I8)')       "  Paires (a,b)     : ", npairs
  WRITE(*,'(A,I12)')      "  Total évaluations : ", npairs * nx * ny * nz
  WRITE(*,'(A,ES12.4,A)') "  FLOPs estimés    : ~", &
    REAL(npairs,DP) * REAL(nx*ny*nz,DP) * 40.0_DP, " FLOPs"
  PRINT *, ""

  ! -------------------------------------------------------------------------
  ! Allocation et construction des listes de paires
  ! -------------------------------------------------------------------------
  ALLOCATE(ibsi_matrix(natoms, natoms))
  ALLOCATE(bond_matrix_cov(natoms, natoms))
  ALLOCATE(bond_matrix_nc(natoms, natoms))
  ALLOCATE(aa_list(npairs), bb_list(npairs))

  ibsi_matrix    = 0.0_DP
  bond_matrix_cov = .FALSE.
  bond_matrix_nc  = .FALSE.

  ipair = 0
  DO a = 1, natoms-1
    DO b = a+1, natoms
      ipair = ipair + 1
      aa_list(ipair) = a
      bb_list(ipair) = b
    END DO
  END DO

  ! =========================================================================
  ! CALCUL GPU PRINCIPAL
  ! =========================================================================
  PRINT *, "=========================================================="
  PRINT *, "  CALCUL GPU - MATRICE IBSI"
  PRINT *, "=========================================================="
  PRINT *, "  [Lancement des noyaux GPU...]"
  PRINT *, "  (Transfert données CPU->GPU inclus dans la mesure)"

  CALL CPU_TIME(t_gpu_start)
#ifdef _OPENMP
  BLOCK
    USE OMP_LIB
    t_gpu_start = OMP_GET_WTIME()
  END BLOCK
#endif

  ! ---------------------------------------------------------------------------
  ! APPEL DU NOYAU GPU
  ! nvfortran avec -stdpar=gpu va :
  !   1) Détecter les tableaux dans le DO CONCURRENT
  !   2) Les allouer en mémoire unifiée (managed memory) ou les copier
  !      explicitement sur le device
  !   3) Générer un noyau CUDA pour compute_ibsi_gpu
  !   4) Synchroniser et rapatrier ibsi_matrix sur l'hôte
  ! ---------------------------------------------------------------------------
  CALL compute_ibsi_gpu(natoms, npairs, aa_list, bb_list, &
                         xyz, ZAatoms,                    &
                         xmin, ymin, zmin, dx, dV,        &
                         nx, ny, nz,                       &
                         ibsi_matrix)

  CALL CPU_TIME(t_gpu_end)
#ifdef _OPENMP
  BLOCK
    USE OMP_LIB
    t_gpu_end = OMP_GET_WTIME()
  END BLOCK
#endif

  ! =========================================================================
  ! RAPPORT DE PERFORMANCES GPU
  ! =========================================================================
  PRINT *, ""
  PRINT *, "=========================================================="
  PRINT *, "  RAPPORT DE PERFORMANCES GPU"
  PRINT *, "=========================================================="
  WRITE(*,'(A,F12.3,A)') "  Temps calcul GPU (DO CONCURRENT)  : ", &
    t_gpu_end - t_gpu_start, " s"
  WRITE(*,'(A,ES12.4,A)') "  Débit GPU                          : ", &
    REAL(npairs,DP)*REAL(nx*ny*nz,DP)*40.0_DP / (t_gpu_end-t_gpu_start), " FLOP/s"
  WRITE(*,'(A,F12.3,A)') "  Temps moyen par paire              : ", &
    (t_gpu_end - t_gpu_start) / MAX(1, npairs) * 1000.0_DP, " ms/paire"
  PRINT *, ""

  IF (tseq_provided .AND. t_seq_ref > 0.0_DP) THEN
    speedup_vs_seq = t_seq_ref / (t_gpu_end - t_gpu_start)
    WRITE(*,'(A,F12.3,A)') "  Temps séquentiel (ref.)            : ", t_seq_ref, " s"
    WRITE(*,'(A,F10.1,A)') "  Speedup GPU vs séquentiel          : ", speedup_vs_seq, "x"
  END IF

  IF (tpar_provided .AND. t_par_ref > 0.0_DP) THEN
    speedup_vs_par = t_par_ref / (t_gpu_end - t_gpu_start)
    WRITE(*,'(A,F12.3,A)') "  Temps OpenMP CPU (ref.)            : ", t_par_ref, " s"
    WRITE(*,'(A,F10.1,A)') "  Speedup GPU vs OpenMP CPU          : ", speedup_vs_par, "x"
  END IF

  IF (tseq_provided .AND. t_seq_ref > 0.0_DP) THEN
    PRINT *, ""
    IF (speedup_vs_seq >= 50.0_DP) THEN
      PRINT *, "  -> Acceleration GPU EXCEPTIONNELLE (>= 50x)"
    ELSE IF (speedup_vs_seq >= 20.0_DP) THEN
      PRINT *, "  -> Acceleration GPU TRES BONNE (>= 20x)"
    ELSE IF (speedup_vs_seq >= 5.0_DP) THEN
      PRINT *, "  -> Acceleration GPU BONNE (>= 5x)"
    ELSE IF (speedup_vs_seq >= 2.0_DP) THEN
      PRINT *, "  -> Acceleration GPU MODEREE (>= 2x)"
    ELSE
      PRINT *, "  -> Acceleration GPU FAIBLE (<2x)"
      PRINT *, "     Verifier: memoire unifiee? Transferts dominants?"
    END IF
  END IF
  PRINT *, "=========================================================="
  PRINT *, ""

  ! =========================================================================
  ! PASSE 1 : Classification des liaisons covalentes
  ! =========================================================================
  PRINT *, "Seuil pour liaisons covalentes (--seuil1): ", threshold_cov
  PRINT *, ""
  PRINT *, "Atome_A    Atome_B    Distance(bohr)    IBSI      Liaison?"
  PRINT *, "--------------------------------------------------------------"

  nbonds_cov = 0
  DO a = 1, natoms-1
    DO b = a+1, natoms
      d_ab = SQRT((xyz(1,a)-xyz(1,b))**2 + &
                  (xyz(2,a)-xyz(2,b))**2 + &
                  (xyz(3,a)-xyz(3,b))**2)
      IF (ibsi_matrix(a,b) > threshold_cov) THEN
        bond_matrix_cov(a,b) = .TRUE.;  bond_matrix_cov(b,a) = .TRUE.
        nbonds_cov = nbonds_cov + 1
        WRITE(*,'(A10,1X,A10,1X,F10.4,6X,F10.6,4X,A)') &
          TRIM(atom_names(a)), TRIM(atom_names(b)), d_ab, ibsi_matrix(a,b), "OUI"
      ELSE
        WRITE(*,'(A10,1X,A10,1X,F10.4,6X,F10.6,4X,A)') &
          TRIM(atom_names(a)), TRIM(atom_names(b)), d_ab, ibsi_matrix(a,b), "NON"
      END IF
    END DO
  END DO

  PRINT *, ""
  PRINT *, "=========================================================="
  PRINT *, "Nombre de liaisons covalentes detectees: ", nbonds_cov
  PRINT *, "=========================================================="
  PRINT *, ""

  PRINT *, "Structure moleculaire (graphe de liaisons covalentes):"
  PRINT *, "----------------------------------------------------------"
  DO a = 1, natoms
    WRITE(*,'(A,I3," (",A,") --> ")', ADVANCE='NO') "Atome ", a, TRIM(atom_names(a))
    DO b = 1, natoms
      IF (bond_matrix_cov(a,b) .AND. b/=a) WRITE(*,'(I4)', ADVANCE='NO') b
    END DO
    PRINT *, ""
  END DO
  PRINT *, ""

  CALL identify_fragments(natoms, bond_matrix_cov, nfrags, fragment_id)
  PRINT *, "Nombre de fragments detectes: ", nfrags
  IF (nfrags > 1) THEN
    DO i = 1, nfrags
      WRITE(*,'(A,I2,A)', ADVANCE='NO') "Fragment ", i, ": Atomes "
      DO j = 1, natoms
        IF (fragment_id(j)==i) WRITE(*,'(I3)', ADVANCE='NO') j
      END DO
      PRINT *, ""
    END DO
  END IF
  PRINT *, ""

  ! =========================================================================
  ! PASSE 2 : liaisons non-covalentes
  ! =========================================================================
  IF (has_fragments .AND. nfrags > 1) THEN
    PRINT *, "=========================================================="
    PRINT *, "PASSE 2: LIAISONS NON-COVALENTES"
    PRINT *, "=========================================================="
    PRINT *, "Fragment 1: atomes", frag1_start, "a", frag1_end
    PRINT *, "Fragment 2: atomes", frag2_start, "a", frag2_end
    PRINT *, ""
    PRINT *, "Seuil (--seuil2): ", threshold_nc
    PRINT *, ""
    PRINT *, "Atome_A    Atome_B    Distance(bohr)    IBSI      Liaison NC?"
    PRINT *, "--------------------------------------------------------------"

    nbonds_nc = 0
    DO a = frag1_start, frag1_end
      DO b = frag2_start, frag2_end
        d_ab = SQRT((xyz(1,a)-xyz(1,b))**2 + &
                    (xyz(2,a)-xyz(2,b))**2 + &
                    (xyz(3,a)-xyz(3,b))**2)
        IF (ibsi_matrix(a,b) > threshold_nc .AND. ibsi_matrix(a,b) <= threshold_cov) THEN
          bond_matrix_nc(a,b) = .TRUE.;  bond_matrix_nc(b,a) = .TRUE.
          nbonds_nc = nbonds_nc + 1
          WRITE(*,'(A10,1X,A10,1X,F10.4,6X,F10.6,4X,A)') &
            TRIM(atom_names(a)), TRIM(atom_names(b)), d_ab, ibsi_matrix(a,b), "OUI"
        ELSE
          WRITE(*,'(A10,1X,A10,1X,F10.4,6X,F10.6,4X,A)') &
            TRIM(atom_names(a)), TRIM(atom_names(b)), d_ab, ibsi_matrix(a,b), "NON"
        END IF
      END DO
    END DO

    PRINT *, ""
    PRINT *, "=========================================================="
    PRINT *, "Nombre de liaisons non-covalentes detectees: ", nbonds_nc
    PRINT *, "=========================================================="
    PRINT *, ""
  END IF

  ! =========================================================================
  ! Fichier PDB
  ! =========================================================================
  OPEN(NEWUNIT=unit_pdb, FILE="molecule.pdb", STATUS="REPLACE")
  DO i = 1, natoms
    SELECT CASE(ZAatoms(i))
      CASE(1);  atom_symbol="H "
      CASE(6);  atom_symbol="C "
      CASE(7);  atom_symbol="N "
      CASE(8);  atom_symbol="O "
      CASE(9);  atom_symbol="F "
      CASE(15); atom_symbol="P "
      CASE(16); atom_symbol="S "
      CASE(17); atom_symbol="Cl"
      CASE DEFAULT; atom_symbol="X "
    END SELECT
    WRITE(unit_pdb,'(A6,I5,2X,A2,2X,A3,1X,A1,I3,4X,3F8.3,2F6.2,10X,A2)') &
      "ATOM  ",i,atom_symbol,"UNK","A",1, &
      xyz(1,i)*0.529177_DP, xyz(2,i)*0.529177_DP, xyz(3,i)*0.529177_DP, &
      1.00_DP, 0.00_DP, atom_symbol
  END DO
  DO i = 1, natoms
    WRITE(unit_pdb,'(A6,I5)',ADVANCE='NO') "CONECT",i
    DO j = 1, natoms
      IF (bond_matrix_cov(i,j) .AND. j/=i) WRITE(unit_pdb,'(I5)',ADVANCE='NO') j
    END DO
    DO j = 1, natoms
      IF (bond_matrix_nc(i,j) .AND. j/=i .AND. .NOT.bond_matrix_cov(i,j)) &
        WRITE(unit_pdb,'(I5)',ADVANCE='NO') j
    END DO
    WRITE(unit_pdb,*) ""
  END DO
  WRITE(unit_pdb,'(A)') "END"
  CLOSE(unit_pdb)

  ! =========================================================================
  ! Fichier résultats
  ! =========================================================================
  OPEN(NEWUNIT=unit_res, FILE="ibsi_results.dat", STATUS="REPLACE")
  WRITE(unit_res,*) "# Resultats IBSI (version GPU - DO CONCURRENT)"
  WRITE(unit_res,*) "# Dossier:", TRIM(input_dir)
  WRITE(unit_res,*) "# Natoms:", natoms, "  Paires:", npairs
  WRITE(unit_res,*) "# Grille:", nx, "x", ny, "x", nz
  WRITE(unit_res,*) "# Seuil covalent:", threshold_cov
  WRITE(unit_res,*) "# Seuil non-covalent:", threshold_nc
  WRITE(unit_res,*) "# Temps GPU (s):", t_gpu_end - t_gpu_start
  WRITE(unit_res,*) "#"
  WRITE(unit_res,*) "# Format: Atome_i Atome_j IBSI_ij Type(0=aucune,1=cov,2=nc)"
  DO a = 1, natoms-1
    DO b = a+1, natoms
      IF (bond_matrix_cov(a,b)) THEN
        WRITE(unit_res,'(I5,I5,F15.8,I5)') a, b, ibsi_matrix(a,b), 1
      ELSE IF (bond_matrix_nc(a,b)) THEN
        WRITE(unit_res,'(I5,I5,F15.8,I5)') a, b, ibsi_matrix(a,b), 2
      ELSE
        WRITE(unit_res,'(I5,I5,F15.8,I5)') a, b, ibsi_matrix(a,b), 0
      END IF
    END DO
  END DO
  CLOSE(unit_res)

  ! =========================================================================
  ! BILAN FINAL
  ! =========================================================================
  CALL CPU_TIME(t_total_end)
#ifdef _OPENMP
  BLOCK
    USE OMP_LIB
    t_total_end = OMP_GET_WTIME()
  END BLOCK
#endif

  PRINT *, "Fichiers ecrits : molecule.pdb  ibsi_results.dat"
  PRINT *, ""
  PRINT *, "=========================================================="
  PRINT *, "  BILAN FINAL"
  PRINT *, "=========================================================="
  WRITE(*,'(A,F12.3,A)') "  Temps total (mur)            : ", &
    t_total_end - t_total_start, " s"
  WRITE(*,'(A,F12.3,A)') "  dont noyau GPU (DO CONCURRENT): ", &
    t_gpu_end - t_gpu_start, " s"
  WRITE(*,'(A,F6.1,A)')  "  Fraction temps GPU           : ", &
    (t_gpu_end-t_gpu_start)/(t_total_end-t_total_start)*100.0_DP, " %"
  IF (tseq_provided .AND. t_seq_ref > 0.0_DP) THEN
    speedup_vs_seq = t_seq_ref / (t_gpu_end - t_gpu_start)
    WRITE(*,'(A,F10.1,A)') "  Speedup final GPU vs seq.    : ", speedup_vs_seq, "x"
  END IF
  IF (tpar_provided .AND. t_par_ref > 0.0_DP) THEN
    speedup_vs_par = t_par_ref / (t_gpu_end - t_gpu_start)
    WRITE(*,'(A,F10.1,A)') "  Speedup final GPU vs OpenMP  : ", speedup_vs_par, "x"
  END IF
  PRINT *, "=========================================================="
  PRINT *, ""
  PRINT *, "ANALYSE GPU TERMINEE AVEC SUCCES!"
  PRINT *, "=========================================================="

END PROGRAM ibsi_gpu_analysis