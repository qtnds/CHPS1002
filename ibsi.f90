MODULE ab_module
  IMPLICIT NONE
  
  INTEGER, PARAMETER :: NB_KNW_ATMS = 54

  REAL(8), PARAMETER :: AA1(NB_KNW_ATMS) = (/ &
    0.2815d0, 2.437d0, &
    11.84d0, 31.34d0, 67.82d0, 120.2d0, 190.9d0, 289.5d0, 406.3d0, 561.3d0, &
    760.8d0, 1016.0d0, 1319.0d0, 1658.0d0, 2042.0d0, 2501.0d0, 3024.0d0, 3625.0d0, &
    422.685d0, 328.056d0, &
    369.212d0, 294.479d0, 525.536d0, 434.298d0, 9346.12d0, 579.606d0, 665.791d0, &
    793.8d0, 897.784d0, 1857.03d0, &
    898.009d0, 1001.24d0, 1101.35d0, 1272.92d0, 1333.8d0, 1459.53d0, &
    0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, &
    0.0d0, 0.0d0, 0.0d0, 457.406d0, 3.23589d0, 643.098d0 /)

  REAL(8), PARAMETER :: AA2(NB_KNW_ATMS) = (/ &
    0.0d0, 0.0d0, &
    0.06332d0, 0.3694d0, 0.8527d0, 1.172d0, 2.247d0, 2.879d0, 3.049d0, 6.984d0, &
    22.42d0, 37.17d0, 57.95d0, 87.16d0, 115.7d0, 158.0d0, 205.5d0, 260.0d0, &
    104.678d0, 48.1693d0, &
    66.7813d0, 64.3627d0, 97.5552d0, 37.8524d0, 166.393d0, 50.2383d0, 51.7033d0, &
    60.5725d0, 58.8879d0, 135.027d0, &
    1.10777d0, 0.855815d0, 901.893d0, 1.20778d0, 1.0722d0, 1.95028d0, &
    0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, &
    0.0d0, 0.0d0, 0.0d0, 0.0648533d0, 4.1956d0, 0.133996d0 /)

  REAL(8), PARAMETER :: AA3(NB_KNW_ATMS) = (/ &
    0.0d0, 0.0d0, &
    0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, &
    0.06358d0, 0.3331d0, 0.8878d0, 0.7888d0, 1.465d0, 2.170d0, 3.369d0, 5.211d0, &
    0.0305251d0, 0.137293d0, &
    0.217304d0, 0.312378d0, 0.288164d0, 0.216041d0, 0.428442d0, 0.301226d0, &
    0.358959d0, 0.384414d0, 0.446463d0, 0.664027d0, &
    0.178217d0, 0.271045d0, 1.43271d0, 0.548474d0, 1.52238d0, 3.12305d0, &
    0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, &
    0.0d0, 0.0d0, 0.0d0, 0.0273883d0, 0.0347855d0, 0.0499776d0 /)

  REAL(8), PARAMETER :: BB1(NB_KNW_ATMS) = (/ &
    1.8910741301059d0, 2.95945546019532d0, &
    5.23012552301255d0, 7.19424460431655d0, 9.44287063267233d0, 11.3122171945701d0, &
    13.0378096479791d0, 14.9476831091181d0, 16.4473684210526d0, 18.2149362477231d0, &
    20.1612903225806d0, 22.271714922049d0, 24.330900243309d0, 26.1780104712042d0, &
    27.9329608938547d0, 29.8507462686567d0, 31.7460317460318d0, 33.7837837837838d0, &
    61.3085d0, 13.6033d0, &
    15.1828d0, 8.87333d0, 18.4925d0, 12.2138d0, 47.0758d0, 13.6638d0, 14.1117d0, &
    15.1832d0, 15.4177d0, 23.5312d0, &
    12.4217d0, 12.8073d0, 13.0865d0, 13.8734d0, 13.8227d0, 14.1814d0, &
    0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, &
    0.0d0, 0.0d0, 0.0d0, 3.82753d0, 23.9761d0, 4.10166d0 /)

  REAL(8), PARAMETER :: BB2(NB_KNW_ATMS) = (/ &
    1.0d0, 1.0d0, &
    1.00080064051241d0, 1.43988480921526d0, 1.88679245283019d0, 1.82481751824818d0, &
    2.20653133274492d0, 2.51635631605435d0, 2.50375563345018d0, 2.90107339715695d0, &
    3.98247710075667d0, 4.65116279069767d0, 5.33617929562433d0, 6.0459492140266d0, &
    6.62690523525514d0, 7.30460189919649d0, 7.9428117553614d0, 8.56164383561644d0, &
    3.81272d0, 3.94253d0, &
    4.24199d0, 4.33369d0, 4.58656d0, 3.81061d0, 5.07878d0, 4.09039d0, 4.17018d0, &
    4.30955d0, 4.35337d0, 5.19033d0, &
    2.00145d0, 1.85496d0, 7.70162d0, 1.88584d0, 2.12925d0, 2.35537d0, &
    0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, &
    0.0d0, 0.0d0, 0.0d0, 1.19985d0, 2.24851d0, 1.25439d0 /)

  REAL(8), PARAMETER :: BB3(NB_KNW_ATMS) = (/ &
    1.0d0, 1.0d0, &
    1.0d0, 1.0d0, 1.0d0, 1.0d0, 1.0d0, 1.0d0, 1.0d0, 1.0d0, &
    0.9769441187964d0, 1.28982329420869d0, 1.67728950016773d0, 1.42959256611866d0, &
    1.70910955392241d0, 1.94212468440474d0, 2.01045436268597d0, 2.26654578422484d0, &
    0.760211d0, 0.972271d0, &
    1.08807d0, 1.18612d0, 1.19053d0, 1.22212d0, 1.3246d0, 1.32999d0, 1.38735d0, &
    1.42114d0, 1.48338d0, 1.54071d0, &
    1.19489d0, 1.31783d0, 1.64735d0, 1.57665d0, 1.79153d0, 1.99148d0, &
    0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0, &
    0.0d0, 0.0d0, 0.0d0, 0.906738d0, 0.856995d0, 1.02511d0 /)

CONTAINS

  SUBROUTINE get_ab_coefficients(ZA, a1, a2, a3, b1, b2, b3)
    INTEGER, INTENT(in) :: ZA
    REAL(8), INTENT(out) :: a1, a2, a3, b1, b2, b3

    IF (ZA < 1 .OR. ZA > NB_KNW_ATMS) THEN
      PRINT *, "ERREUR: Numero atomique hors limites:", ZA
      a1 = 0.0d0
      a2 = 0.0d0
      a3 = 0.0d0
      b1 = 0.0d0
      b2 = 0.0d0
      b3 = 0.0d0
      RETURN
    END IF

    a1 = AA1(ZA)
    a2 = AA2(ZA)
    a3 = AA3(ZA)
    b1 = BB1(ZA)
    b2 = BB2(ZA)
    b3 = BB3(ZA)
  END SUBROUTINE get_ab_coefficients

END MODULE ab_module


MODULE ibsi_module
  USE ab_module
  IMPLICIT NONE

CONTAINS

  FUNCTION rho_atom(x, y, z, ax, ay, az, ZA) RESULT(rho)
    REAL(8), INTENT(in) :: x, y, z, ax, ay, az
    INTEGER, INTENT(in) :: ZA
    REAL(8) :: rho, r, a1, a2, a3, b1, b2, b3

    CALL get_ab_coefficients(ZA, a1, a2, a3, b1, b2, b3)
    r = SQRT((x-ax)**2 + (y-ay)**2 + (z-az)**2)
    rho = a1*EXP(-b1*r) + a2*EXP(-b2*r) + a3*EXP(-b3*r)
  END FUNCTION rho_atom

  SUBROUTINE grad_rho_atom(x, y, z, ax, ay, az, ZA, drho_dx, drho_dy, drho_dz)
    REAL(8), INTENT(in) :: x, y, z, ax, ay, az
    INTEGER, INTENT(in) :: ZA
    REAL(8), INTENT(out) :: drho_dx, drho_dy, drho_dz
    REAL(8) :: r, dx, dy, dz, a1, a2, a3, b1, b2, b3
    REAL(8) :: exp1, exp2, exp3, factor

    CALL get_ab_coefficients(ZA, a1, a2, a3, b1, b2, b3)
    dx = x - ax
    dy = y - ay
    dz = z - az
    r = SQRT(dx**2 + dy**2 + dz**2)

    IF (r < 1.0d-10) THEN
      drho_dx = 0.0d0
      drho_dy = 0.0d0
      drho_dz = 0.0d0
      RETURN
    END IF

    exp1 = EXP(-b1*r)
    exp2 = EXP(-b2*r)
    exp3 = EXP(-b3*r)
    factor = -(a1*b1*exp1 + a2*b2*exp2 + a3*b3*exp3) / r
    drho_dx = factor * dx
    drho_dy = factor * dy
    drho_dz = factor * dz
  END SUBROUTINE grad_rho_atom

  FUNCTION compute_dg_ab(x, y, z, xa, ya, za, ZAa, xb, yb, zb, ZAb) RESULT(dg)
    REAL(8), INTENT(in) :: x, y, z, xa, ya, za, xb, yb, zb
    INTEGER, INTENT(in) :: ZAa, ZAb
    REAL(8) :: dg
    REAL(8) :: drho_a_dx, drho_a_dy, drho_a_dz
    REAL(8) :: drho_b_dx, drho_b_dy, drho_b_dz
    REAL(8) :: norm1, norm2

    CALL grad_rho_atom(x, y, z, xa, ya, za, ZAa, drho_a_dx, drho_a_dy, drho_a_dz)
    CALL grad_rho_atom(x, y, z, xb, yb, zb, ZAb, drho_b_dx, drho_b_dy, drho_b_dz)

    norm1 = SQRT((ABS(drho_a_dx) + ABS(drho_b_dx))**2 + &
                 (ABS(drho_a_dy) + ABS(drho_b_dy))**2 + &
                 (ABS(drho_a_dz) + ABS(drho_b_dz))**2)
    norm2 = SQRT((drho_a_dx + drho_b_dx)**2 + &
                 (drho_a_dy + drho_b_dy)**2 + &
                 (drho_a_dz + drho_b_dz)**2)
    dg = norm1 - norm2
  END FUNCTION compute_dg_ab

END MODULE ibsi_module


MODULE wfx_reader
  IMPLICIT NONE
CONTAINS

  SUBROUTINE read_wfx_geometry(filename, natoms, xyz, ZAatoms, atom_names)
    CHARACTER(len=*), INTENT(in) :: filename
    INTEGER, INTENT(out) :: natoms
    REAL(8), ALLOCATABLE, INTENT(out) :: xyz(:,:)
    INTEGER, ALLOCATABLE, INTENT(out) :: ZAatoms(:)
    CHARACTER(len=10), ALLOCATABLE, INTENT(out) :: atom_names(:)
    INTEGER :: i, unit
    CHARACTER(256) :: line

    OPEN(NEWUNIT=unit, FILE=TRIM(filename), STATUS="OLD", ERR=200)

    DO
      READ(unit,'(A)',END=100) line
      IF (INDEX(line,"<Number of Nuclei>")>0) THEN
        READ(unit,*) natoms
        EXIT
      END IF
    END DO

    REWIND(unit)

    DO
      READ(unit,'(A)',END=100) line
      IF (INDEX(line,"<Nuclear Names>")>0) THEN
        ALLOCATE(atom_names(natoms))
        DO i=1,natoms
          READ(unit,'(A)') atom_names(i)
          atom_names(i) = ADJUSTL(atom_names(i))
        END DO
        EXIT
      END IF
    END DO

    REWIND(unit)

    DO
      READ(unit,'(A)',END=100) line
      IF (INDEX(line,"<Nuclear Cartesian Coordinates>")>0) THEN
        ALLOCATE(xyz(3,natoms))
        DO i=1,natoms
          READ(unit,*) xyz(:,i)
        END DO
        EXIT
      END IF
    END DO

    REWIND(unit)

    DO
      READ(unit,'(A)',END=100) line
      IF (INDEX(line,"<Atomic Numbers>")>0) THEN
        ALLOCATE(ZAatoms(natoms))
        DO i=1,natoms
          READ(unit,*) ZAatoms(i)
        END DO
        EXIT
      END IF
    END DO

100 CLOSE(unit)
    RETURN

200 PRINT *, "ERREUR: Impossible d'ouvrir le fichier: ", TRIM(filename)
    STOP
  END SUBROUTINE read_wfx_geometry

  SUBROUTINE read_xyz_fragments(filename, has_fragments, frag1_start, frag1_end, frag2_start, frag2_end)
    CHARACTER(len=*), INTENT(in) :: filename
    LOGICAL, INTENT(out) :: has_fragments
    INTEGER, INTENT(out) :: frag1_start, frag1_end, frag2_start, frag2_end
    INTEGER :: unit, ios, dash_pos
    CHARACTER(512) :: line, sel_a, sel_b
    INTEGER :: pos_sel_a, pos_sel_b

    has_fragments = .FALSE.
    frag1_start = 0
    frag1_end = 0
    frag2_start = 0
    frag2_end = 0

    OPEN(NEWUNIT=unit, FILE=TRIM(filename), STATUS="OLD", IOSTAT=ios)
    IF (ios /= 0) THEN
      RETURN
    END IF

    READ(unit, '(A)', IOSTAT=ios) line
    READ(unit, '(A)', IOSTAT=ios) line
    CLOSE(unit)

    IF (ios /= 0) RETURN

    line = ADJUSTL(line)
    
    pos_sel_a = INDEX(line, 'selection_a:')
    pos_sel_b = INDEX(line, 'selection_b:')
    
    IF (pos_sel_a == 0 .OR. pos_sel_b == 0) RETURN

    sel_a = line(pos_sel_a+12:pos_sel_b-1)
    sel_a = ADJUSTL(sel_a)
    
    sel_b = line(pos_sel_b+12:)
    sel_b = ADJUSTL(sel_b)

    dash_pos = INDEX(sel_a, '-')
    IF (dash_pos > 0) THEN
      READ(sel_a(1:dash_pos-1), *, IOSTAT=ios) frag1_start
      IF (ios /= 0) RETURN
      READ(sel_a(dash_pos+1:), *, IOSTAT=ios) frag1_end
      IF (ios /= 0) RETURN
    ELSE
      READ(sel_a, *, IOSTAT=ios) frag1_start
      IF (ios /= 0) RETURN
      frag1_end = frag1_start
    END IF

    dash_pos = INDEX(sel_b, '-')
    IF (dash_pos > 0) THEN
      READ(sel_b(1:dash_pos-1), *, IOSTAT=ios) frag2_start
      IF (ios /= 0) RETURN
      READ(sel_b(dash_pos+1:), *, IOSTAT=ios) frag2_end
      IF (ios /= 0) RETURN
    ELSE
      READ(sel_b, *, IOSTAT=ios) frag2_start
      IF (ios /= 0) RETURN
      frag2_end = frag2_start
    END IF

    has_fragments = .TRUE.

  END SUBROUTINE read_xyz_fragments

END MODULE wfx_reader


MODULE fragment_utils
  IMPLICIT NONE
CONTAINS

  SUBROUTINE identify_fragments(natoms, bond_matrix, nfrags, fragment_id)
    INTEGER, INTENT(in) :: natoms
    LOGICAL, INTENT(in) :: bond_matrix(:,:)
    INTEGER, INTENT(out) :: nfrags
    INTEGER, ALLOCATABLE, INTENT(out) :: fragment_id(:)
    INTEGER :: i, j, current_frag
    LOGICAL :: visited(natoms), changed

    ALLOCATE(fragment_id(natoms))
    fragment_id = 0
    visited = .FALSE.
    nfrags = 0

    DO i = 1, natoms
      IF (.NOT. visited(i)) THEN
        nfrags = nfrags + 1
        fragment_id(i) = nfrags
        visited(i) = .TRUE.
        
        changed = .TRUE.
        DO WHILE (changed)
          changed = .FALSE.
          DO j = 1, natoms
            IF (fragment_id(j) == nfrags .AND. .NOT. visited(j)) THEN
              visited(j) = .TRUE.
              changed = .TRUE.
            END IF
            IF (fragment_id(j) == nfrags) THEN
              DO current_frag = 1, natoms
                IF (bond_matrix(j, current_frag) .AND. fragment_id(current_frag) == 0) THEN
                  fragment_id(current_frag) = nfrags
                  changed = .TRUE.
                END IF
              END DO
            END IF
          END DO
        END DO
      END IF
    END DO
  END SUBROUTINE identify_fragments

END MODULE fragment_utils


PROGRAM ibsi_analysis
  USE ibsi_module
  USE wfx_reader
  USE fragment_utils
  IMPLICIT NONE

  INTEGER :: natoms, i, j, k, a, b
  REAL(8), ALLOCATABLE :: xyz(:,:), ibsi_matrix(:,:)
  INTEGER, ALLOCATABLE :: ZAatoms(:)
  CHARACTER(len=10), ALLOCATABLE :: atom_names(:)
  INTEGER :: nbonds_cov, nbonds_nc
  REAL(8) :: xmin, xmax, ymin, ymax, zmin, zmax, dx, eps
  INTEGER :: nx, ny, nz
  REAL(8) :: x, y, z, dg_sum, d_ab, dV, ibsi_ab
  REAL(8) :: threshold_cov, threshold_nc
  INTEGER :: unit_pdb, unit_res, iarg, iostat_val
  CHARACTER(2) :: atom_symbol
  LOGICAL, ALLOCATABLE :: bond_matrix_cov(:,:), bond_matrix_nc(:,:)
  CHARACTER(len=256) :: arg, input_dir, wfx_file, xyz_file, temp_val
  INTEGER :: frag1_start, frag1_end, frag2_start, frag2_end
  INTEGER :: nfrags
  INTEGER, ALLOCATABLE :: fragment_id(:)
  LOGICAL :: has_fragments

  input_dir = ""
  threshold_cov = 0.05d0
  threshold_nc = 0.02d0

  iarg = 1
  DO WHILE (iarg <= COMMAND_ARGUMENT_COUNT())
    CALL GET_COMMAND_ARGUMENT(iarg, arg)
    IF (TRIM(arg) == "--input") THEN
      iarg = iarg + 1
      CALL GET_COMMAND_ARGUMENT(iarg, input_dir)
    ELSE IF (TRIM(arg) == "--seuil1") THEN
      iarg = iarg + 1
      CALL GET_COMMAND_ARGUMENT(iarg, temp_val)
      temp_val = TRIM(temp_val)
      READ(temp_val, *, IOSTAT=iostat_val) threshold_cov
      IF (iostat_val /= 0) THEN
        PRINT *, "ERREUR: Valeur invalide pour --seuil1:", TRIM(temp_val)
        STOP
      END IF
    ELSE IF (TRIM(arg) == "--seuil2") THEN
      iarg = iarg + 1
      CALL GET_COMMAND_ARGUMENT(iarg, temp_val)
      temp_val = TRIM(temp_val)
      READ(temp_val, *, IOSTAT=iostat_val) threshold_nc
      IF (iostat_val /= 0) THEN
        PRINT *, "ERREUR: Valeur invalide pour --seuil2:", TRIM(temp_val)
        STOP
      END IF
    END IF
    iarg = iarg + 1
  END DO

  IF (TRIM(input_dir) == "") THEN
    PRINT *, "Usage: ./ibsi --input <dossier> [--seuil1 <valeur>] [--seuil2 <valeur>]"
    PRINT *, ""
    PRINT *, "Exemple: ./ibsi --input exemples_arbre_molecule/2_dimer --seuil1 0.05 --seuil2 0.009"
    PRINT *, ""
    PRINT *, "Arguments:"
    PRINT *, "  --input   : Dossier contenant mol.wfx (OBLIGATOIRE)"
    PRINT *, "  --seuil1  : Seuil pour liaisons covalentes (defaut: 0.05)"
    PRINT *, "  --seuil2  : Seuil pour liaisons non-covalentes (defaut: 0.02)"
    PRINT *, ""
    PRINT *, "Note: Pour activer la passe 2 (liaisons non-covalentes), creer"
    PRINT *, "      un fichier mol.xyz avec a la ligne 2:"
    PRINT *, "      selection_a: 1-3 selection_b: 4-6"
    STOP
  END IF

  wfx_file = TRIM(input_dir) // "/mol.wfx"
  xyz_file = TRIM(input_dir) // "/mol.xyz"
  
  CALL read_wfx_geometry(wfx_file, natoms, xyz, ZAatoms, atom_names)
  CALL read_xyz_fragments(xyz_file, has_fragments, frag1_start, frag1_end, frag2_start, frag2_end)

  PRINT *, "=========================================================="
  PRINT *, "    ANALYSE IBSI - LIAISONS COVALENTES ET NON-COVALENTES"
  PRINT *, "=========================================================="
  PRINT *, "Dossier d'entree : ", TRIM(input_dir)
  PRINT *, "Fichier WFX      : ", TRIM(wfx_file)
  PRINT *, "Nombre d'atomes  : ", natoms
  PRINT *, ""
  PRINT *, "Parametres:"
  PRINT *, "  Seuil liaisons covalentes     (--seuil1): ", threshold_cov
  PRINT *, "  Seuil liaisons non-covalentes (--seuil2): ", threshold_nc
  PRINT *, ""
  IF (has_fragments) THEN
    PRINT *, "Fichier mol.xyz trouve - Passe 2 activee"
    PRINT *, "  Fragment A: atomes", frag1_start, "a", frag1_end
    PRINT *, "  Fragment B: atomes", frag2_start, "a", frag2_end
    PRINT *, ""
  END IF
  PRINT *, "Liste des atomes (ordre du WFX):"
  DO i = 1, natoms
    WRITE(*,'(I5,". ",A)') i, TRIM(atom_names(i))
  END DO
  PRINT *, ""

  eps = 5.0d0
  dx  = 0.3d0

  xmin = MINVAL(xyz(1,:)) - eps
  xmax = MAXVAL(xyz(1,:)) + eps
  ymin = MINVAL(xyz(2,:)) - eps
  ymax = MAXVAL(xyz(2,:)) + eps
  zmin = MINVAL(xyz(3,:)) - eps
  zmax = MAXVAL(xyz(3,:)) + eps

  nx = INT((xmax-xmin)/dx) + 1
  ny = INT((ymax-ymin)/dx) + 1
  nz = INT((zmax-zmin)/dx) + 1
  dV = dx**3

  PRINT *, "Grille:", nx, "x", ny, "x", nz, "points"
  PRINT *, "Pas de grille (dx):", dx, "bohr"
  PRINT *, "Volume elementaire (dV):", dV, "bohr^3"
  PRINT *, ""

  ALLOCATE(ibsi_matrix(natoms, natoms))
  ALLOCATE(bond_matrix_cov(natoms, natoms))
  ALLOCATE(bond_matrix_nc(natoms, natoms))
  ibsi_matrix = 0.0d0
  bond_matrix_cov = .FALSE.
  bond_matrix_nc = .FALSE.

  PRINT *, "=========================================================="
  PRINT *, "PASSE 1: CALCUL DES LIAISONS COVALENTES"
  PRINT *, "=========================================================="
  PRINT *, ""

  DO a = 1, natoms-1
    DO b = a+1, natoms
      dg_sum = 0.0d0
      d_ab = SQRT((xyz(1,a)-xyz(1,b))**2 + &
                  (xyz(2,a)-xyz(2,b))**2 + &
                  (xyz(3,a)-xyz(3,b))**2)

      DO i = 1, nx
        x = xmin + (i-1)*dx
        DO j = 1, ny
          y = ymin + (j-1)*dx
          DO k = 1, nz
            z = zmin + (k-1)*dx
            dg_sum = dg_sum + compute_dg_ab(x, y, z, &
                      xyz(1,a), xyz(2,a), xyz(3,a), ZAatoms(a), &
                      xyz(1,b), xyz(2,b), xyz(3,b), ZAatoms(b))
          END DO
        END DO
      END DO

      ibsi_ab = (dg_sum * dV) / (d_ab**2)
      ibsi_matrix(a,b) = ibsi_ab
      ibsi_matrix(b,a) = ibsi_ab
    END DO
  END DO

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
        bond_matrix_cov(a,b) = .TRUE.
        bond_matrix_cov(b,a) = .TRUE.
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
      IF (bond_matrix_cov(a,b) .AND. b /= a) WRITE(*,'(I4)', ADVANCE='NO') b
    END DO
    PRINT *, ""
  END DO
  PRINT *, ""

  CALL identify_fragments(natoms, bond_matrix_cov, nfrags, fragment_id)
  
  PRINT *, "Nombre de fragments detectes: ", nfrags
  PRINT *, ""
  IF (nfrags > 1) THEN
    DO i = 1, nfrags
      WRITE(*,'(A,I2,A)', ADVANCE='NO') "Fragment ", i, ": Atomes "
      DO j = 1, natoms
        IF (fragment_id(j) == i) WRITE(*,'(I3)', ADVANCE='NO') j
      END DO
      PRINT *, ""
    END DO
  END IF
  PRINT *, ""

  IF (has_fragments .AND. nfrags > 1) THEN
    PRINT *, "=========================================================="
    PRINT *, "PASSE 2: CALCUL DES LIAISONS NON-COVALENTES"
    PRINT *, "=========================================================="
    PRINT *, ""
    PRINT *, "Fragment 1: atomes", frag1_start, "a", frag1_end
    PRINT *, "Fragment 2: atomes", frag2_start, "a", frag2_end
    PRINT *, ""
    PRINT *, "Seuil pour liaisons non-covalentes (--seuil2): ", threshold_nc
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
          bond_matrix_nc(a,b) = .TRUE.
          bond_matrix_nc(b,a) = .TRUE.
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

  OPEN(NEWUNIT=unit_pdb, FILE="molecule.pdb", STATUS="REPLACE")

  DO i = 1, natoms
    SELECT CASE(ZAatoms(i))
      CASE(1);  atom_symbol = "H "
      CASE(6);  atom_symbol = "C "
      CASE(7);  atom_symbol = "N "
      CASE(8);  atom_symbol = "O "
      CASE(9);  atom_symbol = "F "
      CASE(15); atom_symbol = "P "
      CASE(16); atom_symbol = "S "
      CASE(17); atom_symbol = "Cl"
      CASE DEFAULT; atom_symbol = "X "
    END SELECT
    WRITE(unit_pdb,'(A6,I5,2X,A2,2X,A3,1X,A1,I3,4X,3F8.3,2F6.2,10X,A2)') &
      "ATOM  ", i, atom_symbol, "UNK", "A", 1, &
      xyz(1,i)*0.529177d0, xyz(2,i)*0.529177d0, xyz(3,i)*0.529177d0, &
      1.00d0, 0.00d0, atom_symbol
  END DO

  DO i = 1, natoms
    WRITE(unit_pdb,'(A6,I5)', ADVANCE='NO') "CONECT", i
    DO j = 1, natoms
      IF (bond_matrix_cov(i,j) .AND. j /= i) WRITE(unit_pdb,'(I5)', ADVANCE='NO') j
    END DO
    DO j = 1, natoms
      IF (bond_matrix_nc(i,j) .AND. j /= i .AND. .NOT. bond_matrix_cov(i,j)) THEN
        WRITE(unit_pdb,'(I5)', ADVANCE='NO') j
      END IF
    END DO
    WRITE(unit_pdb,*) ""
  END DO

  WRITE(unit_pdb,'(A)') "END"
  CLOSE(unit_pdb)
  PRINT *, "Fichier PDB ecrit: molecule.pdb"
  PRINT *, "  - Liaisons covalentes: trait plein"
  IF (has_fragments) THEN
    PRINT *, "  - Liaisons non-covalentes: pointilles (dans VMD)"
  END IF

  OPEN(NEWUNIT=unit_res, FILE="ibsi_results.dat", STATUS="REPLACE")
  WRITE(unit_res,*) "# Resultats IBSI - Matrice complete"
  WRITE(unit_res,*) "# Dossier:", TRIM(input_dir)
  WRITE(unit_res,*) "# Nombre d'atomes:", natoms
  WRITE(unit_res,*) "# Seuil liaisons covalentes:", threshold_cov
  WRITE(unit_res,*) "# Seuil liaisons non-covalentes:", threshold_nc
  WRITE(unit_res,*) "#"
  WRITE(unit_res,*) "# Format: Atome_i Atome_j IBSI_ij Type(0=aucune,1=covalente,2=non-cov)"
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
  PRINT *, "Fichier de resultats ecrit: ibsi_results.dat"

  PRINT *, ""
  PRINT *, "=========================================================="
  PRINT *, "ANALYSE TERMINEE AVEC SUCCES!"
  PRINT *, "=========================================================="

END PROGRAM ibsi_analysis