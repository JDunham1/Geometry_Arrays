import emopt
import numpy as np
from numpy import sin
from math import pi
deg_2_rad = pi / 180
from GeometryArrays import (
    Hexagon, Pentagon, ArrayElement, GeometryArray, VCSELGenerator
)
from ga_emopt_bridge import build_eps_from_generator
import matplotlib.pyplot as plt

#%% ------------------------------------------ Build VCSEL Geometries ------------------------------------------
# Build 3 Element in-line array
wavelength = 0.85
psl_lateral_growth = 5
aperture_radius = 3*np.sqrt(2)/2
psl_implant_width = 30
psl_implant_length = 2

aperture_circumradius = aperture_radius / np.cos(np.pi/5)
psl_center_spacing = 10 - aperture_circumradius

inradii_penta = np.array((1,1,1,1,1))*(psl_lateral_growth + aperture_radius)
left_poly = Pentagon.from_inradii(inradii_penta)
right_poly = Pentagon.from_inradii(inradii_penta)
#center element defined via circumradii (as sidelengths are not being directly modified to control detuning)
inradii_hexa = np.ones((6,))*(aperture_radius+psl_lateral_growth)
compensated_lateral_growth = psl_lateral_growth / sin(120*deg_2_rad/2)
#lambda function handles modifying these element's geometry to see its affect on resonance
circumradii_hexa = lambda p: np.ones((6,))*(aperture_radius+compensated_lateral_growth)+ (p,0,0,p,0,0)
center_poly = Hexagon.from_circumradii(circumradii_hexa(0))

#visualize to confirm needed rotations
# left_poly.plot()
# right_poly.plot()
# center_poly.plot()

#convert into oreinted ArrayElements
left = ArrayElement(left_poly,center=(-psl_center_spacing,0),rotation_deg = 0)
right = ArrayElement(right_poly,center=(psl_center_spacing,0),rotation_deg = 180)
center = ArrayElement(center_poly,center=(0,0),rotation_deg=0)

#combine into a GeometryArray for aperture generation
ga = GeometryArray([left,center,right])

#create generator object and generate
vcsel = VCSELGenerator([ga], lateral_growth=psl_lateral_growth)
vcsel.generate_mesa()
vcsel.generate_aperture()
vcsel.plot(show_implant = False,show_contact_partitions=False)

#%% -------------------------------- Build emopt materials from geometries -------------------------------
dx = dy = 0.1
pad = 0.5
n_bg = 1
n_aperture = 3.4
approx_contrast = 0.0134
n_oxide = n_aperture - approx_contrast

eps, mu, domain = build_eps_from_generator(
    vcsel_generator = vcsel,
    dx=dx, dy=dy, pad=pad,
    eps_background=n_bg**2,
    eps_mesa=n_oxide**2,
    eps_aperture=n_aperture**2
)

# sanity check
arr = eps.get_values_in(domain)
bg_val = np.real(n_bg**2)
frac_nonbg = np.count_nonzero(np.abs(np.real(arr) - bg_val) > 1e-9) / arr.size
print(f"[raster check] fraction of cells ≠ background: {frac_nonbg:.6f}")
xb0,xb1,yb0,yb1, *_ = domain.get_bounding_box()
plt.figure()
plt.imshow(np.real(arr),origin='lower',extent=[xb0,xb1,yb0,yb1])
plt.title('epsilon map')
plt.show()

# Solve
# n_modes = 6
# modes = emopt.modes.ModeFullVector(
#     wavelength, eps, mu, domain,
#     n0=n_aperture,
#     neigs = n_modes
# )

# modes.bc = ['0','0']
# modes.build()
# modes.solve()


#%% Assess the default mode vectors across all supermodes
def get_FullVector(solver:emopt.modes.ModeFullVector ,n: int):
    """
    Returns a list of all E and H field Components from the solver.

    Params
    ------
    solver: emopt.modes.ModeFullVector
        A FullVector mode solver
    n: int
        The mode number of the desired E and H field profiles

    Returns
    -------
    Full Vector: list(fieldcomponents)
        In the ordering [Ezs,Exs,Eys,Hzs,Hxs,Hys]
    """
    return [solver.get_field_interp(i=n,component=comp) for comp in ['Ez','Ex','Ey','Hz','Hx','Hy']]

def cavity_complex_phasors(domain, full_vec, x_left=0.0, x_right=0.0, component='Ez'):
    """
    Return complex phasors (L, C, R) for a chosen component, using a power-weighted average
    within each cavity mask. Arrays are expected as (1, Ny, Nx). Masks are split by x.
    
    component: one of 'Ez','Ex','Ey' (you can extend to H* similarly).
    """
    Ez, Ex, Ey, Hz, Hx, Hy = full_vec

    # pick the complex field to represent cavity phase (default Ez)
    comp_map = {'Ez': Ez, 'Ex': Ex, 'Ey': Ey, 'Hz': Hz, 'Hx': Hx, 'Hy': Hy}
    F = np.asarray(comp_map[component])
    if F.ndim != 3 or F.shape[0] != 1:
        raise ValueError(f"{component} must have shape (1, Ny, Nx), got {F.shape}")
    F2D = F[0]  # (Ny, Nx)

    # build |E| for weights (use all E components)
    if Ez.shape[0] == 1 and Ex.shape[0] == 1 and Ey.shape[0] == 1:
        Emag2D = np.sqrt(np.abs(Ez[0])**2 + np.abs(Ex[0])**2 + np.abs(Ey[0])**2)
    else:
        # fallback: weight by |F| if others unavailable
        Emag2D = np.abs(F2D)

    # coordinate grid (Ny, Nx)
    xs, ys = domain.x, domain.y
    X, Y = np.meshgrid(xs, ys, indexing='xy')

    # masks
    Lmask = (X <  x_left)
    Cmask = (X >= x_left) & (X <= x_right)
    Rmask = (X >  x_right)

    def weighted_mean_complex(A, W, M):
        if not M.any():
            return 0.0 + 0.0j
        a = A[M]
        w = W[M]**2  # power weighting
        return (w * a).sum() / (w.sum() + 1e-30)

    L = weighted_mean_complex(F2D, Emag2D, Lmask)
    C = weighted_mean_complex(F2D, Emag2D, Cmask)
    R = weighted_mean_complex(F2D, Emag2D, Rmask)
    return np.array([L, C, R], dtype=complex)

#%% Calculate the in-tune Mode Vector Targets
K = np.array([[0.0, -1.0,  0.0],
              [-1.0, 0.0, -1.0],
              [0.0, -1.0,  0.0]],
                dtype=float
            )
evals, evecs = np.linalg.eigh(K)

#%% Compute the relative error between mode and target value
def best_fit_scale(v, m):
    """Return alpha*, residual norm, and relative error aligning target v to measured m (complex)."""
    v = v.astype(complex)
    m = m.astype(complex)

    denom = np.vdot(v, v) + 1e-30
    alpha = np.vdot(v, m) / denom
    resid = np.linalg.norm(m - alpha * v)
    rel = resid / (np.linalg.norm(m) + 1e-30)
    return alpha, resid, rel

# modevecs = []
# for n in range(n_modes):
#     full_vec = get_FullVector(modes,n)
#     mode_cavity_maximums = cavity_complex_phasors(domain,full_vec,
#                                                   x_left=-psl_center_spacing/2,x_right=psl_center_spacing/2,
#                                                   component='Ex')
#     normalized_mode_maxs = mode_cavity_maximums/np.max(np.abs(mode_cavity_maximums))
#     modevecs.append(normalized_mode_maxs)
#     print(f'mode {n}\n')
#     print(f'magnitude: {np.abs(mode_cavity_maximums)}')
#     print(f'norm mag: {np.abs(normalized_mode_maxs)}\n')
#     print(f'phase: {np.angle(mode_cavity_maximums)}')
#     print(f'norm phase: {np.angle(normalized_mode_maxs)}\n')
#     print(f'target: val={evals[int(n//2)]} vec={evecs[:,int(n//2)]/np.min(evecs[:,int(n//2)])}')
#     _,_,relative_error = best_fit_scale(evecs[:,int(n//2)],normalized_mode_maxs)
#     print(f'Relative_error: {relative_error}')
#     print('\n')

#%% -------------------------- Combine all the previous work to optimize p for minimum mode mismatch -----------------------------------------
def build_mode_targets():
    """
    Computes the target eigenvectors and eigenvalues of a three element in-tune coupled system. With equal coupling between elements.
    """
    K = np.array([[0.0, -1.0,  0.0],
              [-1.0, 0.0, -1.0],
              [0.0, -1.0,  0.0]],
                dtype=float
            )
    evals, evecs = np.linalg.eigh(K)
    targets = [evecs[:, k] for k in range(evecs.shape[1])] #converts column vectors into row vectors
    return evals, targets

# make geometry
def build_geometry(boundary_shift:float) -> GeometryArray:
    """
    Builds the three cavity GeometryArray with the center element modified by the boundary_shift.
    Uses GeometryArray to generate Mesa and Aperture of VCSEL Array.
    """
    inradii_penta = np.array((1,1,1,1,1))*(psl_lateral_growth + aperture_radius)
    left_poly  = Pentagon.from_inradii(inradii_penta)
    right_poly = Pentagon.from_inradii(inradii_penta)

    compensated_lateral_growth = psl_lateral_growth / sin(108*deg_2_rad/2)
    circumradii_hexa = lambda p_: np.ones((6,))*(aperture_radius + compensated_lateral_growth) + (p_,0,0,p_,0,0)
    center_poly = Hexagon.from_circumradii(circumradii_hexa(boundary_shift))

    left   = ArrayElement(left_poly,   center=(-psl_center_spacing, 0), rotation_deg=0)
    right  = ArrayElement(right_poly,  center=( psl_center_spacing, 0), rotation_deg=180)
    center = ArrayElement(center_poly, center=(0, 0), rotation_deg=0)
    ga = GeometryArray([left, center, right])

    vcsel = VCSELGenerator([ga], lateral_growth=psl_lateral_growth)
    vcsel.generate_mesa()
    vcsel.generate_aperture()
    return vcsel

def build_and_solve_emopt(generator:VCSELGenerator,n_modes):
    """
    Uses bridge between emopt and GeometryArrays + simulation settings to build emopt's needed objects:
    1. epsilon matrix 
    2. Mu matrix
    3. Domain Coordinates
    """
    eps, mu, domain = build_eps_from_generator(
        vcsel_generator=generator,
        dx=dx, dy=dy, pad=pad,
        eps_background=n_bg**2,
        eps_mesa=(n_oxide**2),
        eps_aperture=(n_aperture**2)
    )
    solver = emopt.modes.ModeFullVector(
        wavelength, eps, mu, domain,
        n0=n_aperture, neigs=n_modes
    )
    solver.bc = ['0','0']
    solver.build(); solver.solve()
    return solver, domain

def process_cavity_vectors(solver, domain, n_modes=6, component='Ex', xsplit=psl_center_spacing/2):
    """
    Return complex cavity vectors [L,C,R] for the number of modes using cavity_complex_phasors().
    """
    mvecs = []
    for i in range(n_modes):
        full_vec = get_FullVector(solver, i)  # uses your helper
        v = cavity_complex_phasors(domain, full_vec,
                                   x_left=-xsplit, x_right=+xsplit,
                                   component=component)
        # normalize by max magnitude (keeps phase relationships)
        v = v / np.max(np.abs(v))
        mvecs.append(v)
    return mvecs

def extract_global_phase(v, idx=1):
    v = v.astype(complex)
    if np.abs(v[idx]) > 0:
        ang = np.angle(v[idx])
        v = v * np.exp(-1j*ang)
    if v[idx].real < 0:
        v = -v
    return v

def compare_to_targets(target_vectors,measured_vectors):
    error_vector = []
    for i,measure in enumerate(measured_vectors):
        # print(f'target: {target_vectors[int(i//2)]}')
        # print(f'current: {measure}')
        # extract global phase on the center cavity
        t = extract_global_phase(target_vectors[int(i//2)], idx=1)
        m = extract_global_phase(measured_vectors[i],idx=1)
        _,_,relative_error = best_fit_scale(t,m)
        error_vector.append(relative_error)
    return error_vector

def compare_to_targets_best(measured_vectors, target_vectors, anchor_idx=1):
    errs = np.zeros((len(measured_vectors), len(target_vectors)))
    for i, m in enumerate(measured_vectors):
        mA = extract_global_phase(m, idx=anchor_idx)
        for k, t in enumerate(target_vectors):
            tA = extract_global_phase(t, idx=anchor_idx)
            _, _, errs[i, k] = best_fit_scale(tA, mA)
    # best target for each measured mode (ties are fine; polarizations can map to same target)
    best_k = errs.argmin(axis=1) #finds index of target vector with best error (helps with shifting of vector positions)
    best_e = errs[np.arange(len(measured_vectors)), best_k]
    return best_k, best_e, errs

def _aggregate_errors(best_e, kind="l2", weights=None):
    e = np.asarray(best_e, float)
    if weights is not None:
        w = np.asarray(weights, float)
        w = w / (np.sum(w) + 1e-30)
        e = e * w
    if kind == "l2":
        return float(np.sqrt(np.mean(e**2)))      # RMS of per-mode errors
    elif kind == "l1":
        return float(np.mean(e))                  # mean error
    elif kind == "max":
        return float(np.max(e))                   # worst-mode error
    else:
        raise ValueError(f"Unknown aggregation '{kind}'")

def objective(boundary_shift:float) -> float:
    """
    Handles all processes necessary for computing the relative error between target modes and solved modes for a given boundary shift.
    """

    _,target_vectors = build_mode_targets()
    generator = build_geometry(boundary_shift)
    solver,domain = build_and_solve_emopt(generator,n_modes=6)
    measured_vectors = process_cavity_vectors(solver,domain,n_modes=6)
    _, mode_error, _ = compare_to_targets_best(measured_vectors,target_vectors)
    l2_error = _aggregate_errors(mode_error,weights = [1,0,0,0,0,0])
    print(f'Paramters {boundary_shift}')
    print(f'L2 Mode Error {l2_error}')
    return l2_error

# %% Check that the builder matches the previous result
# print('Wrapper Sanity Check. Compare to Previous Output')
# l2_error = objective(0)

#%% Conduct a simple optimization across the parameter space defined
def golden_section_minimize(f, a, b, tol=1e-2, max_iter=64):
    """
    Minimize scalar function f(p) on [a,b] by golden-section search.
    Returns (p_opt, f_opt, history)
    """
    gr = (np.sqrt(5) - 1) / 2  # ~0.618
    c = b - gr*(b - a)
    d = a + gr*(b - a)

    fc = f(c)
    fd = f(d)

    hist = [(a, b, c, d, fc, fd)]
    k = 0
    while (b - a) > tol and k < max_iter:
        if fc > fd:
            a = c
            c = d
            fc = fd
            d = a + gr*(b - a)
            fd = f(d)
        else:
            b = d
            d = c
            fd = fc
            c = b - gr*(b - a)
            fc = f(c)
        hist.append([(a, b, c, d, fc, fd)])
        k += 1

    p_opt = c if fc <= fd else d
    f_opt = min(fc, fd)
    return p_opt, f_opt, hist

# p_optimal, f_optimal, history = golden_section_minimize(objective,0,5)

# print(f'optimal positive {p_optimal}')

p_optimal, f_optimal, history = golden_section_minimize(objective,0,1,max_iter=128)

print(f'optimal boundary shift: {p_optimal} ')

#visually inspect the optimal solution
generator = build_geometry(p_optimal)
solver,domain = build_and_solve_emopt(generator,n_modes=6)

extent = domain.get_bounding_box()
for n in range(6):
    Ez,Ex,Ey,Hz,Hx,Hy = get_FullVector(solver,n)
    fields = {"Ez": Ez[0], "Ex": Ex[0], "Ey": Ey[0], "Hz": Hz[0], "Hx": Hx[0], "Hy": Hy[0]}

    Emax = max(np.max(np.abs(fields[k])) for k in ("Ez", "Ex", "Ey")) or 1.0
    Hmax = max(np.max(np.abs(fields[k])) for k in ("Hz", "Hx", "Hy")) or 1.0
    vmax = {"Ez": Emax, "Ex": Emax, "Ey": Emax, "Hz": Hmax, "Hx": Hmax, "Hy": Hmax}
    vmin = 0.0

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True, sharex=True, sharey=True)
    for ax, key in zip(axes.flat, ("Ez", "Ex", "Ey", "Hz", "Hx", "Hy")):
        img = ax.imshow(np.abs(fields[key]), origin="lower", extent=extent,
                        vmin=vmin, vmax=vmax[key])
        ax.set_title(key)
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    plt.show()


