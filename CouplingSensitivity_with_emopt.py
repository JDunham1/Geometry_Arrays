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
def build_geometry(boundary_shift: float, visualize:bool = False):
    # Build 3 Element in-line array
    psl_lateral_growth = 5
    aperture_radius = 3*np.sqrt(2)/2

    aperture_circumradius = aperture_radius / np.cos(np.pi/5)
    psl_center_spacing = 12 - 2*aperture_circumradius

    inradii_penta = np.array((1,1,1,1,1))*(psl_lateral_growth + aperture_radius)
    compensated_lateral_growth = psl_lateral_growth / sin(108*deg_2_rad/2)
    circumradii_penta = np.ones((5,))*(aperture_radius+compensated_lateral_growth) + (0,0,boundary_shift,boundary_shift,0)
    left_poly = Pentagon.from_circumradii(circumradii_penta)
    right_poly = Pentagon.from_circumradii(circumradii_penta)

    # if visualize:
    # #visualize to confirm correctly selected circumradii
    #     left_poly.plot()
    #     right_poly.plot()

    #convert into oreinted ArrayElements
    left = ArrayElement(left_poly,center=(-psl_center_spacing/2,0),rotation_deg = 90)
    right = ArrayElement(right_poly,center=(psl_center_spacing/2,0),rotation_deg = -90)

    #combine into a GeometryArray for aperture generation
    ga = GeometryArray([left,right])

    #create generator object and generate
    vcsel = VCSELGenerator([ga], lateral_growth=psl_lateral_growth)
    vcsel.generate_mesa()
    vcsel.generate_aperture()
    if visualize:
        vcsel.plot(show_implant = False,show_contact_partitions=False)
    return vcsel

# build_geometry(0,visualize=True)
# build_geometry(-1,visualize=True)
# build_geometry(1,visualize=True)

#%% -------------------------------- Build emopt materials from geometries -------------------------------
dx = dy = 0.1
pad = 0.5
n_bg = 1
n_aperture = 3.4
approx_contrast = 0.0134
n_oxide = n_aperture - approx_contrast
wavelength_global = 0.85

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
        wavelength_global, eps, mu, domain,
        n0=n_aperture, neigs=n_modes
    )
    solver.bc = ['0','0']
    solver.build(); solver.solve()
    return solver, domain

#%% Get Approximate frequency splitting from effective mode indices
def  Approximate_frequency_splitting(solver):
    f0_GHz = 3e8 / (wavelength_global * 1e-6) * 1e-9
    navg = 0.5*(solver.neff[0]+solver.neff[2])
    dneff = abs(solver.neff[2]-solver.neff[0])
    df = f0_GHz * (dneff / max(navg,1e-12))
    return df

def Coupling_Strength(boundary_shift:float,visualize:bool=False) -> float:
    """
    Handles all processes necessary for computing the relative error between target modes and solved modes for a given boundary shift.
    """

    generator = build_geometry(boundary_shift,visualize)
    solver,_ = build_and_solve_emopt(generator,n_modes=4)
    Coupling = Approximate_frequency_splitting(solver)
    return Coupling

p_vals = np.linspace(-2,12,31)
cs_vals = []

for i, p in enumerate(p_vals):
    if i == 0 or i == int(len(p_vals)//2) or i == len(p_vals):
        visualize = True
    else:
        visualize = False
    cs = Coupling_Strength(p,visualize)
    cs_vals.append(cs)

plt.figure()
plt.plot(p_vals, cs_vals, marker = 'o')
plt.xlabel("boundary_shift p (um)")
plt.ylabel("coupling strength (GHz)")
plt.title('Sensitivity of Coupling to Boundary Shifts')
plt.grid(True)
plt.tight_layout()
plt.show()