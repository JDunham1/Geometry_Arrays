# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 16:15:13 2025

@author: Jake Dunham

This script demonstrates the byproducts of the current isotropic lateral growth
simulation. It does so by showing the different approaches to defining the outer
geometry based on desired aperture features:
    1. Inradius
    2. Circumradius
    3. Vertex Positions
"""

import GeometryArrays as GA
import numpy as np
from numpy import sin
import matplotlib.pyplot as plt
plt.close('all')
import gdspy
gdspy.current_library = gdspy.GdsLibrary()
lib = gdspy.GdsLibrary()
pin_cell = lib.new_cell('Poorly Defined Inradius Quadrilateral')
win_cell = lib.new_cell('Well Defined Inradius Quadrilateral')
pcir_cell = lib.new_cell('Poorly Defined Circumradius Quadrilateral')
wcir_cell = lib.new_cell('Well Defined Circumradius Quadrilateral')

deg_2_rad = np.pi/180

growth = 1

#%% quadrilateral definition methods:
# say we have some desired maximum distance from aperture center to vertex
# in this case we would use the circumradii to and add the expected lateral growth 
# to get the full extent
max_dist = 1
# define via diagonals (analogous to circumradii)
diagonals = np.ones((4,))*(max_dist+growth)
quad = GA.Quadrilateral.from_diagonals(diagonals)
cavity = GA.ArrayElement(quad)
array = GA.GeometryArray([cavity])
generator = GA.VCSELGenerator([array],lateral_growth=growth)
generator.generate_aperture()
generator.plot(show_contact_partitions=False,show_implant=False)
for gds in generator.to_gdspy(include_aperture=True):
    pin_cell.add(gds)
# we note that the maximum distance we see here is not what we intended
# this is due to the isotropic growth handling corners differently.

# It is a function of the interior angle determining how much the growth from
# the surrounding edges impinges on the corner
# narrow corners will contribute more and wide corners will contribute less
corrected_corner_growth = growth / sin(quad.angles_deg*deg_2_rad/2)
diagonals_corrected = np.ones((4,))*(max_dist) + corrected_corner_growth
quad_compensated = GA.Quadrilateral.from_diagonals(diagonals_corrected)
cavity_compensated = GA.ArrayElement(quad_compensated)
array_compensated = GA.GeometryArray([cavity_compensated])
generator = GA.VCSELGenerator([array_compensated],lateral_growth = growth)
generator.generate_aperture()
generator.plot(show_contact_partitions=False,show_implant=False)
for gds in generator.to_gdspy(include_aperture=True):
    win_cell.add(gds)


# define via sidelengths
# now say we have a desired sidelength in the aperture we wish to create
# to define these we can utilize the inradii which define the minimum distance
# from the geometries center to the edge (using an 'in'scribed circle)
min_dist = 1
circumradii = np.ones((4,))*(min_dist+growth)
quad = GA.Quadrilateral.from_sidelengths(widths=circumradii[:2],heights=circumradii[2:])
cavity = GA.ArrayElement(quad)
array = GA.GeometryArray([cavity])
generator = GA.VCSELGenerator([array],lateral_growth=growth)
generator.generate_aperture()
generator.plot(show_contact_partitions=False,show_implant=False)
for gds in generator.to_gdspy(include_aperture=True):
    wcir_cell.add(gds)
    
# in this case we note that the growth does not need to be transformed as the 
# lateral_growth definition and the minimum distance definition are colinear


#%% Gds file write
# confirm these dimensions yourself in gds!
lib.write_gds('./gds_files/HOWTO_ApertureDefinedMesas.gds')