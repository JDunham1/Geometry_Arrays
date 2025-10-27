# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 11:11:20 2025

@author: Jake Dunham

Generates geometries for COMSOL simulation of coupling strength. Script accomplishes the following:
    1. Compares Dalir's geometry to mine
    2. Tests perturbed boundary for nominal case
        i. in-line
        ii. angled
"""

import GeometryArrays as GA
import numpy as np
import gdspy
import matplotlib.pyplot as plt
plt.close('all')
gdspy.current_library = gdspy.GdsLibrary()
lib = gdspy.GdsLibrary()
dalir_cell = lib.new_cell('Dalir_array')
psl_cell = lib.new_cell('PSL_2E_array')

#%% create Dalir's Array (with PSL contacts)
dalir_lateral_growth = 6
dalir_aperture_size = 3
dalir_implant_width = 30
dalir_implant_length = 2
center_spacing = 12 - (dalir_aperture_size)*np.sqrt(2)
# diagonals = np.ones((4,))*(dalir_lateral_growth+dalir_aperture_size*np.sqrt(2)*1.2)
# dalir_geom = GA.Quadrilateral.from_diagonals(diagonals)
dalir_inradii = np.ones((4,))*(dalir_aperture_size/2+dalir_lateral_growth)
dalir_geom = GA.Quadrilateral.from_sidelengths(widths=dalir_inradii[:2],
                                               heights=dalir_inradii[2:])
# dalir_geom.plot()

dalir_cavity1 = GA.ArrayElement(dalir_geom,center=(-center_spacing/2,0),rotation_deg=45)
dalir_cavity2 = GA.ArrayElement(dalir_geom,center=(center_spacing/2,0),rotation_deg=45)

dalir_array = GA.GeometryArray(elements=[dalir_cavity1,dalir_cavity2])
# dalir_array.plot()

dalir_generator = GA.VCSELGenerator(geometry_arrays = [dalir_array],
                                    lateral_growth = dalir_lateral_growth,
                                    implant_width = dalir_implant_width,
                                    implant_length = dalir_implant_length)
dalir_generator.generate_all()
# dalir_generator.plot()

for gds in dalir_generator.to_gdspy(include_aperture=True):
    dalir_cell.add(gds) 
lib.write_gds('./gds_files/dalir_array.gds',cells=[dalir_cell])

dalir_generator.write_dxf('./dxf_files/dalir_array.dxf',include_aperture=True)

#%% create PSL Array
psl_lateral_growth = 5
aperture_radius = 3*np.sqrt(2)/2
psl_implant_width = 30
psl_implant_length = 2

aperture_circumradius = aperture_radius / np.cos(np.pi/5)
psl_center_spacing = 12 - 2*aperture_circumradius

inradii_penta = np.array((1,1,1,1,1))*(psl_lateral_growth + aperture_radius)
left_poly = GA.Pentagon.from_inradii(inradii_penta)
right_poly = GA.Pentagon.from_inradii(inradii_penta)
left_poly.plot()
right_poly.plot()

psl_cavity1 = GA.ArrayElement(left_poly,
                            center=(-psl_center_spacing/2,0),
                            rotation_deg=0)
psl_cavity2 = GA.ArrayElement(right_poly,
                             center=(psl_center_spacing/2,0),
                             rotation_deg=180)
# psl_cavity1.plot()
# psl_cavity2.plot()

psl_array = GA.GeometryArray(elements = [psl_cavity1,psl_cavity2])
psl_array.plot()

psl_generator = GA.VCSELGenerator(geometry_arrays=[psl_array],
                                  lateral_growth = psl_lateral_growth,
                                  implant_width = psl_implant_width,
                                  implant_length = psl_implant_length)
psl_generator.generate_all()
psl_generator.plot()

for gds in psl_generator.to_gdspy(include_aperture=True):
    psl_cell.add(gds)
lib.write_gds('./gds_files/psl_2E_array.gds',cells=[psl_cell])

psl_generator.write_dxf('./dxf_files/psl_2E_array.dxf',include_aperture=True)

