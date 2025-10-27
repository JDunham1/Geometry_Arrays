from GeometryArrays import (
    Pentagon, Hexagon, ArrayElement, GeometryArray, VCSELGenerator
)
import numpy as np
deg_2_rad = np.pi/180
from numpy import sin, cos
import gdspy
gdspy.current_library = gdspy.GdsLibrary()
lib = gdspy.GdsLibrary()
cell = lib.new_cell('Center Element Tuning')

wavelength = 0.85
psl_lateral_growth = 5
aperture_radius = 3*np.sqrt(2)/2
psl_implant_width = 30
psl_implant_length = 2

aperture_circumradius = aperture_radius / cos(np.pi/5)
psl_center_spacing = 12 - aperture_circumradius

inradii_penta = np.array((1,1,1,1,1))*(psl_lateral_growth + aperture_radius)
left_poly = Pentagon.from_inradii(inradii_penta)
right_poly = Pentagon.from_inradii(inradii_penta)
#center element defined via circumradii (as sidelengths are not being directly modified to control detuning)
inradii_hexa = np.ones((6,))*(aperture_radius+psl_lateral_growth)
compensated_lateral_growth = psl_lateral_growth / sin(120*deg_2_rad/2)
#lambda function handles modifying these element's geometry to see its affect on resonance
circumradii_hexa = lambda p: np.ones((6,))*(aperture_radius+compensated_lateral_growth)+ (p,0,0,p,0,0)

tuning_boundary_shift = -0.435

center_poly = Hexagon.from_circumradii(circumradii_hexa(-0.435))

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

for gds in vcsel.to_gdspy(include_aperture=True):
    cell.add(gds)
lib.write_gds('./gds_files/Tuned_inline_3E.gds')

vcsel.write_dxf('./dxf_files/Tuned_inline_3E.gds',include_aperture=True)