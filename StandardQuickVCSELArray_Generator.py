import numpy as np
import GeometryArrays as GA
import matplotlib.pyplot as plt
import gdspy

contact_layer = 1
mesa_layer = 2

stnd_lateral_growth = 17
stnd_contact_padding = 2
stnd_contact_width = 10
stnd_tab_width = 5
stnd_tab_length = 10
stnd_pitch = 125
stnd_gauge = 125
leny = 4
lenx = 14
stnd_die_size = np.array((stnd_pitch*lenx,stnd_gauge*leny))
n = leny*lenx
min_mesa_size = 30

# standard_mesas = np.concat((first_row_mesas,second_row_mesas,third_row_mesas,fourth_row_mesas,final_mesa))
standard_mesas = np.array((36.5,36,35.5,35,34.5,34,33.5,33,32.5,32,31.5,31,30.5,30,
                           37,37.5,38,38.5,39,39.5,40,40.5,41,41.5,42,43.5,43,43.5,
                           57,56,55,54,53,52,51,50,49,48,47,46,45,44,
                           58,59,60,61,62,63,64,65,66,67,68,69,70,75))

#%% Generate mask cell structures
plt.close('all')

#quick vcsel unit cell coords
xs = np.linspace(-(lenx-1)/2*stnd_pitch,(lenx-1)/2*stnd_pitch,lenx)
ys = np.linspace(-(leny-1)/2*stnd_gauge,(leny-1)/2*stnd_gauge,leny)
X,Y = np.meshgrid(xs,ys)


stnd_quick_vcsels = []
pull_tabs = []
for coord, mesa in zip(np.column_stack([X.ravel(),Y.ravel()]),standard_mesas):
    Standard_quick_vcsel = GA.Quadrilateral.from_sidelengths(
            widths=[
                    mesa,
                    mesa
                ],
            height = mesa
        )
    cavity = GA.ArrayElement(Standard_quick_vcsel,center=coord)
    stnd_quick_vcsel = GA.GeometryArray([cavity])
    stnd_quick_vcsels.append(stnd_quick_vcsel)
    
    tab_geom = GA.Quadrilateral.from_sidelengths(
            widths=[
                    stnd_tab_width,
                    stnd_tab_width
                ],
            height = stnd_tab_length
        )
    tab = GA.ArrayElement(tab_geom,center=coord+(0,-mesa/2+stnd_contact_padding+10/2))
    pull_tab = GA.GeometryArray([tab])
    pull_tabs.append(pull_tab)
    
    
    
stnd_vcsel_generator = GA.VCSELGenerator(geometry_arrays = stnd_quick_vcsels,
                                         pull_tabs = pull_tabs,
                                        lateral_growth = stnd_lateral_growth,
                                        contact_padding = stnd_contact_padding,
                                        min_contact_area=15,
                                        contact_width=stnd_contact_width)
stnd_vcsel_generator.generate_all()
stnd_vcsel_generator.plot(
                          show_implant=False,
                          show_contact_region=False,
                          )

#%% Output generated structures as GDS files to be used in gds_helper
gdspy.current_library = gdspy.GdsLibrary()
lib = gdspy.GdsLibrary()

stnd_lateral_growth_unitcell = lib.new_cell("Standard Lateral Growth Elements")
for gds in stnd_vcsel_generator.to_gdspy():
    stnd_lateral_growth_unitcell.add(gds)

lib.write_gds('./gds_files/standard_quick_vcsel_elements.gds')