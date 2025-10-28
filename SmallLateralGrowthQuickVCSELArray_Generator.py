import numpy as np
import GeometryArrays as GA
import matplotlib.pyplot as plt
import gdspy 

contact_layer = 1
mesa_layer = 3

#%% Smole Quick VCSEL Parameters
sml_lateral_growth = 5
sml_contact_padding = 1
contact_pad_hole_radius = 1
contact_pad_radius = 15
bridge_length = 7
bridge_width = 3
sml_tab_width = 2
sml_tab_length = 4

sml_pitch = 99 # divisible by 9 to match with the edge shaped array pitch
sml_gauge = 125
leny = 4
lenx = 14
sml_die_size = np.array((sml_pitch*lenx,sml_gauge*leny))
n = leny*lenx
min_mesa_size = sml_lateral_growth*2 + 0.5

step_size_1 = 0.5 
first_half_mesas = min_mesa_size + step_size_1*np.arange(n//2 + 1)

step_size_2 = 1
second_half_mesas = first_half_mesas[-1] + step_size_2*np.arange(1,n//2 - 1)

step_size_final = 5
final_mesa = second_half_mesas[-1] + step_size_final*np.arange(1,2)

sml_mesas = np.concatenate((first_half_mesas,second_half_mesas,final_mesa))

#%% small lateral growth vcsels Generation
xs = np.linspace(-(lenx-1)/2*sml_pitch,(lenx-1)/2*sml_pitch,lenx)
ys = np.linspace(-(leny-1)/2*sml_gauge,(leny-1)/2*sml_gauge,leny)
X,Y = np.meshgrid(xs,ys)

vcsels = []
pull_tabs = []
vcsel_holes = []
for coord, mesa in zip(np.column_stack([X.ravel(),Y.ravel()]),sml_mesas):
    # print(coord)
    # print(mesa)
    #create basic geometries
    sml_quick_vcsel = GA.Quadrilateral.from_sidelengths(
            widths=[
                    mesa/2,
                    mesa/2
                ],
            heights = [mesa/2, mesa/2]
        )
    bridge_geom = GA.Quadrilateral.from_sidelengths(
            widths=[
                    bridge_width/2,
                    bridge_width/2
                ],
            heights = [bridge_length/2, bridge_length/2]
        )
    
    tab_geom = GA.Quadrilateral.from_sidelengths(
            widths=[
                    sml_tab_width/2,
                    sml_tab_width/2
                ],
            heights = [sml_tab_length/2, sml_tab_length/2]
        )
    
    # contact_pad_geom = GA.Quadrilateral.from_sidelengths(
    #         widths=[
    #                 sml_lateral_growth*2,
    #                 sml_lateral_growth*2
    #             ],
    #         height = sml_lateral_growth*2
    #     )
    
    contact_pad_geom = GA.Circle.from_radius(
            r=contact_pad_radius
        )
    
    contact_pad_hole_geom = GA.Circle.from_radius(
            r=contact_pad_hole_radius
        )
    
    pad_hole_quartercircle_geom = GA.Circle.quarter_circle(
            radius = sml_lateral_growth*2-2,
            thickness = contact_pad_hole_radius,
            offset = np.pi/8
        )
    
    #assign centers
    cavity = GA.ArrayElement(sml_quick_vcsel,center=coord)
    bridge = GA.ArrayElement(bridge_geom,center=coord+(0,mesa/2+bridge_length/2))
    contact_pad = GA.ArrayElement(contact_pad_geom,center=coord+(0,mesa/2+contact_pad_radius+bridge_length-0.1))
    
    holes = []
    # contact pad center hole
    holes.append(GA.ArrayElement(contact_pad_hole_geom,
                                 center=coord+(0,+ mesa/2 + contact_pad_radius + bridge_length-0.1)))
    # contact pad quarter circle holes
    for rot in np.linspace(0,270,4):
        holes.append(GA.ArrayElement(pad_hole_quartercircle_geom,
                                     center=coord+(0,+ mesa/2 + contact_pad_radius + bridge_length-0.1),
                                     rotation_deg = rot))
    
    

        
    tab = GA.ArrayElement(tab_geom,center=coord+(0,(sml_lateral_growth-mesa)/2))
    
    #combine into a single small_lateral_growth VCSEL
    vcsel = GA.GeometryArray([cavity,bridge,contact_pad])
    vcsel_hole = GA.GeometryArray(holes)
    pull_tab = GA.GeometryArray([tab])
    
    #optional centering around coord
    # translation = vcsel.zero_array(new_center=coord)
    # vcsel_hole.translate(*translation)
    # pull_tab.translate(*translation)
    
    vcsels.append(vcsel)
    vcsel_holes.append(vcsel_hole)
    pull_tabs.append(pull_tab)

sml_vcsel_generator = GA.VCSELGenerator(geometry_arrays = vcsels,
                                        hole_arrays = vcsel_holes,
                                        pull_tabs = pull_tabs,
                                        lateral_growth = sml_lateral_growth,
                                        hole_lateral_growth = sml_lateral_growth,
                                        contact_padding = sml_contact_padding,
                                        min_contact_area=0,
                                        )
sml_vcsel_generator.generate_all()
sml_vcsel_generator.plot(show_implant=False,
                         show_contact_region=False,
                        )

#%% Output generated structures as GDS files to be used in gds_helper
gdspy.current_library = gdspy.GdsLibrary()
lib = gdspy.GdsLibrary()

sml_lateral_growth_unitcell = lib.new_cell("Small Lateral Growth Elements")
for gds in sml_vcsel_generator.to_gdspy(contact_layer=contact_layer,
                                        mesa_layer=mesa_layer):
    sml_lateral_growth_unitcell.add(gds)

lib.write_gds('./gds_files/small_quick_vcsel_elements.gds')