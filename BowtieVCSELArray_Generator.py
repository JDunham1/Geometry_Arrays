import numpy as np
import GeometryArrays as GA
import gdspy 

contact_layer = 1
mesa_layer = 2
ion_layer = 3

#%% Edge Shaped Array VCSEL Parameters
edge_lateral_growth = 5 #um
edge_pitch = 77*2 #um
edge_gauge = 77 #um, eyeballed, feel free to change
array_contact_padding = 1
array_text_offset = 38
edge_align_padding = 10

aperture_radius = 3*np.sqrt(2)/2
mesa_element_separation_i = 7 #eyeballed, will adjust
mesa_element_separation_a = 8 
sweep_padding = 150 #padding between sweep sets
angled_element_angle = 22.5 * np.pi /180 # radians
implant_length = 2
implant_width = 30

n_swept = 5

# variations in the edge parameters to guarantee operational arrays
# rc_Y_perturbation = np.array([-0.15,-0.1,-0.05,0,0.05,0.1,0.15]) #right-cavity upward
rc_Y_perturbation = np.array([-0.3,-0.2,-0.1,0,0.1,0.2,0.3]) #new right-cavity upward, to account for new mesa structures
rc_Y_perturbation = np.linspace(-3,3,n_swept)
# rc_X_perturbation = np.array([-0.3,-0.2,-0.1,0,0.1,0.2,0.3]) #right-cavity outward
rc_X_perturbation = np.array([-0.6,-0.4,-0.2,0,0.2,0.4,0.6]) #new right-cavity outward, to account for modifying mesa structure
rc_X_perturbation = np.linspace(-3,3,n_swept)
# cca_perturbation = np.array([-0.6,-0.7,-0.8,-0.85,-0.898,-0.95,-1,-1.1,-1.2]) #center-cavity angled
cca_perturbation = np.array([-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]) #new angled perturbations for center cavity, basesd on comsol testing.
cca_perturbation = np.linspace(3,-3,n_swept)
# cci_perturbation = np.array([-0.1,-0.2,-0.3,-0.4,-0.435,-0.47,-0.5,-0.6,-0.7]) #center-cavity in-line
cci_perturbation = np.array([0.2,0.1,0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6]) #new in-line perturbation for centery cavity, based on comsol testing
cci_perturbation = np.linspace(3,-3,n_swept)
growth_perturbation = np.arange(-1,1.5,step=0.5)
# growth_perturbation = np.linspace(3,-3,25)



# mesa defined array vcsels
vcsel_arrays = []
nominal_angled_array = []
nominal_inline_array = []
for i, ox_perturb in enumerate(growth_perturbation):
    element_start = np.array((i*(len(cci_perturbation)*edge_pitch+sweep_padding),0))
    for j, (cci, cca) in enumerate(zip(cci_perturbation,cca_perturbation)):
        element_position = element_start + (j*edge_pitch,0)
        for k, (rc_x, rc_y) in enumerate(zip(rc_X_perturbation,rc_Y_perturbation)):
            # print((i,j,k))
            #generate the starting coordinates for each of the sweep sets
            element_position_ix = element_position + (0,k*edge_gauge)
            element_position_iy = element_position_ix + (0,len(rc_X_perturbation)*edge_gauge+sweep_padding)
            element_position_ax = element_position_iy + (0,len(rc_Y_perturbation)*edge_gauge+sweep_padding)
            element_position_ay = element_position_ax + (0,len(rc_X_perturbation)*edge_gauge+sweep_padding)
            
            #generate the geometries for all sweep sets (angled, inline, x, and y right-cavity perturbation)
            inradii_penta = np.array((1,1,1,1,1))*(edge_lateral_growth + aperture_radius + ox_perturb)
            circumradii_penta = inradii_penta / np.cos(np.pi / 5)
            inradii_hexa = np.array((1,1,1,1,1,1))*(edge_lateral_growth + aperture_radius + ox_perturb)
            circumradii_hexa = inradii_hexa / np.cos(np.pi / 6)
            
            left_poly = GA.Pentagon.from_inradius(inradii_penta[0])
            middle_poly_i = GA.Hexagon.from_circumradii(circumradii_hexa + (cci-ox_perturb, 0, 0, cci-ox_perturb, 0, 0))
            middle_poly_a = GA.Hexagon.from_inradii(inradii_hexa + (cca-ox_perturb,0,0,0,0,0))
            right_poly_x = GA.Pentagon.from_circumradii(circumradii_penta + (rc_x-ox_perturb, 0, 0, 0, 0))
            right_poly_y = GA.Pentagon.from_circumradii(circumradii_penta + (0,rc_y-ox_perturb,0,0,rc_y-ox_perturb))
            
            #convert geometry to array elements (rotate and translate)
            #inline x perturbation
            # left_mesa_ix = GA.ArrayElement(left_poly,
            #                             center=element_position_ix+(-mesa_element_separation_i,0),
            #                             rotation_deg=90)
            # middle_mesa_ix = GA.ArrayElement(middle_poly_i,
            #                               center=element_position_ix+(0,0),
            #                               rotation_deg=0)
            # right_mesa_ix = GA.ArrayElement(right_poly_x,
            #                              center=element_position_ix+(mesa_element_separation_i,0),
            #                              rotation_deg=-90)
            #inline y perturbation
            left_mesa_iy = GA.ArrayElement(left_poly,
                                        center=element_position_iy+(-mesa_element_separation_i,0),
                                        rotation_deg=90)
            middle_mesa_iy = GA.ArrayElement(middle_poly_i,
                                          center=element_position_iy+(0,0),
                                          rotation_deg=0)
            right_mesa_iy = GA.ArrayElement(right_poly_y,
                                           center=element_position_iy+(mesa_element_separation_i,0),
                                           rotation_deg=-90)
            #angled x (parallel) perturbation
            # left_mesa_ax = GA.ArrayElement(left_poly,
            #                                center=element_position_ax-mesa_element_separation_a*np.array((np.cos(angled_element_angle),
            #                                                                                    np.sin(angled_element_angle))
            #                                                                                    ),
            #                                rotation_deg=90+22.5
            #                                )
            # middle_mesa_ax = GA.ArrayElement(middle_poly_a,
            #                                  center=element_position_ax+(0,0),
            #                                  rotation_deg=90
            #                                  )
            # right_mesa_ax = GA.ArrayElement(right_poly_x,
            #                                 center=element_position_ax+mesa_element_separation_a*np.array((np.cos(-angled_element_angle),
            #                                                                                     np.sin(-angled_element_angle))
            #                                                                                     ),
            #                                 rotation_deg = -90-22.5
            #                                 )
            
            #angled y (perpendicular) perturbation
            left_mesa_ay = GA.ArrayElement(left_poly,
                                           center=element_position_ay-mesa_element_separation_a*np.array((np.cos(angled_element_angle),
                                                                                               np.sin(angled_element_angle))
                                                                                               ),
                                           rotation_deg=90+22.5
                                           )
            middle_mesa_ay = GA.ArrayElement(middle_poly_a,
                                             center=element_position_ay+(0,0),
                                             rotation_deg=90
                                             )
            right_mesa_ay = GA.ArrayElement(right_poly_y,
                                            center=element_position_ay+mesa_element_separation_a*np.array((np.cos(-angled_element_angle),
                                                                                                np.sin(-angled_element_angle))
                                                                                                ),
                                            rotation_deg = -90-22.5
                                            )
            
            
            #combine into geometryarray and add to generator list
            # vcsel_array_ix = GA.GeometryArray((left_mesa_ix,middle_mesa_ix,right_mesa_ix))
            # vcsel_arrays.append(vcsel_array_ix)
            vcsel_array_iy = GA.GeometryArray((left_mesa_iy,middle_mesa_iy,right_mesa_iy))
            vcsel_arrays.append(vcsel_array_iy)
            # vcsel_array_ax = GA.GeometryArray((left_mesa_ax,middle_mesa_ax,right_mesa_ax))
            # vcsel_arrays.append(vcsel_array_ax)
            vcsel_array_ay = GA.GeometryArray((left_mesa_ay,middle_mesa_ay,right_mesa_ay))
            vcsel_arrays.append(vcsel_array_ay)
            
            #plotting for troubleshooting
            # if (i == 2 and j == 0 and k == 0) or (i == 2 and j == 4 and k == 3):
            #     vcsel_array_ax.plot()
            #     # vcsel_array.plot()

            #grabbing arrays for em simulation
            # if i == 2 and j == 4 and k == 6:
                # nominal_angled_array.append(vcsel_array_ay)
                # nominal_inline_array.append(vcsel_array_iy)
                
vcsel_array_generator = GA.VCSELGenerator(geometry_arrays = vcsel_arrays,
                                          lateral_growth = edge_lateral_growth,
                                          contact_padding = array_contact_padding,
                                          implant_width = implant_width,
                                          implant_length = implant_length,
                                          min_contact_area = 8)
vcsel_array_generator.generate_all()
vcsel_array_generator.plot(show_implant=False,
                           show_implant_inv = True,
                           show_contact_region=False,)

#%% Output generated structures as GDS files to be used in gds_helper
gdspy.current_library = gdspy.GdsLibrary()
lib = gdspy.GdsLibrary()

array_element_unitcell = lib.new_cell("Bowtie Array Elements")
for gds in vcsel_array_generator.to_gdspy():
    array_element_unitcell.add(gds)

lib.write_gds('./gds_files/bowtie_array_elements.gds')