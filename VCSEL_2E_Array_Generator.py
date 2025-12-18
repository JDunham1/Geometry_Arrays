import numpy as np
import GeometryArrays as GA
import matplotlib.pyplot as plt
import gdspy 

contact_layer = 1
ion_layer = 2
mesa_layer = 3

#%% Shared Parameters between arrays
lateral_growth = 5 #um
edge_pitch = 110 #um
edge_gauge = 60 #um, eyeballed, feel free to change
array_contact_padding = 1
edge_align_padding = 10
aperture_radius = 3*np.sqrt(2)/2
mesa_element_separation_i = 7
mesa_element_separation_a = 8 
sweep_padding = 75 #padding between sweep sets
angled_element_angle = 22.5 * np.pi /180 # radians
implant_length = 3
implant_width = 50
implant_padding = 0

# contact pad dimensions
contact_pad_hole_radius = 5/2
contact_pad_radius = 15.5
pad_hole_distance = (contact_pad_radius+contact_pad_hole_radius)/2
pad_hole_angle = np.pi/4 - np.pi/5.9
bridge_length = 4
bridge_width = 7
bridge_overlap = 0.1 #for making sure the rectangle fully overlaps with the curvature of the contact pad
tab_width = 2
tab_length = 4

growth_perturbation = np.arange(-1,1.5,step=0.5)
#scale_factors = lateral_growth/(lateral_growth+growth_perturbation)
#visually determined due to non-uniform scaling of geometric extent. 
scale_factors = np.array([0.95,1.0,1.0,0.95,0.95])

#%%

def scaled_linspace(center, base_span, s, n, hard_min=None, hard_max=None, if_positive_scan_only=False):
    """
    base_span: half-span at ox=0 (i.e., values go [center - base_span, center + base_span])
    ox_max: maximum |ox| in your growth_perturbation schedule
    s: span scale factor. Modifies the extent of the span according to the oxide_perturbation
    hard_min/hard_max: optional global caps
    """
    span = base_span * s
    if if_positive_scan_only:
        arr = np.linspace(center, center + span, n)
    else:
        arr = np.linspace(center - span, center + span, n)
    if hard_min is not None or hard_max is not None:
        arr = np.clip(arr, hard_min if hard_min is not None else -np.inf,
                           hard_max if hard_max is not None else  np.inf)
    return arr

def create_paramsweep_array_mesas(n_swept,
                            growth_sweep,
                            rc_params,
                            pinch_params,
                            lateral_growth,
                            scale_factors,
                            start = (0,0)
                            ):
    # Paramter Unpacking
    rc_center, rc_base_span = rc_params
    pinch_center,pinch_base_span = pinch_params
    
    
    vcsel_arrays = []
    contacts = []
    contact_holes = []
    tabs = []
    
    for i, (ox_perturb,s) in enumerate(zip(growth_perturbation,scale_factors)):
        print(f'Generating Sweep with Oxide Offset {ox_perturb}, scale_factor {s}')
        # recompute inner arrays for THIS ox_perturb (lengths unchanged = n_swept)
        rc_Y_perturbation = scaled_linspace(rc_center,  rc_base_span,  s, n_swept,if_positive_scan_only=True)
        pinch_perturbation  = scaled_linspace(pinch_center, pinch_base_span, s, n_swept) # modify the "pinch" of the coupling side-length
        element_start = np.array((i*(len(pinch_perturbation)*edge_pitch+sweep_padding),0)) + start
        
        for j, pinch_perturbation in enumerate(pinch_perturbation):
            element_position = element_start + (j*edge_pitch,0)
            for k, rc_y in enumerate(rc_Y_perturbation):
                print((i,j,k))
                #generate the starting coordinates for each of the sweep sets
                element_position_iy = element_position + (0,k*edge_gauge)
                # element_position_ay = element_position_iy + (0,len(rc_Y_perturbation)*edge_gauge+sweep_padding)
                
                #generate the geometries for all sweep sets (angled, inline, x, and y right-cavity perturbation)
                inradii_penta = np.array((1,1,1,1,1))*(lateral_growth + aperture_radius + ox_perturb)
                circumradii_penta = inradii_penta / np.cos(np.pi / 5)
                # inradii_hexa = np.array((1,1,1,1,1,1))*(lateral_growth + aperture_radius + ox_perturb)
                # circumradii_hexa = inradii_hexa / np.cos(np.pi / 6)
                
                
                left_poly = GA.Pentagon.from_circumradii(circumradii_penta + [0,0,pinch_perturbation,pinch_perturbation,0])
                # left_poly = GA.Pentagon.from_circumradii([1,2,3,4,5])
                # middle_poly_i = GA.Hexagon.from_circumradii(circumradii_hexa + (cci, 0, 0, cci, 0, 0))
                # middle_poly_a = GA.Hexagon.from_inradii(inradii_hexa + (cca,0,0,0,0,0))
                right_poly_y = GA.Pentagon.from_circumradii(circumradii_penta + [0,rc_y,pinch_perturbation,pinch_perturbation,rc_y])
                

                #inline y perturbation
                left_center = element_position_iy + (-mesa_element_separation_i/2,0)
                left_mesa_iy = GA.ArrayElement(left_poly,
                                            center=left_center,
                                            rotation_deg=90)
                right_center = element_position_iy + (mesa_element_separation_i/2,0)
                right_mesa_iy = GA.ArrayElement(right_poly_y,
                                               center=right_center,
                                               rotation_deg=-90)
                
                contact_pad_geom = GA.Circle.from_radius(
                    r=contact_pad_radius)
    
                contact_pad_hole_geom = GA.Circle.from_radius(
                    r=contact_pad_hole_radius)
    
                pad_hole_quartercircle_geom = GA.Circle.quarter_circle(
                    radius = pad_hole_distance,
                    thickness = contact_pad_hole_radius,
                    offset = np.pi/4 - pad_hole_angle)

                pad_hole_tab_geom = GA.Quadrilateral.from_dimensions(
                    width = [tab_width/2,tab_width/2],
                    height = [tab_length/2, tab_length/2]
                )
                
                bridge_geom = GA.Quadrilateral.from_sidelengths(
                    widths=[
                        bridge_width/2,
                        bridge_width/2
                    ],
                    heights = [bridge_length/2+bridge_overlap, bridge_length/2+bridge_overlap]
                )

                left_pad_center = left_center - (circumradii_penta[0]+contact_pad_radius,0)
                right_pad_center = right_center + (circumradii_penta[0]+contact_pad_radius,0)
                bridge_left = GA.ArrayElement(bridge_geom,center=left_pad_center + (contact_pad_radius,0))
                bridge_right = GA.ArrayElement(bridge_geom,center=right_pad_center - (contact_pad_radius,0))

                pad_left = GA.ArrayElement(contact_pad_geom,center=left_pad_center)
                pad_right = GA.ArrayElement(contact_pad_geom,center=right_pad_center)

                tab_left = GA.ArrayElement(pad_hole_tab_geom,center=left_pad_center + \
                                           (pad_hole_distance*np.cos(3*np.pi/4)/2,pad_hole_distance*np.sin(3*np.pi/4)/2))
                tab_right = GA.ArrayElement(pad_hole_tab_geom,center=right_pad_center + \
                                           (pad_hole_distance*np.cos(np.pi/4)/2,pad_hole_distance*np.sin(np.pi/4)/2),
                                           rotation_deg = -90)

                holes = []
                holes.append(GA.ArrayElement(contact_pad_hole_geom,
                                 center=left_pad_center))
                holes.append(GA.ArrayElement(contact_pad_hole_geom,
                                 center=right_pad_center))
                
                 # contact pad quarter circle holes
                for rot in np.linspace(0,270,4):
                    holes.append(GA.ArrayElement(pad_hole_quartercircle_geom,
                                     center=left_pad_center,
                                     rotation_deg = rot))
                    holes.append(GA.ArrayElement(pad_hole_quartercircle_geom,
                                     center=right_pad_center,
                                     rotation_deg = rot))
                contact_holes.append(GA.GeometryArray(holes))
                
                #combine into geometryarray and add to generator list
                vcsel_array_iy = GA.GeometryArray((left_mesa_iy,right_mesa_iy))
                vcsel_arrays.append(vcsel_array_iy)        

                #combine contact pads into a geometryarray and add to generator contact list
                pads = GA.GeometryArray((pad_left,bridge_left,pad_right,bridge_right))
                contacts.append(pads)

                #combine pull tabes into a geometryarray and add to generator tab list
                pull_tabs = GA.GeometryArray([tab_left,tab_right])
                tabs.append(pull_tabs)
    
    return vcsel_arrays, contacts, contact_holes, tabs
                

gdspy.current_library = gdspy.GdsLibrary()
lib = gdspy.GdsLibrary()
array_element_unitcell = lib.new_cell(f"2 Element {int(implant_length)}in Arrays")
plt.close('all')


#%% Wide Range sweep (low resolution)
n_swept = 9
rc_center,  rc_base_span  = 0.0, 3.0
pinch_center, pinch_base_span = 0.0, 3

# report swept parameter resolutions, total element count, and estimated simulation time
right_resolution = rc_base_span*2 / n_swept
total_elements = n_swept**2*len(growth_perturbation)
elements_per_oxide = total_elements / len(growth_perturbation)
print(f"""
      Right Element Resolution:
          {right_resolution:.3f} um
      Total Elements:
          {total_elements}
    
      """)

vcsel_arrays, contacts, hole_arrays, tabs = create_paramsweep_array_mesas(n_swept, growth_perturbation,
                                                                          [rc_center, rc_base_span],
                                                                          [pinch_center, pinch_base_span],
                                                                          lateral_growth,
                                                                          scale_factors)

# Generate all difference Array combinations (separate ones at difference lateral growths for visual inspection)
indices_of_ox = len(vcsel_arrays) // len(growth_perturbation)
for i, ox_perturb in enumerate(growth_perturbation):
    print(f"\nGenerating Visualization of oxide offset {ox_perturb} array at {lateral_growth+ox_perturb} lateral growth:")
    visualization_generator = GA.VCSELGenerator(geometry_arrays = vcsel_arrays[i*indices_of_ox:(i+1)*indices_of_ox],
                                                contact_arrays = contacts[i*indices_of_ox:(i+1)*indices_of_ox],
                                                contact_hole_arrays = hole_arrays[i*indices_of_ox:(i+1)*indices_of_ox],
                                                pull_tabs = tabs[i*indices_of_ox:(i+1)*indices_of_ox],
                                                lateral_growth = lateral_growth + ox_perturb,
                                                contact_padding = array_contact_padding,
                                                implant_width = implant_width,
                                                implant_length = implant_length,
                                                min_contact_area = 5)
    visualization_generator.generate_all(inv_fit=False)
    # visualization_generator.plot(show_implant=False,
    #                              show_implant_inv = True,
    #                              show_contact_region = True)

    for gds in visualization_generator.to_gdspy(include_aperture=True,
                                                contact_layer=contact_layer,
                                                mesa_layer=mesa_layer,
                                                ion_layer=ion_layer):
        array_element_unitcell.add(gds)

#%% Medium Range Sweep (Averaged resolution)
n_swept = 9
rc_center,  rc_base_span  = 0.0, 1.75
pinch_center, pinch_base_span = 0.0, 2

# report swept parameter resolutions, total element count, and estimated simulation time
right_resolution = rc_base_span*2 / n_swept
total_elements = n_swept**2*len(growth_perturbation)
print(f"""
      Right Element Resolution:
          {right_resolution:.3f} um
      Total Elements:
          {total_elements}
            
      """)
_, _, _, prev_ymax = visualization_generator.bounding_box
mid_range_start = (0,prev_ymax+2*sweep_padding)

vcsel_arrays, contacts, hole_arrays, tabs = create_paramsweep_array_mesas(n_swept, growth_perturbation, 
                                                                            [rc_center, rc_base_span],
                                                                            [pinch_center, pinch_base_span],
                                                                            lateral_growth,
                                                                            scale_factors,
                                                                            start = mid_range_start)

# Generate all difference Array combinations (separate ones at difference lateral growths for visual inspection)
indices_of_ox = len(vcsel_arrays) // len(growth_perturbation)
for i, ox_perturb in enumerate(growth_perturbation):
    print(f"\nGenerating Visualization of oxide offset {ox_perturb} array at {lateral_growth+ox_perturb} lateral growth:")
    visualization_generator = GA.VCSELGenerator(geometry_arrays = vcsel_arrays[i*indices_of_ox:(i+1)*indices_of_ox],
                                                contact_arrays = contacts[i*indices_of_ox:(i+1)*indices_of_ox],
                                                contact_hole_arrays = hole_arrays[i*indices_of_ox:(i+1)*indices_of_ox],
                                                pull_tabs = tabs[i*indices_of_ox:(i+1)*indices_of_ox],
                                                lateral_growth = lateral_growth + ox_perturb,
                                                contact_padding = array_contact_padding,
                                                implant_width = implant_width,
                                                implant_length = implant_length,
                                                min_contact_area = 5)
    visualization_generator.generate_all(inv_fit=False)
    # visualization_generator.plot(show_implant=False,
    #                              show_implant_inv = True,
    #                              show_contact_region = False)

    for gds in visualization_generator.to_gdspy(include_aperture=True,
                                                contact_layer=contact_layer,
                                                mesa_layer=mesa_layer,
                                                ion_layer=ion_layer):
        array_element_unitcell.add(gds)

#%% Narrow Range Sweep (High Resolution)
n_swept = 9
rc_center,  rc_base_span  = 0.0, 0.5
pinch_center, pinch_base_span = 0.0, 1

# report swept parameter resolutions, total element count, and estimated simulation time
right_resolution = rc_base_span*2 / n_swept
total_elements = n_swept**2*len(growth_perturbation)
print(f"""
      Right Element Resolution:
          {right_resolution:.3f} um
      Total Elements:
          {total_elements}
            
      """)
_, _, _, prev_ymax = visualization_generator.bounding_box
low_range_start = (0,prev_ymax+2*sweep_padding)

vcsel_arrays, contacts, hole_arrays, tabs = create_paramsweep_array_mesas(n_swept, growth_perturbation, 
                                                                            [rc_center, rc_base_span],
                                                                            [pinch_center, pinch_base_span],
                                                                            lateral_growth,
                                                                            scale_factors,
                                                                            start = low_range_start)

# Generate all difference Array combinations (separate ones at difference lateral growths for visual inspection)
indices_of_ox = len(vcsel_arrays) // len(growth_perturbation)
for i, ox_perturb in enumerate(growth_perturbation):
    print(f"\nGenerating Visualization of oxide offset {ox_perturb} array at {lateral_growth+ox_perturb} lateral growth:")
    visualization_generator = GA.VCSELGenerator(geometry_arrays = vcsel_arrays[i*indices_of_ox:(i+1)*indices_of_ox],
                                                contact_arrays = contacts[i*indices_of_ox:(i+1)*indices_of_ox],
                                                hole_arrays = hole_arrays[i*indices_of_ox:(i+1)*indices_of_ox],
                                                pull_tabs = tabs[i*indices_of_ox:(i+1)*indices_of_ox],
                                                lateral_growth = lateral_growth + ox_perturb,
                                                contact_padding = array_contact_padding,
                                                implant_width = implant_width,
                                                implant_length = implant_length,
                                                implant_padding = implant_padding,
                                                min_contact_area = 5)
    visualization_generator.generate_all(inv_fit=False)
    # visualization_generator.plot(show_implant=False,
    #                              show_implant_inv = True,
    #                              show_contact_region = False)

    for gds in visualization_generator.to_gdspy(include_aperture=True,
                                                contact_layer=contact_layer,
                                                mesa_layer=mesa_layer,
                                                ion_layer=ion_layer):
        array_element_unitcell.add(gds)

#%% Outpute File
# lib.write_gds(f'./gds_files/bowtie_{implant_length}inImplant_2E_array_elements.gds')
lib.write_gds(f'../gdspy_helper/gds_files/bowtie_{implant_length}inImplant_2E_array_elements.gds')