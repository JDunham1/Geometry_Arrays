import numpy as np
import GeometryArrays as GA
import matplotlib.pyplot as plt
import gdspy 

contact_layer = 1
mesa_layer = 2
ion_layer = 3

def scaled_linspace(center, base_span, s, n, hard_min=None, hard_max=None):
    """
    base_span: half-span at ox=0 (i.e., values go [center - base_span, center + base_span])
    ox_max: maximum |ox| in your growth_perturbation schedule
    s: span scale factor. Modifies the extent of the span according to the oxide_perturbation
    hard_min/hard_max: optional global caps
    """
    span = base_span * s
    arr = np.linspace(center - span, center + span, n)
    if hard_min is not None or hard_max is not None:
        arr = np.clip(arr, hard_min if hard_min is not None else -np.inf,
                           hard_max if hard_max is not None else  np.inf)
    return arr

plt.close('all')
#%% Edge Shaped Array VCSEL Parameters
lateral_growth = 5 #um
edge_pitch = 77*2 #um
edge_gauge = 77 #um, eyeballed, feel free to change
array_contact_padding = 1
array_text_offset = 38
edge_align_padding = 10

aperture_radius = 3*np.sqrt(2)/2
mesa_element_separation_i = 7 #eyeballed, will adjust
mesa_element_separation_a = 8 
sweep_padding = 75 #padding between sweep sets
angled_element_angle = 22.5 * np.pi /180 # radians
implant_length = 2
implant_width = 30

n_swept = 5

growth_perturbation = np.arange(-1,1.5,step=0.5)

#scale_factors = lateral_growth/(lateral_growth+growth_perturbation)
#visually determined due to non-uniform scaling of geometric extent. 
scale_factors = np.array([0.95,1.0,1.0,0.95,0.95])

# Determine the largest |ox| you will see so scaling behaves predictably
ox_max = np.max(np.abs(growth_perturbation))

# Choose centers and base half-spans from your current choices (edit as you learn bounds)
# (Using your current endpoints to infer reasonable defaults)
# rc_X / rc_Y: current fixed linspace(-3, 3, n_swept)  -> center=0,  base_span=3
# cci:          linspace( 2.5, -2.5, n_swept)         -> center=0,  base_span=2.5  (order doesn't matter; we rebuild)
# cca:          linspace( 3.0, -2.6, n_swept)         -> center≈0.2, base_span≈(3.0-(-2.6))/2 = 2.8
rc_center,  rc_base_span  = 0.0, 3.0
cci_center, cci_base_span = 0.0, 2.6
cca_center, cca_base_span = 0.0, 2.6

vcsel_arrays = []
nominal_angled_array = []
nominal_inline_array = []

for i, (ox_perturb,s) in enumerate(zip(growth_perturbation,scale_factors)):
    print(f'Generating Sweep with Oxide Offset {ox_perturb}, scale_factor {s}')
    # recompute inner arrays for THIS ox_perturb (lengths unchanged = n_swept)
    rc_Y_perturbation = scaled_linspace(rc_center,  rc_base_span,  s, n_swept)
    cci_perturbation  = scaled_linspace(cci_center, cci_base_span, s, n_swept)
    cca_perturbation  = scaled_linspace(cca_center, cca_base_span, s, n_swept)
    element_start = np.array((i*(len(cci_perturbation)*edge_pitch+sweep_padding),0))
    
    for j, (cci, cca) in enumerate(zip(cci_perturbation,cca_perturbation)):
        element_position = element_start + (j*edge_pitch,0)
        for k, rc_y in enumerate(rc_Y_perturbation):
            # print((i,j,k))
            #generate the starting coordinates for each of the sweep sets
            element_position_iy = element_position + (0,k*edge_gauge)
            element_position_ay = element_position_iy + (0,len(rc_Y_perturbation)*edge_gauge+sweep_padding)
            
            #generate the geometries for all sweep sets (angled, inline, x, and y right-cavity perturbation)
            inradii_penta = np.array((1,1,1,1,1))*(lateral_growth + aperture_radius + ox_perturb)
            circumradii_penta = inradii_penta / np.cos(np.pi / 5)
            inradii_hexa = np.array((1,1,1,1,1,1))*(lateral_growth + aperture_radius + ox_perturb)
            circumradii_hexa = inradii_hexa / np.cos(np.pi / 6)
            
            left_poly = GA.Pentagon.from_inradius(inradii_penta[0])
            middle_poly_i = GA.Hexagon.from_circumradii(circumradii_hexa + (cci, 0, 0, cci, 0, 0))
            middle_poly_a = GA.Hexagon.from_inradii(inradii_hexa + (cca,0,0,0,0,0))
            right_poly_y = GA.Pentagon.from_circumradii(circumradii_penta + (0,rc_y,0,0,rc_y))
            

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
            vcsel_array_iy = GA.GeometryArray((left_mesa_iy,middle_mesa_iy,right_mesa_iy))
            vcsel_arrays.append(vcsel_array_iy)
            vcsel_array_ay = GA.GeometryArray((left_mesa_ay,middle_mesa_ay,right_mesa_ay))
            vcsel_arrays.append(vcsel_array_ay)
            

#%% Output generated structures as GDS files to be used in gds_helper
gdspy.current_library = gdspy.GdsLibrary()
lib = gdspy.GdsLibrary()

array_element_unitcell = lib.new_cell("Bowtie Array Elements")

# Generate all difference Array combinations (separate ones at difference lateral growths for visual inspection)
indices_of_ox = len(vcsel_arrays) // len(growth_perturbation)
for i, ox_perturb in enumerate(growth_perturbation):
    print(f"\nGenerating Visualization of oxide offset {ox_perturb} array at {lateral_growth+ox_perturb} lateral growth:")
    visualization_generator = GA.VCSELGenerator(geometry_arrays = vcsel_arrays[i*indices_of_ox:(i+1)*indices_of_ox],
                                                lateral_growth = lateral_growth + ox_perturb,
                                                contact_padding = array_contact_padding,
                                                implant_width = implant_width,
                                                implant_length = implant_length,
                                                min_contact_area = 5)
    visualization_generator.generate_all(inv_fit=False)
    visualization_generator.plot(show_implant=False,
                                 show_implant_inv = True,
                                 show_contact_region = False)

    for gds in visualization_generator.to_gdspy():
        array_element_unitcell.add(gds)

lib.write_gds('./gds_files/bowtie_3E_array_elements.gds')