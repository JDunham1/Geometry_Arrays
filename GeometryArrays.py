# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 12:44:28 2025

@author: Jake Dunham
"""

import math
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
import gdspy

class VCSELGenerator:
    def __init__(self, geometry_arrays: list, lateral_growth: float = None):
        """
        Parameters:
            geometry_arrays (list of GeometryArray objects): Each list entry represents a VCSEL mesa structure
            lateral_growth (float): Defines the expected lateral growth of of the mesa structure
        """
        self.geometry_arrays = geometry_arrays
        self.lateral_growth = lateral_growth
        self._mesa = None
        self._aperture = None

    def generate_mesa(self):
        """
        Unions all elements from all GeometryArrays into a single mesa shape (Shapely).
        """
        all_polygons = []
        for g_array in self.geometry_arrays:
            all_polygons.extend(g_array.to_shapely())
        
        self._mesa = unary_union(all_polygons)
        return self._mesa

    def generate_aperture(self, lateral_growth = None):
        """
        Buffers the mesa inward to create the aperture.
        
        Parameters:
            offset (float): Value for shrinking the mesa.
        
        Returns:
            shapely.geometry.Polygon or MultiPolygon
        """
        #both are none
        if lateral_growth is None and self.lateral_growth is None:
            raise ValueError('DeviceGenerator requires a lateral_growth to generate the aperture.')
        #both are something
        elif lateral_growth is not None and self.lateral_growth is not None:
            UserWarning(f"""Overwriting DeviceGenerator's lateral growth from
                        {self.lateral_growth=} to {lateral_growth}""")
        #one of them is something
        elif self.lateral_growth is None:
            #the object attribute is missing
            self.lateral_growth = lateral_growth
        
        if self.lateral_growth < 0:
            UserWarning("""lateral_growth cannot physically be negative. 
                        Growth is negative relative to the boundary.
                        Handled your input under this assumption.""")
        else:
            self.lateral_growth = -self.lateral_growth
        
        if self._mesa is None:
            self.generate_mesa()
        self._aperture = self._mesa.buffer(self.lateral_growth)
        return self._aperture

    def plot(self, show_mesa=True, show_aperture=True, ax=None, **kwargs):
        """
        Plots the mesa and/or aperture on a single matplotlib axis.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if show_mesa:
            if self._mesa is None:
                self.generate_mesa()
            self._plot_shape(self._mesa, ax, color='gray', alpha=0.5, label="Mesa", **kwargs)

        if show_aperture:
            #if a lateral_growth exists automatically generate the aperture
            if self._aperture is None and self.lateral_growth is not None:
                self.generate_aperture()
            #if a lateral growth doesn't exist inform the user to assign via generate aperture.
            elif self._aperture is None:
                raise ValueError("Call generate_aperture() before plotting the aperture.")
            self._plot_shape(self._aperture, ax, color='blue', alpha=0.4, label="Aperture", **kwargs)

        ax.set_aspect('equal')
        ax.legend()
        plt.show()

    @staticmethod
    def _plot_shape(shape, ax, **kwargs):
        """
        Internal utility to plot a Shapely shape (Polygon or MultiPolygon).
        """
        from shapely.geometry import Polygon, MultiPolygon
        if isinstance(shape, Polygon):
            x, y = shape.exterior.xy
            ax.fill(x, y, **kwargs)
        elif isinstance(shape, MultiPolygon):
            for poly in shape.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, **kwargs)
#%%
class GeometryArray:
    def __init__(self, elements=None):
        self.elements = elements or []  # list of ArrayElement instances

    def add_element(self, element):
        self.elements.append(element)

    def plot(self, ax=None, **kwargs):
        fig, ax = plt.subplots()

        for element in self.elements:
            element.plot(ax=ax)  # Reuse shared axis

        plt.title("Full Geometry Array")
        plt.show()
    
    def center_array(self):
        raise NotImplementedError

    def to_gdspy(self, layer=0):
        gdspy_polygons = []
        for element in self.elements:
            verts = element.get_transformed_vertices()
            gdspy_polygons.append(gdspy.Polygon(verts, layer=layer))
        return gdspy_polygons

    def to_shapely(self):
        shapely_polygons = []
        for element in self.elements:
            verts = element.get_transformed_vertices()
            shapely_polygons.append(ShapelyPolygon(verts))
        return shapely_polygons
    
#%% ArrayElement Class
#array elements hold information on the center, and rotation of a BasePolgyon 
class ArrayElement:
    def __init__(self, polygon, center=(0, 0), rotation_deg=0):
       """
       Initialize an ArrayElement with a polygon, center point, and rotation angle.
       
       Parameters:
           polygon (BasePolygon): The polygon object (must have .vertices).
           center (tuple): (x, y) position to place the polygon's center.
           rotation_deg (float): Rotation angle in degrees around the center.
       """
       self.polygon = polygon
       self.center = np.array(center)
       self.rotation_deg = rotation_deg

    def get_transformed_vertices(self):
       """
       Return the polygon vertices rotated around the origin and translated to the center.
       """
       vertices = np.array(self.polygon.vertices)

       # Step 2: Rotate about origin
       theta = np.deg2rad(self.rotation_deg)
       rotation_matrix = np.array([
           [np.cos(theta), -np.sin(theta)],
           [np.sin(theta),  np.cos(theta)]
       ])
       
       rotated = vertices @ rotation_matrix.T #matrix multiplication

       # Step 3: Translate to specified center
       final_vertices = rotated + self.center
       return final_vertices.tolist()
   
    def resize(self,offset):
        """
        Erodes or dilates the polygon. 
        Does not modify the rotation or translation!
       
       Parameters:
       - offset: float
           Positive value for dilation, negative for erosion.

        """
        # 1. Get polygon vertices in global coordinates
        vertices = self.polygon.vertices
        shapely_poly = ShapelyPolygon(vertices)

        # 2. Erode/dilate using Shapely's buffer
        resized_poly = shapely_poly.buffer(offset)

        if resized_poly.is_empty or not resized_poly.is_valid:
           raise ValueError("Resizing resulted in an invalid or empty polygon.")

        # 3. Convert result to list of (x, y) vertices
        new_vertices = list(resized_poly.exterior.coords)

        # 4. Remove closing vertex (last point == first point)
        if np.allclose(new_vertices[0], new_vertices[-1]):
            new_vertices = new_vertices[:-1]

        # 5. Replace self.polygon with new BasePolygon
        self.polygon = BasePolygon(vertices = new_vertices, t = 'BasePolygon')
        return self
    
    def plot(self, ax=None, **kwargs):
        transformed = self.get_transformed_vertices()
        plot_polygon(transformed, title=f"{self.polygon.t} Element", ax=ax, **kwargs)
        
    def copy_with_transform(self, center=None, rotation_deg=None):
        """
        Returns a new ArrayElement with the same polygon,
        but a new center and/or rotation.
        """
        new_center = center if center is not None else self.center
        new_rotation = rotation_deg if rotation_deg is not None else self.rotation_deg
        return ArrayElement(self.polygon, new_center, new_rotation)

#%% Geometry Classes
#geometry classes check for proper definition of a polygon in terms of its
#vertices. Also computes the side lengths and internal angles if useful
class BasePolygon:
    def __init__(self, vertices, t = 'BasePolygon'):
        if vertices is None or len(vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices.")
        
        self.vertices = vertices    
        
        self.side_lengths, self.angles_deg = polygon_from_vertices(vertices)
        
        if all_sides_equal(self.side_lengths):
            self.t = 'Regular ' + t
        else:
            self.t = 'Irregular ' + t 
    
        
    def plot(self, ax=None, **kwargs):
        plot_polygon(self.vertices, title=self.t, ax=ax, **kwargs)
        
    
class Hexagon(BasePolygon):
    #use the class methods which wrap init in simpler parameters for ease of use
    def __init__(self, vertices):
        
        #checks that vertices meets the min requirements of a hexagon
        if len(vertices) != 6:
            raise ValueError("Hexagon must exactly 6 vertices.")
            
        #assign vertices, type of polygon, and computes side_lengths, and internal angles of polygon
        super().__init__(vertices, 'Hexagon')

    @classmethod
    def from_dimensions(cls,
                        width = [1, 1],
                        height = [sqrt(3)/2, sqrt(3)/2],
                        side_length = [1,1]):
        """
        Creates a flat-topped hexagon from width, height, and side_length

        """
        w1 = width[0]
        w2 = width[1]
        h1 = height[0]
        h2 = height[1]
        sl = side_length[0] / 2
        s2 = side_length[1] / 2
        
        vertices = [
                (-sl,-h1),
                (sl,-h1),
                (w2,0),
                (s2,h2),
                (-s2,h2),
                (-w1,0)
            ]
        
        #calls __init__() with the wrapped vertices
        return cls(vertices)

class Pentagon(BasePolygon):
    def __init__(self, vertices):
        #checks that vertices meets the min requirements of a hexagon
        if len(vertices) != 5:
            raise ValueError("Pentagon must exactly 5 vertices.")
            
        #assign vertices, type of polygon, and computes side_lengths, and internal angles of polygon
        super().__init__(vertices, 'Pentagon')
    
    @classmethod
    def from_dimensions(cls,
                        width = [(1+sqrt(5))/4, (1+sqrt(5))/4],
                        height = [sqrt(5+sqrt(5))/2, (1+sqrt(5))/4],
                        side_length = 1):
        """
        Creates a flat-topped hexagon from width, height, and side_length

        """
        w1 = width[0]
        w2 = width[1]
        h1 = height[0]
        h2 = height[1]
        sl = side_length / 2
        
        vertices = [
                (-sl,-h1),
                (sl,-h1),
                (w2,0),
                (0,h2),
                (-w1,0)
            ]
        
        #calls __init__() with the wrapped vertices
        return cls(vertices)
    
    @classmethod
    def from_inradius(cls,inradius):
        """
        Creates a regular pentagon from the defined inradius of the geoemtry
        """
        # Number of sides and angle step
        n = 5
        angle_step = 2 * np.pi / n

        # Convert inradius to circumradius
        R = inradius / np.cos(np.pi / n)

        # Align the bottom edge to be horizontal (flat on x-axis)
        # Rotate so that one edge lies flat: shift angle by -π/2 - π/n
        start_angle = -np.pi / 2 - np.pi / n

        vertices = []
        for i in range(n):
            theta = start_angle + i * angle_step
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            vertices.append((x, y))

        return cls(vertices)

class Quadrilateral(BasePolygon):
    #use the class methods which wrap init in simpler parameters for ease of use
    def __init__(self, vertices):

        #checks that the parameters meet the min requirements of a quadrilateral
        if len(vertices) != 4:
            raise ValueError("Quadrilateral must have exactly 4 vertices.")
         
        #assign vertices, type of polygon, and computes side_lengths, and internal angles of polygon
        super().__init__(vertices, 'Quadrilateral')
            
    @classmethod
    def from_diagonals(cls, diagonals=[1,1,1,1]):
        """
        Creates a rotated quadrilateral with diagonal_1 = diagonals[0] + diagonals[1]
        and diagonal_2 = diagonals[2] + diagonals[3]

        """
        d1 = diagonals[0]
        d2 = diagonals[1]
        d3 = diagonals[2]
        d4 = diagonals[3]
        
        vertices = [
                (0,-d1),
                (d3,0),
                (0,d2),
                (-d4,0)
            ]
        
        #calls __init__() with the wrapped vertices
        return cls(vertices)
    
    @classmethod
    def from_dimensions(cls, width=[1,1], height=[1,1]):
        """
        Creates a rotated quadrilateral with width = width[0] + width[1]
        and height = height[0] + height[1]
        """
        w1 = width[0]
        w2 = width[1]
        h1 = height[0]
        h2 = height[1]
        
        vertices = np.array(
            [(-w1,-h1),
             (w2,-h1),
             (w2,h2),
             (-w1,h2)]
            )
        
        theta = np.deg2rad(45)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        rotated = vertices @ rotation_matrix.T
        return cls(rotated)

#%% Polygon Measurement Functions

def all_sides_equal(side_lengths, tol=1e-8):
    arr = np.array(side_lengths)
    return np.all(np.abs(arr - arr[0]) < tol)

def polygon_from_vertices(vertices):
    """
    Given a list of 2D vertices [(x0, y0), (x1, y1), ..., (xn, yn)],
    returns:
    - side_lengths: list of side lengths
    - angles_deg: list of internal angles in degrees
    Skips consecutive duplicate vertices.
    """
    # Remove consecutive duplicates
    cleaned_vertices = []
    for v in vertices:
        if not cleaned_vertices or not np.allclose(v, cleaned_vertices[-1]):
            cleaned_vertices.append(v)

    # Ensure the polygon is still valid
    if len(cleaned_vertices) < 3:
        raise ValueError("Polygon must have at least 3 distinct vertices.")

    n = len(cleaned_vertices)
    side_lengths = []
    angles_deg = []

    for i in range(n):
        p0 = cleaned_vertices[i - 1]
        p1 = cleaned_vertices[i]
        p2 = cleaned_vertices[(i + 1) % n]

        # ---- Side length (p1 to p2)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        side_lengths.append(length)

        # ---- Internal angle at p1 between vectors (p0->p1) and (p1->p2)
        v1 = (p0[0] - p1[0], p0[1] - p1[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])

        norm1 = math.hypot(*v1)
        norm2 = math.hypot(*v2)

        # Guard against zero-length vectors
        if norm1 == 0 or norm2 == 0:
            angles_deg.append(0)
            continue

        dot = v1[0]*v2[0] + v1[1]*v2[1]
        cos_theta = max(min(dot / (norm1 * norm2), 1), -1)
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)
        angles_deg.append(angle_deg)

    return side_lengths, angles_deg

#%% Polygon Plotting Function
def plot_polygon(vertices, color='lightblue', edgecolor='black', show_vertices=True,
                 title=None, ax=None, label=None):
    """Plot a polygon on the given axis (or create a new one)."""
    # Create figure and axis only if none is provided
    if ax is None:
        fig, ax = plt.subplots()
        new_fig = True
    else:
        new_fig = False

    patch = MplPolygon(vertices, closed=True, facecolor=color, edgecolor=edgecolor, label=label)
    ax.add_patch(patch)

    if show_vertices:
        xs, ys = zip(*vertices)
        ax.plot(xs, ys, 'o', color='red', markersize=4)

    if title and new_fig:
        ax.set_title(title)

    ax.set_aspect('equal')
    ax.autoscale()
    ax.grid(True)

    # Only call plt.show() if we created the figure ourselves
    if new_fig:
        plt.show()

    return ax  # Always return the axis for further use

#%% Testing space
if __name__ == "__main__":
    plt.close('all')
    
    #generic geometry testing
    regular_hexagon = Hexagon.from_dimensions()
    regular_hexagon.plot()
    print(regular_hexagon.side_lengths)
    print(regular_hexagon.angles_deg)
    
    regular_pentagon = Pentagon.from_dimensions()
    regular_pentagon.plot()
    print(regular_pentagon.side_lengths)
    print(regular_pentagon.angles_deg)
    
    irregular_hexagon = Hexagon.from_dimensions(width=[1,0.5])
    irregular_hexagon.plot()
    print(irregular_hexagon.side_lengths)
    print(irregular_hexagon.angles_deg)
    
    trapezoid = Hexagon.from_dimensions(height=[0,sqrt(3)/2])
    trapezoid.plot()
    print(trapezoid.side_lengths)
    print(trapezoid.angles_deg)
    
    diamond = Quadrilateral.from_dimensions()
    diamond.plot()
    print(diamond.side_lengths)
    print(diamond.angles_deg)
    
    #ArrayElement testing
    hexagon_element = ArrayElement(regular_hexagon,center=(5,0), rotation_deg=0)
    hexagon_element_2 = hexagon_element.copy_with_transform(center=(0,5), rotation_deg=-45)
    hexagon_element_3 = hexagon_element.copy_with_transform(center= (0,-5), rotation_deg = -105).resize(0.5)
    hexagon_element_4 = hexagon_element.copy_with_transform(center=(0,0),rotation_deg = 45).resize(-0.5)
    
    #GeometryArray testing
    array = GeometryArray(elements=[hexagon_element,hexagon_element_2,hexagon_element_3,hexagon_element_4])
    array.plot()
    
    print(array.to_gdspy())
    print(array.to_shapely())
    
    #VCSELGenerator testing
    #build Dalir Mesa and erode it
    diamond = Quadrilateral.from_dimensions(width=[7.5,7.5],height=[7.5,7.5])
    diamond.plot()
    #check to make sure side lengths equal 2*6 (lateral_growth) + 3 (aperture_size)
    print(diamond.side_lengths)
    dalir_element_1 = ArrayElement(diamond)
    dalir_element_2 = dalir_element_1.copy_with_transform(center=(7.757,0))
    dalir_geometries = GeometryArray(elements=[dalir_element_1,dalir_element_2])
    
    dalir_device = VCSELGenerator([dalir_geometries], lateral_growth = 6)
    dalir_device.plot()
    
    #build 45 angled mesa structure
    edge_cavity = Pentagon.from_inradius(5+1.5) #inradius = lateral_growth + aperture/2
    edge_cavity.plot()
    left_edge = ArrayElement(edge_cavity,center=(-8.5,-3),
                             rotation_deg=90+22.5)
    right_edge = ArrayElement(edge_cavity,center=(8.5,-3),
                              rotation_deg=-90-22.5)
    
    center_cavity = Hexagon.from_dimensions(width=[12.35/2,12.35/2],
                                            height=[11.029,2.025],
                                            side_length=[2.947,4.703])
    center_cavity.plot()
    centerElement = ArrayElement(center_cavity,center = (0,4.5))
    
    angled_array = GeometryArray([left_edge,centerElement,right_edge])
    angled_array.plot()