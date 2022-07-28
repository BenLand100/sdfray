#    Copyright 2022 by Benjamin J. Land (a.k.a. BenLand100)
#
#    This file is part of sdfray.
#
#    sdfray is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    sdfray is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with sdfray.  If not, see <https://www.gnu.org/licenses/>.

from .util import *
from .surface import *
from scipy.spatial.transform import Rotation as R
import numpy as np

default_surface = UniformSurface(SurfaceProp())

class SDF:
    '''Base class for all geometry primatives, implementing common functionality'''
    
    def __init__(self,surface=default_surface,translate=None,rotate_seq='xyz',rotate=None,rounding=None):
        if rotate is None:
            self.rotate = None
        else:
            self.rotate = R.from_euler(seq=rotate_seq,angles=rotate).as_matrix()
        if translate is None:
            self.translate = None
        else:
            self.translate = A(translate)
        self.surface = surface
        self.rounding = rounding
        
    def __call__(self,pts,properties=False):
        '''Transform coordinates for this primative AND
           Evaluate its signed distance function OR
           Evaluate its surface at the specified points'''
        pts = self.transform(pts)
        if properties:
            return self.props(pts)
        else:
            val = self.fn(pts)
            return val if self.rounding is None else val-self.rounding
            
    def transform(self,pts):
        if self.rotate is not None:
            pts = (self.rotate @ pts.T).T
        if self.translate is not None:
            pts = pts - self.translate
        return pts
            
    def fn(self,pts):
        '''Implements the signed distance function for the primitive'''
        raise Exception('SDF base class cannot be evaluated')
        
    def props(self,pts):
        '''Calls the configured surface to obtain properties'''
        return self.surface(pts)
        
            
class Intersection(SDF):
    '''Defines a SDF for the intersection of two SDFs'''
    
    def __init__(self,a,b,surface=None,**kwargs):
        '''surface can be specified to override properties of a and b'''
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.surface = surface
        
    def fn(self,pts): 
        return np.maximum(self.a(pts),self.b(pts))
        
    def props(self,pts):
        if self.surface is not None:
            return self.surface(pts)
        mask = self.a(pts) >= self.b(pts)
        props = np.empty(len(pts),dtype=SurfaceProp)
        props[mask] = self.a.props(pts[mask])
        mask = ~mask
        props[mask] = self.b.props(pts[mask])
        return props
        
class Union(SDF):
    '''Defines a SDF for the union of two SDFs'''
    
    def __init__(self,a,b,surface=None,**kwargs):
        '''surface can be specified to override properties of a and b'''
        super().__init__(self,**kwargs)
        self.a = a
        self.b = b
        self.surface = surface
        
    def fn(self,pts): 
        return np.minimum(self.a(pts),self.b(pts))
        
    def props(self,pts):
        if self.surface is not None:
            return self.surface(pts)
        mask = self.a(pts) <= self.b(pts)
        props = np.empty(len(pts),dtype=SurfaceProp)
        props[mask] = self.a.props(pts[mask])
        mask = ~mask
        props[mask] = self.b.props(pts[mask])
        return props

class Subtraction(SDF):
    '''Defines a SDF for the subtraction of two SDFs'''
    
    def __init__(self,a,b,surface=None,**kwargs):
        '''surface can be specified to override properties of a and b'''
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.surface = surface
    def fn(self,pts): 
        return np.maximum(self.a(pts),-self.b(pts))
        
    def props(self,pts):
        if self.surface is not None:
            return self.surface(pts)
        mask = self.a(pts) >= -self.b(pts)
        props = np.empty(len(pts),dtype=SurfaceProp)
        props[mask] = self.a.props(pts[mask])
        mask = ~mask
        props[mask] = self.b.props(pts[mask])
        return props
