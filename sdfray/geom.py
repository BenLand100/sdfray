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
        
    def glsl(self,tx=None,rot=None):
        tx,rot,glsl_tx,glsl_rot = self.glsl_transform(tx,rot)
        geo,gfrags = self.glsl_geo(glsl_tx,glsl_rot)
        prop,sfrags = self.glsl_prop(geo)
        return geo,prop,gfrags+sfrags
        
    def glsl_geo(self,tx,rx):
        raise Exception(f'{type(self)} does not implement glsl_Geo')
    
    def glsl_prop(self,geo):
        if self.surface is None:
            sfrags = []
            prop = geo
        else:
            sglsl,sfrags = self.surface.glsl()
            prop = f'wrap({geo},{sglsl})'
        return prop,sfrags
        
    def glsl_transform(self,tx,rot):
        if self.translate is not None:
            if rot is None:
                tx = self.translate + tx if tx is not None else self.translate
            else:
                tx = (rot.T @ self.translate) + tx if tx is not None else (rot.T @ self.translate)
        if self.rotate is not None:
            rot = self.rotate @ rot if rot is not None else self.rotate
        glsl_tx = 'vec3(0.,0.,0.)' if tx is None else f'vec3({tx[0]},{tx[1]},{tx[2]})'
        glsl_rot = 'mat3(1.,0.,0.,0.,1.,0.,0.,0.,1.)' if rot is None else \
            f'mat3({rot[0,0]},{rot[1,0]},{rot[2,0]},{rot[0,1]},{rot[1,1]},{rot[2,1]},{rot[0,2]},{rot[1,2]},{rot[2,2]})'
        return tx,rot,glsl_tx,glsl_rot
        
            
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
    
    def glsl(self,tx=None,rot=None):
        tx,rot,glsl_tx,glsl_rot = self.glsl_transform(tx,rot)
        ageo,aprop,afrags = self.a.glsl(tx=tx,rot=rot)
        bgeo,bprop,bfrags = self.b.glsl(tx=tx,rot=rot)
        geo = f'intersect({ageo},{bgeo})'
        prop = f'intersect({aprop},{bprop})'
        prop,sfrags = self.glsl_prop(prop)
        fragments = [Intersection.glsl_function]+afrags+bfrags+sfrags
        return geo,prop,fragments
        
    glsl_function = '''
        float intersect(float a, float b) {
            return max(a,b);
        }
        
        GeoInfo intersect(GeoInfo a, GeoInfo b) {
            if (a.sdf > b.sdf) {
                return a;
            } else {
                return b;
            }
        }
    '''
        
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
        
    def glsl(self,tx=None,rot=None):
        tx,rot,glsl_tx,glsl_rot = self.glsl_transform(tx,rot)
        ageo,aprop,afrags = self.a.glsl(tx=tx,rot=rot)
        bgeo,bprop,bfrags = self.b.glsl(tx=tx,rot=rot)
        geo = f'join({ageo},{bgeo})'
        prop = f'join({aprop},{bprop})'
        prop,sfrags = self.glsl_prop(prop)
        fragments = [Union.glsl_function]+afrags+bfrags+sfrags
        return geo,prop,fragments
        
    glsl_function = '''
        float join(float a, float b) {
            return min(a,b);
        }
        
        GeoInfo join(GeoInfo a, GeoInfo b) {
            if (a.sdf < b.sdf) {
                return a;
            } else {
                return b;
            }
        }
    '''

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
        
    def glsl(self,tx=None,rot=None):
        tx,rot,glsl_tx,glsl_rot = self.glsl_transform(tx,rot)
        ageo,aprop,afrags = self.a.glsl(tx=tx,rot=rot)
        bgeo,bprop,bfrags = self.b.glsl(tx=tx,rot=rot)
        geo = f'subtract({ageo},{bgeo})'
        prop = f'subtract({aprop},{bprop})'
        prop,sfrags = self.glsl_prop(prop)
        fragments = [Subtraction.glsl_function]+afrags+bfrags+sfrags
        return geo,prop,fragments
        
    glsl_function = '''
        float subtract(float a, float b) {
            return max(a,-b);
        }
        
        GeoInfo subtract(GeoInfo a, GeoInfo b) {
            if (a.sdf > -b.sdf) {
                return a;
            } else {
                return b;
            }
        }

    '''
