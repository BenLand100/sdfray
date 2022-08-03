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
from .geom import *
import numpy as np

class Sphere(SDF):

    def __init__(self,radius=1,**kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        
    def fn(self,pts):
        return L(pts) - self.radius
    
    def glsl(self):
        tx = self.translate if self.translate is not None else A([0,0,0])
        geo = f'sphere(p,vec3({tx[0]},{tx[1]},{tx[2]}),{float(self.radius)})'
        if self.surface is None:
            prop = geo
            sfrags = []
        else:
            sglsl,sfrags = self.surface.glsl()
            prop = f'wrap({geo},{sglsl})'
        frags = [Sphere.glsl_function]+sfrags
        return geo,prop,frags
        
    glsl_function = '''
        float sphere(vec3 p, vec3 loc, float radius) {
            return length(p - loc) - radius;
        }
    '''

class Box(SDF):

    def __init__(self,width=1,height=1,depth=1,**kwargs):
        super().__init__(**kwargs)
        self.dims = A([width,height,depth])
        
    def fn(self,pts):
        deltas = np.abs(pts)-self.dims/2
        return L(np.maximum(deltas,0.0)) + np.minimum(np.max(deltas,axis=-1),0.0)
    
    def glsl(self):
        tx = self.translate if self.translate is not None else A([0,0,0])
        whd = f'vec3({float(self.dims[0]),float(self.dims[1]),float(self.dims[2])})'
        geo = f'box(p,vec3({tx[0]},{tx[1]},{tx[2]}),{whd})'
        if self.surface is None:
            prop = geo
            sfrags = []
        else:
            sglsl,sfrags = self.surface.glsl()
            prop = f'wrap({geo},{sglsl})'
        frags = [Box.glsl_function]+sfrags
        return geo,prop,frags
        
    glsl_function = '''
        float box(vec3 p, vec3 loc, vec3 whd) {
            vec3 del = abs(p-loc)-whd/2.;
            float mval = max(max(del.x,del.y),del.z);
            return length(vec3(max(del.x,0.),max(del.y,0.),max(del.z,0.)))+min(mval,0.);
        }
    '''

class Cylinder(SDF):

    def __init__(self,height=1,radius=1,**kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.radius = radius
        
    def fn(self,pts):
        a = L(pts[...,[0,1]]) - self.radius
        b = np.abs(pts[...,2]) - self.height/2
        return np.minimum(np.maximum(a,b),0.0) + L(np.maximum(A([a,b]).T,0))
        
    def glsl(self):
        tx = self.translate if self.translate is not None else A([0,0,0])
        geo = f'cylinder(p,vec3({tx[0]},{tx[1]},{tx[2]}),{float(self.height)},{float(self.radius)})'
        if self.surface is None:
            prop = geo
            sfrags = []
        else:
            sglsl,sfrags = self.surface.glsl()
            prop = f'wrap({geo},{sglsl})'
        frags = [Cylinder.glsl_function]+sfrags
        return geo,prop,frags
        
    glsl_function = '''
        float cylinder(vec3 p, vec3 loc, float height, float radius) {
            p -= loc;
            float a = length(p.xz)-radius;
            float b = abs(p.y)-height/2.;
            return min(max(a,b),0.) + length(vec2(max(a,0.),max(b,0.)));
        }
    '''

class Plane(SDF):

    def __init__(self,anchor=A([0,-1,0]),normal=A([0,1,0]),**kwargs):
        super().__init__(**kwargs)
        self.anchor = anchor
        self.normal = N(normal)
        
    def fn(self,pts):
        return np.sum((pts - self.anchor)*self.normal,axis=-1)
        
    def glsl(self):
        tx = self.translate if self.translate is not None else A([0,0,0])
        norm = f'vec3({self.normal[0]},{self.normal[1]},{self.normal[2]})'
        geo = f'plane(p,vec3({tx[0]},{tx[1]},{tx[2]}),{norm})'
        if self.surface is None:
            prop = geo
            sfrags = []
        else:
            sglsl,sfrags = self.surface.glsl()
            prop = f'wrap({geo},{sglsl})'
        frags = [Plane.glsl_function]+sfrags
        return geo,prop,frags
        
    glsl_fragment = '''
        float plane(vec3 p, vec3 loc, vec3 norm) {
            return dot(p-loc,norm);
        }
    '''
