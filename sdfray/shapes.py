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
    
    def glsl_geo(self,tx,rot):
        geo = f'sphere(p,{tx},{glsl_float(self.radius)})'
        frags = [Sphere.glsl_function]
        return geo,frags
        
    glsl_function = '''
        float sphere(vec3 p, vec3 tx, float radius) {
            return length(p - tx) - radius;
        }
    '''

class Box(SDF):

    def __init__(self,width=1,height=1,depth=1,**kwargs):
        super().__init__(**kwargs)
        self.dims = A([width,height,depth])
        
    def fn(self,pts):
        deltas = np.abs(pts)-self.dims/2
        return L(np.maximum(deltas,0.0)) + np.minimum(np.max(deltas,axis=-1),0.0)
    
    def glsl_geo(self,tx,rot):
        whd = glsl_vec3(self.dims)
        geo = f'box(p,{tx},{rot},{whd})'
        frags = [Box.glsl_function]
        return geo,frags
        
    glsl_function = '''
        float box(vec3 p, vec3 tx, mat3 rot, vec3 whd) {
            p = rot*(p-tx);
            vec3 del = abs(p)-whd/2.;
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
        a = L(pts[...,[0,2]]) - self.radius
        b = np.abs(pts[...,1]) - self.height/2
        return np.minimum(np.maximum(a,b),0.0) + L(np.maximum(A([a,b]).T,0))
        
    def glsl_geo(self,tx,rot):
        geo = f'cylinder(p,{tx},{rot},{glsl_float(self.height)},{glsl_float(self.radius)})'
        frags = [Cylinder.glsl_function]
        return geo,frags
        
    glsl_function = '''
        float cylinder(vec3 p, vec3 tx, mat3 rot, float height, float radius) {
            p = rot*(p-tx);
            float a = length(p.xz)-radius;
            float b = abs(p.y)-height/2.;
            return min(max(a,b),0.) + length(vec2(max(a,0.),max(b,0.)));
        }
    '''

class Plane(SDF):

    def __init__(self,anchor=A([0,-1,0]),normal=A([0,1,0]),**kwargs):
        super().__init__(**kwargs)
        self.anchor = A(anchor)
        self.normal = N(A(normal))
        
    def fn(self,pts):
        return np.sum((pts - self.anchor)*self.normal,axis=-1)
        
    def glsl_geo(self,tx,rot):
        norm = f'vec3({self.normal[0]},{self.normal[1]},{self.normal[2]})'
        anchor = f'vec3({self.anchor[0]},{self.anchor[1]},{self.anchor[2]})'
        geo = f'plane(p,{anchor},{norm})'
        if self.surface is None:
            prop = geo
            sfrags = []
        else:
            sglsl,sfrags = self.surface.glsl()
            prop = f'wrap({geo},{sglsl})'
        frags = [Plane.glsl_function]+sfrags
        return geo,frags
        
    glsl_function = '''
        float plane(vec3 p, vec3 anchor, vec3 norm) {
            return dot(p-anchor,norm);
        }
    '''
