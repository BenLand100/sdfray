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

class Box(SDF):

    def __init__(self,width=1,height=1,depth=1,**kwargs):
        super().__init__(**kwargs)
        self.dims = A([width,height,depth])
        
    def fn(self,pts):
        deltas = np.abs(pts)-self.dims/2
        return L(np.maximum(deltas,0.0)) + np.minimum(np.max(deltas,axis=-1),0.0)

class Cylinder(SDF):

    def __init__(self,height=1,radius=1,**kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.radius = radius
        
    def fn(self,pts):
        a = L(pts[...,[0,1]]) - self.radius
        b = np.abs(pts[...,2]) - self.height/2
        return np.minimum(np.maximum(a,b),0.0) + L(np.maximum(A([a,b]).T,0))

class Plane(SDF):

    def __init__(self,anchor=A([0,-1,0]),normal=A([0,1,0]),**kwargs):
        super().__init__(**kwargs)
        self.anchor = anchor
        self.normal = N(normal)
        
    def fn(self,pts):
        return np.sum((pts - self.anchor)*self.normal,axis=-1)
