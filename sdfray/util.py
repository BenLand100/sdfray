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

from .parameter import *
import numpy as np
    
    
def glsl_float(val):
    '''Converts a float value or Parameter to a GLSL statement'''
    return str(val) if isinstance(val,Parameter) else f'{float(val)}'
    
def glsl_vec3(listlike):
    '''Converts a length-3 listlike of float or Parameter values to a GLSL vec3'''
    assert len(listlike) == 3, 'vec3 must have 3 elements'
    elems = ','.join([glsl_float(l) for l in listlike])
    return f'vec3({elems})'
    
def glsl_mat3(rot):
    '''Converts a 3x3 ndarray-compatible object of float or Parameter values to a GLSL mat3'''
    assert rot.shape == (3,3), 'mat3 must have shape (3,3)'
    return f'mat3({glsl_float(rot[0,0])},{glsl_float(rot[1,0])},{glsl_float(rot[2,0])},{glsl_float(rot[0,1])},{glsl_float(rot[1,1])},{glsl_float(rot[2,1])},{glsl_float(rot[0,2])},{glsl_float(rot[1,2])},{glsl_float(rot[2,2])})'

def A(listlike,dtype=np.float64):
    '''Convenience function to convert to numpy array. Preserves Parameters.'''
    params = [l for l in listlike if isinstance(l,Parameter)]
    if len(params) > 0:
        dtype = Parameter
        listlike = [params[0].wrap(l) for l in listlike]
    return np.asarray(listlike,dtype=dtype)

def L(arr):
    '''Computes the euclidean length across the last axis'''
    return np.sqrt(np.sum(arr*arr,axis=-1))

def N(arr):
    '''Normalizes vectors across the last axis'''
    return (arr.T/L(arr)).T
    
def XROT(ang):
    '''3D Rotation matrix about X axis'''
    ca,sa = np.cos(ang),np.sin(ang)
    return np.asarray([A([1,0,0]),A([0,ca,-sa]),A([0,sa,ca])])
    
def YROT(ang):
    '''3D Rotation matrix about Y axis'''
    ca,sa = np.cos(ang),np.sin(ang)
    return np.asarray([A([ca,0,sa]),A([0,1,0]),A([-sa,0,ca])])
    
def ZROT(ang):
    '''3D Rotation matrix about Z axis'''
    ca,sa = np.cos(ang),np.sin(ang)
    return np.asarray([A([ca,-sa,0]),A([sa,ca,0]),A([0,0,1])])

D_ = 1e-4
DX = A([D_,0,0])
DY = A([0,D_,0])
DZ = A([0,0,D_])

def G(sdf,pts):
    '''Computes the gradient of the SDF scalar field'''
    return A([sdf(pts+DX)-sdf(pts-DX),
              sdf(pts+DY)-sdf(pts-DY),
              sdf(pts+DZ)-sdf(pts-DZ)]).T/(2*D_)
              
class Rays:
    '''A sometimes-used class for storing the [p]osition and [d]irection of some rays'''
    def __init__(self,p=A([[0,0,0]]),d=A([[0,0,1]])):
        self.p = p
        self.d = d
