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

import numpy as np

def A(listlike,dtype=np.float64):
    '''Convenience function to convert to numpy array'''
    return np.asarray(listlike,dtype=dtype)

def L(arr):
    '''Computes the euclidean length across the last axis'''
    return np.sqrt(np.sum(arr*arr,axis=-1))

def N(arr):
    '''Normalizes vectors across the last axis'''
    return (arr.T/L(arr)).T

D_ = 1e-4
DX = A([D_,0,0])
DY = A([0,D_,0])
DZ = A([0,0,D_])

def G(sdf,pts):
    '''Computes the gradient of the SDF'''
    return A([sdf(pts+DX)-sdf(pts-DX),
              sdf(pts+DY)-sdf(pts-DY),
              sdf(pts+DZ)-sdf(pts-DZ)]).T/(2*D_)
              
class Rays:
    '''A sometimes-used class for storing the [p]osition and [d]irection of some rays'''
    def __init__(self,p=A([[0,0,0]]),d=A([[0,0,1]])):
        self.p = p
        self.d = d
