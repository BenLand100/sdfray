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
from .render import *
import numpy as np

class Light:
    '''Base class for implenting lighting effects. Implements lighting surfaces
       based on orientation to an illumination direction (or ambient, with no 
       orientation).'''

    def __init__(self):
        pass
        
    def pointing(self,pts):
        '''Should return a direction vector from the pts to the source'''
        raise Exception('Use a Light implementation, instead!')
        
    def color(self,in_dirs):
        '''Should return the color of the source given an angle of incidence'''
        raise Exception('Use a Light implementation, instead!')
        
    def light(self,pts,surfaces,in_dirs,normals,sdf,colors=None,lights=[],prescale=None):
        '''Checks to see if the light is not occluded, and if not, calculates 
           the light reflected from that surface'''
        #print('Lighting',self)
        if colors is None:
            colors = np.zeros(pts.shape,dtype=np.float32)
        pointing = self.pointing(pts)
        if pointing is None:
            light_colors = self.illumination(pts,None)
            surf_colors = A([s.color for s in surfaces])
            colors += (light_colors*surf_colors)
        else:
            lr = Rays(p=pts,d=N(pointing))
            #print('Calculating visibility')
            _,_,lr_blocked = next_surface(lr,sdf,lighting=True)
            m = ~lr_blocked #point has visibility to light source
        
            if np.count_nonzero(m):
                norms = normals[m]
                in_d = in_dirs[m]
                out_d = lr.d[m]
                p_illum = lr.p[m]
                light_colors = self.illumination(p_illum,pointing[m])
                surfs = surfaces[m]
                surf_colors = A([s.color for s in surfs])
                
                out_d_dot_normal = np.sum(out_d*norms,axis=-1)
                colors[m] += ((light_colors*surf_colors).T*out_d_dot_normal).T
            
        return colors
        
    def glsl(self):
        raise Exception(f'{type(self)} does not implement glsl')

class AmbientLight(Light):
    '''A light that is _everywhere_'''

    def __init__(self,color):
        self.color = A(color)
        
    def pointing(self,pts):
        return None
        
    def illumination(self,pts,pointing):
        return np.tile(self.color,(len(pts),1))
        
    def glsl(self):
        return f'vec3({self.color[0]},{self.color[1]},{self.color[2]})',[]

class DistantLight(Light):
    '''A light that is far from the scene, and comes from a particular direction'''

    def __init__(self,color,direction):
        self.color = A(color)
        self.direction = A(direction)
        
    def pointing(self,pts):
        return np.tile(self.direction,(len(pts),1))
        
    def illumination(self,pts,pointing):
        return np.tile(self.color,(len(pts),1))
        
    def glsl(self):
        direction = f'vec3({self.direction[0]},{self.direction[1]},{self.direction[2]})'
        color = f'vec3({self.color[0]},{self.color[1]},{self.color[2]})'
        light = f'distant_light(p,d,n,{direction},{color})'
        frags = [DistantLight.glsl_function]
        return light,frags
        
    glsl_function = '''
        vec3 distant_light(vec3 p, vec3 d, vec3 n, vec3 d_light, vec3 c_light)  {
            vec3 d_l = normalize(d_light);
            vec3 p_l = p;
            float cosa = dot(d_l,n);
            vec3 g;
            if (cosa > 0. && !next_surface(p_l,d_l,g)) {
                return cosa*c_light;
            }
            return vec3(0.,0.,0.);
        }
    '''
    
class PointLight(Light):
    '''A light that is directional within the scene, and obeys 1/r^2'''

    def __init__(self,color,position):
        self.color = A(color)
        self.position = A(position)
        
    def pointing(self,pts):
        return self.position - pts
        
    def illumination(self,pts,pointing):
        intensity = 1/L(pointing)**2
        return np.outer(intensity,self.color)
    
    
    def glsl(self):
        position = f'vec3({self.position[0]},{self.position[1]},{self.position[2]})'
        color = f'vec3({self.color[0]},{self.color[1]},{self.color[2]})'
        light = f'point_light(p,d,n,{position},{color})'
        frags = [PointLight.glsl_function]
        return light,frags
        
    glsl_function = '''
        vec3 point_light(vec3 p, vec3 d, vec3 n, vec3 p_light, vec3 c_light)  {
            vec3 d_light = p_light-p;
            float dist = length(d_light);
            if (dist > 1e3) return vec3(0.,0.,0.);
            vec3 d_l = d_light/dist;
            vec3 p_l = p;
            float cosa = dot(d_l,n);
            vec3 g;
            if (cosa > 0. && !next_surface(p_l,d_l,g,p_light)) {
                return cosa/(dist*dist)*c_light;
            }
            return vec3(0.,0.,0.);
        }
    '''
    
        
