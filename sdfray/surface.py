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
import numpy as np

class SurfaceProp:
    '''Defines properties of a surface'''
    
    def __init__(self,color=A([1.0,1.0,1.0]),diffuse=0.8,specular=0.0,refractive_index=1.0,transmit=0.0,emittance=A([0.0,0.0,0.0])):
        '''Most quantities are multipliers for effects. [0,1] is realistic, but not required
           Refractive index is the usual way, where the speed of light in the medium is c/n
           Default surface is 80% diffuse reflective'''
        self.diffuse = diffuse
        self.specular = specular
        self.refractive_index = refractive_index
        self.transmit = transmit
        self.color = color
        self.emittance = emittance
        self.name = f'var_{hash((diffuse,specular,refractive_index,transmit,tuple(color),tuple(emittance))) % 100000000}'
        
    def glsl(self):
        color = f'vec3({self.color[0]},{self.color[1]},{self.color[2]})'
        emittance = f'vec3({self.emittance[0]},{self.emittance[1]},{self.emittance[2]})'
        frags = [f'''
            const Property {self.name} = Property({self.diffuse},{self.specular},{self.transmit},{self.refractive_index},{color},{emittance});
        ''']
        return f'{self.name}',frags
        
class Surface:
    '''A paramaterized surface that returns SurfaceProp as a function of position'''
    
    def __init__(self):
        pass
        
    def __call__(self,pts):
        return self.fn(pts)
        
    def fn(self,pts):
        raise Exception('Use a Surface implementation, instead!')
        
    def glsl(self):
        raise Exception(f'{type(self)} does not implement glsl')
        
    
class UniformSurface(Surface):
    '''A surface that has the same properties everywhere'''
    
    def __init__(self,prop,**kwargs):
        super().__init__(**kwargs)
        self.prop = prop
        
    def fn(self,pts):
        return np.full(len(pts),self.prop)
        
    def glsl(self):
        prop,frags = self.prop.glsl()
        frags.append(UniformSurface.glsl_function)
        return f'uniform_surf(p,d,{prop})',frags
        
    glsl_function = '''
        Property uniform_surf(vec3 p, vec3 d, Property surf) {
            return surf;
        }
    '''

class CheckerSurface(Surface):
    '''The standard graphics demo checker pattern'''

    def __init__(self,checker_size=1.0,a_v=[1,0,0],a=SurfaceProp(diffuse=0.95),b_v=[0,0,1],b=SurfaceProp(diffuse=0.05)):
        self.checker_size = checker_size
        self.a = a
        self.b = b
        self.a_v = N(A(a_v))
        self.b_v = N(A(b_v))
        
    def fn(self,pts):
        a_c = np.sum(pts*self.a_v,axis=-1)
        b_c = np.sum(pts*self.b_v,axis=-1)
        a_c = np.mod(a_c,2*self.checker_size)
        b_c = np.mod(b_c,2*self.checker_size)
        a_odd = a_c >= self.checker_size
        b_odd = b_c >= self.checker_size
        return np.where(a_odd == b_odd,self.a,self.b)
        
    def glsl(self): #FIXME
        aprop,afrags = self.a.glsl()
        bprop,bfrags = self.b.glsl()
        frags = afrags+bfrags+[CheckerSurface.glsl_function]
        return f'uniform_surf(p,d,{prop})',frags
        
    glsl_function = '''
        Property checker_surf(vec3 p, vec3 d, vec3 anchor, vec3 norm, float checker_size, Property a, Property b) {
            bool a_odd = mod(p.x,2.*checker_size) >= checker_size;
            bool b_odd = mod(p.z,2.*checker_size) >= checker_size;
            if (a_odd == b_odd) {
                return a;
            } else {
                return b;
            }
        }
    ''' #FIXME
        
class LimbDarkening(Surface):
    def __init__(self,emittance=[1,0.7,0]):
        self.emittance = A(emittance)
    def fn(self,pts,dirs):
        perpness = np.sum(-N(pts)*dirs,axis=-1) #only works for spheres... [0,1]
        return np.asarray([SurfaceProp(emittance=x**0.7*self.emittance) for x in perpness])
        
class PerlinSurface(Surface):
    def __init__(self,emittance,length_scale=1.0,feature_count=10):
        self.emittance = emittance
        self.length_scale = length_scale
        self.feature_count = feature_count
        lattice_points = feature_count**3
        self.lattice_vecs = N(np.random.normal(0,1,size=lattice_points*3).reshape((feature_count,feature_count,feature_count,3)))
        #fixups
        self.lattice_vecs[-1,:,:] = self.lattice_vecs[0,:,:]
        self.lattice_vecs[:,-1,:] = self.lattice_vecs[:,0,:]
        self.lattice_vecs[:,:,-1] = self.lattice_vecs[:,:,0]
        self.lattice_vecs = self.lattice_vecs.reshape(lattice_points,3)
        self.dim = np.linspace(0,length_scale,feature_count)
        self.lattice = A([(self.dim[i],self.dim[j],self.dim[k]) for i in range(feature_count) for j in range(feature_count) for k in range(feature_count)])
        self.idx = np.arange(feature_count)
        self.indexer = A([1,feature_count,feature_count*feature_count],np.int32)
        
    def fn(self,pts,dirs):
        p_mod = np.mod(pts,self.length_scale)
        p_lat = np.interp(p_mod,self.dim,self.idx)
        p_cell,p_lat_l = np.modf(p_lat)
        p_lat_l = p_lat_l.astype(np.int32)
        dx = 1
        dy = self.feature_count
        dz = self.feature_count**2
        a = np.sum(p_lat_l*self.indexer,axis=-1)
        b = a+dx
        c = a+dy
        d = a+dx+dy
        e = a+dz
        f = a+dx+dz
        g = a+dy+dz
        h = a+dx+dy+dz
        
        a = self.vec_helper(pts,a,b,p_cell[:,0])
        b = self.vec_helper(pts,c,d,p_cell[:,0])
        c = self.vec_helper(pts,e,f,p_cell[:,0])
        d = self.vec_helper(pts,g,h,p_cell[:,0])
        
        a = self.interpolate(a,b,p_cell[:,1])
        b = self.interpolate(c,d,p_cell[:,1])
        
        result = self.interpolate(a,b,p_cell[:,2])
        result = np.maximum(result/4/np.sqrt(2)+0.5,0) # 0-1
        result = result*0.5+0.5
        #print(np.min(result),np.mean(result),np.max(result))
        return np.asarray([SurfaceProp(emittance=r*self.emittance) for r in result])
    
    def vec_helper(self,pts,a,b,frac):
        a = np.sum(self.lattice_vecs[a]*(pts-self.lattice[a]),axis=-1)
        b = np.sum(self.lattice_vecs[b]*(pts-self.lattice[b]),axis=-1)
        return self.interpolate(a,b,frac)
    
    def interpolate(self,a,b,frac):
        #return a*frac+b*(1.0-frac)
        return (b - a) * (3.0 - frac * 2.0) * frac * frac + a;
        
def _sphere_to_cube(xyz):
    '''ratio between the distance along a line from the center to the surface of a unit cube
       in the same direction as a given ray to the surface of a unit sphere.'''
    return np.sqrt(np.sum(np.square(xyz),axis=-1))/np.max(np.abs(xyz),axis=-1)

def sphere_to_cube(theta,phi):
    '''spherical coordinates theta,phi to x,y,z positions on a unit cube'''
    xyz = A([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).T
    return xyz*_sphere_to_cube(xyz)

class SphereCubeMap(Surface):
    def __init__(self,cube_map):
        self.cube_map = A(np.asarray(cube_map)[:,:,:3])/255
    def fn(self,pts,dirs):
        xyz = 512*(pts.T*_sphere_to_cube(pts)).T
        xyz[xyz>512] = 512
        xyz[xyz<-512] = -512
        print(np.min(xyz),np.max(xyz))
        axis = np.argmax(np.abs(xyz),axis=-1)
        pos = np.sign(np.take_along_axis(xyz,np.expand_dims(axis, axis=-1), axis=-1).squeeze()) > 0
        xy = np.empty((len(xyz),2))
        if np.any(m_front := (axis == 1) & pos):
            xy[m_front] = xyz[:,[0,2]][m_front] + A([0,1024])
        if np.any(m_right := (axis == 0) & pos):
            xy[m_right] = xyz[:,[1,2]][m_right] + A([1024,1024])
        if np.any(m_back := (axis == 1) & ~pos):
            xy[m_back] = xyz[:,[0,2]][m_back] + A([2048,1024])
        if np.any(m_left := (axis == 0) & ~pos):
            xy[m_left] = xyz[:,[1,2]][m_left] + A([3096,1024])
        if np.any(m_top := (axis == 2) & pos):
            xy[m_top] = xyz[:,[1,0]][m_top] + A([2048,0])
        if np.any(m_bot := (axis == 2) & ~pos):
            xy[m_bot] = xyz[:,[1,0]][m_bot] + A([0,2048])

        pixels = np.floor(xy+512).astype(np.uint32)
        return np.asarray([SurfaceProp(diffuse=0,specular=0,emittance=e) for e in self.cube_map[pixels[:,1],pixels[:,0]]])
