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
from scipy.spatial.transform import Rotation as R
from .render import *
from functools import partial
from PIL import Image
import numpy as np
        
class Camera:
    '''Abstracts a camera or viewer as a collection of rays through a viewscreen'''
    def __init__(self,
                 width_px=500, 
                 aspect_ratio = 1.618,    
                 screen_width = 0.1,
                 viewing_dist = 0.1,
                 camera_orig = A([0,0,-10]),
                 camera_pitch = 0,
                 camera_yaw = 0,
                 camera_roll = 0):
    
        self.width_px = width_px
        self.height_px = int(round(width_px/aspect_ratio))
        self.viewing_dist = viewing_dist
        x = np.linspace(-0.5,0.5,self.width_px)
        y = np.linspace(-1/2/aspect_ratio,1/2/aspect_ratio,self.height_px)
        self.X,self.Y = np.meshgrid(x,-y)
        
        self.pixel_locations_cam = A([screen_width/2*self.X.flatten(),
                                      screen_width/2*self.Y.flatten(),
                                      np.full(self.width_px*self.height_px,self.viewing_dist)]).T
        self.camera_orig = camera_orig
        self.camera_pitch = camera_pitch
        self.camera_yaw = camera_yaw
        self.camera_roll = camera_roll
        self.adjust()
        
    def adjust(self,
               camera_orig = None,
               camera_pitch = None,
               camera_yaw = None,
               camera_roll = None):
        '''For moving a camera orientation, and recomputing rays'''
        
        if camera_orig is not None:
            self.camera_orig = camera_orig
        if camera_pitch is not None:
            self.camera_pitch = camera_pitch
        if camera_yaw is not None:
            self.camera_yaw = camera_yaw
        if camera_roll is not None:
            self.camera_roll = camera_roll
        
        self.proj = R.from_euler(seq='XYZ',angles=[self.camera_pitch,self.camera_yaw,self.camera_roll]).as_matrix()

        self.pixel_directions_world = (self.proj @ self.pixel_locations_cam.T).T

        self.rays = Rays(p=np.tile(self.camera_orig,(len(self.pixel_directions_world),1)),d=N(self.pixel_directions_world))

class Scene:
    '''A renderable scene consists of a SDF geometry, some number of lights, and 
       a Camera to specifiy what perspective to render'''
    def __init__(self,sdf,lights,cam=Camera()):
        self.sdf = sdf
        self.lights = lights
        self.cam = cam
        
    def render(self,antialias=None,ang_res=0.02):
        '''Heavy lifting is done in the `render` module'''
        out_shape = (self.cam.height_px,self.cam.width_px,3)
        if antialias is not None:
            colors = np.zeros(out_shape,dtype=np.uint32)
            fn = partial(multipass_antialias,self.cam.rays,self.sdf,self.lights,ang_res)
            seeds = np.random.randint(2**32,size=antialias,dtype=np.uint64).astype(np.int64)
            for i,c in enumerate(map(fn,seeds)):
                colors += c.reshape(out_shape)
            return Image.fromarray((colors/antialias).astype(np.uint8))
        else:
            return Image.fromarray(march_many(self.cam.rays,self.sdf,self.lights).reshape(out_shape))

