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
import re
        
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
        self.screen_width = screen_width
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

def deduplicate(fragments):
    included = set()
    result = []
    for f in fragments:
        if f in included:
            continue
        included.add(f)
        result.append(f)
    return result

class Scene:
    '''A renderable scene consists of a SDF geometry, some number of lights, and 
       a Camera to specifiy what perspective to render'''
    def __init__(self,sdf,lights,cam=Camera()):
        self.sdf = sdf
        self.lights = lights
        self.cam = cam
        
    def cpu_render(self,antialias=None,ang_res=0.02):
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
            
    
    def render(self,ctx=None,antialias=None,ang_res=0.02):
        import moderngl
        if ctx is None:
            ctx = moderngl.create_standalone_context()
        vertex_shader = '''
            in vec2 position;
            void main() {
                gl_Position = vec4(position.xy,0.,1.);
            }
        '''
        fragment_shader = self.glsl()
        
        glpg = ctx.program(vertex_shader=vertex_shader,fragment_shader=fragment_shader)
        
        data = np.asarray([-1,1,-1,-1,1,1,1,-1],dtype=np.float32)
        vbo = ctx.buffer(data.tobytes())
        res = (self.cam.width_px,self.cam.height_px)
        glpg['u_resolution'] = res
        glpg['u_time'] = 0.

        vao = ctx.simple_vertex_array(glpg, vbo, 'position')
        fbo = ctx.simple_framebuffer(res)
        fbo.use()
        fbo.clear(0.,0.,0.,1.)
        vao.render(moderngl.TRIANGLE_STRIP)
        return Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1)
        
            
    def glsl(self):
        geo,prop,scafolding = self.sdf.glsl()
        
        geo = f'''
            float sdf(vec3 p) {{
                return {geo};
            }}
        '''
        
        prop = f'''
            Property prop(vec3 p, vec3 d) {{
                return {prop}.prop;
            }}
        '''
        
        lights = []
        lighting = []
        for li in self.lights:
            light,frags = li.glsl()
            lighting.extend(frags)
            lights.append(light)
        
        light = '''
            vec3 light(vec3 p, vec3 d, vec3 n) {
                vec3 color = vec3(0.0,0.0,0.0);'''+('\n'.join([f'''
                color += {li};''' for li in lights]))+'''
                return color;
            }
        '''
        
        entrypoint = f'''
            void main() {{
                vec2 st = gl_FragCoord.xy/u_resolution;
                //seed_rand(st);
                float ratio = u_resolution.x/u_resolution.y;
                if (ratio >= 1.0) {{
                    st.x *= ratio;
                    st.x += (1. - ratio)/2.;
                }} else {{
                    st.y /= ratio;
                    st.y += (1. - 1./ratio)/2.;
                }}
                st = {float(self.cam.screen_width)}*(st-0.5)/2.;
                
                float view_dist = 0.1;
                vec3 px_cam = vec3(st.x,st.y,{self.cam.viewing_dist});
                
                /*
                float cost = cos(0.4*cos(u_time/2.));
                float sint = sin(0.4*cos(u_time/2.));
                */
                float cost = cos(u_time/2.);
                float sint = sin(u_time/2.);

                mat3 cam_proj = mat3(cost,0.0,-sint,0.0,1.0,0.0,sint,0.0,cost);
                vec3 cam_orig = vec3(-8.0*sint,0.0,-8.*cost);
                
                vec3 color = vec3(0.0,0.0,0.0);
                const int passes = 1;
                for (int i = 0; i < passes; i++) {{
                    //color += cast_ray_rt(cam_orig,normalize(cam_proj*px_cam))/float(passes);
                    color += cast_ray_bt(cam_orig,normalize(cam_proj*px_cam))/float(passes);
                }}
                gl_FragColor = vec4(color,1.0);
            }}
        '''
        
        scafolding = deduplicate(scafolding)
        lighting = deduplicate(lighting)
        elements = [glsl_core]+scafolding+[geo,prop,glsl_tracking]+lighting+[light,glsl_render_backtrace,entrypoint]
        r = re.compile(r'( +).*')
        def process(elem):
            elems = elem.split('\n')
            for e in elems:
                if len(e.strip()) > 0:
                    if (m:=r.match(e)):
                        prefix = m.group(1)
                    else:
                        prefix = ''
                    break
            rep = re.compile(f'^{prefix}')
            elems = [rep.sub('',e) for e in elems]
            return '\n'.join(elems)
        fragment_shader = ''.join([process(e) for e in elements])
        return fragment_shader

            


