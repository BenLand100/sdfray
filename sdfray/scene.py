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
from .light import PointLight
from .shapes import Sphere
from .geom import Union
from .surface import UniformSurface,SurfaceProp
from functools import partial
from PIL import Image
import numpy as np
import re
        
class Camera:
    '''Abstracts a camera or viewer as a collection of rays through a viewscreen'''
    def __init__(self,
                 width_px=500, 
                 height_px=None,
                 aspect_ratio = 1.618,    
                 screen_width = 0.1,
                 viewing_dist = 0.1,
                 camera_orig = A([0,0,-10]),
                 camera_pitch = 0,
                 camera_yaw = 0,
                 camera_roll = 0):
    
        self.width_px = width_px
        self.height_px = height_px if height_px is not None else int(round(width_px/aspect_ratio))
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
        
        self.proj = ZROT(self.camera_roll)@YROT(self.camera_yaw)@XROT(self.camera_pitch)
        #self.proj = R.from_euler(seq='XYZ',angles=[self.camera_pitch,self.camera_yaw,self.camera_roll]).as_matrix()

        self.pixel_directions_world = (np.asarray(self.proj,dtype=np.float64) @ np.asarray(self.pixel_locations_cam,dtype=np.float64).T).T

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
        self._glpg = None
        self._res = None
        self._ctx = None
        
    def cpu_render(self,antialias=None,ang_res=0.):
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
            
    def clear_cache(self):
        self._glpg = None
        self._vbo = None
        self._fbo = None
        self._res = None
    
    def render(self,antialias=None,time=0.,passes=1,batching=10,**kwargs):
        import moderngl
        if self._ctx is None:
            ctx = moderngl.create_standalone_context()
            self._ctx = ctx
        else:
            ctx = self._ctx
        vertex_shader = '''
            in vec2 position;
            void main() {
                gl_Position = vec4(position.xy,0.,1.);
            }
        '''
        res = (self.cam.width_px,self.cam.height_px)
        if self._glpg is None:
            fragment_shader = self.glsl(**kwargs)
            try:
                self._glpg  = ctx.program(vertex_shader=vertex_shader,fragment_shader=fragment_shader)
            except:
                print('\n'.join([f'{i:04} {l}' for i,l in enumerate(fragment_shader.split('\n'))]))
                raise
        
            data = np.asarray([-1,1,-1,-1,1,1,1,-1],dtype=np.float32)
            self._vbo = ctx.buffer(data.tobytes())
        if self._res is None or self._res != res:
            self._vao = ctx.simple_vertex_array(self._glpg, self._vbo, 'position')
            self._fbo = ctx.simple_framebuffer(res, dtype='f4')
            self._res = res

        self._glpg['u_resolution'] = res
        try:
            self._glpg['u_time'] = time
        except:
            pass #nothing using u_time
        self._fbo.use()
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.ONE, moderngl.ONE
        self._ctx.disable(moderngl.DEPTH_TEST)
        loops = (passes-1) // batching + 1
        img = np.zeros((self._res[1],self._res[0],3),dtype=np.float64)
        for l in range(loops):
            iters = min(passes,batching)
            self._glpg['u_alpha'] = 1/iters;
            self._fbo.clear(0.,0.,0.,1.)
            for i in range(iters):
                try:
                    self._glpg['u_nonce'] = 100*np.random.random()
                except:
                    pass
                self._vao.render(moderngl.TRIANGLE_STRIP)
            frame = np.frombuffer(self._fbo.read(dtype='f4'), dtype=np.float32).reshape((self._res[1],self._res[0],3))
            m = iters/batching
            img = (l*img + m*frame)/(l+m)
            passes = max(0,passes-batching)
        return Image.fromarray(np.uint8(np.minimum(np.flip(img,0),1.0)*255))
        
            
    def glsl(self,ang_res=0.,true_optics=False):
        
        if true_optics:
            renderer = 'vec3 color = cast_ray_rt(cam_orig,normalize(cam_proj*px_cam_i));'
            raycaster = glsl_render_raytrace
            light = ''
            lighting = []
            sdf = self.sdf
            for li in self.lights:
                if isinstance(li,PointLight):
                    obj = Sphere(translate=li.position,radius=0.5,surface=UniformSurface(SurfaceProp(diffuse=0.,specular=0.,transmit=0.,emittance=li.color)))
                    sdf = Union(obj,sdf)
        else:
            renderer = 'vec3 color = cast_ray_bt(cam_orig,normalize(cam_proj*px_cam_i));'
            raycaster = glsl_render_backtrace      
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
            sdf = self.sdf
            
        geo,prop,scafolding = sdf.glsl()
        
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
        
        entrypoint = f'''
            void main() {{
                vec2 st = gl_FragCoord.xy/u_resolution;
                seed_rand(st);
                float ratio = u_resolution.x/u_resolution.y;
                if (ratio >= 1.0) {{
                    st.x *= ratio;
                    st.x += (1. - ratio)/2.;
                }} else {{
                    st.y /= ratio;
                    st.y += (1. - 1./ratio)/2.;
                }}
                st = {float(self.cam.screen_width)}*(st-0.5)/2.;
                
                vec3 px_cam = vec3(st.x,st.y,{self.cam.viewing_dist});
                const float ang_res = {ang_res};
                
                mat3 cam_proj = {glsl_mat3(self.cam.proj)};
                vec3 cam_orig = {glsl_vec3(self.cam.camera_orig)};
                
                vec3 px_cam_i;
                if (ang_res > 0.) {{
                    vec3 different;
                    if (abs(px_cam.x) > 0.5) {{
                        different = vec3(0.,1.,0.);
                    }} else {{
                        different = vec3(1.,0.,0.);
                    }}
                    vec3 p1 = cross(px_cam,different);
                    vec3 p2 = cross(px_cam,p1);
                    float theta = rand_normal()*ang_res*3.14159/180.;
                    float phi = rand()*2.*3.14159;
                    float cth = cos(theta);
                    float sth = sin(theta);
                    px_cam_i = cos(theta)*px_cam + sin(theta)*(p1*cos(phi) + p2*sin(phi));
                }} else {{
                    px_cam_i = px_cam;
                }}
                {renderer}
                gl_FragColor = vec4(u_alpha*color,1.);
            }}
        '''
        
        scafolding = deduplicate(scafolding)
        lighting = deduplicate(lighting)
        elements = [glsl_core]+scafolding+[geo,prop,glsl_tracking]+lighting+[light,raycaster,entrypoint]
        r = re.compile(r'( +).*')
        def process(elem):
            elems = elem.split('\n')
            prefix = ''
            for e in elems:
                if len(e.strip()) > 0:
                    if (m:=r.match(e)):
                        prefix = m.group(1)
                    break
            rep = re.compile(f'^{prefix}')
            elems = [rep.sub('',e) for e in elems]
            return '\n'.join(elems)
        fragment_shader = ''.join([process(e) for e in elements])
        return fragment_shader

            


