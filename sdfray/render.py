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

WORLD_MAX = 1000
WORLD_RES = 1e-4
BACKGROUND = A([0,0,0])

def negate(sdf):
    return lambda *args,**kwargs: -sdf(*args,**kwargs)
    
def resolve_transmission(sdf,n1,n2,p,in_d,n):
    '''For each ray, figure out where and in what direction it leaves the interior of a transparent shape'''
    out_d = np.empty_like(in_d)
    out_p = np.copy(p)
    
    nratio = n1/n2
    perp_oblique = np.cross(in_d,n,axis=-1)
    internal = np.square(nratio)*np.sum(perp_oblique*perp_oblique,axis=-1)
    
    #these stay outside
    totaled = internal > 1
    in_d_dot_n = np.sum((in_d_t:=in_d[totaled])*(n_t:=n[totaled]),axis=-1)
    out_d[totaled] = N(in_d_t - (2*in_d_dot_n*n_t.T).T)
    
    #these go inside
    processing = ~totaled
    out_d = np.empty_like(in_d)
    out_d[processing] = (nratio[processing] * np.cross((n_p:=n[processing]),perp_oblique[processing],axis=-1).T - n_p.T*np.sqrt(1-internal[processing])).T
    
    #print('Finding refracted path')
    #find where they come out
    inner_sdf = negate(sdf)
    nratio = 1/nratio
    i = 0
    while np.any(processing): #total internal reflection loop
        i = i+1
        if i > 5:
            #print('Assuming totally internally reflected')
            break
        #print('Refracted:',i)
        inner_lr = Rays(p=out_p[processing],d=out_d[processing])
        out_p[processing],out_g,valid = next_surface(inner_lr,inner_sdf)
        processing[processing] = valid
        out_n = N(out_g[valid])
        nr = nratio[processing]
        perp_oblique = np.cross(out_d[processing],out_n,axis=-1)
        internal = np.square(nr)*np.sum(perp_oblique*perp_oblique,axis=-1)
        tot = internal > 1
        esc = ~tot
        escaped = np.copy(processing)
        escaped[escaped] = esc
        out_d[escaped] = (nr[esc] * np.cross(out_n[esc],perp_oblique[esc],axis=-1).T - out_n[esc].T*np.sqrt(1-internal[esc])).T
        processing[processing] = tot # these keep going
        out_d_dot_out_n = np.sum(out_d[processing]*(out_n_t:=out_n[tot]) ,axis=-1)
        out_d[processing] = N(out_d[processing] - (2*out_d_dot_out_n*out_n_t.T).T)
    return Rays(p=out_p[mask:=~processing],d=out_d[mask]),mask
        
def next_surface(rays,sdf,stop_at=None,lighting=False):
    '''For each ray, return the position and gradient of the next surface intersection in the SDF'''
    p = np.copy(rays.p)
    #print('Intersecting:',len(p))
    alive = np.arange(len(p))
    g_res = np.empty_like(p) if not lighting else False
    intersected = np.zeros(len(p),dtype=bool)
    i = 0
    while len(alive) > 0:
        sd = sdf(p[alive])
        if np.any(m := (sd < 0)):
            sd[m] = np.abs(sd[m])+WORLD_RES
        #if np.any(m := (sd < -WORLD_RES)):
        #    m = ~m
        #    alive = alive[m]
        #    p_alive = p_alive[m]
        #    sd = sd[m]
        if np.any(m := (sd < WORLD_RES)):
            zombie = alive[m]
            g_zombie = G(sdf,p[zombie])
            moving_towards = np.sum(g_zombie*rays.d[zombie],axis=-1) < 0
            dead = zombie[moving_towards]
            if not lighting:
                g_res[dead] = g_zombie[moving_towards]
            intersected[dead] = True
            m[m] = moving_towards
            m = ~m
            alive = alive[m]
            sd = sd[m]
        if np.any(m := L(p[alive]) > WORLD_MAX):
            m = ~m
            alive = alive[m]
            sd = sd[m]
        if stop_at is not None:
            if np.any(m := (np.sum((stop_at-p_alive)*rays.d[alive],axis=-1) < 0)):
                m = ~m
                alive = alive[m]
                p_alive = p_alive[m]
                sd = sd[m]
        i = i+1
        if i > 10000:
            print('TOO MANY STEPS')
            return p,g_res,intersected
        p[alive,:] += (rays.d[alive].T*np.abs(sd)).T
    #print(i,'STEPS')
    return p,g_res,intersected
    
def march_many(rays,sdf,lights,prescale=None):
    '''Run the optical simulation to compute the observed colors along each ray'''
    #print('Projecting foreground')
    p,g,foreground = next_surface(rays,sdf)
    p,g = p[foreground],g[foreground]
    n = N(g)
    in_d = rays.d[foreground]
    prescale = prescale[foreground] if prescale is not None else None
    
    colors = np.zeros((len(foreground),3),dtype=np.float32)
    colors[~foreground] = BACKGROUND
    
    surfs = sdf(p,properties=True)
    
    specular = A([s.specular for s in surfs])
    diffuse = A([s.diffuse for s in surfs])
    transmit = A([s.transmit for s in surfs])
    refractive = A([s.refractive_index for s in surfs])
    
    # Transmission
    transmit_mask = transmit > 0
    if np.any(transmit_mask):
        transmit_scale = transmit[transmit_mask]
        if prescale is None:
            absolute = transmit_scale
        else:
            #print('S',np.mean(transmit_scale),np.max(transmit_scale),np.min(transmit_scale))
            #print('P',np.mean(prescale[transmit_mask]),np.max(prescale[transmit_mask]),np.min(prescale[transmit_mask]))
            absolute = transmit_scale*prescale[transmit_mask]
        #print('A',np.mean(absolute),np.max(absolute),np.min(absolute))
        if np.any(m:=(absolute>1e-3)):   
            transmit_mask[transmit_mask] = m #destructive
            trx_p = p[transmit_mask]
            trx_n = n[transmit_mask]
            trx_in_d = in_d[transmit_mask]
            
            n1 = np.ones(len(trx_p))
            n2 = refractive[transmit_mask]
            
            #print('Calculating transmission')
            tr,valid = resolve_transmission(sdf,n1,n2,trx_p,trx_in_d,trx_n)
            transmit_mask[transmit_mask] = valid
            transmit_colors = march_many(tr,sdf,lights,prescale=absolute[m][valid])
            fg = np.copy(foreground)
            fg[fg] = transmit_mask
            
            colors[fg] += (transmit_colors.T*transmit_scale[m][valid]).T

    # Diffuse reflectivity
    diffuse_mask = diffuse > 0
    if np.any(diffuse_mask):
        diffuse_scale = diffuse[diffuse_mask]
        ref_p = p[diffuse_mask]
        ref_n = n[diffuse_mask]
        ref_surf = surfs[diffuse_mask]
        ref_in_d = in_d[diffuse_mask]
        diffuse_light = np.zeros((len(ref_p),3),dtype=np.float32)
        for li in lights:
            li.light(ref_p,ref_surf,ref_in_d,ref_n,sdf,diffuse_light,lights=lights,prescale=diffuse_scale)
        fg = np.copy(foreground)
        fg[fg] = diffuse_mask
        colors[fg] += (diffuse_light.T*diffuse_scale).T
        
    # Specular reflectivity
    spec_mask = specular > 0
    if np.any(spec_mask):
        specular_scale = specular[spec_mask]
        if prescale is None:
            absolute = specular_scale
        else:
            #print('S',np.mean(specular_scale),np.max(specular_scale),np.min(specular_scale))
            #print('P',np.mean(prescale[spec_mask]),np.max(prescale[spec_mask]),np.min(prescale[spec_mask]))
            absolute = specular_scale*prescale[spec_mask]
        #print('A',np.mean(absolute),np.max(absolute),np.min(absolute))
        if np.any(m:=(absolute>1e-3)):
            spec_mask[spec_mask] = m
            ref_p = p[spec_mask]
            ref_n = n[spec_mask]
            ref_in_d = in_d[spec_mask]
            
            ref_in_d_dot_n = np.sum(ref_in_d*ref_n ,axis=-1)
            ref_d = N(ref_in_d - (2*ref_in_d_dot_n*ref_n.T).T)
            sr = Rays(p=ref_p,d=ref_d)
            #print('Calculating reflection')
            specular_colors = march_many(sr,sdf,lights,prescale=absolute[m])
            fg = np.copy(foreground)
            fg[fg] = spec_mask
            colors[fg] += (specular_colors.T*specular_scale[m]).T
    
    return colors if prescale is not None else np.minimum(colors*255,255).astype(np.uint8)
    
def multipass_antialias(rays,sdf,lights,ang_res,seed):
    '''Simple anti-aliasing algorithm that samples random perturbations around the specified rays'''
    np.random.seed(seed)
    if ang_res is not None and ang_res > 0:
        different = np.empty_like(rays.d)
        m = np.abs(rays.d[:,0])<0.5
        different[m] = A([1,0,0])
        different[~m] = A([0,1,0]);
        perp_1 = np.cross(rays.d,different)
        perp_2 = np.cross(rays.d,perp_1)
        angular_error = np.random.normal(0,ang_res*np.pi/180,len(rays.d))
        phi = np.random.random(len(rays.d))*np.pi*2
        costheta = np.cos(angular_error)
        sintheta = np.sin(angular_error)
        perturbed = ((costheta*rays.d.T)+sintheta*(perp_1.T*np.cos(phi)+perp_2.T*np.sin(phi))).T
        rays = Rays(p=rays.p,d=perturbed)
    return march_many(rays,sdf,lights)
    
    
glsl_core = '''
    //Autogenerated by sdfray

    #ifdef GL_ES
    precision highp float;
    #endif

    uniform vec2 u_resolution;
    uniform vec2 u_mouse;
    uniform float u_time;
    
    struct Property {
        float diffuse, specular, transmit, refractive_index;
        vec3 color, emittance;
    };
    
    struct GeoInfo {
        float sdf;
        Property prop;
    };
    
    GeoInfo wrap(float sdf, Property prop) {
        return GeoInfo(sdf,prop);
    }
    
    GeoInfo wrap(GeoInfo info, Property prop) {
        return GeoInfo(info.sdf,prop);
    }

'''

glsl_tracking = '''
    const float D_ = 1e-4;
    const vec3 DX = vec3(D_,0.0,0.0);
    const vec3 DY = vec3(0.0,D_,0.0);
    const vec3 DZ = vec3(0.0,0.0,D_);
    vec3 gradient(vec3 p) {
        return vec3(sdf(p+DX)-sdf(p-DX),
                    sdf(p+DY)-sdf(p-DY),
                    sdf(p+DZ)-sdf(p-DZ))/(2.0*D_);
    }

    const float WORLD_RES = 1e-4;
    const float WORLD_MAX = 1e4;
    bool next_surface(inout vec3 p, inout vec3 d, out vec3 g, bool inside) {
        for (int i = 0; i < 1000; i++) {
        	float v = inside ? -sdf(p) : sdf(p);
            if (v <= 0.) {
                v = WORLD_RES - v;
            } else if (v < WORLD_RES) {
                g = inside ? -gradient(p) : gradient(p);
                if (dot(g,d) < 0.0) {
                    return true;
                }
            } else if (v > WORLD_MAX) {
                return false;
            }
            p += v*d;
        }
        return false;
    }

    bool next_surface(inout vec3 p, inout vec3 d, out vec3 g) {
        return next_surface(p,d,g,false);
    }

    bool next_surface(inout vec3 p, vec3 d, out vec3 g, vec3 stop_at) {
        for (int i = 0; i < 1000; i++) {
        	float v = sdf(p);
            if (v <= 0.) {
                v = WORLD_RES - v;
            } else if (v < WORLD_RES) {
                g = gradient(p);
                if (dot(g,d) < 0.0) {
                    return true;
                }
            } else if (v > WORLD_MAX) {
                return false;
            } else if (dot(stop_at-p,d) < 0.0) {
                return false;
            }
            p += v*d;
        }
        return false;
    }
    
    bool resolve_transmission(float n1, float n2, inout vec3 p, inout vec3 d, inout vec3 n) {
        float nratio = n1/n2;
        vec3 perp_oblique = cross(d,n);
        float internal = nratio*nratio*dot(perp_oblique,perp_oblique);
        if (internal > 1.) {
            d = reflect(d,n);
            return true;
        } else {
            d = refract(d,n,nratio);
            nratio = 1./nratio;
            for (int i = 0; i < 100; i++) {
                vec3 g;
                bool hit = next_surface(p,d,g,true);
                if (!hit) return false;
                n = normalize(g);
                perp_oblique = cross(d,n);
                internal = nratio*nratio*dot(perp_oblique,perp_oblique);
                if (internal > 1.) {
                    d = reflect(d,n);
                } else {
                    d = refract(d,n,nratio);
                    return true;
                }
            }
        }
        return false;
    }
'''

glsl_render_backtrace = '''
    vec3 cast_ray_bt(vec3 p, vec3 d) {
        float prescale = 1.0;
        vec3 color = vec3(0.0,0.0,0.0);
        float atten = 1.0;
        vec3 p_stack[10];
        vec3 d_stack[10];
        float prescale_stack[10];
        int sp = 0;
        for (int i = 0; i < 1000; i++) {
            vec3 g;
            bool hit = next_surface(p,d,g);
            if (hit) {
                Property s = prop(p,d);
                vec3 n = normalize(g);
		        bool keep_going = false;
                
                color += prescale*s.emittance;
                
                if (s.transmit > 0.0) {
			        float n1 = 1.0;
                    float n2 = s.refractive_index;
                    vec3 ref_p = p;
                    vec3 ref_d = d; 
                    vec3 ref_n = n;
                    bool exited = resolve_transmission(n1,n2,ref_p,ref_d,ref_n);
                    if (exited) {
                		Property ref_s = prop(ref_p,ref_d);
                        color += s.transmit*prescale*ref_s.diffuse*light(ref_p,ref_d,ref_n)*ref_s.color;
                        if (sp < 5) {
                            #define _PUSH(i) if (sp == i) {\\
                                p_stack[i] = ref_p;\\
                                d_stack[i] = ref_d;\\
                                prescale_stack[i] = prescale*s.transmit;\\
                            }
                            #define PUSH(i) else _PUSH(i)
                           _PUSH(0)
                            PUSH(1)
                            PUSH(2)
                            PUSH(3)
                            PUSH(4)
                            PUSH(5)
                            PUSH(6)
                            PUSH(7)
                            PUSH(8)
                            PUSH(9)
                            sp++;
                        }
                    }
                }

                if (s.diffuse > 0.0) {
                    color += prescale*s.diffuse*light(p,d,n)*s.color;
                }

                if (s.specular > 0.0) {
                    prescale = prescale*s.specular;
                    d = reflect(d,n);
                    keep_going = true;
                }
                
		        if (keep_going && prescale > 1e-3) continue;
            }
	        if (sp > 0) {
                sp--;
                #define _POP(i) if (sp == i) {\\
                    p = p_stack[i];\\
                    d = d_stack[i];\\
                    prescale = prescale_stack[i];\\
                }
                #define POP(i) else _POP(i) 
               _POP(0)
                POP(1)  
                POP(2)
                POP(3)
                POP(4)
                POP(5)
                POP(6)
                POP(7)
                POP(8)
                POP(9) 
                continue;
            }
            return color;
        }
    }
'''

glsl_render_raytrace = '''
    vec2 rnd_state;
    void seed_rand(vec2 state) {
        rnd_state = mat2(cos(u_time),-sin(u_time),sin(u_time),cos(u_time))*state;
    }

    float rand() {
        float res = fract(sin(dot(rnd_state, vec2(12.9898,78.233))) * 43758.5453123);
      	rnd_state.x = rnd_state.y;
        rnd_state.y = res;
        return res;
    }
    
    vec3 cast_ray_rt(vec3 p, vec3 d) {
        vec3 prescale = vec3(1.,1.,1.);
        vec3 color = vec3(0.,0.,0.);
        for (int i = 0; i < 1000; i++) {
            vec3 g;
            bool hit = next_surface(p,d,g);
            if (hit) {
                Property s = prop(p,d);
                color += prescale*s.emittance;
                vec3 n = normalize(g);
		        bool keep_going = false;
                
                float rnd = rand();
                if (rnd <= s.transmit) {
			        float n1 = 1.0;
                    float n2 = s.refractive_index;
                    bool exited = resolve_transmission(n1,n2,p,d,n);
                    if (exited) {
                        prescale *= s.transmit;
                		Property ref_s = prop(p,d);
                        color += prescale*ref_s.emittance;
                        keep_going = true;
                    }
                } else if (rnd <= s.diffuse+s.transmit) {
                    prescale *= s.diffuse;\
                    vec3 hemi = n;
                    vec3 diff = abs(hemi.x) < 0.5 ? vec3(1.,0.,0.) : vec3(0.,1.,0.);
                    vec3 p1 = cross(hemi,diff);
                    vec3 p2 = cross(hemi,p1);
                    float phi = 2.*3.141592658*rand();
                    float costheta = rand();
                    float sintheta = sqrt(1.-costheta*costheta);
                    d = costheta*hemi + sintheta*(p1*cos(phi)+p2*sin(phi));
                    keep_going = true;
                } else if (rnd <= s.diffuse+s.transmit+s.specular) {
                    prescale *= s.specular;
                    d = reflect(d,n);
                    keep_going = true;
            	} // else absorbed
                
		        if (keep_going && max(max(prescale.x,prescale.y),prescale.z) > 1e-3) continue;
            }
            return color;
        }
    }
'''
