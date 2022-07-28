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
