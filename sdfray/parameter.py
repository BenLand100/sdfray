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
import math
    
class Parameter:
    ''' This class represents a frame-by-frame runtime value and loosely masquerades
        as a floating point number from the POV of the Numpy API and Python math 
        operators. Create them within a Context, which also manages their values 
        when evaluating on a CPU.'''
        
    def __init__(self,ctx,contents,noparen=False):
        self.ctx = ctx
        self.contents = tuple(contents) if isinstance(contents,list) else contents
        self.noparen = noparen
        
    def __str__(self):
        match self.contents:
            case str() as s:
                return s
            case int() as i:
                return f'{i}.'
            case float() as f:
                return f'{f}'
            case con:
                if self.noparen:
                    return ' '.join([str(c) for c in con])
                else:
                    return '(' + ' '.join([str(c) for c in con]) + ')'
                
                
    def __repr__(self):
        return f'<Parameter {str(self)}>'    
            
    def wrap(self,o):
        if isinstance(o,Parameter):
            assert o.ctx == self.ctx, 'Cannot mix parameter contexts!'
            return o
        else:
            return Parameter(self.ctx,o)
            
    def __float__(self):
        return eval(str(self),self.ctx.globals)
        
    def __eq__(self,o):
        if isinstance(o,Parameter):
            return self.contents == o.contents
        else:
            return self.contents == o
            
    def __req__(self,o):
        if isinstance(o,Parameter):
            return self.contents == o.contents
        else:
            return self.contents == o
            
    def __hash__(self):
        return hash(self.contents)^hash(self.noparen)
    
    def __add__(self, o):
        if o == 0:
            return self
        if self == 0:
            return o
        return Parameter(self.ctx,[self,'+',self.wrap(o)])
        
    def __radd__(self, o):
        if o == 0:
            return self
        if self == 0:
            return o
        return Parameter(self.ctx,[self.wrap(o),'+',self])
    
    def __sub__(self, o):
        if o == 0:
            return self
        if self == 0:
            return o
        return Parameter(self.ctx,[self,'-',self.wrap(o)])
        
    def __rsub__(self, o):
        if o == 0:
            return self
        if self == 0:
            return o
        return Parameter(self.ctx,[self.wrap(o),'-',self])
    
    def __mul__(self, o):
        if o == 0 or self == 0:
            return Parameter(self.ctx,0)
        if o == 1:
            return self
        if self == 1:
            return o
        return Parameter(self.ctx,[self,'*',self.wrap(o)],noparen=True)
        
    def __rmul__(self, o):
        if o == 0 or self == 0:
            return Parameter(self.ctx,0)
        if o == 1:
            return self
        if self == 1:
            return o
        return Parameter(self.ctx,[self.wrap(o),'*',self],noparen=True)
    
    def __div__(self, o):
        if o == 1:
            return self
        return Parameter(self.ctx,[self,'/',self.wrap(o)])
    __truediv__ = __div__
        
    def __rdiv__(self, o):
        if self == 1:
            return o
        return Parameter(self.ctx,[self.wrap(o),'/',self])
    __rtruediv__ = __rdiv__
    
    def __neg__(self):
        return Parameter(self.ctx,['-',self],noparen=True)
        
    def __abs__(self):
        return Parameter(self.ctx,['abs(',self,')'],noparen=True)
        
    def cos(self):
        return Parameter(self.ctx,['cos(',self,')'],noparen=True)
        
    def sin(self):
        return Parameter(self.ctx,['sin(',self,')'],noparen=True)
        
    def tan(self):
        return Parameter(self.ctx,['tan(',self,')'],noparen=True)
        
    def sqrt(self):
        return Parameter(self.ctx,['sqrt(',self,')'],noparen=True)
        
    def square(self):
        return Parameter(self.ctx,['pow(',self,',2.)'],noparen=True)
        
    def __pow__(self,o):
        return Parameter(self.ctx,['pow(',self,',',self.wrap(o),')'],noparen=True)
        
    def __rpow__(self,o):
        return Parameter(self.ctx,['pow(',self.wrap(o),',',self,')'],noparen=True)
        
class Context:
    ''' A collection of runtime frame-by-frame parameters. '''

    def __init__(self):
        self.globals = dict( #required functions for eval
            abs=np.abs,
            sin=np.sin,
            cos=np.cos,
            tan=np.tan,
            sqrt=np.sqrt,
            pow=np.power)
    
    def __getitem__(self,key):
        if key not in self.globals:
            self.globals[key] = 0.
        return Parameter(self,key)
            
    def __setitem__(self,key,value):
        self.globals[key] = value
