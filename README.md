# SDFRay

`sdfray` started as a python package for doing high quality rendering of 
geometries described by signed distance functions on a CPU using pure Python 
w/ Numpy, because I wanted to learn more about rendering techniques, and why
signed distance functions were not as widely-known as triangular meshes. 

This capability still exists, for reasons of curiousity, while the default
rendering method now is to generate GLSL shader code to render scenes, which are 
compiled and run by [ModernGL](https://moderngl.readthedocs.io/en/latest/). 

I plan to slowly add to this until it's making hyper-realistic renders.

## Features

Beta/WIP; capabilities and examples are in Jupyter notebooks.

* Primitive objects for spheres, boxes, cylinders, ...
* Constructive solid geometry support for subtraction, union, intersection
* Arbitrary translations and rotations
* Transmission (refraction) and reflection (diffuse, specular)
* Real time backtracking of optical paths
* Full support for shadows (some limitations w.r.t. occlusion by transmissive objects)
* Alternate rendering engine that does full reverse raytracing with 

## Special Thanks

[Inigo Quilez](https://iquilezles.org/articles/distfunctions/) has an excellent
blog about signed distance functions and related topics. Getting a PhD in 
experimental particle physics also really helped understanding how light works
well enough to homebrew a rendering engine.

## Copying

Copyright 2022 Benjamin Land (BenLand100)

Released under GPLv3

