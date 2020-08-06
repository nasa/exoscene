# exoscene

**exoscene** is a library of classes and utility functions for simulating direct 
images of exoplanetary systems.

A jupyter notebook providing usage examples for much of the functionality is included under the docs subdirectory:
[exoscene/docs/notebooks/Roman-CGI_scene_demo.ipynb](exoscene/docs/notebooks/Roman-CGI_scene_demo.ipynb)

The functions are organized in 3 modules: [exoscene/planet.py](exoscene/planet.py), 
[exoscene/star.py](exoscene/star.py), and [exoscene/image.py](exoscene/image.py).

## 1. planet.py

* a Planet() class with a simple data structure for containing the basic physical 
parameters of a planet, its orbit, its host star, and methods for computing its 
relative astrometry ephemeris, its phase function, and flux ratio.

* A function for modeling the orbital position and Lambert sphere phase based
on the Keplerian orbital elements and the date defined in the Planet class 
attributes.

* A function for mapping the time-dependent sky-projected position and 
Lambert phase factor.


