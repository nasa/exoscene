# exoscene

Installation: `pip install exoscene`

**exoscene** is a library of classes and utility functions for simulating
direct images of exoplanetary systems. The package was developed by Neil
Zimmerman (NASA/GSFC), with source code contributions from Maxime Rizzo,
Christopher Stark, and Ell Bogat. This work was funded in part by a WFIRST/Roman Science
Investigation Team contract (PI: Margaret Turnbull). 

**exoscene** makes significant use of the Astropy, NumPy, SciPy, and 
Scikit-image packages.

A jupyter notebook providing usage examples for much of the functionality is included under the docs subdirectory:
[exoscene/docs/notebooks/Roman-CGI_scene_demo.ipynb](exoscene/docs/notebooks/Roman-CGI_scene_demo.ipynb)

The functions are organized in 3 modules: [exoscene/planet.py](exoscene/planet.py), 
[exoscene/star.py](exoscene/star.py), and [exoscene/image.py](exoscene/image.py).

## 1. [exoscene/planet.py](exoscene/planet.py)

* a Planet() class with a data structure for containing the basic physical 
parameters of a planet, its orbit, its host star, and associated methods for
computing its relative astrometry ephemeris, its phase function, and flux ratio.

* A function for modeling the orbital position and the Lambert sphere phase function,
based on the Keplerian orbital elements and date of observation.

* A function for mapping the time-dependent sky-projected position and 
Lambert phase factor.

## 2. [exoscene/star.py](exoscene/star.py)

* Functions for computing the band-integrated irradiance of a star based on its 
apparent magnitude and spectral type, and instrument bandpass, using the built-in 
Bruzual-Persson-Gunn-Stryker (BPGS) Spectral Atlas (under 
[exoscene/data/bpgs/](exoscene/data/bpgs/))

* A function for computing the approximate parallax and proper motion offset for 
a star, based on the celestial coordinates and observing dates.

## 3. [exoscene/image.py](exoscene/image.py)

* A function for accurately resampling an image model array to a detector array.

* Functions for translating a coronagraph PSF model to an arbitrary field point,
taking into account position-dependent properties included in the model.

* Functions for applying a noise model to a detector intensity map, to simulate
an image with photon counting noise, read noise, and dark current, for a given 
integration time.

Copyright © 2020 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Other Rights Reserved.
