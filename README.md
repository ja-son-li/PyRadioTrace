# PyTrace
Atmospheric and Ionospheric raytracing 

### Installation 
1. Clone the repo.

   ``git clone git@github.com:ja-son-li/PyRadioTrace.git``
  
3. Install the package (ideally in a virtual environment)

   ``cd PyRadioTrace``
   
   ``pip intall -e .`` 

### General Use
1. Defining enviromental model functions
Every model for defining refractivity from the neutral atmosphere should have the following inputs.
```
test_lat = 30
test_lon = 100
test_alt = 30
n, n_derivatives = neutral(test_lat, test_lon, test_alt)
```
n is the index of refaction and n_derivatives is an array of length 3 defining the spatial derivatives of the index of refraction (r, theta, phi).
For ionospheric models, the output is the electron density (in count/m^3) and its spatial derivatives. 
For the purposes of the repo, a simple scale height function is provided. 

2. Raytracing
   
Instantiate a raytracer object with the environmental model functions.
``Raytrace  = Raytracer(iono = iono, magneto = None, neutral = neutral)``
Call the raytracer for a specific pointing direction.
```
rtol = 1e-10
atol = 1e-8
transmit_lat = 0
transmit_lon = 0.5
transmit_alt = 800e3
f = 1575.42e6
az = 90
el = -10
distances_to_evaluate = np.arange(0, 100e3, 1)

ray_solution = Raytrace.ray_propagate(transmit_lat,
                                    transmit_lon, 
                                    transmit_alt,
                                    az,
                                    el,
                                    f, 
                                    group_path_distances = distances_to_evaluate,
                                    rtol = rtol,
                                    atol = atol)
```
3. Rayhoming

```
receive_lat = 0
receive_lon = 105
receive_alt = 20200e3
solution = Raytrace.ray_home_onedim(transmit_lat,
                                    transmit_lon,
                                    transmit_alt,
                                    receive_lat, 
                                    receive_lon,
                                    receive_alt,
                                    f, xatol=1e-10, resolution_m = 0.1, rtol = rtol, atol = atol)
```

### Examples
See examples folder
