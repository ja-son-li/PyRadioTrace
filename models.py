import PyIRI
import numpy as np 
import os 
import pyproj
import xarray as xr 
import datetime as dt
import ppigrf
import pymsis
import pymap3d as pm

class GeoModel():
    def __init__(self):
        pass

    @staticmethod
    def convert_lla_to_ecef(lat, lon, alt):
        """
        Vectorized conversion from latitude, longitude, altitude (in meters) to ECEF coordinates.
        """
        transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
        x, y, z = transformer.transform(lon, lat, alt)
        return x, y, z
    # @staticmethod
    # def convert_ecef_to_lla(x, y, z):
    #     """
    #     Vectorized conversion from ECEF coordinates to latitude, longitude, and altitude.
    #     """
    #     transformer = pyproj.Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
    #     lat, lon, alt = transformer.transform(x, y, z)
    #     return lat, lon, alt
    @staticmethod
    def geodetic2aer(lat_tx, lon_tx, alt_tx, lat_rx, lon_rx, alt_rx):
        """
        Converts geodetic coordinates to azimuth, elevation, and range.
        """
        az, el, range_val = pm.geodetic2aer(lat_rx, lon_rx, alt_rx, lat_tx, lon_tx, alt_tx)
        
        return az, el, range_val

    @staticmethod
    def aer_to_ecef(lat_obs, lon_obs, alt_obs, az, el, range_val):
        """
        Converts azimuth, elevation, range to ECEF coordinates.

        Args:
            lat_obs: Observer latitude (degrees).
            lon_obs: Observer longitude (degrees).
            alt_obs: Observer altitude (meters).
            az: Azimuth (degrees).
            el: Elevation (degrees).
            range_val: Range (meters).

        Returns:
            Tuple: (x, y, z) ECEF coordinates (meters).
        """
        x, y, z = pm.aer2ecef(az, el, range_val, lat_obs, lon_obs, alt_obs)
        return x, y, z
    @staticmethod
    def aer_to_lla_ranges(lat_obs, lon_obs, alt_obs, az, el, ranges):
        
        lats = np.full_like(ranges,np.nan)
        lons = np.full_like(ranges,np.nan)
        alts = np.full_like(ranges,np.nan)

        for ind, slant_range in enumerate(ranges):

            X_plus, Y_plus, Z_plus = GeoModel.aer_to_ecef(lat_obs, lon_obs, alt_obs, az, el, slant_range)
            straight_lat,  straight_lon, straight_alt = GeoModel.convert_ecef_to_lla(X_plus, Y_plus, Z_plus)
            lats[ind] = straight_lat 
            lons[ind] = straight_lon 
            alts[ind] = straight_alt

        return lats, lons, alts
    
    def wgs84_to_ecef(lat, lon, alt):
        """
        Converts WGS 84 coordinates (latitude, longitude, altitude) to ECEF 
        coordinates (x, y, z).

        Args:
            lat (float or array-like): Latitude in degrees.
            lon (float or array-like): Longitude in degrees.
            alt (float or array-like): Altitude in meters.

        Returns:
            tuple: A tuple containing the x, y, and z coordinates in ECEF.
        """

        ecef_proj = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        lla_proj = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        
        x, y, z = pyproj.transform(lla_proj, ecef_proj, lon, lat, alt)
        return x, y, z
    @staticmethod
    def convert_ecef_to_spherical(x, y, z):
        """
        Convert ECEF coordinates to spherical coordinates (m, rad, rad)
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        phi = np.sign(y)*np.arccos(x/np.sqrt(x**2 + y**2))

        return r, theta, phi
    
    @staticmethod
    def convert_spherical_to_ecef(r, theta, phi):
        """
        Convert spherical coordinates to ECEF coordinates
        """
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        
        return x, y, z
    @staticmethod
    def convert_ecef_to_lla(x,y,z):

        transformer = pyproj.Transformer.from_crs(
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
            )

        lon1, lat1, alt1 = transformer.transform(x,y,z,radians=False)
        return lat1, lon1, alt1 

    # @staticmethod
    # def convert_ecef_to_lla(x, y, z):
    #     """
    #     Convert ECEF coordinates to latitude, longitude, and altitude
    #     """
    #     a = 6378137.0          # Semi-major axis of WGS84 ellipsoid
    #     e2 = 6.69437999014e-03  # Eccentricity squared
        
    #     p = np.sqrt(x**2 + y**2)
    #     theta = np.arctan2(z * a, p * (1 - e2))
        
    #     lat_rad = np.arctan2(z + e2 * a * np.sin(theta)**3, p - e2 * a * np.cos(theta)**3)
    #     lon_rad = np.arctan2(y, x)
    #     N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    #     alt = p / np.cos(lat_rad) - N

    #     lat_deg = np.degrees(lat_rad)
    #     lon_deg = np.degrees(lon_rad)
        
    #     return lat_deg, lon_deg, alt
    @staticmethod
    def convert_spherical_to_lla(r, theta, phi):
        """
        Convert spherical coordinates to latitude, longitude, and altitude
        """

        x, y, z = GeoModel.convert_spherical_to_ecef(r, theta, phi)
        lat, lon, alt = GeoModel.convert_ecef_to_lla(x, y, z)
        return lat, lon, alt

class IRIModel(GeoModel):
    def __init__(self, dt_UTC):
        """
        
        """
        # find coefficients directory
        coeff_dir = os.path.join(os.path.dirname(PyIRI.__file__), 'coefficients')

        # get date and time
        self.dt_UTC = dt_UTC
        self.year = dt_UTC.year
        self.month = dt_UTC.month
        self.day = dt_UTC.day
        self.aUT = np.asarray([dt_UTC.hour + dt_UTC.minute/60.])


        dlat=1       #resolution of geographic latitude {degrees} (integer or float)
        dlon=1       #resolution of geographic longitude {degrees} (integer or float)
        dalt=10      #resolution of altitude {km} (integer or float)
        alt_min=0    #minimum altitude {km} (integer or float)
        alt_max=1000  #maximum altitude {km} (integer or float)
        alon, alat, alon_2d, alat_2d = PyIRI.main_library.set_geo_grid(dlon, dlat)
        lats = np.arange(-90, 90+dlat, dlat)
        lons = np.arange(-180, 180+dlon, dlon)
        aalt=np.arange(alt_min, alt_max, dalt)

        # call PyIRI 
        F2, F1, E, Es, sun, mag= PyIRI.main_library.IRI_monthly_mean_par(self.year, self.month, self.aUT, alon, alat, coeff_dir)  
        min_max_EDP = PyIRI.main_library.reconstruct_density_from_parameters(F2, F1, E, aalt)
        
        # take the max or min of the electron density
        edp = np.squeeze(min_max_EDP[0,:,:,:]) # aalt x alon/alat
        
        # reshape to 3D grid 
        edp = edp.reshape(len(aalt), len(lons), len(lats))

        # put into xarray dataset
        self.edp_ds = xr.Dataset( data_vars = dict( 
                                edp = (["alt", "lon" ,"lat"], edp),
                            ),
                            coords= dict(lon = ('lon',lons),
                                        lat = ('lat',lats),
                                        alt = ('alt', aalt))
                            )


    def __call__(self, lat, lon, alt, with_derivatives = True):
        """
        Interpolates the electron density at a given latitude (deg), longitude (deg), and altitude (m).
        count per meter cubed
        """
        interpolate_result = self.edp_ds.interp(lat= lat, lon= lon, alt=  alt/1e3, method='linear')
        ne = interpolate_result['edp'].values
        
        # spherical derivatives 
        if with_derivatives:
            x, y, z = self.convert_lla_to_ecef(lat, lon, alt)
            r, theta, phi = self.convert_ecef_to_spherical(x, y, z)

            delta_r = 1e3 # m
            delta_theta = 0.01 # rad 
            delta_phi = 0.01 # rad 

            # calculate the partial derivatives of the electron density
            lat_deltar, lon_deltar, alt_deltar = self.convert_spherical_to_lla(r+delta_r, theta, phi)
            ne_deltar = (self.edp_ds.interp(lat= lat_deltar, lon= lon_deltar, alt=  alt_deltar/1e3, method='linear'))['edp'].values
            deltane_deltar = (ne_deltar - ne)/delta_r

            lat_deltatheta, lon_deltatheta, alt_deltatheta = self.convert_spherical_to_lla(r, theta+delta_theta, phi)
            ne_deltatheta = (self.edp_ds.interp(lat= lat_deltatheta, lon= lon_deltatheta, alt=  alt_deltatheta/1e3, method='linear'))['edp'].values
            deltane_deltatheta = (ne_deltatheta - ne)/delta_theta

            lat_deltaphi, lon_deltaphi, alt_deltaphi = self.convert_spherical_to_lla(r, theta, phi+delta_phi)
            ne_deltaphi = (self.edp_ds.interp(lat= lat_deltaphi, lon= lon_deltaphi, alt=  alt_deltaphi/1e3, method='linear'))['edp'].values
            deltane_deltaphi = (ne_deltaphi - ne)/delta_phi

            ne_derivatives = np.array([deltane_deltar, deltane_deltatheta, deltane_deltaphi])

            return ne, ne_derivatives
        else:
            return ne
    
class IGRF(GeoModel):
    def __init__(self, dt_UTC):
        """
        """
        self.dt_UTC = dt_UTC
    def __call__(self, lat, lon, alt, with_derivatives = True):
        """ 
        Return spherical coordinate components of the magnetic field (nT) at a given latitude, longitude, and altitude 
        """
        # convert geodetic coordinates to spherical coordinates 
        x, y, z = self.convert_lla_to_ecef(lat, lon, alt)
        lat, lon, r = self.convert_ecef_to_spherical(x, y, z)
        colatitude = 90 - lat 
        # r     = 6500 # kilometers from center of Earht
        # theta = 30   # colatitude in degrees
        # phi   = 4    # degrees east (same as lon)
        # r in km, colatitude in deg, lon in deg
        Br, Btheta, Bphi = ppigrf.igrf_gc(r/1e3, colatitude, lon, self.dt_UTC) # returns radial, south, east
        B_vec = np.array([Br, Btheta, Bphi])

        # calculate the derivatives of the magnetic field
        if with_derivatives:
            delta_r = 1e3 # m 
            delta_theta = 0.01 # rad 
            delta_phi = 0.01 # rad 

            # calculate the partial derivatives of the magnetic field 
            Br_with_deltar, Btheta_with_deltar, Bphi_with_deltar = ppigrf.igrf_gc(r/1e3+delta_r, colatitude, lon, self.dt_UTC) # returns radial, south, east
            deltaBr_deltar = (Br_with_deltar - Br)/delta_r 
            deltaBtheta_deltar = (Btheta_with_deltar - Btheta)/delta_r
            deltaBphi_deltar = (Bphi_with_deltar - Bphi)/delta_r
            magnitude_deriv_deltar = np.sqrt( (Br_with_deltar - Br)**2 + (Btheta_with_deltar - Btheta)**2 + (Bphi_with_deltar - Bphi)**2) /delta_r

            Br_with_deltatheta, Btheta_with_deltatheta, Bphi_with_deltatheta = ppigrf.igrf_gc(r/1e3, colatitude + delta_theta, lon, self.dt_UTC) # returns radial, south, east 
            deltaBr_deltatheta = (Br_with_deltatheta - Br)/delta_theta
            deltaBtheta_deltatheta = (Btheta_with_deltatheta - Btheta)/delta_theta
            deltaBphi_deltatheta = (Bphi_with_deltatheta - Bphi)/delta_theta
            magnitude_deriv_deltatheta = np.sqrt( (Br_with_deltatheta - Br)**2 + (Btheta_with_deltatheta - Btheta)**2 + (Bphi_with_deltatheta - Bphi)**2) /delta_theta


            Br_with_deltaphi, Btheta_with_deltaphi, Bphi_with_deltaphi = ppigrf.igrf_gc(r/1e3, colatitude, lon + delta_phi, self.dt_UTC) # returns radial, south, east
            deltaBr_deltaphi = (Br_with_deltaphi - Br)/delta_phi
            deltaBtheta_deltaphi = (Btheta_with_deltaphi - Btheta)/delta_phi    
            deltaBphi_deltaphi = (Bphi_with_deltaphi - Bphi)/delta_phi
            magnitude_deriv_deltaphi = np.sqrt( (Br_with_deltaphi - Br)**2 + (Btheta_with_deltaphi - Btheta)**2 + (Bphi_with_deltaphi - Bphi)**2) /delta_phi

            B_deriv_mat = np.array([[deltaBr_deltar, deltaBr_deltatheta, deltaBr_deltaphi],
                                    [deltaBtheta_deltar, deltaBtheta_deltatheta, deltaBtheta_deltaphi],
                                    [deltaBphi_deltar, deltaBphi_deltatheta, deltaBphi_deltaphi]])
            
            B_mag_deriv = np.array([magnitude_deriv_deltar, magnitude_deriv_deltatheta, magnitude_deriv_deltaphi])

            return B_vec, B_deriv_mat, B_mag_deriv
        
        else:
            return B_vec
        
    
class MSIS(GeoModel):
    def __init__(self, dt_UTC):
        """
        """
        dlat=1       #resolution of geographic latitude {degrees} (integer or float)
        dlon=1       #resolution of geographic longitude {degrees} (integer or float)
        dalt=2     #resolution of altitude {km} (integer or float)
        alt_min=0    #minimum altitude {km} (integer or float)
        alt_max=100 #maximum altitude {km} (integer or float)
        lats = np.arange(-90, 90+dlat, dlat)
        lons = np.arange(-180, 180+dlon, dlon)
        aalt=np.arange(alt_min, alt_max, dalt)
        date64 = np.datetime64(dt_UTC)

        data = pymsis.calculate(date64, lons, lats, aalt, geomagnetic_activity=-1)   #[ndates, nlons, nlats, nalts, 11]

        density = data[...,0] # kg/ m^-3
        temperature = data[...,-1] #temperature in K
        volume = 1 #m^3 
        R = 8.31446261815324 # J/(mol K)
        n = 28.96e-3 # kg/mol
        # convert density to pressure via ideal gas law
        pressure = density/n * R * temperature / volume # Pa
        
        # calculate refractivity 
        Nd = 77.64*pressure*1e-2/temperature
        n = Nd/1e6 + 1 
        # import pdb
        # pdb.set_trace()
        # put into xarray dataset 
        self.msis_ds = xr.Dataset( data_vars = dict( 
                                density = (["lon","lat","alt"], np.squeeze(density)),
                                temperature = (["lon","lat","alt"], np.squeeze(temperature)),
                                pressure = (["lon","lat","alt"], np.squeeze(pressure)),
                                refractivity = (["lon","lat","alt"], np.squeeze(n))
                            ),
                            coords= dict(lon = ('lon',lons),
                                        lat = ('lat',lats),
                                        alt = ('alt', aalt))
                            )

        self.dt_UTC = dt_UTC


    def __call__(self, lat, lon, alt, with_derivatives = True):
        """
        """
        N0 = 300
        H = 7e3 
        N = N0*np.exp(-alt/H)
        
        n = 1 + N/1e6 # refractivity
        n_derivatives = np.zeros(3)
        n_derivatives[0] = N0 * np.exp(-alt/H)/1e6 * (-1/H) # derivative with respect to altitude
        return n, n_derivatives
    
        # n = self.msis_ds.interp(lat= lat, lon= lon, alt= alt/1e3, method='linear')['refractivity'].values
        # if np.isnan(n):
        #     n = 0

        # # spherical derivatives 
        # if with_derivatives:
        #     x, y, z = self.convert_lla_to_ecef(lat, lon, alt)
        #     r, theta, phi = self.convert_ecef_to_spherical(x, y, z)

        #     delta_r = 10. # m
        #     delta_theta = 0.002 # rad 
        #     delta_phi = 0.002 # rad 

        #     # calculate the partial derivatives of the electron density
        #     lat_deltar, lon_deltar, alt_deltar = self.convert_spherical_to_lla(r+delta_r, theta, phi)
        #     n_deltar = (self.msis_ds.interp(lat= lat_deltar, lon= lon_deltar, alt=  alt_deltar/1e3, method='linear'))['refractivity'].values
        #     n_deltar = 0 if np.isnan(n_deltar) else n_deltar
        #     deltan_deltar = (n_deltar - n)/delta_r


        #     lat_deltatheta, lon_deltatheta, alt_deltatheta = self.convert_spherical_to_lla(r, theta+delta_theta, phi)
        #     n_deltatheta = (self.msis_ds.interp(lat= lat_deltatheta, lon= lon_deltatheta, alt=  alt_deltatheta/1e3, method='linear'))['refractivity'].values
        #     n_deltatheta = 0 if np.isnan(n_deltatheta) else n_deltar
        #     deltane_deltatheta = (n_deltatheta - n)/delta_theta

        #     lat_deltaphi, lon_deltaphi, alt_deltaphi = self.convert_spherical_to_lla(r, theta, phi+delta_phi)
        #     n_deltaphi = (self.msis_ds.interp(lat= lat_deltaphi, lon= lon_deltaphi, alt=  alt_deltaphi/1e3, method='linear'))['refractivity'].values
        #     n_deltaphi = 0 if np.isnan(n_deltaphi) else n_deltar
        #     deltane_deltaphi = (n_deltaphi - n)/delta_phi

        #     n_derivatives = np.array([deltan_deltar, deltane_deltatheta, deltane_deltaphi])

        #     return n, n_derivatives
        # else:
        #     return n
