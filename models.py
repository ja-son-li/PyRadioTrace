import PyIRI
import numpy as np 
import os 
import pyproj
import xarray as xr 
import datetime as dt
import ppigrf
import pymsis
import pymap3d as pm
from numba import jit

transformer_lla_ecef = pyproj.Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
transformer_ecef_lla = pyproj.Transformer.from_crs(
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
            )

class GeoModel():
    def __init__(self):
        pass

    @staticmethod
    def convert_lla_to_ecef(lat, lon, alt):
        """
        Vectorized conversion from latitude, longitude, altitude (in meters) to ECEF coordinates.
        """
        x, y, z = transformer_lla_ecef.transform(lon, lat, alt)
        return x, y, z

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
        
        # lats = np.full_like(ranges,np.nan)
        # lons = np.full_like(ranges,np.nan)
        # alts = np.full_like(ranges,np.nan)
        X_plus, Y_plus, Z_plus = GeoModel.aer_to_ecef(lat_obs, lon_obs, alt_obs, az, el, ranges)
        straight_lat,  straight_lon, straight_alt = GeoModel.convert_ecef_to_lla(X_plus, Y_plus, Z_plus)

        # for ind, slant_range in enumerate(ranges):

        #     straight_lat,  straight_lon, straight_alt = GeoModel.convert_ecef_to_lla(X_plus, Y_plus, Z_plus)
        #     lats[ind] = straight_lat 
        #     lons[ind] = straight_lon 
        #     alts[ind] = straight_alt

        return straight_lat,  straight_lon, straight_alt 
    
    @staticmethod
    @jit(nopython=True)
    def convert_ecef_to_spherical(x, y, z):
        """
        Convert ECEF coordinates to spherical coordinates (m, rad, rad)
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)
        phi = np.sign(y)*np.arccos(x/np.sqrt(x**2 + y**2))

        return r, theta, phi
    
    @staticmethod
    @jit(nopython=True)
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

        # transformer = pyproj.Transformer.from_crs(
        #     {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        #     {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        #     )

        lon1, lat1, alt1 = transformer_ecef_lla.transform(x,y,z,radians=False)
        return lat1, lon1, alt1 
    
    @staticmethod
    @jit(nopython=True)
    def angle_between_spherical_vectors(r1, theta1, phi1, r2, theta2, phi2):
        """
        Calculate the angle between two spherical vectors.
        """
        x1 = r1*np.sin(phi1)*np.cos(theta1)
        y1 = r1*np.sin(theta1)*np.sin(phi1)
        z1 = r1*np.cos(phi1)

        x2 = r1*np.sin(phi2)*np.cos(theta2)
        y2 = r1*np.sin(theta2)*np.sin(phi2)
        z2 = r1*np.cos(phi2)

        return np.arccos( (x1*x2 + y1*y2 + z1*z2)  / (r1 * r2)) 
    
    @staticmethod
    @jit(nopython=True)
    def distance_between_two_spherical_vectors(r1, theta1, phi1, r2, theta2, phi2):
        """
        Calculate the distance between two spherical vectors.
        r1 : radius of first vector (m)
        theta1 : azimuth of first vector (rad)
        phi1 : elevation of first vector (rad)
        r2 : radius of second vector (m)    
        theta2 : azimuth of second vector (rad)
        phi2 : elevation of second vector (rad)
        """

        return np.sqrt( r1**2 + r2**2 
                       - 2*r1*r2*(np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2)) 
                       + np.cos(theta1)*np.cos(theta2))
    
    @staticmethod
    def azel_to_spherical_vector(lat, lon, alt, az, el):
        """
        lat : latitude (deg)
        lon : longitude (deg)
        alt : altitude (m)
        az : azimuth East of North (deg) 
        el : elevation off horizon (deg)
        """
        
        x, y, z = GeoModel.convert_lla_to_ecef(lat, lon, alt)
        r, theta, phi = GeoModel.convert_ecef_to_spherical(x, y, z)
            
        # convert observer location to ECEF 
        X_plus, Y_plus, Z_plus =GeoModel.aer_to_ecef(lat, lon, alt, az, el, 1)

        delta_X = X_plus - x
        delta_Y = Y_plus - y
        delta_Z = Z_plus - z


        delta_r = np.sin(theta)*np.cos(phi)*delta_X + np.sin(theta)*np.sin(phi)*delta_Y + np.cos(theta)*delta_Z
        delta_theta = np.cos(theta)*np.cos(phi)*delta_X + np.cos(theta)*np.sin(phi)*delta_Y - np.sin(theta)*delta_Z
        delta_phi = -np.sin(phi)*delta_X + np.cos(phi)*delta_Y

        return  delta_r, delta_theta, delta_phi

    @staticmethod
    def convert_spherical_to_lla(r, theta, phi):
        """
        Convert spherical coordinates to latitude, longitude, and altitude
        """

        x, y, z = GeoModel.convert_spherical_to_ecef(r, theta, phi)
        lat, lon, alt = GeoModel.convert_ecef_to_lla(x, y, z)
        return lat, lon, alt


class EpsteinLayersModel(GeoModel):
    def __init__(self, nm_list : list[float] = [1.4e12], 
                        hm_list : list[float]= [300], 
                        Bbottom_list:list[float] = [23.5], 
                        Btop_list: list[float]= [35]):
        """
        nm_list : list of peak densities (m^-3)
        hm_list : list of peak heights (km)
        B_list : list of thicknesses (km)
        alt_list : list of altitudes (km)
        """
        self.nm_list = nm_list
        self.hm_list = hm_list
        self.Bbottom_list = Bbottom_list
        self.Btop_list = Btop_list

    def __call__(self, lat, lon, alt, with_derivatives = True):
        """
        Interpolates the electron density at a given latitude (deg), longitude (deg), and altitude (m).
        count per meter cubed
        """
        # convert alt to km
        alt = alt/1e3
        n_e = 0
        n_e_plus = 0
        n_derivatives = np.zeros(3)

        if alt < 2000:
            for layer_ind in range(len(self.nm_list)):
                n_e += self.epstein(self.nm_list[layer_ind], 
                                    self.hm_list[layer_ind], 
                                    self.Btop_list[layer_ind], 
                                    self.Bbottom_list[layer_ind], 
                                    alt)

                delta_alt = 0.1 # km 
                n_e_plus += self.epstein(self.nm_list[layer_ind], 
                                        self.hm_list[layer_ind], 
                                        self.Btop_list[layer_ind], 
                                        self.Bbottom_list[layer_ind], 
                                        alt+delta_alt)

            n_derivatives[0] = (n_e_plus - n_e)/(delta_alt * 1e3) # derivative with respect to altitude

        if with_derivatives:
            return n_e, n_derivatives
        else:
            return n_e
    
    @staticmethod
    def epstein(Nm, hm, Btop, Bbottom, alt):
        """Calculate Epstein function for given parameters.

        Parameters
        ----------
        Nm : array-like
            Peak density in m-3.
        hm : array-like
            Height of peak density in km.
        B : array-like
            Thickness of the layer in km.
        alt : array-like
            Altitude array in km.

        Returns
        -------
        res : array-like
            Constructed Epstein profile in m-3.

        Notes
        -----
        This function returns epstein function for given parameters.
        In a typical Epstein function: X = Nm, Y = hm, Z = B, and W = alt.

        """
        if alt > hm:
            B = Btop
        else:
            B = Bbottom

        aexp = EpsteinLayersModel.fexp((alt - hm) / B)
        res = Nm * aexp / (1 + aexp)**2
        return res
    
    @staticmethod
    def fexp(x):
        """Calculate exponent without overflow.

        Parameters
        ----------
        x : array-like
            Any input.

        Returns
        -------
        y : array-like
            Exponent of x.

        Notes
        -----
        This function function caclulates exp(x) with restrictions to not cause
        overflow.

        """
        if (isinstance(x, float)) or (isinstance(x, int)):
            if x > 80:
                y = 5.5406E34
            if x < -80:
                y = 1.8049E-35
            else:
                y = np.exp(x)

        if isinstance(x, np.ndarray):
            y = np.zeros((x.shape))
            a = np.where(x > 80.)
            b = np.where(x < -80.)
            c = np.where((x >= -80.) & (x <= 80.))
            y[a] = 5.5406E34
            y[b] = 1.8049E-35
            y[c] = np.exp(x[c])

        return y


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


        # dlat=1       #resolution of geographic latitude {degrees} (integer or float)
        # dlon=1       #resolution of geographic longitude {degrees} (integer or float)
        # dalt=10      #resolution of altitude {km} (integer or float)
        # alt_min=0    #minimum altitude {km} (integer or float)
        # alt_max=1000  #maximum altitude {km} (integer or float)
        # alon, alat, alon_2d, alat_2d = PyIRI.main_library.set_geo_grid(dlon, dlat)
        # lats = np.arange(-90, 90+dlat, dlat)
        # lons = np.arange(-180, 180+dlon, dlon)
        # aalt=np.arange(alt_min, alt_max, dalt)

        # # call PyIRI 
        # F2, F1, E, Es, sun, mag= PyIRI.main_library.IRI_monthly_mean_par(self.year, self.month, self.aUT, alon, alat, coeff_dir)  
        # min_max_EDP = PyIRI.main_library.reconstruct_density_from_parameters(F2, F1, E, aalt)
        
        # # take the max or min of the electron density
        # edp = np.squeeze(min_max_EDP[0,:,:,:]) # aalt x alon/alat
        
        # # reshape to 3D grid 
        # edp = edp.reshape(len(aalt), len(lons), len(lats))

        # # put into xarray dataset
        # self.edp_ds = xr.Dataset( data_vars = dict( 
        #                         edp = (["alt", "lon" ,"lat"], edp),
        #                     ),
        #                     coords= dict(lon = ('lon',lons),
        #                                 lat = ('lat',lats),
        #                                 alt = ('alt', aalt))
        #                     )
    @staticmethod
    def epstein(Nm, hm, Btop, Bbottom, alt):
        """Calculate Epstein function for given parameters.

        Parameters
        ----------
        Nm : array-like
            Peak density in m-3.
        hm : array-like
            Height of peak density in km.
        B : array-like
            Thickness of the layer in km.
        alt : array-like
            Altitude array in km.

        Returns
        -------
        res : array-like
            Constructed Epstein profile in m-3.

        Notes
        -----
        This function returns epstein function for given parameters.
        In a typical Epstein function: X = Nm, Y = hm, Z = B, and W = alt.

        """
        if alt > hm:
            B = Btop
        else:
            B = Bbottom

        aexp = IRIModel.fexp((alt - hm) / B)
        res = Nm * aexp / (1 + aexp)**2
        return res
    
    @staticmethod
    def fexp(x):
        """Calculate exponent without overflow.

        Parameters
        ----------
        x : array-like
            Any input.

        Returns
        -------
        y : array-like
            Exponent of x.

        Notes
        -----
        This function function caclulates exp(x) with restrictions to not cause
        overflow.

        """
        if (isinstance(x, float)) or (isinstance(x, int)):
            if x > 80:
                y = 5.5406E34
            if x < -80:
                y = 1.8049E-35
            else:
                y = np.exp(x)

        if isinstance(x, np.ndarray):
            y = np.zeros((x.shape))
            a = np.where(x > 80.)
            b = np.where(x < -80.)
            c = np.where((x >= -80.) & (x <= 80.))
            y[a] = 5.5406E34
            y[b] = 1.8049E-35
            y[c] = np.exp(x[c])

        return y
    
    def __call__(self, lat, lon, alt, with_derivatives = True):
        """
        Interpolates the electron density at a given latitude (deg), longitude (deg), and altitude (m).
        count per meter cubed
        """
        # convert alt to km
        alt = alt/1e3

        # triple chapman layer 
        hm_F2 = 300 # km
        nm_F2 = 2.28050755e11 # m^-3
        nm_F2 = 1.4e12 
        Btop = 35 # km
        Bbottom = 23.5 #km 

        n_e = 0
        n_derivatives = np.zeros(3)

        if alt < 2000:
            n_e = self.epstein(nm_F2, hm_F2, Btop, Bbottom, alt)

            delta_alt = 0.1 # km 
            n_e_plus = self.epstein(nm_F2, hm_F2, Btop, Bbottom, alt+delta_alt)

            n_derivatives[0] = (n_e_plus - n_e)/(delta_alt * 1e3) # derivative with respect to altitude

        return n_e, n_derivatives
       
        # interpolate_result = self.edp_ds.interp(lat= lat, lon= lon, alt=  alt/1e3, method='linear')
        # ne = interpolate_result['edp'].values
        
        # # spherical derivatives 
        # if with_derivatives:
        #     x, y, z = self.convert_lla_to_ecef(lat, lon, alt)
        #     r, theta, phi = self.convert_ecef_to_spherical(x, y, z)

        #     delta_r = 1e3 # m
        #     delta_theta = 0.01 # rad 
        #     delta_phi = 0.01 # rad 

        #     # calculate the partial derivatives of the electron density
        #     lat_deltar, lon_deltar, alt_deltar = self.convert_spherical_to_lla(r+delta_r, theta, phi)
        #     ne_deltar = (self.edp_ds.interp(lat= lat_deltar, lon= lon_deltar, alt=  alt_deltar/1e3, method='linear'))['edp'].values
        #     deltane_deltar = (ne_deltar - ne)/delta_r

        #     lat_deltatheta, lon_deltatheta, alt_deltatheta = self.convert_spherical_to_lla(r, theta+delta_theta, phi)
        #     ne_deltatheta = (self.edp_ds.interp(lat= lat_deltatheta, lon= lon_deltatheta, alt=  alt_deltatheta/1e3, method='linear'))['edp'].values
        #     deltane_deltatheta = (ne_deltatheta - ne)/delta_theta

        #     lat_deltaphi, lon_deltaphi, alt_deltaphi = self.convert_spherical_to_lla(r, theta, phi+delta_phi)
        #     ne_deltaphi = (self.edp_ds.interp(lat= lat_deltaphi, lon= lon_deltaphi, alt=  alt_deltaphi/1e3, method='linear'))['edp'].values
        #     deltane_deltaphi = (ne_deltaphi - ne)/delta_phi

        #     ne_derivatives = np.array([deltane_deltar, deltane_deltatheta, deltane_deltaphi])

        #     return ne, ne_derivatives
        # else:
        #     return ne
    
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
        
class ScaleHeight(GeoModel):
    def __init__(self, N0: float = 400, H: float = 7e3):
        """
        N0 : float
        H : float
        """
        self.N0 = N0
        self.H = H

    def __call__(self, lat: float, lon: float, alt: float, with_derivatives: bool = True):
        N0 = self.N0
        H = self.H
        
        n = 1
        n_derivatives = np.zeros(3)

        if alt < 100e3 and alt > 0:
            N = N0*np.exp(-alt/H)
            n = 1 + N/1e6 # refractivity
            n_derivatives = np.zeros(3)
            n_derivatives[0] = N0 * np.exp(-alt/H)/1e6 * (-1/H) # derivative with respect to altitude

        if with_derivatives:
            return n, n_derivatives
        else:
            return n
    
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

        # data = pymsis.calculate(date64, lons, lats, aalt, geomagnetic_activity=-1)   #[ndates, nlons, nlats, nalts, 11]

        # density = data[...,0] # kg/ m^-3
        # temperature = data[...,-1] #temperature in K
        # volume = 1 #m^3 
        # R = 8.31446261815324 # J/(mol K)
        # n = 28.96e-3 # kg/mol
        # # convert density to pressure via ideal gas law
        # pressure = density/n * R * temperature / volume # Pa
        
        # # calculate refractivity 
        # Nd = 77.64*pressure*1e-2/temperature
        # n = Nd/1e6 + 1 
        # # import pdb
        # # pdb.set_trace()
        # # put into xarray dataset 
        # self.msis_ds = xr.Dataset( data_vars = dict( 
        #                         density = (["lon","lat","alt"], np.squeeze(density)),
        #                         temperature = (["lon","lat","alt"], np.squeeze(temperature)),
        #                         pressure = (["lon","lat","alt"], np.squeeze(pressure)),
        #                         refractivity = (["lon","lat","alt"], np.squeeze(n))
        #                     ),
        #                     coords= dict(lon = ('lon',lons),
        #                                 lat = ('lat',lats),
        #                                 alt = ('alt', aalt))
        #                     )

        # self.dt_UTC = dt_UTC


    def __call__(self, lat, lon, alt, with_derivatives = True):
        """
        """
        # N0 = 300
        N0 = 700

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
