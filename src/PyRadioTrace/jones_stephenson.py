import numpy as np
from PyRadioTrace.models import GeoModel
import scipy.constants as const
from scipy.integrate import odeint, solve_ivp 
from numba import jit
import scipy.optimize

class Raytracer():

    def __init__(self, iono = None, magneto = None, neutral = None, o_mode = True):
        """"
        state space is [1. r : spherical coordinate radius (m), 
                        2. theta : spherical coordinate longitude (rad), 
                        3. phi : spherical coordinate colatitude (rad),
                        4. kr, (m/m)
                        5. ktheta, (m/m)
                        6. kphi : spherical coordinate wave normal  (m/m)
                        7. P : total phase (m), 
                        8. s: geometric path length (m)]
        TODO: ADD TEC 

        kr ktheta kphi are normalized such that kr**2 + ktheta**2 + kphi**2 = omega**2/c**2 where omega = 2*pi*f
        """
        self.iono = iono
        self.magneto = magneto
        self.neutral = neutral
        
    def ray_home_onedim(self, transmit_lat:np.float64,
                            transmit_lon:np.float64,
                            transmit_alt:np.float64,
                            receive_lat:np.float64, 
                            receive_lon:np.float64, 
                            receive_alt:np.float64, 
                            f : np.float64, 
                            resolution_m:np.float64 = 0.01,
                            xatol = 1e-8,
                            rtol = 1e-5, 
                            atol = 1e-6,
                            elevation_bounds = None,
                            max_bounce = 1) -> scipy.optimize.OptimizeResult:
        """
        transmit_lat : latitude of transmitter (deg)
        transmit_lon : longitude of transmitter (deg)
        transmit_alt : altitude of transmitter (m)
        receive_lat : latitude of receiver (deg)
        receive_lon : longitude of receiver (deg)
        receive_alt : altitude of receiver (m)
        f : frequency (Hz)
        hmin : minimum step size of ODE solver (m)
        resolution_m : minimum interation difference for cost function(m)
        """
        
        # first guess 
        az, el, range_val = GeoModel.geodetic2aer(transmit_lat, transmit_lon, transmit_alt, receive_lat, receive_lon, receive_alt)
        group_path_distances = np.arange(range_val-20e3,range_val+20e3, resolution_m/2)

        # initial guess is straight line of sight pointing 
        initial_guess = el

        if elevation_bounds is None:
            elevation_bounds = [initial_guess-3, initial_guess+3]

        # minimization function
        def optimization_function(el_guess) -> float:
            return self.ray_distance_to_target(transmit_lat, transmit_lon, transmit_alt, 
                                                receive_lat, receive_lon, receive_alt, 
                                                az, el_guess, f, 
                                                group_path_distances=group_path_distances, 
                                                rtol=rtol, atol=atol,
                                                max_bounce=max_bounce)
        
        result = scipy.optimize.minimize_scalar(optimization_function, bounds = elevation_bounds, 
                                                                        method='bounded', 
                                                                        options = {'xatol' : xatol})

        return result 
    
    def ray_home(self, transmit_lat:np.float64, 
                transmit_lon:np.float64,
                transmit_alt:np.float64,
                receive_lat:np.float64, 
                receive_lon:np.float64, 
                receive_alt:np.float64, 
                f : np.float64, 
                resolution_m:np.float64 = 0.01,
                rtol = 1e-5, 
                atol = 1e-6,
                max_bounce = 1) -> scipy.optimize.OptimizeResult:
        """
        transmit_lat : latitude of transmitter (deg)
        transmit_lon : longitude of transmitter (deg)
        transmit_alt : altitude of transmitter (m)
        receive_lat : latitude of receiver (deg)
        receive_lon : longitude of receiver (deg)
        receive_alt : altitude of receiver (m)
        f : frequency (Hz)
        hmin : minimum step size of ODE solver (m)
        resolution_m : minimum interation difference for cost function(m)
        resolution_angle_deg : minimum interation difference azimuth and elevation (deg)
        """
        
        # first guess 
        az, el, range_val = GeoModel.geodetic2aer(transmit_lat, transmit_lon, transmit_alt, receive_lat, receive_lon, receive_alt)
        group_path_distances = np.arange(range_val-20e3,range_val+20e3, resolution_m/2)

        # initial guess is straight line of sight pointing 
        initial_guess = [az, el]

        # minimization function
        def optimization_function(azel_guess:tuple[float]) -> float:
            return self.ray_distance_to_target(transmit_lat, transmit_lon, transmit_alt, 
                                                receive_lat, receive_lon, receive_alt, 
                                                azel_guess[0], azel_guess[1], f, 
                                                group_path_distances=group_path_distances, 
                                                rtol=rtol, atol=atol, max_bounce=max_bounce)
        
        result = scipy.optimize.minimize(optimization_function, initial_guess, method='Nelder-Mead', options={'fatol': resolution_m/10, 
                                                                                                              'maxiter':1e3,
                                                                                                              'disp': True})
        return result

    def ray_distance_to_target(self, transmit_lat:np.float64, transmit_lon:np.float64, transmit_alt:np.float64, 
                                    receiver_lat:np.float64, receiver_lon:np.float64, receiver_alt:np.float64,
                                    az:np.float64, el:np.float64, f:np.float64, 
                                    group_path_distances:np.ndarray = np.arange(0,2000,5), rtol = 1e-3, atol = 1e-6,
                                    max_bounce = 1) -> np.float64:
        """
        INPUT:
        transmit_lat : latitude of transmitter (deg)
        transmit_lon : longitude of transmitter (deg)
        transmit_alt : altitude of transmitter (m)
        receiver_lat : latitude of receiver (deg)
        receiver_lon : longitude of receiver (deg)
        receiver_alt : altitude of receiver (m)
        az : azimuth of look direction (deg)
        el : elevation off horizon (deg)
        f : frequency (Hz)

        OUTPUT:
        Minimum distance of ray to target (m)
        """

        if max_bounce == 0:
            solution = self.ray_propagate(transmit_lat, transmit_lon, transmit_alt,
                                          az, el, f,
                                          group_path_distances = group_path_distances,
                                          atol = atol, rtol = rtol, stop_at_surface=False)
        else:
            solution = self.ray_propagate_with_bounce(transmit_lat, transmit_lon, transmit_alt,
                                                      az, el, f,
                                                      group_path_distances = group_path_distances,
                                                      atol = atol, rtol = rtol,
                                                      max_bounce = max_bounce)

        # convert geodetic coordinates to spherical coordinates
        x, y, z = GeoModel.convert_lla_to_ecef(receiver_lat, receiver_lon, receiver_alt)
        r_rx, theta_rx, phi_rx = GeoModel.convert_ecef_to_spherical(x, y, z)

        distances = GeoModel.distance_between_two_spherical_vectors(solution.y[0,:], solution.y[1,:], solution.y[2,:], r_rx, theta_rx, phi_rx)

        min_distance = np.nanmin(distances)
        return min_distance
    
    def ray_propagate_with_bounce(self, transmit_lat: np.float64, 
                            transmit_lon: np.float64,
                            transmit_alt: np.float64, 
                            az: np.float64,
                            el: np.float64,
                            f : np.float64,
                            group_path_distances:np.ndarray = np.arange(0,2000,5),
                            rtol : float = 1e-5, 
                            atol : float = 1e-6,
                            max_bounce : int = 1):
        """
        Propagate a ray from the transmitter to a look direction
        transmit_lat : latitude of transmitter (deg)
        transmit_lon : longitude of transmitter (deg)
        transmit_alt : altitude of transmitter (m)
        az : azimuth of look direction (deg)
        el : elevation of look direction (deg)
        f : frequency (Hz)
        max_bounce : maximum number of 
        """

        bounce_init_lat = transmit_lat
        bounce_init_lon = transmit_lon
        bounce_init_alt = transmit_alt
        bounce_init_az = az
        bounce_init_el = el
        bounce_init_range = group_path_distances[0]

        end_path_range = group_path_distances[-1]


        full_solution = lambda: None
        full_solution.t = np.empty((0), dtype = float)
        full_solution.y = np.empty((8,0), dtype = float)


        # repeat raytracing for each ground bounce 
        for bounce_ind in range(max_bounce+1):
            
            group_path_distance_for_bounce = group_path_distances[group_path_distances < end_path_range - bounce_init_range]
            
            # propagate ray 
            solution = self.ray_propagate(bounce_init_lat, bounce_init_lon, bounce_init_alt, 
                                      bounce_init_az, bounce_init_el, f, 
                                      group_path_distances = group_path_distance_for_bounce, rtol = rtol, atol = atol)

            # append ray state evaluations 
            if len(solution.t) > 0:
                full_solution.t = np.append(full_solution.t, solution.t + bounce_init_range)
                full_solution.y = np.append(full_solution.y, solution.y, axis = 1)


            if len(solution.t_events[0]) > 0:
                # if the ray altitude is 0, reinitialize the ray state to "bounce away" from the ground 

                # get the ray state as it hits the Earth surface
                bounce_init_range = solution.t_events[0][0]
                ray_state_at_bounce = solution.y_events[0][0]
                r, theta, phi, kr, ktheta, kphi, P, s = ray_state_at_bounce

                # convert to geodetic coordinates
                x_ecef, y_ecef, z_ecef = GeoModel.convert_spherical_to_ecef(r, theta, phi)
                bounce_init_lat, bounce_init_lon, bounce_init_alt = GeoModel.convert_ecef_to_lla(x_ecef, y_ecef, z_ecef)

                # convert pointing to az el 
                bounce_init_az, bounce_init_el = GeoModel.spherical_vector_azel(r, theta, phi, kr, ktheta, kphi)

                kr = np.abs(kr) 

                # reflect elevation to bounce back up
                bounce_init_el = np.abs(bounce_init_el)

            else:
                # stop the loop is there is no ground bounce 
                break
    
        return full_solution 
    
    def ray_propagate(self, transmit_lat: np.float64, 
                            transmit_lon: np.float64,
                            transmit_alt: np.float64, 
                            az: np.float64,
                            el: np.float64,
                            f : np.float64,
                            group_path_distances:np.ndarray = np.arange(0,2000,5),
                            rtol = 1e-5, 
                            atol = 1e-6,
                            stop_at_surface= True):
        """
        Propagate a ray from the transmitter to a look direction
        transmit_lat : latitude of transmitter (deg)
        transmit_lon : longitude of transmitter (deg)
        transmit_alt : altitude of transmitter (m)
        az : azimuth of look direction (deg)
        el : elevation of look direction (deg)
        f : frequency (Hz)
        """

        # convert geodetic coordinates to spherical coordinates
        x, y, z = GeoModel.convert_lla_to_ecef(transmit_lat, transmit_lon, transmit_alt)
        r, theta, phi = GeoModel.convert_ecef_to_spherical(x, y, z)

        # convert az, el look angle to spherical vectors 
        omega = 2*np.pi*f
        
        # renormalize the propagation vector 
        kr, ktheta, kphi = GeoModel.azel_to_spherical_vector(transmit_lat, transmit_lon, transmit_alt, az, el)
        norm = np.sqrt(kr**2 + ktheta**2 + kphi**2)
        kr = kr/norm * omega/const.c
        ktheta = ktheta/norm *omega/const.c
        kphi = kphi/norm *omega/const.c

        # initalize state vector
        x0 = [r, theta, phi, kr, ktheta, kphi, 0,0]

        t_span = [0, np.max(group_path_distances)]

        # set up ground bounce event 
        event_func = self.get_geodetic_altitude
        event_func.terminal = stop_at_surface
        event_func.direction = -1 # only stop when decreasing altitude 


        if self.neutral is not None and self.iono is not None:
            # solution = solve_ivp(self.derivatives, t_span, x0, method = 'RK45',t_eval = group_path_distances, args=(f,), rtol = rtol, atol = atol)
            solution = solve_ivp(self.derivatives, t_span, x0, method = 'LSODA',t_eval = group_path_distances, events = [event_func], args=(f,), rtol = rtol, atol = atol)
        elif self.neutral is not None:
            solution = solve_ivp(self.derivatives_neutral, t_span, x0,  method = 'RK45', t_eval = group_path_distances, args=(f,), rtol=rtol, atol=atol)
        elif self.iono is not None:
            solution = solve_ivp(self.derivatives_ionosphere, t_span, x0,  method = 'LSODA', t_eval = group_path_distances, args=(f,), rtol=rtol, atol=atol)

        # print failure message 
        if solution.success is False:
            print('ODE solver failed to converge')
            print(solution.message)
            return solution

        return solution
    
    @staticmethod 
    def get_geodetic_altitude(t, y, f):
        """
        
        """
        r, theta, phi, kr, ktheta, kphi, P, s = y
        # convert to geodetic coordinates
        x_ecef, y_ecef, z_ecef = GeoModel.convert_spherical_to_ecef(r, theta, phi)
        lat, lon, alt = GeoModel.convert_ecef_to_lla(x_ecef, y_ecef, z_ecef)
        return alt
    
    def derivatives_neutral(self, Pgroup:np.float64, x:tuple[np.float64], f:np.float64):
        """
        x : state vector
        Pgroup : group path length (m)
        f : frequency (Hz)
        """
        # unpack state space
        r, theta, phi, kr, ktheta, kphi, P, s = x

        omega = 2*np.pi*f 

        # calculate geodetic coordinates 
        lat, lon, alt = GeoModel.convert_spherical_to_lla(r, theta, phi) # deg deg m

        # calculate index of refraction and spatial derivatives
        if alt > 70e3:
            n_refract, n_derivatives = 1, np.zeros(3)
        else:
            n_refract, n_derivatives = self.neutral(lat, lon, alt)

        # calculate hamiltonian derivatives
        deltaH_deltar, deltaH_deltatheta, deltaH_deltaphi, deltaH_deltaomega, deltaH_deltakr, deltaH_deltaktheta, deltaH_deltakphi = self.hamiltonian_neutral_atmosphere(x, f, n_refract, n_derivatives)
 
        return self.group_path_derivatives(x, omega, 
                                            deltaH_deltar, deltaH_deltatheta, deltaH_deltaphi, 
                                            deltaH_deltaomega, 
                                            deltaH_deltakr, deltaH_deltaktheta, deltaH_deltakphi)
    
    def derivatives_ionosphere(self, Pgroup:np.float64, x:np.ndarray, f:np.float64, o_mode = True):
        
        # unpack state space
        r, theta, phi, kr, ktheta, kphi, P, s = x
        omega = 2*np.pi*f 

        lat, lon, alt = GeoModel.convert_spherical_to_lla(r, theta, phi) # deg deg m

        # get electron density 
        n_e, n_e_derivatives = self.iono(lat, lon, alt)
        
        # calculate hamiltonian derivatives
        deltaH_deltar, deltaH_deltatheta, deltaH_deltaphi, \
                deltaH_deltaomega, \
                deltaH_deltakr, deltaH_deltaktheta, deltaH_deltakphi = Raytracer.hamiltonian_ionosphere_no_field_no_collisions(x, f, n_e, n_e_derivatives)

        return Raytracer.group_path_derivatives(x, omega, 
                                            deltaH_deltar, deltaH_deltatheta, deltaH_deltaphi, 
                                            deltaH_deltaomega, 
                                            deltaH_deltakr, deltaH_deltaktheta, deltaH_deltakphi)


    def derivatives(self, Pgroup:np.float64, x:np.ndarray, f:np.float64):
        """
        derivative with respect to group path length
        x is state space
        f is center frequency (Hz) 
        """
        omega = 2*np.pi*f 

        # unpack state space
        r, theta, phi, kr, ktheta, kphi, P, s = x
        lat, lon, alt = GeoModel.convert_spherical_to_lla(r, theta, phi) # deg deg m

        # get electron density 
        n_e, n_e_derivatives = self.iono(lat, lon, alt)

        # get index of refraction 
        n_refract, n_derivatives = self.neutral(lat, lon, alt)

        # calculate hamiltonian derivatives
        iono_hamiltonian_derivatives = Raytracer.hamiltonian_ionosphere_no_field_no_collisions(x, f, n_e, n_e_derivatives)
        neutral_hamiltonian_derivatives = Raytracer.hamiltonian_neutral_atmosphere(x, f, n_refract, n_derivatives)

        # sum hamiltonian derivatives 
        sum_hamiltonian_derivatives = tuple(x + y for x, y in zip(iono_hamiltonian_derivatives, neutral_hamiltonian_derivatives))
        deltaH_deltar, deltaH_deltatheta, deltaH_deltaphi, \
                        deltaH_deltaomega, \
                        deltaH_deltakr, deltaH_deltaktheta, deltaH_deltakphi = sum_hamiltonian_derivatives
        
        # if alt > 60e3:
        #     # get iono derivatives 
        #     derivatives = self.derivatives_ionosphere(Pgroup, x, f)
        # else: 
        #     # neutral atmosphere case 
        #     derivatives = self.derivatives_neutral(Pgroup,x, f)

        # return derivatives

        
        return Raytracer.group_path_derivatives(x, omega, 
                                            deltaH_deltar, deltaH_deltatheta, deltaH_deltaphi, 
                                            deltaH_deltaomega, 
                                            deltaH_deltakr, deltaH_deltaktheta, deltaH_deltakphi)


    
    @staticmethod
    @jit(nopython=True)
    def hamiltonian_neutral_atmosphere(x : list[float], f : float, n : float,  n_derivatives : np.ndarray):
        """
        x : state vector 
        f : frequency (Hz)
        n : refractive index of neutral atmosphere
        """

        # unpack statespace 
        r, theta, phi, kr, ktheta, kphi, P, s = x

        # assume neutral atmosphere is not dispersive 
        nprime = n 
        deltan_deltaVr = 0
        deltan_deltaVtheta = 0
        deltan_deltaVphi = 0
 
        omega = 2*np.pi*f 
 
        # eq 24 
        deltan_deltar = n_derivatives[0]
        deltaH_deltar = -n*deltan_deltar 

        # eq 25 
        deltan_deltatheta = n_derivatives[1]
        deltaH_deltatheta = -n*deltan_deltatheta

        # eq 26
        deltan_deltaphi = n_derivatives[2]
        deltaH_deltaphi = -n*deltan_deltaphi

        # eq 27
        deltaH_deltaomega = -n*nprime/omega

        # eq 28
        deltaH_deltakr = const.c**2/(omega**2) * kr - const.c/omega * n * deltan_deltaVr

        # eq 29
        deltaH_deltaktheta = const.c**2/(omega**2) * ktheta - const.c/omega * n * deltan_deltaVtheta

        # eq 30
        deltaH_deltakphi = const.c**2/(omega**2) * kphi - const.c/omega * n * deltan_deltaVphi

        return  deltaH_deltar, deltaH_deltatheta, deltaH_deltaphi, deltaH_deltaomega, deltaH_deltakr, deltaH_deltaktheta, deltaH_deltakphi
    
    @staticmethod
    @jit(nopython=True)
    def hamiltonian_ionosphere_no_field_no_collisions(x : list[float], f : float, n_e : float, n_e_derivatives : np.ndarray):
        """
        x : state vector
        f : frequency (Hz)
        n_e : electron density (m^-3)
        n_e_derivatives : electron density spatial derivatives [d_ne/dr, d_ne/dtheta, d_ne/dphi](m^-3) 
        """
        # unpack state space
        r, theta, phi, kr, ktheta, kphi, P, s = x
        omega = 2*np.pi*f
        # omega = np.sqrt(const.c**2 * (kr**2 + ktheta**2 + kphi**2) )

        deltaX_deltar =    const.e**2/(const.epsilon_0 * const.m_e * omega**2) * n_e_derivatives[0]
        deltaX_deltatheta =  const.e**2/(const.epsilon_0 * const.m_e * omega**2) * n_e_derivatives[1]
        deltaX_deltaphi =  const.e**2/(const.epsilon_0 * const.m_e * omega**2) * n_e_derivatives[2]

        nnprime = 1 
        
        ndeltan_deltaX = -0.5 # eq 82 
        ndeltan_deltar = ndeltan_deltaX * deltaX_deltar
        ndeltan_deltatheta = ndeltan_deltaX * deltaX_deltatheta
        ndeltan_deltaphi =  ndeltan_deltaX * deltaX_deltaphi
        ndeltan_deltaVr = 0
        ndeltan_deltaVtheta = 0
        ndeltan_deltaVphi = 0

        # eq 24 
        deltaH_deltar = -ndeltan_deltar 
        # eq 25 
        deltaH_deltatheta = -ndeltan_deltatheta
        # eq 26
        deltaH_deltaphi = -ndeltan_deltaphi
        # eq 27
        deltaH_deltaomega = -nnprime/omega
        # eq 28
        deltaH_deltakr = const.c**2/(omega**2) * kr - const.c/omega * ndeltan_deltaVr
        # eq 29
        deltaH_deltaktheta = const.c**2/(omega**2) * ktheta - const.c/omega * ndeltan_deltaVtheta
        # eq 30
        deltaH_deltakphi = const.c**2/(omega**2) * kphi - const.c/omega * ndeltan_deltaVphi

        return  deltaH_deltar, deltaH_deltatheta, deltaH_deltaphi, deltaH_deltaomega, deltaH_deltakr, deltaH_deltaktheta, deltaH_deltakphi

    @staticmethod
    @jit(nopython=True)
    def group_path_derivatives(x : list[float], omega : float, 
                                deltaH_deltar : float, deltaH_deltatheta : float, deltaH_deltaphi : float, 
                                deltaH_deltaomega : float, 
                                deltaH_deltakr : float, deltaH_deltaktheta : float, deltaH_deltakphi : float) -> list[float]:
        
        r, theta, phi, kr, ktheta, kphi, P, s = x

        # equation 9 
        dr_dPgroup = -1 /(const.c) * deltaH_deltakr / deltaH_deltaomega

        # equation 10 
        dtheta_dPgroup = -1 /(r*const.c) * deltaH_deltaktheta / deltaH_deltaomega

        # equation 11 
        dphi_dPgroup = -1 /(r*const.c*np.sin(theta)) * deltaH_deltakphi / deltaH_deltaomega 

        # equation 12 
        deltakr_dPgroup =  1/(const.c) * deltaH_deltar / deltaH_deltaomega \
                            + ktheta * dtheta_dPgroup \
                            + kphi * np.sin(theta) * dphi_dPgroup

        # equation 13 
        deltaktheta_dPgroup = 1/(r) * ( (1/const.c) * deltaH_deltatheta / deltaH_deltaomega \
                                        -ktheta * dr_dPgroup  \
                                        + kphi * r* np.cos(theta) * dphi_dPgroup)

        # equation 14 
        deltakphi_dPgroup = 1/(r*np.sin(theta)) * ( (1/const.c) * deltaH_deltaphi / deltaH_deltaomega \
                                                    - kphi * np.sin(theta) * dr_dPgroup \
                                                    - kphi * r * np.cos(theta) * dtheta_dPgroup)


        # equation 16 
        dP_dPgroup = const.c / omega * (kr * dr_dPgroup \
                                        + ktheta * r * dtheta_dPgroup \
                                        + kphi * r * np.sin(theta)*dphi_dPgroup) 

        # equation 18
        # ds_dPgroup = np.sqrt( (dr_dPgroup)**2 + r**2 * (dtheta_dPgroup)**2 + (r*np.sin(theta))**2 * (dphi_dPgroup)**2 )
        ds_dPgroup = np.sqrt( (dr_dPgroup)**2 + r**2*(dtheta_dPgroup)**2 + (r*np.sin(theta))**2 *(dphi_dPgroup)**2 ) 

        return [dr_dPgroup, dtheta_dPgroup, dphi_dPgroup, deltakr_dPgroup, deltaktheta_dPgroup, deltakphi_dPgroup, dP_dPgroup, ds_dPgroup]
    
    def hamiltonian_ionosphere_with_field_no_collisions(x, f, omega0, omegaH, n_e_derivatives, Bfield, Bfield_mag_derivatives, mode_sign = 1 ):
        """
        x : state vector
        f : frequency (Hz)
        omega0 : plasma frequency (Hz)
        omegaH : gyro frequency (Hz)
        """

        # ignore collisions 
        deltaZ_deltar = 0 
        deltaZ_deltatheta = 0
        deltaZ_deltaphi = 0
        
        # unpack state space
        r, theta, phi, kr, ktheta, kphi, P, s = x
        
        # gyrofrequency components
        omega = 2*np.pi*f
        omegaH_r = np.abs(const.e) * Bfield[0] / const.m_e
        omegaH_theta = np.abs(const.e) * Bfield[1] / const.m_e
        omegaH_phi = np.abs(const.e) * Bfield[2] / const.m_e

        # gyrofrequency ratio 
        Y = omegaH/omega # 36
        Yr = omegaH_r/omega 
        Ytheta = omegaH_theta/omega 
        Yphi = omegaH_phi/omega 


        # plasma frequency derivatives
        deltaX_deltar =    const.e**2/(const.epsilon_0 * const.m_e * omega**2) * n_e_derivatives[0]
        deltaX_deltatheta =  const.e**2/(const.epsilon_0 * const.m_e * omega**2) * n_e_derivatives[1]
        deltaX_deltaphi =  const.e**2/(const.epsilon_0 * const.m_e * omega**2) * n_e_derivatives[2]

        # gyrofrequency derivatives
        deltaY_deltar = Bfield_mag_derivatives[0] *  np.abs(const.e)/(omega * const.m_e)
        deltaY_deltatheta = Bfield_mag_derivatives[1] *  np.abs(const.e)/(omega * const.m_e)
        deltaY_deltaphi = Bfield_mag_derivatives[2] *  np.abs(const.e)/(omega * const.m_e)

        psi = GeoModel.angle_between_spherical_vectors(kr, ktheta, kphi, Bfield[0], Bfield[1], Bfield[2])       
        # deltan_deltaf = 1
        
        nsquared = 1 - 2*X *( 1 - X) /(2*(1-X) - Y_T**2 + mode_sign*np.sqrt(Y_T**4 + 4*Y_L**2 *(1-X)**2) ) # eq 34
        wave_array = np.asarray([kr, ktheta, kphi])
        
        # 
        V_R= 1
        V = 1

        X = omega0**2/(omega**2) # eq 35
        Y = omegaH/omega # 36
        Y_T = Y * np.sin(psi) # eq 38
        Y_L = Y*np.cos(psi) # eq 39
        U = 1 # eq 46 
        RAD = mode_sign* np.sqrt(Y_T**4 + 4*Y_L**2 *(U-X)**2) # eq 47
        D = 2*U*(U-X) - Y_T**2 + RAD # eq 48

        # eq 50
        ndeltan_deltaYLYTdeltaPsi = 2*X*(U-X)*(-1 + (Y_T**2 - 2*(U-X)**2)/RAD)/D**2

        ndeltan_deltaX = -( 2*U*(U-X) -Y_T**2 *(U-2*X) + (Y_T**4 * (U-2*X) + 4*Y_L**2 *(U-X)**3)/RAD   ) /D**2 #eq 54
        ndeltan_deltaY = (2*X*(U-X)/(D**2 + Y)) * ( -Y_T**2 + (Y_T**4 + 2*Y_L**2 *(U-X)**2)/RAD    ) #eq 55
        ndeltan_deltaZ = 1j *X/D**2 *(-2*(U-X)**2 - Y_T**2 + Y_T**4/RAD ) # eq 56
        
        # eq 57
        ndeltan_deltar = ndeltan_deltaX * deltaX_deltar \
                            + ndeltan_deltaY * deltaY_deltar \
                            + ndeltan_deltaZ * deltaZ_deltar \
                            + ndeltan_deltaYLYTdeltaPsi * Y_T * Y_L * deltapsi_deltar 
        
        # eq 58
        ndeltan_deltatheta = ndeltan_deltaX*deltaX_deltatheta \
                            + ndeltan_deltaY*deltaY_deltatheta \
                            + ndeltan_deltaZ*deltaZ_deltatheta \
                            + ndeltan_deltaYLYTdeltaPsi * Y_T * Y_L * deltapsi_deltatheta 
        # eq 59
        ndeltan_deltaphi = ndeltan_deltaX*deltaX_deltaphi \
                            + ndeltan_deltaY*deltaY_deltaphi \
                            + ndeltan_deltaZ*deltaZ_deltaphi \
                            + ndeltan_deltaYLYTdeltaPsi * Y_T * Y_L * deltapsi_deltaphi 
        # eq 60 
        ndeltan_deltaVr = ndeltan_deltaYLYTdeltaPsi *(V_R * Y_L**2/V**2 - (Y_L/V)*Yr)

        # eq 61
        ndeltan_deltaVtheta = ndeltan_deltaYLYTdeltaPsi *(V_theta * Y_L**2/V**2 - (Y_L/V)*Ytheta)

        # eq 62
        ndeltan_deltaVphi = ndeltan_deltaYLYTdeltaPsi *(V_phi * Y_L**2/V**2 - (Y_L/V)*Yphi)

        # eq 24 
        deltaH_deltar = -ndeltan_deltar 
        # eq 25 
        deltaH_deltatheta = -ndeltan_deltatheta
        # eq 26
        deltaH_deltaphi = -ndeltan_deltaphi
        # eq 27
        deltaH_deltaomega = -nnprime/omega
        # eq 28
        deltaH_deltakr = const.c**2/(omega**2) * kr - const.c/omega * n * ndeltan_deltaVr
        # eq 29
        deltaH_deltaktheta = const.c**2/(omega**2) * ktheta - const.c/omega * n * ndeltan_deltaVtheta
        # eq 30
        deltaH_deltakphi = const.c**2/(omega**2) * kphi - const.c/omega * n * ndeltan_deltaVphi

        return deltaH_deltar, deltaH_deltatheta, deltaH_deltaphi, deltaH_deltaomega, deltaH_deltakr, deltaH_deltaktheta, deltaH_deltakphi

    @staticmethod
    def hamiltonian_derivatives_Booker_quartic_with_field_no_collisions(x, f, n_e, n_e_derivatives, Bfield, Bfield_mag_derivatives, Bfield_jacobian):
        """
        
        """

        #unpack state space 
        r, theta, phi, kr, ktheta, kphi, P, s = x
        k = np.sqrt(kr**2 + ktheta**2 + kphi**2)
        U = 1 
        omega = 2*np.pi*f 

        # calculate gyro frequency and vector components
        B0 = np.linalg.norm(Bfield) # mag field strength 
        omegaH = np.abs(const.e) * B0 / const.m_e
        omegaH_r = np.abs(const.e) * Bfield[0] / const.m_e
        omegaH_theta = np.abs(const.e) * Bfield[1] / const.m_e
        omegaH_phi = np.abs(const.e) * Bfield[2] / const.m_e

        Y = omegaH/omega # 36
        Yr = omegaH_r/omega 
        Ytheta = omegaH_theta/omega 
        Yphi = omegaH_phi/omega 

        # magnetic field spatial derivatives
        delta_Y_jacobian = Bfield_jacobian * np.abs(const.e)/(omega * const.m_e) #

        # gyrofrequency derivatives
        deltaYr_deltar = delta_Y_jacobian[0,0]
        deltaYtheta_deltar = delta_Y_jacobian[1,0]
        deltaYphi_deltar = delta_Y_jacobian[2,0]
        deltaYr_deltatheta = delta_Y_jacobian[0,1]
        deltaYtheta_deltatheta = delta_Y_jacobian[1,1]
        deltaYphi_deltatheta = delta_Y_jacobian[2,1]
        deltaYr_deltaphi = delta_Y_jacobian[0,2]
        deltaYtheta_deltaphi = delta_Y_jacobian[1,2]
        deltaYphi_deltaphi = delta_Y_jacobian[2,2]
        
        deltaY_deltar = Bfield_mag_derivatives[0] *  np.abs(const.e)/(omega * const.m_e)
        deltaY_deltatheta = Bfield_mag_derivatives[1] *  np.abs(const.e)/(omega * const.m_e)
        deltaY_deltaphi = Bfield_mag_derivatives[2] *  np.abs(const.e)/(omega * const.m_e)


        # plasma frequency derivatives
        omega0 = np.sqrt(n_e * const.e**2 / (const.epsilon_0 * const.m_e) )
        X = omega0**2/(omega**2) # eq 35

        deltaX_deltar =    const.e**2/(const.epsilon_0 * const.m_e * omega**2) * n_e_derivatives[0]
        deltaX_deltatheta =  const.e**2/(const.epsilon_0 * const.m_e * omega**2) * n_e_derivatives[1]
        deltaX_deltaphi =  const.e**2/(const.epsilon_0 * const.m_e * omega**2) * n_e_derivatives[2]

        # eq 93 
        k_dot_Y = kr*Yr + ktheta*Ytheta + kphi*Yphi
        
        # eq 95
        A = (U-X)*U**2 - U*Y**2 

        # eq 96
        B = -2*U*(U-X)**2 + Y**2 * (2*U-X)

        # eq 97
        alpha = A*const.c**4*k**4 + X*(k_dot_Y)**2 *const.c**4 *k**2

        # eq 98
        beta = B*const.c**2 * k**2 *omega**2 - X*(k_dot_Y)**2*const.c**2*omega**2 

        # eq 99
        gamma = ((U-X)**2 - Y**2 )*(U-X)*omega**4

        # eq 100
        H = alpha+beta+gamma 

        # eq 101
        deltaH_deltaX = -U**2 *const.c**4 *k**4 \
                        + (k_dot_Y)**2 *const.c**4 *k**2 \
                        + (4*U*(U-X) - Y**2)*const.c**2 *k**2 *omega**2 \
                        - (k_dot_Y)**2 *const.c**2 *omega**2 \
                        + (-3*(U-X)**2 + Y**2)*omega**4

        # eq 102
        deltaH_deltaYsquared = -U*const.c**4 *k**4 \
                                + (2*U-X)*const.c**2 *k**2 *omega**2 \
                                - (U-X)*omega

        # eq 103 
        deltaH_deltakdotYsquared = X*const.c**2*(const.c**2*k**2 - omega**2)

        # eq 106
        deltaH_deltaksquared = 2*A*const.c**4*k**2 \
                                + X*(k_dot_Y)**2*const.c**4 \
                                +B*const.c**2*omega**2 

        #eq 108
        deltaH_deltar = deltaH_deltaX * deltaX_deltar \
                        + 2* deltaH_deltaYsquared * Y * deltaY_deltar  \
                        + 2 * deltaH_deltakdotYsquared *(k_dot_Y) * (kr * deltaYr_deltar + ktheta*deltaYtheta_deltar + kphi*deltaYphi_deltar)

        #eq 109
        deltaH_deltatheta = deltaH_deltaX * deltaX_deltatheta \
                            + 2* deltaH_deltaYsquared * Y * deltaY_deltatheta  \
                            + 2 * deltaH_deltakdotYsquared *(k_dot_Y) * (kr * deltaYr_deltatheta + ktheta*deltaYtheta_deltatheta + kphi*deltaYphi_deltatheta)

        # eq 110
        deltaH_deltaphi = deltaH_deltaX * deltaX_deltaphi + \
                            2* deltaH_deltaYsquared * Y * deltaY_deltaphi  + \
                            2 * deltaH_deltakdotYsquared *(k_dot_Y) * (kr * deltaYr_deltaphi + ktheta*deltaYtheta_deltaphi + kphi*deltaYphi_deltaphi)

        # eq 111
        deltaH_deltaomega = (2*beta + 4*omega)/omega \
                            - 2*deltaH_deltaX * X/omega \
                            - 2*deltaH_deltaYsquared * Y**2/omega \
                            - 2*deltaH_deltakdotYsquared * (k_dot_Y)**2/omega

        # eq 112 
        deltaH_deltakr = 2*deltaH_deltaksquared*kr \
                            + 2*k_dot_Y*deltaH_deltakdotYsquared*Yr 

        # eq 113 
        deltaH_deltaktheta = 2*deltaH_deltaksquared*ktheta \
                            + 2*k_dot_Y*deltaH_deltakdotYsquared*Ytheta

        # eq 114
        deltaH_deltakphi = 2*deltaH_deltaksquared*kphi \
                            + 2*k_dot_Y*deltaH_deltakdotYsquared*Yphi


        return deltaH_deltar, deltaH_deltatheta, deltaH_deltaphi, deltaH_deltaomega, deltaH_deltakr, deltaH_deltaktheta, deltaH_deltakphi

    def hamiltonian_appleton_hartree(self,x, f, n, psi, omega0, omegaH, O_mode = True):
        """
        x : state vector 
        f : frequency (Hz)
        n : refractive index of neutral atmosphere
        psi: angle between wavenormal and magnetic field (rad)
        """

        # unpack state space
        r, theta, phi, kr, ktheta, kphi, P, s = x


        omega = 2*np.pi*f 

        
        # calculate Vr, Vtheta, Vphi from equation 33 

        # equation 32 
        nprime = n + f* deltan_deltaf 
        
        X = omega0**2/(omega**2) # eq 35
        Y = omegaH/omega # 36

        Y_T = Y * np.sin(psi) # eq 38
        Y_L = Y*np.cos(psi) # eq 39

        U = 1 # eq 46 

        RAD = mode_sign* np.sqrt(Y_T**4 + 4*Y_L**2 *(U-X)**2) # eq 47

        D = 2*U*(U-X) - Y_T**2 + RAD # eq 48
        
        ndeltan_deltaX = -( 2*U*(U-X) -Y_T**2 *(U-2*X) + (Y_T**4 * (U-2*X) + 4*Y_L**2 *(U-X)**3)/RAD   ) /D**2 #eq 54

        ndeltan_deltaY = (2*X*(U-X)/(D**2 + Y)) * ( -Y_T**2 + (Y_T**4 + 2*Y_L**2 *(U-X)**2)/RAD    ) #eq 55

        # ndeltan_deltaZ = 1j #eq 56

        ndeltan_deltar = ndeltan_deltaX *deltaX_deltar + ndeltan_detlaY *deltaY_deltar + n/(Y_L*Y_T)*deltan_deltapsi *Y_L*Y_T*deltapsi_deltar    # eq 57 

        # eq 38

        # eq 24 
        deltaH_deltar = -ndeltan_deltar 

        # eq 25 
        deltaH_deltatheta = -ndeltan_deltatheta

        # eq 26
        deltaH_deltaphi = -ndeltan_deltaphi

        # eq 27
        deltaH_deltaomega = -n*nprime/omega

        # eq 28
        deltaH_deltakr = const.c**2/(omega**2) * kr - const.c/omega * n * deltan_deltaVr

        # eq 29
        deltaH_deltaktheta = const.c**2/(omega**2) * ktheta - const.c/omega * n * deltan_deltaVtheta

        # eq 30
        deltaH_deltakphi = const.c**2/(omega**2) * kphi - const.c/omega * n * deltan_deltaVphi

        return  deltaH_deltar, deltaH_deltatheta, deltaH_deltaphi, deltaH_deltaomega, deltaH_deltakr, deltaH_deltaktheta, deltaH_deltakphi
    
