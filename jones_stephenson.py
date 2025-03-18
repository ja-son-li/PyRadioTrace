import numpy as np
from models import GeoModel
import scipy.constants as const
from scipy.integrate import odeint 
from numba import jit
import scipy.optimize

class Raytracer():

    def __init__(self, iono, magneto, neutral, **kwargs):
        self.iono = iono
        self.magneto = magneto
        self.neutral = neutral
        self.c = const.speed_of_light
    
        # state space is [1. r : spherical coordinate radius (m), 
        #                 2. theta : spherical coordinate longitude (rad), 
        #                 3. phi : spherical coordinate colatitude (rad),
        #                 4. kr, 
        #                 5. ktheta, 
        #                 6. kphi : spherical coordinate wave normal  
        #                 7. P : path length (m), 
        #                 8. s: geometric path length (m)]
        # 
        # kr ktheta kphi are normalized such that kr**2 + ktheta**2 + kphi**2 = omega**2/c**2

    def ray_home(self, transmit_lat, transmit_lon, transmit_alt, receive_lat, receive_lon, receive_alt, f, hmin = 1, resolution_m = 5, resolution_angle_deg = 0.1):
        """
        
        """

        # convert geodetic coordinates to spherical coordinates
        x, y, z = GeoModel.convert_lla_to_ecef(transmit_lat, transmit_lon, transmit_alt)
        r, theta, phi = GeoModel.convert_ecef_to_spherical(x, y, z)


        # convert az, el look angle to spherical vectors 
        omega = 2*np.pi*f
        
        # first guess 
        az, el, range_val = GeoModel.geodetic2aer(transmit_lat, transmit_lon, transmit_alt, receive_lat, receive_lon, receive_alt)
        group_path_distances = np.arange(0,range_val+1e3, resolution_m)
        initial_guess = [az, el]
        #optimize 
        optimize_func = lambda az_el_list : self.ray_distance_to_target(transmit_lat, transmit_lon, transmit_alt, 
                                                                                receive_lat, receive_lon, receive_alt, 
                                                                                az_el_list[0], az_el_list[1], f,  group_path_distances=group_path_distances, hmin = hmin)
        
        # nelder mead simplex optimization for minimum distance 
        minimum = scipy.optimize.fmin(optimize_func, initial_guess,  xtol=resolution_angle_deg, ftol=resolution_m, disp=True)

        return minimum
     
    def ray_distance_to_target(self, transmit_lat, transmit_lon, transmit_alt, 
                               receiver_lat, receiver_lon, receiver_alt,
                                az, el, f, group_path_distances=np.arange(0,2000,5), hmin = 100):
        
        solution = self.ray_propagate(transmit_lat, transmit_lon, transmit_alt, az, el, f, group_path_distances, hmin)

        # convert geodetic coordinates to spherical coordinates
        x, y, z = GeoModel.convert_lla_to_ecef(receiver_lat, receiver_lon, receiver_alt)
        r_rx, theta_rx, phi_rx = GeoModel.convert_ecef_to_spherical(x, y, z)
        distances = self.distance_between_two_spherical_vectors(solution[:,0], solution[:,1], solution[:,2], r_rx, theta_rx, phi_rx)

        return np.min(distances) 
    
    @staticmethod
    def distance_between_two_spherical_vectors(r1, theta1, phi1, r2, theta2, phi2):
        """
        Calculate the distance between two spherical vectors.
        """

        return np.sqrt( r1**2 + r2**2 
                       - 2*r1*r2*(np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2)) 
                       + np.cos(theta1)*np.cos(theta2))
    

    def ray_propagate(self, transmit_lat, transmit_lon, transmit_alt, az, el, f, group_path_distances=np.arange(0,2000,5), hmin = 100):
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
        kr, ktheta, kphi = self.azel_to_spherical_vector(transmit_lat, transmit_lon, transmit_alt, az, el)
        norm = np.sqrt(kr**2 + ktheta**2 + kphi**2)
        kr = kr/norm * omega/const.c
        ktheta = ktheta/norm *omega/const.c
        kphi = kphi/norm *omega/const.c

        x0 = [r, theta, phi, kr, ktheta, kphi, 0,0]

        solution = odeint(self.derivatives_neutral, x0, group_path_distances, args=(f,), rtol=1e-6, atol=1e-6, hmin =hmin)

        return solution
    
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
    
    def derivatives_neutral(self, x, Pgroup, f):
        
        # import pdb; pdb.set_trace()
        # unpack state space
        r, theta, phi, kr, ktheta, kphi, P, s = x

        omega = 2*np.pi*f 

        # calculate geodetic coordinates 
        # x_ecef, y_ecef, z_ecef = GeoModel.convert_spherical_to_ecef(r, np.rad2deg(theta), 90 - np.rad2deg(phi))

        x_ecef, y_ecef, z_ecef = GeoModel.convert_spherical_to_ecef(r, theta, phi)
        lat, lon, alt = GeoModel.convert_ecef_to_lla(x_ecef, y_ecef, z_ecef) # deg deg m 

        # calculate index of refraction and spatial derivatives
        if alt > 60e3:
        # if False:
            n_refract, n_derivatives = 1, np.zeros(3)
        else:
            n_refract, n_derivatives = self.neutral(lat, lon, alt)

            # dr_dPgroup = kr /omega * const.c  
            # dtheta_dPgroup = ktheta /omega * const.c  
            # dphi_dPgroup = kphi /omega * const.c  
            # deltakr_dPgroup = 0
            # deltaktheta_dPgroup = 0
            # deltakphi_dPgroup = 0
            # dP_dPgroup = -const.c / omega * (kr * dr_dPgroup + ktheta * r * dtheta_dPgroup + kphi * r * np.sin(theta)*dphi_dPgroup) 
            # ds_dPgroup = np.sqrt( (dr_dPgroup)**2 + r**2 * (dtheta_dPgroup)**2 + (r*np.sin(theta))**2 * (dphi_dPgroup)**2 )

        # calculate hamiltonian derivatives
        deltaH_deltar, deltaH_deltatheta, deltaH_deltaphi, deltaH_deltaomega, deltaH_deltakr, deltaH_deltaktheta, deltaH_deltakphi = self.hamiltonian_neutral_atmosphere(x, f, n_refract, n_derivatives)

        # equation 9 
        dr_dPgroup = - 1 /(self.c) * deltaH_deltakr / deltaH_deltaomega

        # equation 10 
        dtheta_dPgroup = - 1 /(r*self.c) * deltaH_deltaktheta / deltaH_deltaomega

        # equation 11 
        dphi_dPgroup = - 1 /(r*self.c*np.sin(theta)) * deltaH_deltakphi / deltaH_deltaomega 

        # equation 12 
        deltakr_dPgroup =  1/(self.c) * deltaH_deltar / deltaH_deltaomega + ktheta * dtheta_dPgroup + kphi * np.sin(theta) * dphi_dPgroup

        # equation 13 
        deltaktheta_dPgroup = 1/(r) * ( (1/self.c)* deltaH_deltatheta / deltaH_deltaomega - ktheta * dr_dPgroup + kphi * r* np.cos(theta) * dphi_dPgroup)

        # equation 14 
        deltakphi_dPgroup = 1/(r*np.sin(theta)) * ( (1/self.c) * deltaH_deltaphi / deltaH_deltaomega - kphi * np.sin(theta) * dr_dPgroup - kphi * r * np.cos(theta) * dtheta_dPgroup)

        # equation 16 
        dP_dPgroup = -const.c / omega * (kr * dr_dPgroup + ktheta * r * dtheta_dPgroup + kphi * r * np.sin(theta)*dphi_dPgroup) 

        # equation 18
        ds_dPgroup = np.sqrt( (dr_dPgroup)**2 + r**2 * (dtheta_dPgroup)**2 + (r*np.sin(theta))**2 * (dphi_dPgroup)**2 )

        return [dr_dPgroup, dtheta_dPgroup, dphi_dPgroup, deltakr_dPgroup, deltaktheta_dPgroup, deltakphi_dPgroup, dP_dPgroup, ds_dPgroup]
    
    

    def derivatives(self, x, f, o_mode = True):
        """
        derivative with respect to group path length
        x is state space
        f is center frequency (Hz) 
        """
        # O mode 
        if o_mode:
            mode_sign = 1
        else:
            mode_sign = -1


        # unpack state space
        r, theta, phi, kr, ktheta, kphi, P, s = x

        # calculate geodetic coordinates 
        x, y, z = GeoModel.convert_spherical_to_ecef(r, np.rad2deg(theta), 90 - np.rad2deg(phi))
        lat, lon, alt = GeoModel.convert_ecef_to_lla(x, y, z) # deg deg m 

        if alt > 50e3:

        # get the magnetic field
        Br, Btheta, Bphi  = self.magneto.get_magnetic_field(lat, lon, alt/1e3)
        B0 = np.sqrt(Br**2 + Btheta**2 + Bphi**2)

        # get electron density 
        n_e = self.iono.get_electron_density(lat, lon, alt/1e3)

        # calculate the angle between the magnetic field and the ray 
        angle_between_magfield_wavevector = self.angle_between_spherical_vectors(ktheta, kphi, kr, Btheta, Bphi, Br) # rad

        # calculate the refractive index of ionosphere
        omega = np.sqrt(const.c*(kr**2 + ktheta**2 + kphi**2))

        # omega = 2*np.pi*f 

        # electron plasma frequency 
        omega0 = np.sqrt(n_e * const.e**2 / (const.epsilon_0 * const.m_e) )
        
        # electron gyrofrequency 
        omegaH = B0 * const.e / (const.m_e)
        X = omega0**2/(omega**2) 
        Y = omegaH/omega
        Yr = Br/omega
        Ytheta = Btheta/omega
        Yphi = Bphi/omega



        # n2 = 1 - ( X*(1-X)
        #           / (1 - X - 0.5*Y**2 * (np.sin(angle_between_magfield_wavevector))**2 
        #                      + mode_sign*np.sqrt(  (0.5*Y**2 * (np.sin(angle_between_magfield_wavevector))**2)**2  
        #                                     +(1-X)**2 *Y**2 * (np.cos(angle_between_magfield_wavevector))**2)  )   )
        # x mode 

        # calculate the refractive index of neutral atmosphere 


        # determine which regime the ray is in 

        use_iono = True 
        if use_iono:
            deltaH_deltar, deltaH_deltatheta, deltaH_deltaphi, deltaH_deltaomega, deltaH_deltakr, deltaH_deltaktheta, deltaH_deltakphi = self.hamiltonian_derivatives_Booker_quartic_with_field_no_collisions(x, f,angle_between_magfield_wavevector, omega0, omegaH)
        else:
            partials = self.hamiltonian_neutral_atmosphere(x, f)

        # equation 9 
        dr_dPgroup = - 1 /(self.c) * deltaH_deltakr / deltaH_deltaomega

        # equation 10 
        dtheta_dPgroup = - 1 /(r*self.c) * deltaH_deltaktheta / deltaH_deltaomega

        # equation 11 
        dphi_dPgroup = - 1 /(r*self.c*np.sin(theta)) * deltaH_deltakphi / deltaH_deltaomega 

        # equation 12 
        deltakr_dPgroup =  1/(self.c) * deltaH_deltar / deltaH_deltaomega + ktheta * dtheta_dPgroup + kphi * np.sin(theta) * dphi_dPgroup

        # equation 13 
        deltaktheta_dPgroup = 1/(r) * ( (1/self.c)* deltaH_deltatheta / deltaH_deltaomega - ktheta * dr_dPgroup + kphi * r* np.cos(theta) * dphi_dPgroup)

        # equation 14 
        deltakphi_dPgroup = 1/(r*np.sin(theta)) * ( (1/self.c) * deltaH_deltaphi / deltaH_deltaomega - kphi * np.sin(theta) * dr_dPgroup - kphi * r * np.cos(theta) * dtheta_dPgroup)

        # equation 16 
        dP_dPgroup = -const.c / omega * (kr * dr_dPgroup + ktheta * r * dtheta_dPgroup + kphi * r * np.sin(theta)*dphi_dPgroup) 

        # equation 18
        ds_dPgroup = np.sqrt( (dr_dPgroup)**2 + r**2 * (dtheta_dPgroup)**2 + (r*np.sin(theta))**2 * (dphi_dPgroup)**2 )

        return [dr_dPgroup, dtheta_dPgroup, dphi_dPgroup, deltakr_dPgroup, deltaktheta_dPgroup, deltakphi_dPgroup, dP_dPgroup, ds_dPgroup]
    
    @staticmethod
    def hamiltonian_neutral_atmosphere(x, f, n,  n_derivatives):
        """
        x : state vector 
        f : frequency (Hz)
        n : refractive index of neutral atmosphere
        """

        # unpack statespace 
        r, theta, phi, kr, ktheta, kphi, P, s = x

        # assume 
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

    def hamiltonian_derivatives_Booker_quartic_with_field_no_collisions(self, omega0,omegaH, Yr, Ytheta, Yphi, f, x):
        r, theta, phi, kr, ktheta, kphi, P, s = x
        k = np.sqrt(kr**2 + ktheta**2 + kphi**2)
        U = 1 
        omega = 2*np.pi*f 

        X = omega0**2/(omega**2) # eq 35
        Y = omegaH/omega # 36


        # 
        deltaX_deltar = 1
        deltaX_deltatheta = 1
        deltaX_deltaphi = 1


        # magnetic field spatial derivatives 
        dBr_dr, dBr_dtheta, dBr_dphi, dBtheta_dr, dBtheta_dtheta, dBtheta_dphi, dBphi_dr, dBphi_dtheta, dBphi_dphi = self.mag_field_derivative(x)

        # electron plasma frequency spatial derivatives
        deltaYr_deltar = 1
        deltaYtheta_deltar = 1
        deltaYphi_deltar = 1
        
        deltaYr_deltatheta = 1
        deltaYtheta_deltatheta = 1
        deltaYphi_deltatheta = 1

        deltaYr_deltaphi = 1
        deltaYtheta_deltaphi = 1
        deltaYphi_deltaphi = 1
        
        deltaY_deltar = 1
        deltaY_deltatheta = 1
        deltaY_deltaphi = 1

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
    
    # def ne_derivative(self, x):

    #     return dNe_dr, dNe_dtheta, dNe_dphi
    
    # def mag_field_derivative(self,r, theta, phi):
        
    #     delta_r = 0.01 
    #     delta_theta = 0.01 
    #     delta_phi = 0.01 

    #     # r derivatives
   
   
    #     return dBr_dr, dBr_dtheta, dBr_dphi, dBtheta_dr, dBtheta_dtheta, dBtheta_dphi, dBphi_dr, dBphi_dtheta, dBphi_dphi


    @staticmethod
    def angle_between_spherical_vectors(theta1, phi1, r1, theta2, phi2, r2):
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

