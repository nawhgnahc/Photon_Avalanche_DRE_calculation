# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:28:59 2020

@author: Changhwan Lee
"""

import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt


################ Runge-Kutta 4th-order method
def runge_kutta_4(f, ti, yi, dt, *args) :
     k1 = dt * f(ti         , yi, *args)
     k2 = dt * f(ti + .5*dt , yi + .5*k1, *args) 
     k3 = dt * f(ti + .5*dt , yi + .5*k2, *args) 
     k4 = dt * f(ti + dt    , yi + k3, *args)
     yf = yi + (1./6.) * ( k1 + 2.*k2 + 2.*k3 + k4 )
     return yf


################ Calculation of the maximum rate paramter of the total rate tensor under pumping 
def rhs_calc_fordt(n, A_fl, A_pump_coef, Q22, Q23, s3):
    
    # Energy trnasfer rate tensor 
    A1_cr = np.array([[0.0, (Q22+ Q23)*n[1]       , -s3*n[0] ],\
                      [0.0, -(Q22 + 2.0*Q23)*n[1] ,2*s3*n[0] ],\
                      [0.0,  Q23*n[1]             , -s3*n[0] ]])
    
    
    
    A_total = A_fl + A1_cr + A_pump_coef    # total rate tensor 
    
    A_max = np.amax(np.absolute(A_total))   # maximum rate parameter of total rate tensor 
    
    return  A_max

################ Calculation of the total rate tensor under pumping 
def rhs_calc(t, n, A_fl, A_pump_coef, Q22, Q23, s3):
    

    A1_cr = np.array([[0.0, (Q22+ Q23)*n[1]       , -s3*n[0] ],\
                      [0.0, -(Q22 + 2.0*Q23)*n[1] ,2*s3*n[0] ],\
                      [0.0,  Q23*n[1]             , -s3*n[0] ]])  # Energy transfer rate tensor in [1/s] 

    A_total = A_fl + A1_cr + A_pump_coef    # total rate tensor in [1/s]
    
    RHS = np.matmul(A_total, n)             # Time derivative of the populations in [1/s]
    
    return RHS

################ Calculation of differential rate equation model
def tr_rate_equation_simul_ext(c_Tm, f_ab_1, f_ab_2, f_s3, f_q22, f_q23, 
                              f_w2_nr, f_w3nr,
                              dt_coeff, time_target, num_pump, p_start, p_end):
    
    
    P_pump_array_power = np.linspace(p_start, p_end, num=num_pump)  # The base-10 logarithm of pumping power density

    P_pump_array = 10**(P_pump_array_power) # Pumping power density in [W/m2]
    
    w2_r = 162.60                       # Radiative relaxation rate constant at the 3F4 state in [1/s]
    w2_nr = 162.60*f_w2_nr+ 0.00239     # Non-radiative relaxation rate constant at the 3F4 state in [1/s]  
    w3_r = 636.01                       # Radiative relaxation rate constant at the 3H4 state in [1/s]  
    w3_nr = 6.02 + 162.60*f_w3nr        # Non-radiative relaxation rate constant at the 3H4 state in [1/s]  
    
    w2 = w2_r + w2_nr                   # Total relaxation rate constant at the 3F4 state in [1/s]  

    print("c_Tm: ", c_Tm)

    b32 = 0.144                         # branching ratio from the 3H4 state to the 3F4 state
    
    v_pump  = np.array([9398])          # excitation wavenumber in [1/cm]
    dt_coeff = 0.4                      # time-step constant for variable step-size method
    hp=6.626e-34                        # Planck constant in [J*s]
    c=2.998e8                           # the speed of light in [m/s]
    Eph   = v_pump*1e2*c*hp             # photon energy in [J]
    Fai_pump_1 = P_pump_array/Eph[0]    # photon incident flux at Gaussian peak in [#/m^2/s]

    ab_cs_12_1 = f_ab_1*0.12e-25        # the ground-state absorption cross section for 3H6-3H4 transition in [cm^2]
    ab_cs_23_1 = f_ab_2*3.2e-25         # tne excited-stete absorption cross section for 3F4-3H4 transition in [cm^2]

    s3 = f_s3*(1.6*c_Tm**2)*1000.0      # the cross relaxation rate parameter in [1/s]

    Q22 = f_q22*((0.32*c_Tm**3)/(c_Tm**2 + 4.3**2))*1000.0 # energy tranfser upconversion rate parameter (3F4 + 3F4 -> 3H6 + 3H4) in [1/s]
    Q23 = f_q23*((0.09*c_Tm**3)/(c_Tm**2 + 4.3**2))*1000.0 # energy tranfser rate parameter (3F4 + 3F4 -> 3H6 + 3F5) in [1/s]

    index = 0
    
    
    A_fl_array =  np.array([[0, w2  , (1.0-b32)*w3_r  ],
                            [0, -w2 , b32*w3_r + w3_nr],
                            [0, 0   , -w3_r-w3_nr     ]]) # Total relaxation rate parameter tensor in [1/s]
    
    time_array = []                 # time data in [s]
    population_array = []           # populations with time, Te sum of population is set to 1
    population_steady_array = []    # populations at steady state

    for Fai_1 in Fai_pump_1:
        index += 1
        n_array = np.array([1.0, 0.0, 0.0]) # initial populations when time is zero, all the ions are at the ground state 3H6
        pump_coef = np.array([[-ab_cs_12_1,  0.0          , 0.0 ],\
                              [ ab_cs_12_1, -ab_cs_23_1   , 0.0 ],\
                              [0.0        ,  ab_cs_23_1   , 0.0 ]]) * Fai_1 # absorption rate tensor in [1/s]
        
        time_now = 0.0              #time at initial state in [s]
        data_now = n_array          #pupulation at time 0

        time_array_i = []           # time data under photon influx Fai_1 in [s]
        population_array_i = []     # pupulation with time under photon influx Fai_1
        
        time_array_i.append(time_now)
        population_array_i.append(data_now)
        print(index, time_target)
        
        ##### Calculation of time-resolved population until time_target with laser pumping
        while time_now < time_target:
            
            A_max_i = rhs_calc_fordt(n_array, A_fl_array, pump_coef, Q22, Q23, s3)  # Maximum rate parameter in rate parameter tensor in [1/s]
            
            dt = dt_coeff/A_max_i   # time step in [s]

            y_dt = runge_kutta_4(rhs_calc, time_now, data_now, dt,
                                 A_fl_array, pump_coef, Q22, Q23, s3)               # populations after the time step t + dt
  
            data_last = data_now
            data_now = y_dt
            time_last = time_now
            time_now = time_last + dt
            
            time_array_i.append(time_now)
            population_array_i.append(data_now)

        
        population_steady_array.append(data_now)
        
        pump_coef = np.array([[-ab_cs_12_1,  0.0          , 0.0 ],\
                      [ ab_cs_12_1, -ab_cs_23_1   , 0.0 ],\
                      [0.0        ,  ab_cs_23_1   , 0.0 ]]) * 0.0                   # After time_target, pump_power is assumed to be zero
        
        ##### Calculation of time-resolved population until time_target_2 without laser pumping
        while time_now < time_target_2:
            
            A_max_i = rhs_calc_fordt(n_array, A_fl_array, pump_coef, Q22, Q23, s3)  # Maximum rate parameter in rate parameter tensor in [1/s]

            dt = dt_coeff/A_max_i # time step in [s]

            y_dt = runge_kutta_4(rhs_calc, time_now, data_now, dt,
                                 A_fl_array, pump_coef, Q22, Q23, s3)               # populations after the time step t + dt
  
            data_last = data_now
            data_now = y_dt
            time_last = time_now
            time_now = time_last + dt
            
            time_array_i.append(time_now)
            population_array_i.append(data_now)

            k += 1
        
        time_array.append(np.array(time_array_i))
        population_array.append(np.array(population_array_i))

    time_array = np.array(time_array)
    population_array = np.array(population_array)
    P_pump_array = np.array(P_pump_array)
    population_steady_array = np.array(population_steady_array)
    
    population_3H4_array = population_steady_array[:, 2]
    
    ##### Integrated over the Gaussian profile to account for the Gaussian profile of the laser beam
    f_I = population_3H4_array/P_pump_array
    
    population_3H4_gaussian_array = cumtrapz(f_I, P_pump_array, initial=min(P_pump_array))
    
    population_3H4_gaussian_array = np.array(population_3H4_gaussian_array)
    return time_array, population_array, P_pump_array, population_steady_array, population_3H4_gaussian_array

#c_Tm_array = [2.25, 8.0, 8.0, 8.0, 20.0, 20.0, 100.0]

#label_array = np.array(["4%, Sample No. 2", "8%, Sample No. 3", 
#                        "8%, Sample No. 4", "8%, Sample No. 5", 
#                        "20%, Sample No. 6", "20%, Sample No. 7",
#                        , "100%, Sample No. 8"])
    
#f_w2_nr_array = np.array([0.35, 3.25, 0.25, 0.0, 6.0, 3.6, 5.3])
#f_w3_nr_array = np.array([0.6, 6.3, 0.5, 0.0, 14.0, 7.2, 10.6])
    
c_Tm_array = [2.25, 8.0, 8.0, 8.0, 20.0, 20.0]
label_array = np.array(["4%, Sample No. 2", "8%, Sample No. 3", 
                        "8%, Sample No. 4", "8%, Sample No. 5", 
                        "20%, Sample No. 6", "20%, Sample No. 7"])

f_ab_1 = 0.005  # a factor multiplied with the ground-state absorption cross section for 3H6-3H4 transition
f_ab_2 = 2.0    # a factor multiplied with tne excited-stete absorption cross section for 3F4-3H4 transition
f_s3 = 0.1      # a factor multiplied with the cross relaxation rate parameter
f_q22 = 0.08    # a factor multiplied with the energy tranfser rate parameter (3F4 + 3F4 -> 3H6 + 3H4)
f_q23 = 0.1     # a factor multiplied with the energy tranfser rate parameter (3F4 + 3F4 -> 3H6 + 3F4)

f_w2_nr_array = np.array([0.35, 3.25, 0.25, 0.0, 6.0, 3.6]) # a factor array multiplied with the non-radiative relaxation rate at the 3F4 level
f_w3_nr_array = np.array([0.6, 6.3, 0.5, 0.0, 14.0, 7.2])   # a factor array multiplied with the non-radiative relaxation rate at the 3H5 level


dt_coeff= 0.4                       # a factor for adaptive time step size
time_target = 1.0                   # calculation time for steady state under laser excitation in [s]
time_target_2 = time_target + 0.005 # calculation time for PL decay lifetime calculation
num_pump = 43                       # the size of pumping power array
p_start = 7.0                       # the base-10 logarithm of the lowest pumping power density
p_end = 9.1                         # the base-10 logarithm of the highest pumping power density
time_array = []                     # array of time data for all the samples in [s] 
population_array = []               # array of populations with time for all the samples
P_pump_array = []                   # array of pumping power density for all the samples
population_steady_array = []        # array of populations at steady state for all the samples 
population_3H4_gaussian_array = []  # array of populations at steady state after Gaussian beam correction for all the samples 



fig1 = plt.figure(figsize = (3.5, 3.5))
ax1 = fig1.add_subplot(1,1,1)

# Simulation of time-resolved PL of avalanching nanoparticles (ANPs) with different configurations
for i in np.arange(len(c_Tm_array)):
    c_Tm = c_Tm_array[i]
    f_w2_nr = f_w2_nr_array[i]
    f_w3_nr = f_w3_nr_array[i]
    print(label_array[i])
    x1, y1, x2, y2, y2_1 = tr_rate_equation_simul_ext(c_Tm, f_ab_1, f_ab_2, f_s3, f_q22, f_q23, f_w2_nr, f_w3_nr,
                                                  dt_coeff, time_target, num_pump, p_start, p_end)
    time_array.append(x1)
    population_array.append(y1)
    P_pump_array.append(x2)
    population_steady_array.append(y2)
    population_3H4_gaussian_array.append(y2_1)
    
    # plotting of the power-dependent luminescence of ANPs
    ax1.plot(x2, c_Tm*y2_1, label = label_array[i]) 


fontsize = 8
label_loc = 2

ax1.set_xlabel(r'Excitation Power (W/cm$^2$)', fontsize = fontsize)
ax1.set_ylabel(r'Emission intensity (a.u)' , fontsize = fontsize)
ax1.legend(markerscale=1, loc=label_loc ,frameon=False, numpoints=1, ncol=2, fontsize = fontsize)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(10**7.5, 10**9.134450)
plt.show()
      
