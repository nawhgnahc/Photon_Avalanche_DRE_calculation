import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt



def runge_kutta_4(f, ti, yi, dt, *args) :
     k1 = dt * f(ti         , yi, *args)
     k2 = dt * f(ti + .5*dt , yi + .5*k1, *args) 
     k3 = dt * f(ti + .5*dt , yi + .5*k2, *args) 
     k4 = dt * f(ti + dt    , yi + k3, *args)
     yf = yi + (1./6.) * ( k1 + 2.*k2 + 2.*k3 + k4 )
     return yf

def rhs_calc_fordt(n, A_fl, A_pump_coef, Q22, Q23, s3):

    A1_cr = np.array([[0.0, (Q22+ Q23)*n[1]       , -s3*n[0] ],\
                      [0.0, -(Q22 + 2.0*Q23)*n[1] ,2*s3*n[0] ],\
                      [0.0,  Q23*n[1]             , -s3*n[0] ]])
    
    

    A_total = A_fl + A1_cr + A_pump_coef
    
    A_max = np.amax(np.absolute(A_total))
    
    return  A_max

def rhs_calc(t, n, A_fl, A_pump_coef, Q22, Q23, s3):

    A1_cr = np.array([[0.0, (Q22+ Q23)*n[1]       , -s3*n[0] ],\
                      [0.0, -(Q22 + 2.0*Q23)*n[1] ,2*s3*n[0] ],\
                      [0.0,  Q23*n[1]             , -s3*n[0] ]])

    A_total = A_fl + A1_cr + A_pump_coef
    
    RHS = np.matmul(A_total, n)
    
    return RHS

def tr_rate_equation_simul_ext(c_Tm, f_ab_1, f_ab_2, f_s3, f_q22, f_q23, 
                              f_w2_nr, f_w3nr,
                              dt_coeff, time_target, num_pump, p_start, p_end):

    P_pump_array_power = np.linspace(p_start, p_end, num=num_pump)  # in [W/m2]

    P_pump_array = 10**(P_pump_array_power)
    
    w2_r = 162.60  
    w2_nr = 162.60*f_w2_nr+ 0.00239     
    w3_r = 636.01
    w3_nr = 6.02 + 162.60*f_w3nr
    
    w2 = w2_r + w2_nr

    print("c_Tm: ", c_Tm)

    #b32 = 0.18
    b32 = 0.144
    
    v_pump  = np.array([9398])
    dt_coeff = 0.4
    hp=6.626e-34 #in [J*s]cal
    c=2.998e8 #in [m/s]
    Eph   = v_pump*1e2*c*hp
    Fai_pump_1 = P_pump_array/Eph[0]

    ab_cs_12_1 = f_ab_1*0.12e-25
    ab_cs_23_1 = f_ab_2*3.2e-25

    s3 = f_s3*(1.6*c_Tm**2)*1000.0

    Q22 = f_q22*((0.32*c_Tm**3)/(c_Tm**2 + 4.3**2))*1000.0
    Q23 = f_q23*((0.09*c_Tm**3)/(c_Tm**2 + 4.3**2))*1000.0

    index = 0
    
    A_fl_array =  np.array([[0, w2  , (1.0-b32)*w3_r  ],
                            [0, -w2 , b32*w3_r + w3_nr],
                            [0, 0   , -w3_r-w3_nr     ]])
    
    time_array = []
    population_array = []
    population_steady_array = []

    
    for Fai_1 in Fai_pump_1:
        index += 1
        n_array = np.array([1.0, 0.0, 0.0])
        pump_coef = np.array([[-ab_cs_12_1,  0.0          , 0.0 ],\
                              [ ab_cs_12_1, -ab_cs_23_1   , 0.0 ],\
                              [0.0        ,  ab_cs_23_1   , 0.0 ]]) * Fai_1 
        
        time_now = 0.0
        data_now = n_array
        k = 0
        
        time_array_i = []
        population_array_i = []
        
        time_array_i.append(time_now)
        population_array_i.append(data_now)
        print(time_now, time_target)
        while time_now < time_target:
            
            A_max_i = rhs_calc_fordt(n_array, A_fl_array, pump_coef, Q22, Q23, s3)

            dt = dt_coeff/A_max_i
            #print("dt: ", dt)
            y_dt = runge_kutta_4(rhs_calc, time_now, data_now, dt,
                                 A_fl_array, pump_coef, Q22, Q23, s3)
  
            data_last = data_now
            data_now = y_dt
            time_last = time_now
            time_now = time_last + dt
            
            time_array_i.append(time_now)
            population_array_i.append(data_now)
            #print("k, time: ", k, time_now)
            k += 1
        
        population_steady_array.append(data_now)
        
        pump_coef = np.array([[-ab_cs_12_1,  0.0          , 0.0 ],\
                      [ ab_cs_12_1, -ab_cs_23_1   , 0.0 ],\
                      [0.0        ,  ab_cs_23_1   , 0.0 ]]) * 0.0
        
        while time_now < time_target_2:
            
            A_max_i = rhs_calc_fordt(n_array, A_fl_array, pump_coef, Q22, Q23, s3)

            dt = dt_coeff/A_max_i
            #print("dt: ", dt)
            y_dt = runge_kutta_4(rhs_calc, time_now, data_now, dt,
                                 A_fl_array, pump_coef, Q22, Q23, s3)
  
            data_last = data_now
            data_now = y_dt
            time_last = time_now
            time_now = time_last + dt
            
            time_array_i.append(time_now)
            population_array_i.append(data_now)
            #print("k, time: ", k, time_now)
            k += 1
        
        time_array.append(np.array(time_array_i))
        population_array.append(np.array(population_array_i))

    time_array = np.array(time_array)
    population_array = np.array(population_array)
    P_pump_array = np.array(P_pump_array)
    population_steady_array = np.array(population_steady_array)
    
    population_3H4_array = population_steady_array[:, 2]
    
    print(population_array)
    print(P_pump_array)
    
    f_I = population_3H4_array/P_pump_array
    
    population_3H4_gaussian_array = cumtrapz(f_I, P_pump_array, initial=min(P_pump_array))
    
    population_3H4_gaussian_array = np.array(population_3H4_gaussian_array)
    return time_array, population_array, P_pump_array, population_steady_array, population_3H4_gaussian_array

c_Tm_array = [2.25, 8.0, 8.0, 8.0, 20.0, 20.0, 100.0]
#c_Tm_array = [4.0, 8.0]
f_ab_1 = 0.005
f_ab_2 = 2.0
f_s3 = 0.1
f_q22 = 0.08
f_q23 = 0.1

f_w2_nr_array = np.array([0.35, 3.25, 0.25, 0.0, 6.0, 3.6, 5.3])
f_w3_nr_array = np.array([0.6, 6.3, 0.5, 0.0, 14.0, 7.2, 10.6])

#f_w2_nr_array = np.array([0.35, 3.15])
#f_w3_nr_array = np.array([0.7, 6.3])

dt_coeff= 0.4
time_target = 1.0
time_target_2 = 1.0 + 0.005
num_pump = 43
p_start = 7.0
p_end = 9.1
time_array = []
population_array = []
P_pump_array = []
population_steady_array = []
population_3H4_gaussian_array = []

fig1 = plt.figure(figsize = (3.5, 3.5))
ax1 = fig1.add_subplot(1,1,1)
for i in np.arange(len(c_Tm_array)):
    c_Tm = c_Tm_array[i]
    f_w2_nr = f_w2_nr_array[i]
    f_w3_nr = f_w3_nr_array[i]
    x1, y1, x2, y2, y2_1 = tr_rate_equation_simul_ext(c_Tm, f_ab_1, f_ab_2, f_s3, f_q22, f_q23, f_w2_nr, f_w3_nr,
                                                  dt_coeff, time_target, num_pump, p_start, p_end)
    time_array.append(x1)
    population_array.append(y1)
    P_pump_array.append(x2)
    population_steady_array.append(y2)
    population_3H4_gaussian_array.append(y2_1)
    
    ax1.plot(x2, y2_1)
#fig1 = plt.figure(figsize = (3.5, 3.5))
#ax1 = fig1.add_subplot(1,1,1)
#ax1.plot(P_pump_array[0], population_3H4_gaussian_array[0])
#ax1.plot(P_pump_array[1], population_3H4_gaussian_array[1])
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(10**7.5, 10**9.134450)
plt.show()
      
    
                    
