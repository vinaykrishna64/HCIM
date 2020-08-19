# -*- coding: utf-8 -*-
"""
Created on Fri May 29 22:28:28 2020

@author: vinay
"""
# checking if dependencies are installed, if not installing them
import os
import sys
import subprocess
import pkg_resources
import datetime
required = {'tqdm', 'numpy','scipy'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
        print('Installing missing libraries in this folder')
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing,'-t','.'], stdout=subprocess.DEVNULL)


# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import SWIC_IO_module as IO # input-output module
import SWIC_core_module as CORE# core module   

from tqdm import tqdm         
if os.path.exists('LOG_FILE.txt'):
  open('LOG_FILE.txt', 'w').close()
else:
  pass

# loading inputs
current_time = datetime.datetime.now()  
IO.write_log('#'*50)
IO.write_log('started execution at\t\t ::' + str(current_time))
IO.write_log('#'*50)
IO.write_log('Hybrid Coastal Inundation Model')
IO.write_log('Author: Vinay krishna')
IO.write_log('#'*50)
IO.load_data()

IO.display_inputs()
IO.write_inputs()
IO.make_folder('outputs')
IO.write_log('--->\t outputs folder is created')


#Initializing computatonal grid variables


if IO.inpt['mode'] == '1D':
    [nx,X,d,h_flow,elv_cu,U_cu,elv_pr,U_pr,wave_break,wet,Cf] = CORE.intialize_Cgrid(IO.inpt)
    IO.write_log('--->\t Grid matrices Initialized')
    [d,Cf] = IO.interpolate_data(X,d,Cf)
    IO.write_log('--->\t Depth and Cf interpolated')
elif IO.inpt['mode'] == '2D':
    [nx,ny,X,Y,d,h_flow_x,h_flow_y,elv_cu,U_cu,V_cu,elv_pr,U_pr,V_pr,wave_break,wet,Cf] = CORE.intialize_Cgrid(IO.inpt)
    IO.write_log('--->\t Grid matrices Initialized')
    [d,Cf] = IO.interpolate_data(X,d,Cf,Y)
    IO.write_log('--->\t Depth and Cf interpolated')

if IO.inpt['mode'] == '1D':
    bot_grad = np.gradient(d)
    ind = np.where(X >= IO.inpt['Otop'] )[0][0]
    IO.inpt['beach_slope'] = np.mean(bot_grad[0:ind+1])
    IO.inpt['N_i'] = IO.inpt['beach_slope']/ np.sqrt(2*IO.inpt['amp']/(1.56 * IO.inpt['TP']**2))
    IO.inpt['gamma'] = 1.16*(IO.inpt['N_i']**0.22)
    IO.inpt['bifur_depth'] = -(0.798*4*(IO.inpt['amp']**2) + 3.514) * np.exp(-1.459*IO.inpt['gamma'])
    adv = np.zeros(len(U_pr))
    adv[0:ind+1] = np.ones((ind+1))
    N_i = np.ones(len(U_pr))
    gamma = np.ones(len(U_pr))
    N_i[0:ind] = bot_grad[0:ind]/ np.sqrt(2*IO.inpt['amp']/(1.56 * IO.inpt['TP']**2))
    gamma[0:ind] = 1.16*(N_i[0:ind]**0.22)

if IO.inpt['mode'] == '1D':
    [h_flow,elv_pr,wet] = CORE.intial_set_elvetaions_n_heights(d,elv_pr,wet,h_flow,IO.inpt['mode'])
elif IO.inpt['mode'] == '2D':
    [h_flow_x,h_flow_y,elv_pr,wet] = CORE.intial_set_elvetaions_n_heights(d,elv_pr,wet,h_flow_x,IO.inpt['mode'],h_flow_y)
IO.write_log('--->\t Intial water height are set')



##--------------------------------------------------------main loop--------------------------------------------------------


IO.write_log('--->\t Model is being simulated')



if IO.inpt['mode'] == '1D':
    
    iter = 0    # iteration counter
    t = 0       # curretn time
    T = np.array(0) # time tracker
    del_t = IO.inpt['CFL'] * IO.inpt['del_x']/np.sqrt(IO.inpt['g']* np.max(h_flow))  # initial time step for CFL = CFL_set
    IO.write_log('--->\t initial del_t set as --->\t' +str(del_t))
    CFL = IO.inpt['CFL']
    tq = t- del_t/2
    # for progress bar
    progress_bar = tqdm(total = np.ceil(IO.inpt['total_time']/del_t), position = 0, leave = False)


    IO.write_outputs([h_flow,U_pr,elv_pr,wave_break],['h_flow','U','zeta','break_switch'],t)
    IO.write_log('--->\t Outputs written for simulation time\t' + str(t))

    if  IO.inpt['write_interval'] < del_t:
         IO.inpt['write_interval'] = del_t
    next_write = 0 + IO.inpt['write_interval']
    nu_eddy = np.zeros((len(elv_cu)))
    P_nh_pr = np.zeros((len(elv_cu)))
    P_nh_cu = np.zeros((len(elv_cu)))
    W_cu = np.zeros((len(elv_cu)))
    W_pr = np.zeros((len(elv_cu)))
    W_b_cu = np.zeros((len(elv_cu)))
    W_b_pr = np.zeros((len(elv_cu)))
    h = np.zeros((len(elv_cu)))
    
    while True:
        
                                                      ## boundary condition
        # incoming flow and open BC for weak reflection
        
        elv_plus = CORE.Zeta_plus(t-del_t)
          
        u_plus = np.sqrt(IO.inpt['g']/h_flow[0]) * elv_plus
        u_b =   np.sqrt(IO.inpt['g']/h_flow[0]) *(2* elv_plus - elv_pr[0] )

       
        U_pr[0] = u_b
        
        elv_plus = CORE.Zeta_plus(t)
          
        u_plus = np.sqrt(IO.inpt['g']/h_flow[0]) * elv_plus
        u_b =   np.sqrt(IO.inpt['g']/h_flow[0]) *(2* elv_plus - elv_pr[0] )
        
        U_cu[0] = u_b 
        #solve fluxes
        [U_cu,P_nh_cu,W_cu,W_b_cu] = CORE.Solve_1D_NHs_velocities(U_cu,elv_pr,U_pr,h_flow,d,wet,del_t,wave_break,nu_eddy,Cf,P_nh_cu,P_nh_pr,W_cu,W_pr,bot_grad,W_b_pr,W_b_cu,adv)
        #solve elevation
        elv_cu = CORE.Solve_1D_elevation(elv_cu,elv_pr,U_cu,wet,del_t,h_flow)
      
        [h_flow,elv_cu,wet,U_cu] = CORE.set_water_height(d,elv_cu,wet,h_flow,U_cu,'1D')  

        [wet,U_cu,h_flow,elv_cu] = CORE.check_flooding(wet,U_cu,d,h_flow,elv_cu,'1D')
        ## wet check
        

        if IO.inpt['exit_BC'] == 'O':        #open boundary for flat bed
            U_cu[-1] = U_cu[-2]
        elif IO.inpt['exit_BC'] == 'C':     #closed boundary for flat bed
            U_cu[-1] = 0

        
        if IO.inpt['break'] == 1: 
            nu_eddy = CORE.eddy_viscosity_model(nu_eddy,U_cu,IO.inpt['del_x'],'1D',d)           
            [wave_break] = CORE.check_onset_of_breaking(d,U_cu,elv_cu,wave_break,h_flow,X,wet,gamma)       
           
        # updating the previous time step to current time step       
        U_pr = U_cu
        elv_pr = elv_cu
        W_pr = W_cu
        W_b_pr = W_b_cu
        P_nh_pr = P_nh_cu
       
        
             
                                        ## necessary counters and trackers

        iter += 1 #iteration counter update
        
        if del_t >IO.inpt['CFL'] * IO.inpt['del_x']/np.sqrt(IO.inpt['g']* np.max(h_flow)): 
            del_t = IO.inpt['CFL'] * IO.inpt['del_x']/np.sqrt(IO.inpt['g']* np.max(h_flow))  
            CFL = del_t/(IO.inpt['del_x']/np.sqrt(IO.inpt['g']* np.max(h_flow))) # CFL number
            IO.write_log('--->\t del_t changed to '+ str(del_t) + '\t !!')
        
        t = t + del_t #update time
        tq = t- del_t/2
        T = np.append(T,t) #time tracker update

        if (T[iter] == next_write) or (T[iter-1] < next_write and t+del_t > next_write) :
            next_write = next_write + IO.inpt['write_interval']
            IO.write_outputs([h_flow,U_pr,elv_pr,wave_break,nu_eddy],['h_flow','U','zeta','break_switch','nu_eddy'],t)
            IO.write_log('--->\t Outputs written for simulation time\t' + str(t))
                
        # checking for abnormal values and breaking the simulation       
        CORE.check_values(CFL,elv_pr)

        progress_bar.total = np.ceil(iter + (IO.inpt['total_time']-T[iter])/del_t)
        progress_bar.refresh()
        progress_bar.set_description("Running....".format(iter))
        progress_bar.update(1)
        # Exit conditon after the end of simulation time
        IO.write_log('--->\t\tSIMULATION TIME\t\t' + str(t))
        if t > IO.inpt['total_time']:
            IO.write_log('Finished running without errors')
            progress_bar.close()
            break
    
elif IO.inpt['mode'] == '2D':
    iter = 0    # iteration counter
    t = 0       # curretn time
    T = np.array(0) # time tracker
    del_t = IO.inpt['CFL'] * IO.inpt['del_x']/np.sqrt(IO.inpt['g']* np.max([np.max(h_flow_x),np.max(h_flow_y)]))  # initial time step for CFL = CFL_set
    CFL = IO.inpt['CFL']

    # for progress bar
    progress_bar = tqdm(total = np.ceil(IO.inpt['total_time']/del_t), position = 0, leave = False)


    IO.write_outputs([h_flow_x,h_flow_y,U_pr,V_pr,elv_pr,wave_break],['h_flow_x','h_flow_y','U','V','zeta','break_switch'],t)
    if  IO.inpt['write_interval'] < del_t:
         IO.inpt['write_interval'] = del_t
    next_write = 0 + IO.inpt['write_interval']
    nu_eddy = np.zeros((np.size(elv_pr,0),np.size(elv_pr,1)))
    viscous_term = np.zeros((np.size(elv_pr,0),np.size(elv_pr,1)))
    elv_bc = np.zeros((np.size(elv_pr,1)))
    while True:
        
                                                        ## boundary condition
        # incoming flow and open BC for weak reflection
        
        
        
        elv_plus = CORE.Zeta_plus(t-del_t)
        
        
        u_plus = np.sqrt(IO.inpt['g']/h_flow_x[0,:]) * elv_plus

        u_b =  2* u_plus - np.sqrt(IO.inpt['g']/h_flow_x[0,:]) *  elv_pr[0,:]

        U[0,:] = u_b * np.ones((np.size(elv_pr,1)))
        U_pr[0,:] = u_b * h_flow_x[0,:] * np.ones((np.size(elv_pr,1)))
        
            
        
        U_cu[0,:] = U_pr[0,:] * np.ones((np.size(elv_pr,1)))
            

                                                ## solving for elevation
        for i in range(0,nx):
            for j in range(0,ny):
                if wet[i,j] == 1:
                    elv_cu[i,j] = elv_pr[i,j] - del_t * (((U_pr[i+1,j] - U_pr[i,j])/IO.inpt['del_x']) + ((V_pr[i,j+1] - V_pr[i,j])/IO.inpt['del_y'])) 

                                              ## solving for fluxes
        for j in range(0,ny):
            for i in range(1,nx):  #neglect inlfow
                if wet[i,j]: #needs the cell on the right to be wet???
                    weighted_flux = ((IO.inpt['flux_weight'] * U_pr[i,j]) + ((1 - IO.inpt['flux_weight'])* (U_pr[i-1,j] + U_pr[i+1,j])/2))
                    viscous_term[i,j] =  nu_eddy[i,j] *del_t *( (((h_flow_x[i,j] - h_flow_x[i-1,j])/IO.inpt['del_x'])*((U[i,j] - U[i-1,j])/IO.inpt['del_x'])) + ((U[i+1,j] - 2*U[i,j] + U[i-1,j])/IO.inpt['del_x']**2))
                    numerator =  weighted_flux - (IO.inpt['g'] * h_flow_x[i,j] * del_t * (elv_pr[i,j] - elv_pr[i-1,j])/IO.inpt['del_x']) + wave_break[i,j] * viscous_term[i,j]
                    denominator = 1 + (IO.inpt['g'] *  del_t * (Cf[i,j]**2) * abs(U_pr[i,j]) / (h_flow_x[i,j]**(7/3)))
                    U_cu[i,j] = numerator/denominator
                    
        for i in range(0,nx):
            for j in range(1,ny):  #neglect inlfow
                if wet[i,j]: #needs the cell on the right to be wet???
                    weighted_flux = ((IO.inpt['flux_weight'] * V_pr[i,j]) + ((1 - IO.inpt['flux_weight'])* (V_pr[i,j-1] + V_pr[i,j+1])/2))
                    viscous_term[i,j] =  nu_eddy[i,j] *del_t *( (((h_flow_y[i,j] - h_flow_y[i,j-1])/IO.inpt['del_y'])*((V[i,j] - V[i,j-1])/IO.inpt['del_y'])) + ((V[i,j+1] - 2*V[i,j] + V[i,j-1])/IO.inpt['del_y']**2))
                    numerator =  weighted_flux - (IO.inpt['g'] * h_flow_y[i,j] * del_t * (elv_pr[i,j] - elv_pr[i,j-1])/IO.inpt['del_y']) + wave_break[i,j] * viscous_term[i,j]
                    denominator = 1 + (IO.inpt['g'] *  del_t * (Cf[i,j]**2) * abs(U_pr[i,j]) / (h_flow_y[i,j]**(7/3)))
                    V_cu[i,j] = numerator/denominator
        #keeping boundaries open
        U_cu[0,:] = U_cu[1,:]
        V_cu[:,0] = V_cu[:,1]
        V_cu[:,-1] = V_cu[:,-2]

        ## calculating depth with respect to new elevation           

        [h_flow_x,h_flow_y,elv_cu,wet,U_cu,V_cu] = CORE.set_water_height(d,elv_cu,wet,h_flow_x,U_cu,'2D',h_flow_y,V_cu)  
        
        [wet,U_cu,h_flow_x,elv_cu,h_flow_y,V_cu] = CORE.check_flooding(wet,U_cu,d,h_flow_x,elv_cu,'2D',h_flow_y,V_cu)


        
        if IO.inpt['exit_BC'] == 'O':        #open boundary for flat bed
            U_cu[-1,:] = U_cu[-2,:]
        elif IO.inpt['exit_BC'] == 'C':     #closed boundary for flat bed
            U_cu[-1,:] = 0
                
                
        


             
           
        if IO.inpt['break'] == 1: 
            nu_eddy = CORE.eddy_viscosity_model(nu_eddy,U_cu,IO.inpt['del_x'],'2D',d,V_cu,IO.inpt['del_y'] )      
            wave_break = CORE.check_onset_of_breaking_2D(d,U_cu,elv_cu,wave_break,h_flow_x,X,wet) 
        # updating the previous time step to current time step       
        U_pr = U_cu
        V_pr = V_cu
        elv_pr = elv_cu
        
        
             
                                        ## necessary counters and trackers

        iter += 1 #iteration counter update
        
        if del_t >IO.inpt['CFL'] * IO.inpt['del_x']/np.sqrt(IO.inpt['g']* np.max([np.max(h_flow_x),np.max(h_flow_y)])): 
            del_t = IO.inpt['CFL'] * IO.inpt['del_x']/np.sqrt(IO.inpt['g']* np.max([np.max(h_flow_x),np.max(h_flow_y)])) 
            CFL = del_t/(IO.inpt['del_x']/np.sqrt(IO.inpt['g']* np.max([np.max(h_flow_x),np.max(h_flow_y)])) ) # CFL number
        
        t = t + del_t #update time
        T = np.append(T,t) #time tracker update

        if (T[iter] == next_write) or (T[iter-1] < next_write and t+del_t > next_write) :
            next_write = next_write + IO.inpt['write_interval']
            IO.write_outputs([h_flow_x,h_flow_y,U_pr,V_pr,elv_pr,wave_break,nu_eddy],['h_flow_x','h_flow_y','U','V','zeta','break_switch','nu_eddy'],t)
        
                
        # checking for abnormal values and breaking the simulation       
        CORE.check_values(CFL,elv_pr)

        progress_bar.total = np.ceil(iter + (IO.inpt['total_time']-T[iter])/del_t)
        progress_bar.refresh()
        progress_bar.set_description("Running....".format(iter))
        progress_bar.update(1)
        # Exit conditon after the end of simulation time
        if t > IO.inpt['total_time']:
            logging.info('Finished running without erros')
            progress_bar.close()
            break

