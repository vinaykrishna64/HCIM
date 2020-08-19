
# ----------module for core solver functions

import numpy as np
import SWIC_IO_module as IO # input-output module
import sys
from statistics import mode
import scipy.sparse
import scipy.sparse.linalg
def fill_zeros_with_last(arr): #fills zeros with last non_zero value
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]
#--------------------------------------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#---------------------------------Initializing computatonal grid variables-------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------------------------------------------------------

def intialize_Cgrid(inpt):
    if inpt['mode'] == '1D':
        nx = int(IO.inpt['Lx']/IO.inpt['del_x'])+1 # number of points in X

        d = np.zeros((nx)) # depth
        h_flow = np.zeros((nx+1)) # h at cell edges water height

        #current time step
        elv_cu = np.zeros((nx)) # elevtation at current time step
        U_cu = np.zeros((nx+1)) # flux in X at current time step

        # previous time step
        elv_pr = np.zeros((nx)) # elevtation at previous time step
        U_pr = np.zeros((nx+1)) # flux in X at previous time step

        ## assigning flags for trakcing breaking and dry-wet condtions

        wave_break = np.zeros((nx)) # wave breaking flag of the cell
        wet = np.zeros((nx)) # flag for dry-wet status of the cell

        #initiating velocity array
        U = np.zeros((nx+1))
        #X-co-ordinates
        X = np.zeros((nx))
        X[0] = IO.inpt['start_x']
        for i in range(1,nx):
            X[i] = X[i-1] + IO.inpt['del_x']
        #if constant manning is choosen populate manning value        
        Cf = np.ones((nx)) 
        return [nx,X,d,h_flow,elv_cu,U_cu,elv_pr,U_pr,wave_break,wet,Cf]
    elif inpt['mode'] == '2D':
        nx = int(IO.inpt['Lx']/IO.inpt['del_x']) +1# number of points in X
        ny = int(IO.inpt['Ly']/IO.inpt['del_y']) +1# number of points in y

        d = np.zeros((nx,ny)) # depth
        h_flow_x = np.zeros((nx+1,ny)) # h at cell edges water height
        h_flow_y = np.zeros((nx,ny+1)) # h at cell edges water height
        #current time step
        elv_cu = np.zeros((nx,ny)) # elevtation at current time step
        U_cu = np.zeros((nx+1,ny)) # flux in X at current time step
        V_cu = np.zeros((nx,ny+1)) # flux in Y at current time step
        # previous time step
        elv_pr = np.zeros((nx,ny)) # elevtation at previous time step
        U_pr = np.zeros((nx+1,ny)) # flux in X at previous time step
        V_pr = np.zeros((nx,ny+1)) # flux in Y at previous time step
        ## assigning flags for trakcing breaking and dry-wet condtions

        wave_break = np.zeros((nx,ny)) # wave breaking flag of the cell
        wet = np.zeros((nx,ny)) # flag for dry-wet status of the cell

        #initiating velocity array
        U = np.zeros((nx+1,ny))
        V = np.zeros((nx,ny+1))
        #X-co-ordinates
        X = np.zeros((nx,ny))
        for j in range(0,ny):
            X[0,j] = IO.inpt['start_x']
            for i in range(1,nx):
                X[i,j] = X[i-1,j] + IO.inpt['del_x']
        #Y-co-ordinates
        Y = np.zeros((nx,ny))
        for i in range(0,nx):
            Y[i,0] = IO.inpt['start_y']
            for j in range(1,ny):
                Y[i,j] = Y[i,j-1] + IO.inpt['del_y']

        Cf = np.ones((nx,ny))
        return [nx,ny,X,Y,d,h_flow_x,h_flow_y,elv_cu,U_cu,V_cu,elv_pr,U_pr,V_pr,wave_break,wet,Cf]
        


def intial_set_elvetaions_n_heights(d,elv,wet,h_flow,mode,h_flow_y = []):
    if mode == '1D':
        nx = len(h_flow)

        for i in range(0,nx-1):
            if d[i] <=0:
                elv[i] = 0
            else:
                elv[i] = d[i]

        h_flow[0] = elv[0] - d[0]
        h_flow[nx-1] = elv[-1] - d[-1]
        for i in range(1,nx-1):
            if IO.inpt['Z_opt'] == 'mean': #mean scheme
                h_flow[i] = (np.sum(elv[i:i+2]) - np.sum(d[i:i+2]))/2
            elif  IO.inpt['Z_opt'] == 'max': #max scheme
                h_flow[i] = max(elv[i:i+2]) - max(d[i:i+2])
            elif IO.inpt['Z_opt'] == 'upwind': #upwind scheme
                h_flow[i] = max(elv[i:i+2]) - max(d[i:i+2])    

        # update wet-dry cell flags in the first calculation   
        if np.any(wet == 0):
            wet[ np.where(h_flow[0:len(h_flow)-1] >= IO.inpt['depth_threshold'])] = 1 #where depth > threshold flag the cell as wet
        return [h_flow,elv,wet]
    elif mode == '2D':
        nx = np.size(d, 0)
        ny = np.size(d, 1)
        for j in range(0,ny):
            for i in range(0,nx):
                if d[i,j] <=0:
                    elv[i,j] = 0
                else:
                    elv[i,j] = d[i,j]
        nx = np.size(h_flow, 0)
        ny = np.size(h_flow, 1)
        h_flow[0,:] = elv[0,:] - d[0,:]
        h_flow[nx-1,:] = elv[-1,:] - d[-1,:]
        for j in range(0,ny):
            for i in range(1,nx-1):
                if IO.inpt['Z_opt'] == 'mean': #mean scheme
                    h_flow[i,j] = (np.sum(elv[i:i+2,j]) - np.sum(d[i:i+2,j]))/2
                elif  IO.inpt['Z_opt'] == 'max': #max scheme
                    h_flow[i,j] = max(elv[i:i+2,j]) - max(d[i:i+2,j])
                elif IO.inpt['Z_opt'] == 'upwind': #upwind scheme
                    h_flow[i,j] = max(elv[i:i+2,j]) - max(d[i:i+2,j])    

        nx = np.size(h_flow_y, 0)
        ny = np.size(h_flow_y, 1)
        
        h_flow_y[:,0] = elv[:,0] - d[:,0]
        h_flow_y[:,ny-1] = elv[:,-1] - d[:,-1]

        for i in range(0,nx):
            for j in range(1,ny-1):
                if IO.inpt['Z_opt'] == 'mean': #mean scheme
                    h_flow_y[i,j] = (np.sum(elv[i,j:j+2]) - np.sum(d[i,j:j+2]))/2
                elif  IO.inpt['Z_opt'] == 'max': #max scheme
                    h_flow_y[i,j] = max(elv[i,j:j+2]) - max(d[i,j:j+2])
                elif IO.inpt['Z_opt'] == 'upwind': #upwind scheme
                    h_flow_y[i,j] = max(elv[i,j:j+2]) - max(d[i,j:j+2])
        # update wet-dry cell flags in the first calculation                  
        wet[ np.where(h_flow[0:-1] >= IO.inpt['depth_threshold'])] = 1 #where depth > threshold flag the cell as wet
        return [h_flow,h_flow_y,elv,wet]





#--------------------------------------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#-------------------------------------------Ramp function------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------------------------------------------------------
def ramp(t,TP):
    ramp_time = TP
    if t < ramp_time:
        ramp = 0.5 * (1 - np.cos((2 * np.pi /TP )* (t/2)))
    else:
        ramp = 1
    return ramp
def Zeta_plus(t):
    if IO.inpt['w_type'] == 'Linear':
        elv_plus = IO.inpt['amp'] * np.sin(t *2* np.pi * IO.inpt['f'] ) #incoming wave elevtion
    ret_val = elv_plus * ramp(t,IO.inpt['TP']) #apply a ramp function
    return ret_val
#--------------------------------------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#-------------------------------------------drying algorithm------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------------------------------------------------------
def Drying_algorithm(h_flow,wet,elv,d,U_cu,mode,h_flow_y = [],V_cu = []):
    if  mode == '1D':
        nx = len(elv)
        for i in range(1,nx):
            if h_flow[i] < IO.inpt['depth_threshold']/2: #for hysteris
                elv[i] = d[i]
                h_flow[i] = elv[i] - d[i]
                U_cu[i] = 0
                wet[i] = 0
        return [h_flow,elv,wet,U_cu]
    elif  mode == '2D':
        nx = np.size(elv,0)
        ny = np.size(elv,1)
        for i in range(0,nx):
            for j in range(0,ny):
                if ((h_flow[i,j] + h_flow[i+1,j] + h_flow_y[i,j] + h_flow_y[i,j+1]) /4) < IO.inpt['depth_threshold']/2: #for hysteris
                    elv[i,j] = d[i,j]
                    h_flow[i,j] = elv[i,j] - d[i,j]
                    h_flow_y[i,j] = elv[i,j] - d[i,j]
                    h_flow_y[i,j+1] = elv[i,j] - d[i,j]
                    U_cu[i,j] = 0
                    V_cu[i,j] = 0
                    V_cu[i,j+1] = 0
                    wet[i,j] = 0
        return [h_flow,h_flow_y,elv,wet,U_cu,V_cu]

#--------------------------------------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#---------------------------------computing water depth grid -----------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------------------------------------------------------

## h_flow calculation -- h at cell edges (water height h = zeta - d)
def set_water_height(d,elv,wet,h_flow,U_cu,mode,h_flow_y = [],V_cu = []):
    if  mode == '1D':
        nx = len(h_flow)
        h_flow[0] = elv[0] - d[0]
        h_flow[nx-1] = h_flow[nx-2]
        for i in range(1,nx-1):
            if IO.inpt['Z_opt'] == 'mean': #mean scheme
                h_flow[i] = (np.sum(elv[i:i+2]) - np.sum(d[i:i+2]))/2
            elif  IO.inpt['Z_opt'] == 'max': #max scheme
                h_flow[i] = max(elv[i:i+2]) - max(d[i:i+2])
            elif IO.inpt['Z_opt'] == 'upwind': #upwind scheme
                if  U_cu[i] > 0:
                    h_flow[i] = elv[i-1] - d[i-1]
                elif  U_cu[i] < 0:
                    h_flow[i] = elv[i] - d[i]
                elif U_cu[i] == 0:
                    h_flow[i] = max(elv[i:i+2]) - max(d[i:i+2])    
        [h_flow,elv,wet,U_cu] =  Drying_algorithm(h_flow,wet,elv,d,U_cu,mode)    
        return [h_flow,elv,wet,U_cu]
    elif  mode == '2D':
        nx = np.size(h_flow, 0)
        ny = np.size(h_flow, 1)
        h_flow[0,:] = elv[0,:] - d[0,:]
        h_flow[nx-1,:] = elv[-1,:] - d[-1,:]
        for j in range(0,ny):
            for i in range(1,nx-1):
                if IO.inpt['Z_opt'] == 'mean': #mean scheme
                    h_flow[i,j] = (np.sum(elv[i:i+2,j]) - np.sum(d[i:i+2,j]))/2
                elif  IO.inpt['Z_opt'] == 'max': #max scheme
                    h_flow[i,j] = max(elv[i:i+2,j]) - max(d[i:i+2,j])
                elif IO.inpt['Z_opt'] == 'upwind': #upwind scheme
                    h_flow[i,j] = max(elv[i:i+2,j]) - max(d[i:i+2,j])    

        nx = np.size(h_flow_y, 0)
        ny = np.size(h_flow_y, 1)
        
        h_flow_y[:,0] = elv[:,0] - d[:,0]
        h_flow_y[:,ny-1] = elv[:,-1] - d[:,-1]

        for i in range(0,nx):
            for j in range(1,ny-1):
                if IO.inpt['Z_opt'] == 'mean': #mean scheme
                    h_flow_y[i,j] = (np.sum(elv[i,j:j+2]) - np.sum(d[i,j:j+2]))/2
                elif  IO.inpt['Z_opt'] == 'max': #max scheme
                    h_flow_y[i,j] = max(elv[i,j:j+2]) - max(d[i,j:j+2])
                elif IO.inpt['Z_opt'] == 'upwind': #upwind scheme
                    h_flow_y[i,j] = max(elv[i,j:j+2]) - max(d[i,j:j+2])

        [h_flow,h_flow_y,elv,wet,U_cu,V_cu] =  Drying_algorithm(h_flow,wet,elv,d,U_cu,mode,h_flow_y,V_cu)    
        return [h_flow,h_flow_y,elv,wet,U_cu,V_cu]
            
#--------------------------------------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#----------------------------------Flooding algorithm----------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------------------------------------------------------

def check_flooding(wet,U_cu,d,h_flow,elv_cu,mode,h_flow_y = [],V_cu = []):
## FLOODING algortihm
    if  np.any(wet==0):
        if mode == '1D':
            # assuming 1 cell at a time gets flooded
            i_flood = np.where(wet == 0)[0][0] #first dry cel

            dry_cell_flux = U_cu[i_flood-1] #flux in cell before it 

            if dry_cell_flux > 0: #given positive flow towards the dry cell 
                # yamazaki method
                new_depth =  elv_cu[i_flood-1] - d[i_flood] #check the expected depth
                if new_depth >= IO.inpt['depth_threshold']: #if the expected depth is greater than the thrshold
                    h_flow[i_flood] = new_depth # assign the expected water depth
                    U_cu[i_flood] = U_cu[i_flood -1] # assign +ve flux
                    wet[i_flood] = 1 # assign wet status
            return [wet,U_cu,h_flow,elv_cu]
        elif mode == '2D':
            # assuming 1 cell at a time gets flooded
            wet_bool = (wet[1::] == 0) * (wet[0:-1] == 1)
            i_flood =  np.where(wet_bool == True)#first dry cell
            Xs = np.array(i_flood[0])+1
            Ys = np.array(i_flood[1])
            edges = np.where( Ys == np.size(wet,1)-1 )[0]
            Xs = np.delete(Xs,edges)
            Ys = np.delete(Ys,edges)
            multplier = np.array([0,0])
            for i in range(0,len(Xs)):
                dry_cell_flux = U_cu[Xs[i]-1,Ys[i]] #flux in cell before it 
                x_flooding = 0
                if dry_cell_flux > 0  and (wet[Xs[i],Ys[i]-1] == 0 or wet[Xs[i],Ys[i]+1] == 0) : #given positive flow towards the dry cell 
                    # yamazaki method
                    new_depth =  elv_cu[Xs[i]-1,Ys[i]] - d[Xs[i]-1,Ys[i]] #check the expected depth
                    if new_depth >= IO.inpt['depth_threshold']: #if the expected depth is greater than the thrshold
                        h_flow[Xs[i],Ys[i]] = new_depth # assign the expected water depth
                        h_flow_y[Xs[i],Ys[i]+1] = new_depth # assign the expected water depth
                        h_flow_y[Xs[i],Ys[i]] = new_depth # assign the expected water depth
                        U_cu[Xs[i],Ys[i]] = U_cu[Xs[i]-1,Ys[i]] # assign +ve flux
                        V_cu[Xs[i],Ys[i]+1] = V_cu[Xs[i]-1,Ys[i]+1] # assign +ve flux
                        V_cu[Xs[i],Ys[i]] = V_cu[Xs[i]-1,Ys[i]] # assign +ve flux
                        wet[Xs[i],Ys[i]] = 1 # assign wet status
                        x_flooding = 1
                elif (wet[Xs[i],Ys[i]-1] == 1 or wet[Xs[i],Ys[i]+1] == 1) :
                    if V_cu[Xs[i],Ys[i]-1] > 0:
                        multplier[0] = 1
                    elif  V_cu[Xs[i],Ys[i]+2] < 0:
                        multplier[1] = 1    
                    if x_flooding:
                        new_depth =  (elv_cu[Xs[i]-1,Ys[i]] + multplier[0] * V_cu[Xs[i],Ys[i]-1] + multplier[1] * V_cu[Xs[i],Ys[i]+2])/(1 + np.sum(multplier)) - d[Xs[i]-1,Ys[i]] #check the expected depth
                    else:
                        new_depth =  ( multplier[0] * V_cu[Xs[i],Ys[i]-1] + multplier[1] * V_cu[Xs[i],Ys[i]+2])/np.sum(multplier) - d[Xs[i]-1,Ys[i]] #check the expected depth
                        U_cu[Xs[i],Ys[i]] = U_cu[Xs[i]-1,Ys[i]]
                        h_flow[Xs[i],Ys[i]] = h_flow[Xs[i]-1,Ys[i]] 
                    V_cu[Xs[i],Ys[i]+1] = V_cu[Xs[i],Ys[i]+2] # assign +ve flux
                    V_cu[Xs[i],Ys[i]] = V_cu[Xs[i],Ys[i]-1] # assign +ve flux
                    wet[Xs[i],Ys[i]] = 1        
                        

            return [wet,U_cu,h_flow,elv_cu,h_flow_y,V_cu]
    else:
        if mode == '1D':
            return [wet,U_cu,h_flow,elv_cu]
        elif mode == '2D':
            return [wet,U_cu,h_flow,elv_cu,h_flow_y,V_cu]


    

#--------------------------------------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#---------------------------------check error function to stop simulation which might not ---------------------
#----------------------------------raise python errors but the physics is unreasonable-------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------------------------------------------------------
def check_values(CFL,elv_pr):
    if  CFL > 0.75  or  np.any(np.isnan(elv_pr)) or np.any(abs(elv_pr) > 20):
        IO.write_log("This program is forcefully stopped as some values are abnormal or they will cause instabilies later on")
        if CFL >0.75:
            IO.write_log('CFL = %02f : must be less than 0.75 to ensure stability' %CFL)
        elif   np.any(np.isnan(elv_pr)) or np.any(abs(elv_pr) > 10**2):
            IO.write_log('Abnormal values detected in wave elevation')

        sys.exit()


def eddy_viscosity_model(nu_eddy,U,dx,mode,d,V = [],dy = []) :
    Cs = 0.1;
    if mode == '1D':
        du = np.gradient(U) / dx
        nu_eddy = Cs**2 * np.sqrt(3) * dx * abs(du[0:-1])
    elif mode == '2D':
        du = np.gradient(U,axis=0)/dx
        dv = np.gradient(V,axis=1)/dy
        nu_eddy = Cs**2 * np.sqrt(2) * dx * dy * np.sqrt(du[0:-1,:]**2 + dv[:,0:-1]**2 + 0.5 * (du[0:-1,:]+dv[:,0:-1])**2)
    return nu_eddy



#--------------------------------------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------checking the onset of breaking--------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------------------------------------------------------

def check_onset_of_breaking(d,U,elv_cu,wave_break,h_flow,X,wet,gamma):
    
    wave_break = np.zeros(len(elv_cu)) #setting all flags to zero for re evaluation
    #checking differnt breaking criterion
    for i in range(0,len(elv_cu)):
        if h_flow[i] > 0.5:
        # wave steepness
            if abs(elv_cu[i])*2 >= 0.8 * abs(d[i]):
                wave_break[i] = 1
                break
            # wave celerity:
            elif abs(U[i]) > np.sqrt(IO.inpt['g'] * abs(d[i])): 
                wave_break[i] = 1    
                break    
            # new wave parameter
            elif abs(U[i]) > 0.8* np.sqrt(IO.inpt['g'] * abs(d[i])):
                wave_break[i] = 1
                break
            elif abs(elv_cu[i])*2 >=  gamma[i]* abs(d[i]): 
                wave_break[i] = 1
                break
    wave_break = fill_zeros_with_last(wave_break) #makes sure the wave is dissipated along 
                                                #the whole wave just the crest from onset of breaking
    return [wave_break]

def check_onset_of_breaking_2D(d,U,elv_cu,wave_break,h_flow_x,X,wet):
    nx = np.size(d,0)
    ny = np.size(d,1)
    wave_break = np.zeros((nx,ny))
    for j in range(0,ny):
        [wave_break[:,j]] = check_onset_of_breaking(d[:,j],U[:,j],elv_cu[:,j],wave_break[:,j],h_flow_x[:,j],X[:,j],wet[:,j])

    return wave_break

#--------------------------------------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------solver functions--------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------------------------------------------------------

def Solve_1D_elevation(elv_cu,elv_pr,U_pr,wet,del_t,h_flow): #solver for elevation
    nx = len(elv_cu)
    elv_cu = np.zeros((nx))
    for i in range(0,nx):
        if wet[i] == 1:
            elv_cu[i] = elv_pr[i] - (del_t * (U_pr[i+1]*h_flow[i+1] - U_pr[i]*h_flow[i])/IO.inpt['del_x']) 
    return elv_cu
def make_flux_n_cel_vel(U_pr,h_flow):
    FLU_p = np.zeros((len(U_pr)))
    FLU_n = np.zeros((len(U_pr)))
    U_p = np.zeros((len(U_pr)))
    U_n = np.zeros((len(U_pr)))
    nx = len(U_pr)
    for i in range(0,nx):
    	if  i == 1 or i == nx-1:
    		FLU_p[i] = U_pr[i] * h_flow[i] 
    		FLU_n[i] = U_pr[i] * h_flow[i] 
    		U_p[i] = FLU_p[i]/h_flow[i]
    		U_n[i] = FLU_n[i]/h_flow[i]
    	else:
    		if U_pr[i-1] >0:
    			if i >1:
    				FLU_p[i] = (U_pr[i] + U_pr[i-1])*0.5*0.5*(h_flow[i-1]+h_flow[1-2])
    			else:
    				FLU_p[i] = (U_pr[i] + U_pr[i-1])*0.5*h_flow[i-1]
    		elif U_pr[i-1] <= 0 :
    			FLU_p[i] = (U_pr[i] + U_pr[i-1])*0.5*h_flow[i-1]

    		if U_pr[i+1] >0:
    			FLU_p[i] = (U_pr[i] + U_pr[i+1])*0.5*0.5*(h_flow[i]+h_flow[1+1])		
    		elif U_pr[i-1] <= 0 :
    			FLU_p[i] = (U_pr[i] + U_pr[i+1])*0.5*h_flow[i]

    		U_p[i] = 2*FLU_p[i]/(h_flow[i] +h_flow[i-1])
    		U_n[i] = 2*FLU_n[i]/(h_flow[i] +h_flow[i-1])

    U_p = 0.5*(U_p + abs(U_p))
    U_n = 0.5*(U_n - abs(U_n))
    return [U_p,U_n]
def make_flux_n_cel_vel2(U,h_flow):
	flux = 0.5*(U[0:-1]*h_flow[0:-1] + U[1:]*h_flow[1:])
	cell_vel = 0.5*(U[0:-1] + U[1:])
	return [flux,cell_vel]
def Solve_1D_Hs_velocities(U_cu,elv_pr,U_pr,h_flow,d,wet,del_t,wave_break,nu_eddy,Cf,adv): #hydrostatic flux solver
    nx = len(U_pr)
    #[U_p,U_n] = make_flux_n_cel_vel(U_pr,h_flow)
    [flux,cell_vel] = make_flux_n_cel_vel2(U_pr,h_flow)
    for i in range(1,nx-1):  #neglect inlfow
        if wet[i]: 
            if adv[i] == 1:
                advection = (1/h_flow[i])*(flux[i]*cell_vel[i] - flux[i-1]*cell_vel[i-1])/IO.inpt['del_x'] -  (U_pr[i]/h_flow[i])*(flux[i] - flux[i-1])/IO.inpt['del_x']
                #advection = U_p[i]*(U_pr[i] - U_pr[i-1])/IO.inpt['del_x'] +  U_n[i]*(U_pr[i+1] - U_pr[i])/IO.inpt['del_x']
                weighted_flux = (IO.inpt['flux_weight'] * U_pr[i]) + ((1 - IO.inpt['flux_weight'])* (U_pr[i-1] + U_pr[i+1])/2)
                viscous_term = 0
                if wave_break[i] :
                    viscous_term =  del_t * (nu_eddy[i+1] *(U_pr[i+1] - U_pr[i]) - nu_eddy[i] *(U_pr[i] - U_pr[i-1]))/(h_flow[i] * IO.inpt['del_x']**2)
                numerator = weighted_flux - del_t*advection - (IO.inpt['g'] *  del_t* (elv_pr[i] - elv_pr[i-1])/IO.inpt['del_x']) +  viscous_term  
                denominator = 1 + (  del_t * Cf[i] * abs(U_pr[i]) / (h_flow[i])) #
                U_cu[i] = numerator/denominator
            else:
                weighted_flux = (IO.inpt['flux_weight'] * U_pr[i]) + ((1 - IO.inpt['flux_weight'])* (U_pr[i-1] + U_pr[i+1])/2)
                viscous_term = 0
                if wave_break[i] :
                    viscous_term =  del_t * (nu_eddy[i+1] *(U_pr[i+1] - U_pr[i]) - nu_eddy[i] *(U_pr[i] - U_pr[i-1]))/(h_flow[i] * IO.inpt['del_x']**2)
                numerator = weighted_flux - (IO.inpt['g'] *  del_t* (elv_pr[i] - elv_pr[i-1])/IO.inpt['del_x']) +  viscous_term  
                denominator = 1 + (  del_t * Cf[i] * abs(U_pr[i]) / (h_flow[i]))
                U_cu[i] = numerator/denominator
    return U_cu


def TDMA(a,b,c,d):
    n = len(d)
    w= np.zeros(n-1,float)
    g= np.zeros(n, float)
    p = np.zeros(n,float)

    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    p[n-1] = g[n-1]
    for i in range(n-1,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i]
    return p


def Solve_1D_NHs_velocities(U_cu,elv_pr,U_pr,h_flow,d,wet,del_t,wave_break,nu_eddy,Cf,P_nh_cu,P_nh_pr,W_cu,W_pr,bot_grad,W_b_pr,W_b_cu,adv): #non-hydrostatic flux solver
    P_nh_cu = np.zeros((len(elv_pr))) 
    W_cu = np.zeros((len(elv_pr)))
    W_b_cu = np.zeros((len(elv_pr)))
    U_cu_half = np.zeros(len(U_cu)) 
    U_cu_half[0] = U_cu[0] 
    U_cu_half = Solve_1D_Hs_velocities(U_cu,elv_pr,U_pr,h_flow,d,wet,del_t,wave_break,nu_eddy,Cf,adv)
    
    ## calculate W_cu
    nx =  len(d)
    for i in range(0,nx):
        if wet[i] :
            if i == 0:
                W_b_cu[i] = - U_cu_half[i]*bot_grad[i]
            else:
                W_b_cu[i] = - 0.5*(U_cu_half[i]+abs(U_cu_half[i]))*bot_grad[i] -  0.5*(U_cu_half[i]-abs(U_cu_half[i]))*bot_grad[i-1]
   
    if np.any(bot_grad != 0):
        nx_cut = min([np.where(d >= IO.inpt['bifur_depth'])[0][0], np.where(wet == 0)[0][0]]) -1# min([np.where(d >= -2)[0][0], np.where(wet == 0)[0][0]])
    else:
        nx_cut = nx-1
    
    if np.any(wave_break == 1):
        if np.where(wave_break == 1)[0][0] < nx_cut:
            nx_cut = np.where(wave_break == 1)[0][0] 

    A_fac = (np.gradient(elv_pr[0:nx_cut+1]) - bot_grad[0:nx_cut+1])/h_flow[0:nx_cut+1]

    A = np.eye((nx_cut))#tridaigonal matrix
    B = np.zeros((nx_cut)) #sytem of equations AX=B
    f1 = del_t/(2*(IO.inpt['del_x']**2))
    a = np.zeros((nx_cut))
    b = np.ones((nx_cut))
    c = np.zeros((nx_cut))
    # a[0:nx_cut] = f1*(-1 + A_fac[0:nx_cut])
    # b[0:nx_cut] = f1*(2 + A_fac[0:nx_cut] -A_fac[1:nx_cut+1]) + 2*del_t/(0.5*(h_flow[0:nx_cut]+h_flow[1:nx_cut+1]))**2
    # c[0:nx_cut] = f1*(-1 - A_fac[0:nx_cut])
    for i in range(0,nx_cut):     
        a[i] = f1*(-1 + A_fac[i])
        b[i] = f1*(2 + A_fac[i] -A_fac[i+1]) + 2*del_t/(0.5*(h_flow[i]+h_flow[i+1]))**2
        c[i] = f1*(-1 - A_fac[i+1])
        A[i,i] = f1*(2 + A_fac[i] -A_fac[i+1]) + 2*del_t/(0.5*(h_flow[i]+h_flow[i+1]))**2
        if i != 0:
        	A[i,i-1] = f1*(-1 + A_fac[i])
        if i != nx_cut-1:
        	A[i,i+1] = f1*(-1 - A_fac[i])
        B[i] = - (U_cu_half[i+1] - U_cu_half[i])/IO.inpt['del_x'] - 2*(W_pr[i] + W_b_pr[i] - 2*W_b_cu[i])/(h_flow[i]+h_flow[i+1])
    
    
    
    ## solve P_nh_cu
    M2 = scipy.sparse.linalg.spilu(A)
    M = scipy.sparse.linalg.LinearOperator((nx_cut,nx_cut), M2.solve)
    P_nh_cu[0:nx_cut], exitCode = scipy.sparse.linalg.bicgstab(A, B,M=M,tol=10**-8)
    #P_nh_cu = np.linalg.inv(A).dot(B)    
    #P_nh_cu[0:nx_cut] =  TDMA(a,b,c,B)
    nx = len(U_cu)
    for i in range(1,nx_cut+1):  #neglect inlfow
        
        
        pressure_term =   0.5*(P_nh_cu[i] - P_nh_cu[i-1])/IO.inpt['del_x']  +  0.5*((P_nh_cu[i]+P_nh_cu[i-1])*(elv_pr[i] - elv_pr[i-1] - d[i] + d[i-1])/( (h_flow[i]+h_flow[i+1])  *IO.inpt['del_x']))
            
        U_cu[i]  =  U_cu_half[i]  - pressure_term  *  del_t   

    U_cu[nx_cut+1:] = U_cu_half[nx_cut+1:]


    nx =  len(d)       
    for i in range(0,nx):
        if wet[i] :

            W_cu[i] = W_pr[i] - (W_b_cu[i] - W_b_pr[i]) + 2*del_t*P_nh_cu[i]/h_flow[i]    
           
    
    return [U_cu,P_nh_cu,W_cu,W_b_cu]

