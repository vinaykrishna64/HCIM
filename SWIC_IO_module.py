
# ----------module for import and output functions

import numpy as np
import os
import shutil
import csv
import sys
#--------------------------------------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------functions for reading inputs----------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------------------------------------------------------
def write_log(write_str):
    LOGfile =  open("LOG_FILE.txt", "a")
    LOGfile.write(write_str + '\n')
    LOGfile.close()
    
global inpt 
inpt =  {'init':'init'}

def assign_values(key,values):
    if key == 'wave':
        inpt['amp'] = float(values[0])
        inpt['TP'] = float(values[1])
        # linear wave signal
        inpt['f'] = 1/float(values[1]) # signal frequency
        inpt['w_type'] = 'Linear'
    elif key == 'cgrid':
        if len(values) == 3:
            inpt['Lx'] = float(values[0])
            inpt['del_x'] = float(values[1])
            inpt['start_x'] = float(values[2])
        elif len(values) == 6:
            inpt['Lx'] = float(values[0])
            inpt['del_x'] = float(values[1])
            inpt['start_x'] = float(values[2])
            inpt['Ly'] = float(values[3])
            inpt['del_y'] = float(values[4])
            inpt['start_y'] = float(values[5])
    elif key == 'bottom':
        inpt['d_file'] = values[0]
        inpt['exit_BC'] = str(values[1])
        inpt['x_file'] = values[2]
        if inpt['mode'] == '2D':
            inpt['y_file'] = values[3]
        
    elif key == 'sim_time':
        inpt['total_time'] = float(values[0])
        inpt['CFL'] = float(values[1])
        inpt['write_interval'] = float(values[2])
    elif key == 'OPT': 
        inpt['Z_opt'] = values[0]
        inpt['flux_weight'] = float(values[1])
        inpt['depth_threshold'] = float(values[2])
        inpt['manning_file'] = str(values[3])
        inpt['g'] = float(values[4])
        inpt['rho'] = float(values[5])
    elif key == 'mode':
        inpt['mode'] = values[0]
    elif key == 'Break':
        inpt['break'] = float(values[0])
    elif key == 'Hydrostatic':
        inpt['HS'] = float(values[0])
    elif key == 'Overtop':
        inpt['Otop'] = float(values[0])
    return inpt
def display_inputs(x = inpt):
    keys = list(x.keys())
    values = list(x.values())
    for i in range(0,len(keys)):
        print('\n'+keys[i] + '    <------->    '+ str(values[i])+'\n')

def load_data(file = "input.inpt"):
    f = open(file, "r")
    for x in f:
        if x[0] == '$':#ignore comment
            continue
        else: #find the key
            x = x.split()
            key = x[0]
            values = x[1:]
            assign_values(key,values)
    f.close()

def load_file(file,mode):
    nx = 0;
    with open(file, 'r') as d_file:
     csv_reader = csv.reader(d_file)
     for line in csv_reader:
         if nx == 0:
             first_line = line
         nx = nx+1     
    
    ny = len(first_line)
    if mode == '1D':
        d = np.zeros((ny))
    elif mode == '2D':
        d = np.zeros((nx,ny))
    with open(file,'r') as d_file:
        csv_reader = csv.reader(d_file)
        if mode == '1D':
            for line in csv_reader:
                for i in range(0,len(line)):
                    d[i] = float(line[i])
        elif mode == '2D':
            count = 0
            for line in csv_reader:
                for i in range(0,len(line)):
                    d[count,i] = float(line[i])
                count += 1
        return d   


           
#--------------------------------------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------functions for output storage----------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------------------------------------------------------

def make_folder(folder_name): 
    #creates the requested folder locally
    path = os.path.abspath(os.getcwd()) # current path 

    if not os.path.exists(folder_name): #if the folder doesn't exist, create it 
        os.mkdir(folder_name)
    else:                                  #if the folder already exists, delete it and create it (to empty it) 
        shutil.rmtree(os.path.join(path,folder_name))
        os.mkdir(folder_name)

def write_outputs(variables,var_names,t):
    folder_name = os.path.join('outputs',str(t)) 
    make_folder(folder_name)
    for i  in range(0,len(variables)):
        file_name = var_names[i] + '.txt'
        np.savetxt(os.path.join(folder_name,file_name),variables[i])           
    





#--------------------------------------------------------------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#------------------------------------interpolate files---------------------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------------------------------------------------------

def interpolate_data(X,d,Cf,Y=[]):
    d_inp = load_file( inpt['d_file'], inpt['mode'])
    X_inp = load_file( inpt['x_file'], inpt['mode'])
    if inpt['mode'] == '2D':
        Y_inp = load_file( inpt['y_file'], inpt['mode'])
        x_pos = np.reshape(X_inp, (np.size(X_inp,0)*np.size(X_inp,1),1))
        y_pos = np.reshape(Y_inp, (np.size(Y_inp,0)*np.size(Y_inp,1),1))
        d_val = np.reshape(d_inp, (np.size(d_inp,0)*np.size(d_inp,1)))
        if not (len(x_pos) == len(y_pos) and len(x_pos) == len(d_val)):
                write_log('invalid input drid')
                write_log('check Mesh grid')
                sys.exit()
        positions  = np.concatenate((x_pos, y_pos), axis=1)
        d = griddata(positions, d_val, (X, Y), method='linear')
        if np.any(np.isnan(d)):
            write_log(' This function only linear interpolates!! please give additional points')
            write_log('check Mesh grid')
            sys.exit()
        np.savetxt('d.txt',d)
        np.savetxt('x.txt',X)
        np.savetxt('y.txt',Y)
    elif inpt['mode'] == '1D':
        if not (len(X_inp) == len(d_inp)):
             write_log('invalid input grid')
             write_log('check grid')
             sys.exit()
        d = np.interp(X, X_inp, d_inp)
        np.savetxt('d.txt',d)
        np.savetxt('x.txt',X)


    Cf_inp = load_file( inpt['manning_file'], inpt['mode'])
    X_inp = load_file( inpt['x_file'], inpt['mode'])
    if inpt['mode'] == '2D':
        Y_inp = load_file( inpt['y_file'], inpt['mode'])
        x_pos = np.reshape(X_inp, (np.size(X_inp,0)*np.size(X_inp,1),1))
        y_pos = np.reshape(Y_inp, (np.size(Y_inp,0)*np.size(Y_inp,1),1))
        Cf_val = np.reshape(d_inp, (np.size(Cf_inp,0)*np.size(Cf_inp,1)))
        if not (len(x_pos) == len(y_pos) and len(x_pos) == len(Cf_val)):
                write_log('invalid input drid')
                write_log('check Mesh grid')
                sys.exit()
        positions  = np.concatenate((x_pos, y_pos), axis=1)
        Cf = griddata(positions, Cf_val, (X, Y), method='linear')
        if np.any(np.isnan(Cf)):
            write_log(' This function only linear interpolates!! please give additional points')
            write_log('check Mesh grid')
            sys.exit()
        np.savetxt('Cf.txt',Cf)
        np.savetxt('x.txt',X)
        np.savetxt('y.txt',Y)
    elif inpt['mode'] == '1D':
        if not (len(X_inp) == len(Cf_inp)):
             write_log('invalid input grid')
             write_log('check grid')
             sys.exit()
        Cf = np.interp(X, X_inp, Cf_inp)
        np.savetxt('Cf.txt',Cf)
        np.savetxt('x.txt',X)

    return [d,Cf]




def write_inputs(a_dict = inpt):
    write_log('')
    write_log('')
    write_log('#'*50)
    write_log('\t\t INPUTS PROVIDED  \t\t')
 
    write_log('')
    write_log('\\'+'-'*50 + '//')
    write_log('')
    write_log('wave type  \t::' +  inpt['w_type'])
    write_log('wave amplitude\t::' +  str(inpt['amp'])+ 'm' + '\t<--->\t' + 'wave period\t::' +  str(inpt['TP'])+ 's')
    write_log('If spectrum was selected, these will be significant values')
   
    write_log('')
    write_log('\\'+'-'*50 + '//')
    write_log('')
    if inpt['break'] == 0:
        write_log('Breaking \t::\t NO')
    if inpt['break'] == 1:
        write_log('Breaking \t::\t YES')
        write_log('ad hoc eddy viscocity using the smagorinsky model with Cs =0.01 will be employed')
        write_log('current model does not provide any control over breaking criteria to the user')
      
    write_log('')
    write_log('\\'+'-'*50 + '//')
    write_log('')
    
    write_log('X_file \t::' +  inpt['x_file'])
    write_log('d_file \t::' +  inpt['d_file'])
    write_log('Cf_file \t::' +  inpt['manning_file'])
    if inpt['mode'] == '2D':
        write_log('Y_file \t::' +  inpt['y_file'])
    
    write_log('')
    write_log('\\'+'-'*50 + '//')
    write_log('')
    write_log('g \t::' +  str(inpt['g']))
    write_log('rho \t::' +  str(inpt['rho']))
    write_log('depth interpolation mode \t::' +  inpt['Z_opt'])
    write_log('depth_threshold \t::' +  str(inpt['depth_threshold'])+ 'm')
    write_log('flux_weight \t::' +  str(inpt['flux_weight']))

    write_log('')
    write_log('\\'+'-'*50 + '//')
    write_log('')
    write_log('Time to simulate \t::' +  str(inpt['total_time'])+ 's')
    write_log('CFL \t::' +  str(inpt['CFL']))
    write_log('Outputs write interval \t::' +  str(inpt['write_interval'])+ 's')

    write_log('')
    write_log('\\'+'-'*50 + '//')
    write_log('')
    
    if str(inpt['exit_BC']) == 'C':
        write_log('exit boundary condition\t::' + '\t Closed'  )
    else:
        write_log('exit boundary condition\t::' + '\t Open'  )

    # write_log('beach_slope \t::' +  str(inpt['beach_slope']))
    # write_log('Irribarren number estimated \t::' +  str(inpt['N_i']))
    # write_log('Gamma estimated \t::' +  str(inpt['gamma']))

    write_log('')
    write_log('\\'+'-'*50 + '//')
    write_log('')
    write_log('mode \t::' +  str(inpt['mode']))
    if inpt['mode'] == '1D':
            write_log('Lx \t::'+str(inpt['Lx'])+ 'm')
            write_log('dx \t::'+str(inpt['del_x'])+ 'm')
            write_log('starting x \t::'+str(inpt['start_x'])+ 'm')
    elif inpt['mode'] == '2D':
            write_log('Lx \t::'+str(inpt['Lx'])+ 'm')
            write_log('dx \t::'+str(inpt['del_x'])+ 'm')
            write_log('starting x \t::'+str(inpt['start_x'])+ 'm')
            write_log('Ly \t::'+str(inpt['Ly'])+ 'm')
            write_log('dy \t::'+str(inpt['del_y'])+ 'm')
            write_log('starting y \t::'+str(inpt['start_y'])+ 'm')
    write_log('Overtopping point \t::'+str(inpt['Otop']) + 'm')
    write_log('')
    write_log('\\'+'-'*50 + '//')
    write_log('')