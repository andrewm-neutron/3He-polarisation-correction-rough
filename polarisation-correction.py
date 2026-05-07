# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:01:07 2022

@author: andrewm
"""

##############################################################
#
#            IMPORTS
#
##############################################################

import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import curve_fit
from scipy import optimize
from datetime import datetime


##############################################################
#
#            CONSTANT DEFINITIONS
#
##############################################################

trans_per_window = 0.9754   # for 1 cm of Si at 5.22A -> should generalise later
cell_length = 10            # cm for Si window cells

# done by hand at 0.55 pm 0.05
del_invXi = np.matrix([[-9.87696537,  4.56716489],[ 4.56716489, -9.87696537]])
del_P_He = 0.05



##############################################################
#
#            FUNCTION DEFINITIONS
#
##############################################################

# Define model function to be used to fit to the data:
def gauss(x, *p):
    A, mu, sigma, bg = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + bg

# Enter sequence of user inputs
def input_entry():
    inputs = np.zeros(46)
    inputs[0] = int(input("Instrument - 1=PEL, 2=PLA, 3=QKK, 4=SIK, 5=TAI, 6=WOM: "))
    inputs[1] = int(input("Enter the experiment number: "))
    inputs[2] = int(input("Enter the first file number: ")) 
    inputs[3] = int(input("Enter the last file number: "))
    inputs[4] = int(input("Enter the highest number of q points measured in this set (typ. 21): "))
    inputs[5] = int(input("Enter the number of spin filter cells: "))
    inputs[6] = float(input("Enter the cell fill pressure: "))
    inputs[7] = int(input("Plots? 0=no, 1=results only, 2=all: "))
    inputs[8] = int(input("Number of cell fills (typ. one/day): "))
    inputs[9] = int(input("File number of calibration, no cells: "))
    inputs[35] = int(input("File number of calibration, one cell: "))
    print("\n------ First run ------")
    inputs[10] = int(input("File number of calibration, ++ start: "))
    inputs[11] = int(input("File number of calibration, +- start: "))
    inputs[36] = int(input("File number of sample, start: "))
    inputs[37] = int(input("File number of sample, end: "))
    inputs[12] = int(input("File number of calibration, ++ end: "))
    inputs[13] = int(input("File number of calibration, +- end: "))
    inputs[14] = int(input("File number of calibration, depol end: "))
    if inputs[8] > 1:
        print("\n------ Second run ------")
        inputs[15] = int(input("File number of calibration, ++ start: "))
        inputs[16] = int(input("File number of calibration, +- start: "))
        inputs[38] = int(input("File number of sample, start: "))
        inputs[39] = int(input("File number of sample, end: "))
        inputs[17] = int(input("File number of calibration, ++ end: "))
        inputs[18] = int(input("File number of calibration, +- end: "))
        inputs[19] = int(input("File number of calibration, depol end: "))        
    if inputs[8] > 2:
        print("\n------ Third run ------")
        inputs[20] = int(input("File number of calibration, ++ start: "))
        inputs[21] = int(input("File number of calibration, +- start: "))
        inputs[40] = int(input("File number of sample, start: "))
        inputs[41] = int(input("File number of sample, end: "))
        inputs[22] = int(input("File number of calibration, ++ end: "))
        inputs[23] = int(input("File number of calibration, +- end: "))
        inputs[24] = int(input("File number of calibration, depol end: "))    
    if inputs[8] > 3:
        print("\n------ Fourth run ------")
        inputs[25] = int(input("File number of calibration, ++ start: "))
        inputs[26] = int(input("File number of calibration, +- start: "))
        inputs[42] = int(input("File number of sample, start: "))
        inputs[43] = int(input("File number of sample, end: "))
        inputs[27] = int(input("File number of calibration, ++ end: "))
        inputs[28] = int(input("File number of calibration, +- end: "))
        inputs[29] = int(input("File number of calibration, depol end: "))    
    if inputs[8] > 4:
        print("\n------ Fifth run ------")
        inputs[30] = int(input("File number of calibration, ++ start: "))
        inputs[31] = int(input("File number of calibration, +- start: "))
        inputs[44] = int(input("File number of sample, start: "))
        inputs[45] = int(input("File number of sample, end: "))
        inputs[32] = int(input("File number of calibration, ++ end: "))
        inputs[33] = int(input("File number of calibration, +- end: "))
        inputs[34] = int(input("File number of calibration, depol end: "))  
        
    print("\n... Please wait...\n")
    return inputs

# Read in raw data files
def read_raw_data(exp_no: int, firstfile: int, lastfile: int, maxnumqvals: int, plotflag: int):
    numfiles = lastfile - firstfile + 1
    q = np.zeros((int(numfiles),int(maxnumqvals)))
    count = np.zeros((int(numfiles),int(maxnumqvals)), dtype=int)
    beamMon = np.zeros(int(numfiles), dtype=int)
    s1 = np.zeros((int(numfiles),int(maxnumqvals)))
    s2 = np.zeros(int(numfiles))
    date_object = np.empty(int(numfiles), dtype=datetime)
    numqvals = np.zeros(int(numfiles), dtype=int) 
    wavelength = np.zeros(int(numfiles))
    
    file_number = np.zeros(int(numfiles), dtype=int)
    flip_arr = np.full(int(numfiles), False)        # 0 = NSF, 1 = SF
    analyser_arr = np.full(int(numfiles), False)    # 0 = diffraction, 1 = EFW or INS
    s1_arr = np.zeros(int(numfiles), dtype=int)         # 0 = no match, 1 = first S1, 2 = second S1
    s1_first = -0.001
    s1_second = -9.9989
    substring_flip = "NSF"
    substring_analyser = "elastic"
    
    slash = os.path.normcase('/')
    
    position_counter = 0
    for j in range(int(firstfile)-1,int(lastfile)):
        position_counter +=1  
        
        filename = ".%sexp%d%sDatafiles%s%04d_%06d.dat"% (slash, exp_no, slash, slash, exp_no, j+1)
        file_number[position_counter-1] = j+1
        
        raw_data = np.loadtxt(filename, unpack = True)
        numqvals[j] = np.size(raw_data,1)
        numqvalsnow = np.size(raw_data,1)
        
        #print(raw_data.ndim)
        #print(raw_data[0,:])
        
        #Pt = raw_data[0,:];
        q[j,:numqvalsnow] = raw_data[1,:];
        #time = raw_data[2,:];
        count[j,:numqvalsnow] = raw_data[3,:];
        beamMon[j] = np.average(raw_data[4,:]);
        ei = float(np.average(raw_data[7,:]));
        s1[j,:numqvalsnow] = raw_data[16,:];
        s2[j] = np.average(raw_data[17,:]);
        wavelength[j] = 9.045/np.sqrt(ei)
        
        # print(s1)
        # print(s2)
        #print(np.average(polariser))
        
        f = open(filename, "r")
        content = f.readlines()
        f.close()

        timeline = content[2]
        timestring = timeline[9:17]
        dateline = content[1]
        datestring = dateline[9:19]
        date_object[j] = datetime.strptime(datestring+timestring, "%Y-%m-%d%H:%M:%S")
        #print(date_object)
        
        labelstring = content[10]
        #print(labelstring)
        if substring_flip in labelstring:
            flip_arr[position_counter-1] = False
        else:
            flip_arr[position_counter-1] = True  
        if substring_analyser in labelstring:
            analyser_arr[position_counter-1] = True
        else:
            analyser_arr[position_counter-1] = False
        if np.abs(np.average(s1[j,:numqvalsnow]) - s1_first) < 0.1:
            s1_arr[position_counter-1] = 1;
        elif np.abs(np.average(s1[j,:numqvalsnow]) - s1_second) < 0.1:
            s1_arr[position_counter-1] = 2;
        else:
            s1_arr[position_counter-1] = 0;
            print("S1 angle not matching expected values")
            input("----- Press Enter to Continue -----")
        
        if plotflag == 2:
            plotname = "Experiment %d, file %d"% (int(exp_no), j+1)
            plt.figure(j+1)
            plt.clf()
            plt.plot(q[j,:numqvalsnow],count[j,:numqvalsnow], 'bo-')
            plt.title(plotname)
            plt.xlabel('Q (inv Å)')
            plt.ylabel('Counts (30k on monitor)')
            plt.show()
      
    return q, count, beamMon, s1, s2, date_object, wavelength, numqvals, flip_arr, analyser_arr, s1_arr, file_number

# Use calibration scan to characterise spin filter cells
def calibrate_cells(file_nocells: int, file_onecell: int, file_Ipp_start_alumina: int, file_Ipm_start_alumina: int, file_Ipp_end_alumina: int, file_Ipm_end_alumina: int, file_dep_end_alumina: int, q, count, numqvals, exp_no, wavelength, cell_pressure, plotflag):
    files_alumina = [int(file_nocells)-1, int(file_onecell)-1, int(file_Ipp_start_alumina)-1, int(file_Ipm_start_alumina)-1, int(file_Ipp_end_alumina)-1, int(file_Ipm_end_alumina)-1, int(file_dep_end_alumina)-1]
    maxcountrate_alumina = np.zeros(np.size(files_alumina,0))
    q_alumina = np.zeros(np.size(files_alumina,0))
    time_alumina = np.empty(np.size(files_alumina,0), dtype=datetime)

    counter = 0
    for j in files_alumina:      
        hist = count[j,:int(numqvals[j])]
        bin_centres = q[j,:int(numqvals[j])]   
        
        # p0 is the initial guess for the fitting coefficients (A, mu, sigma, bg above)
        p0 = [np.amax(hist), np.average(bin_centres), 0.01*np.average(bin_centres), hist[0]]
        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
        hist_fit = gauss(bin_centres, *coeff)
  
        if plotflag > 0:
            plotname = "P9906 exp%d file %d"% (exp_no,j+1)
            plt.figure(j+1)
            plt.clf()
            #plt.plot(q[j,:numqvalsnow],count[j,:numqvalsnow], 'bo-')
            plt.plot(bin_centres, hist, 'bo-', label='Test data')
            plt.plot(bin_centres, hist_fit, 'r--', label='Fitted data')
            plt.title(plotname)
            plt.xlabel('Q (inv Å)')
            plt.ylabel('Counts (30k on monitor)')
            plt.show()
        
        maxcountrate_alumina[counter] = coeff[0]
        q_alumina[counter] = coeff[1]
        time_alumina[counter] = date_object[j]
        
        counter +=1
        
    countrate_nocells = maxcountrate_alumina[0]
    countrate_onecell = maxcountrate_alumina[1]
    countrate_Ipp_new = maxcountrate_alumina[2]
    countrate_Ipm_new = maxcountrate_alumina[3]
    countrate_Ipp_old = maxcountrate_alumina[4]
    countrate_Ipm_old = maxcountrate_alumina[5]   
    countrate_depol = maxcountrate_alumina[6]   

    opacity_theo = 0.0732*np.average(wavelength)*cell_length*cell_pressure    
    opacity_theo_str = "Theoretical filter opacity: \t\t\t%.4f"% (opacity_theo) 
    
    P_He_onecell = np.log(np.power(2*countrate_onecell/countrate_nocells*np.exp(opacity_theo)/(trans_per_window*trans_per_window),(1/opacity_theo))) 
    P_He_onecell_str = "Helium polarisation for one cell: \t\t%.4f"% (P_He_onecell)

    
    opacity_depol = -0.5*np.log(countrate_depol/countrate_nocells*np.power(trans_per_window,-4))
    opacity_depol_str = "Filter opacity from depolarised cells: \t%.4f"% (opacity_depol)
    print(opacity_theo_str)  
    print(opacity_depol_str)
 
    def Ipp_new_solve(P_He_var):
        return (0.5*(np.exp(-2*opacity_depol*(1-P_He_var))+np.exp(-2*opacity_depol*(1+P_He_var))) - countrate_Ipp_new/countrate_nocells*np.power(trans_per_window,-4))**2
    P_He_guess = 0.5
    P_He_new_empirical = optimize.fmin(Ipp_new_solve, P_He_guess, xtol=1e-8, disp=False)

    def Ipp_old_solve(P_He_var):
        return (0.5*(np.exp(-2*opacity_depol*(1-P_He_var))+np.exp(-2*opacity_depol*(1+P_He_var))) - countrate_Ipp_old/countrate_nocells*np.power(trans_per_window,-4))**2
    P_He_guess = 0.2
    P_He_old_empirical = optimize.fmin(Ipp_old_solve, P_He_guess, xtol=1e-8, disp=False)

    interval = time_alumina[4] - time_alumina[2]
    interval_hours = interval.seconds/(60*60) + interval.days*24

    lifetime = -interval_hours/np.log(P_He_old_empirical/P_He_new_empirical)
    lifetime_str = "The T1 lifetime is: \t\t\t\t\t%.2f hours"% (lifetime)
    print(lifetime_str)
    print(P_He_onecell_str)
    P_He_new_empirical_str = "Helium polarisation at start: \t\t\t%.4f\n"% (P_He_new_empirical)
    print(P_He_new_empirical_str)
       
    time_start = time_alumina[2]
    time_end = time_alumina[4]

    return lifetime, opacity_theo, opacity_depol, time_start, time_end, P_He_old_empirical, P_He_new_empirical

# Correct data scans for spin filter cell behaviour
def correct_scans(lifetime, time_start, time_end, P_He_new_empirical, first_data_file, last_data_file, date_object, opacity_depol, count, flip_arr, analyser_arr, s1_arr, file_number):

    numrows = int((last_data_file-first_data_file)/8)+2
    numcols = int(np.amax(numqvals))
    
    Sigma_pp_diff_s1_1_out = np.zeros((numrows,numcols))
    Sigma_pm_diff_s1_1_out = np.zeros((numrows,numcols))
    Sigma_pp_diff_s1_1_out_file = np.zeros(numrows)
    Sigma_pm_diff_s1_1_out_file = np.zeros(numrows)   
    Sigma_pp_diff_s1_1_counter = 0
    Sigma_pm_diff_s1_1_counter = 0
    Sigma_pp_efw_s1_1_out = np.zeros((numrows,numcols))
    Sigma_pm_efw_s1_1_out = np.zeros((numrows,numcols))
    Sigma_pp_efw_s1_1_out_file = np.zeros(numrows)
    Sigma_pm_efw_s1_1_out_file = np.zeros(numrows)   
    Sigma_pp_efw_s1_1_counter = 0
    Sigma_pm_efw_s1_1_counter = 0
    
    Sigma_pp_diff_s1_2_out = np.zeros((numrows,numcols))
    Sigma_pm_diff_s1_2_out = np.zeros((numrows,numcols))
    Sigma_pp_diff_s1_2_out_file = np.zeros(numrows)
    Sigma_pm_diff_s1_2_out_file = np.zeros(numrows)   
    Sigma_pp_diff_s1_2_counter = 0
    Sigma_pm_diff_s1_2_counter = 0
    Sigma_pp_efw_s1_2_out = np.zeros((numrows,numcols))
    Sigma_pm_efw_s1_2_out = np.zeros((numrows,numcols))
    Sigma_pp_efw_s1_2_out_file = np.zeros(numrows)
    Sigma_pm_efw_s1_2_out_file = np.zeros(numrows)   
    Sigma_pp_efw_s1_2_counter = 0
    Sigma_pm_efw_s1_2_counter = 0
    
    # ERRORS
    Sigma_pp_err_diff_s1_1_out = np.zeros((numrows,numcols))
    Sigma_pm_err_diff_s1_1_out = np.zeros((numrows,numcols))
    Sigma_pp_err_efw_s1_1_out = np.zeros((numrows,numcols))
    Sigma_pm_err_efw_s1_1_out = np.zeros((numrows,numcols)) 
    Sigma_pp_err_diff_s1_2_out = np.zeros((numrows,numcols))
    Sigma_pm_err_diff_s1_2_out = np.zeros((numrows,numcols))
    Sigma_pp_err_efw_s1_2_out = np.zeros((numrows,numcols))
    Sigma_pm_err_efw_s1_2_out = np.zeros((numrows,numcols))
    
    Sigma_pp_err_diff_s1_1_out, Sigma_pm_err_diff_s1_1_out, Sigma_pp_err_efw_s1_1_out, Sigma_pm_err_efw_s1_1_out, Sigma_pp_err_diff_s1_2_out, Sigma_pm_err_diff_s1_2_out, Sigma_pp_err_efw_s1_2_out, Sigma_pm_err_efw_s1_2_out
      
    loop_counter = 0
    for j in range(int(first_data_file)-1,int(last_data_file)):

        loop_counter +=1
        
        if int(s1_arr[j]) == 1:
            if flip_arr[j] == False:
                if analyser_arr[j] == True:
                    print("NSF, s1 1, EFW %d"% (int(file_number[j])))
                    file_Ipp_efw_s1_1_sample = j
                    Sigma_pp_efw_s1_1_out_file[Sigma_pp_efw_s1_1_counter] = file_Ipp_efw_s1_1_sample
                    interval_Ipp = date_object[j] - time_start
                    interval_Ipp_hours = interval_Ipp.seconds/(60*60) + interval_Ipp.days*24
                    P_He_Ipp_efw_s1_1 = P_He_new_empirical*np.exp(-interval_Ipp_hours/lifetime)                   
                    Sigma_pp_efw_s1_1_counter +=1
                else:
                    print("NSF, s1 1, diff %d"% (int(file_number[j])))  
                    file_Ipp_diff_s1_1_sample = j
                    Sigma_pp_diff_s1_1_out_file[Sigma_pp_diff_s1_1_counter] = file_Ipp_diff_s1_1_sample
                    interval_Ipp = date_object[j] - time_start
                    interval_Ipp_hours = interval_Ipp.seconds/(60*60) + interval_Ipp.days*24
                    P_He_Ipp_diff_s1_1 = P_He_new_empirical*np.exp(-interval_Ipp_hours/lifetime)                   
                    Sigma_pp_diff_s1_1_counter +=1            
            if flip_arr[j] == True:
                if analyser_arr[j] == True:
                    print("SF, s1 1, EFW %d"% (int(file_number[j])))
                    file_Ipm_efw_s1_1_sample = j
                    Sigma_pm_efw_s1_1_out_file[Sigma_pm_efw_s1_1_counter] = file_Ipm_efw_s1_1_sample
                    interval_Ipm = date_object[j] - time_start
                    interval_Ipm_hours = interval_Ipm.seconds/(60*60) + interval_Ipm.days*24
                    P_He_Ipm_efw_s1_1 = P_He_new_empirical*np.exp(-interval_Ipm_hours/lifetime)                   
                    
                    Sigma_pm_efw_s1_1_counter +=1
                else:
                    print("SF, s1 1, diff %d"% (int(file_number[j])))
                    file_Ipm_diff_s1_1_sample = j
                    Sigma_pm_diff_s1_1_out_file[Sigma_pm_diff_s1_1_counter] = file_Ipm_diff_s1_1_sample
                    interval_Ipm = date_object[j] - time_start
                    interval_Ipm_hours = interval_Ipm.seconds/(60*60) + interval_Ipm.days*24
                    P_He_Ipm_diff_s1_1 = P_He_new_empirical*np.exp(-interval_Ipm_hours/lifetime)                   
                                                        
                    Sigma_pm_diff_s1_1_counter +=1
        elif int(s1_arr[j]) == 2:
            if flip_arr[j] == False:
                if analyser_arr[j] == True:
                    print("NSF, s1 2, EFW %d"% (int(file_number[j])))
                    file_Ipp_efw_s1_2_sample = j
                    Sigma_pp_efw_s1_2_out_file[Sigma_pp_efw_s1_2_counter] = file_Ipp_efw_s1_2_sample
                    interval_Ipp = date_object[j] - time_start
                    interval_Ipp_hours = interval_Ipp.seconds/(60*60) + interval_Ipp.days*24
                    P_He_Ipp_efw_s1_2 = P_He_new_empirical*np.exp(-interval_Ipp_hours/lifetime)                   
                    Sigma_pp_efw_s1_2_counter +=1
                else:
                    print("NSF, s1 2, diff %d"% (int(file_number[j]))) 
                    file_Ipp_diff_s1_2_sample = j
                    Sigma_pp_diff_s1_2_out_file[Sigma_pp_diff_s1_2_counter] = file_Ipp_diff_s1_2_sample
                    interval_Ipp = date_object[j] - time_start
                    interval_Ipp_hours = interval_Ipp.seconds/(60*60) + interval_Ipp.days*24
                    P_He_Ipp_diff_s1_2 = P_He_new_empirical*np.exp(-interval_Ipp_hours/lifetime)                   
                    Sigma_pp_diff_s1_2_counter +=1
            if flip_arr[j] == True:
                if analyser_arr[j] == True:
                    print("SF, s1 2, EFW %d"% (int(file_number[j])))
                    file_Ipm_efw_s1_2_sample = j
                    Sigma_pm_efw_s1_2_out_file[Sigma_pm_efw_s1_2_counter] = file_Ipm_efw_s1_2_sample
                    interval_Ipm = date_object[j] - time_start
                    interval_Ipm_hours = interval_Ipm.seconds/(60*60) + interval_Ipm.days*24
                    P_He_Ipm_efw_s1_2 = P_He_new_empirical*np.exp(-interval_Ipm_hours/lifetime)                   
                   
                    Sigma_pm_efw_s1_2_counter +=1
                else:
                    print("SF, s1 2, diff %d"% (int(file_number[j])))
                    file_Ipm_diff_s1_2_sample = j
                    Sigma_pm_diff_s1_2_out_file[Sigma_pm_diff_s1_2_counter] = file_Ipm_diff_s1_2_sample
                    interval_Ipm = date_object[j] - time_start
                    interval_Ipm_hours = interval_Ipm.seconds/(60*60) + interval_Ipm.days*24
                    P_He_Ipm_diff_s1_2 = P_He_new_empirical*np.exp(-interval_Ipm_hours/lifetime)                   
               
                    Sigma_pm_diff_s1_2_counter +=1
        else:
            print("NSF, s1 ?, diff %d"% (int(file_number[j])))
        
        
        if loop_counter%8 == 0:
            
            Xi_11 = 0.25*(np.exp(-2*opacity_depol*(1-P_He_Ipp_efw_s1_1))+np.exp(-2*opacity_depol*(1+P_He_Ipp_efw_s1_1)))
            Xi_12 = 0.5*np.exp(-opacity_depol*(1-P_He_Ipp_efw_s1_1))*np.exp(-opacity_depol*(1+P_He_Ipp_efw_s1_1))
            Xi_21 = 0.5*np.exp(-opacity_depol*(1-P_He_Ipm_efw_s1_1))*np.exp(-opacity_depol*(1+P_He_Ipm_efw_s1_1))
            Xi_22 = 0.25*(np.exp(-2*opacity_depol*(1-P_He_Ipm_efw_s1_1))+np.exp(-2*opacity_depol*(1+P_He_Ipm_efw_s1_1)))
            Xi = np.matrix([[float(Xi_11), float(Xi_12)], [float(Xi_21), float(Xi_22)]])
            invXi = inv(Xi)
            
            Sigmapp_efw_s1_1 = count[file_Ipp_efw_s1_1_sample,:int(numqvals[file_Ipp_efw_s1_1_sample])] * invXi[0,0] + count[file_Ipp_efw_s1_1_sample,:int(numqvals[file_Ipm_efw_s1_1_sample])] * invXi[0,1]
            Sigmapm_efw_s1_1 = count[file_Ipm_efw_s1_1_sample,:int(numqvals[file_Ipm_efw_s1_1_sample])] * invXi[1,0] + count[file_Ipm_efw_s1_1_sample,:int(numqvals[file_Ipp_efw_s1_1_sample])] * invXi[1,1]
            Sigma_pp_efw_s1_1_out[Sigma_pp_efw_s1_1_counter,:int(numqvals[file_Ipp_efw_s1_1_sample])] = Sigmapp_efw_s1_1
            Sigma_pm_efw_s1_1_out[Sigma_pm_efw_s1_1_counter,:int(numqvals[file_Ipm_efw_s1_1_sample])] = Sigmapm_efw_s1_1
            
            Sigmapp_err_efw_s1_1 = np.sqrt(count[file_Ipp_efw_s1_1_sample,:int(numqvals[file_Ipp_efw_s1_1_sample])] * np.power(invXi[0,0],2) + count[file_Ipp_efw_s1_1_sample,:int(numqvals[file_Ipm_efw_s1_1_sample])] *  np.power(invXi[0,1],2) + del_invXi[0,0]*del_P_He*count[file_Ipp_efw_s1_1_sample,:int(numqvals[file_Ipp_efw_s1_1_sample])] + del_invXi[0,1]*del_P_He*count[file_Ipm_efw_s1_1_sample,:int(numqvals[file_Ipm_efw_s1_1_sample])])
            Sigmapm_err_efw_s1_1 = np.sqrt(count[file_Ipm_efw_s1_1_sample,:int(numqvals[file_Ipm_efw_s1_1_sample])] * np.power(invXi[1,0],2) + count[file_Ipm_efw_s1_1_sample,:int(numqvals[file_Ipp_efw_s1_1_sample])] *  np.power(invXi[1,1],2) + del_invXi[0,1]*del_P_He*count[file_Ipp_efw_s1_1_sample,:int(numqvals[file_Ipp_efw_s1_1_sample])] + del_invXi[0,0]*del_P_He*count[file_Ipm_efw_s1_1_sample,:int(numqvals[file_Ipm_efw_s1_1_sample])])       
            Sigma_pp_err_efw_s1_1_out[Sigma_pp_efw_s1_1_counter,:int(numqvals[file_Ipp_efw_s1_1_sample])] = Sigmapp_err_efw_s1_1
            Sigma_pm_err_efw_s1_1_out[Sigma_pm_efw_s1_1_counter,:int(numqvals[file_Ipm_efw_s1_1_sample])] = Sigmapm_err_efw_s1_1

            plotname = "Sigma for diffraction (first S1) files %d and %d"% (file_Ipp_efw_s1_1_sample+1, file_Ipm_efw_s1_1_sample+1)
            plt.figure(j+1000)
            plt.clf()
            plt.yscale('log')
            # plt.plot(q[file_Ipp_efw_s1_1_sample,:int(numqvals[file_Ipp_efw_s1_1_sample])], Sigmapp_efw_s1_1, 'b-', label='Sigma++')
            # plt.plot(q[file_Ipp_efw_s1_1_sample,:int(numqvals[file_Ipm_efw_s1_1_sample])], Sigmapm_efw_s1_1, 'r-', label='Sigma+-')
            plt.plot(q[file_Ipp_efw_s1_1_sample,:int(numqvals[file_Ipp_efw_s1_1_sample])], count[file_Ipp_efw_s1_1_sample,:int(numqvals[file_Ipp_efw_s1_1_sample])], 'b--', label='I++ EFW S1 1')
            plt.plot(q[file_Ipp_efw_s1_1_sample,:int(numqvals[file_Ipm_efw_s1_1_sample])], count[file_Ipm_efw_s1_1_sample,:int(numqvals[file_Ipm_efw_s1_1_sample])], 'r--', label='I+- EFW S1 2')
            plt.errorbar(q[file_Ipp_efw_s1_1_sample,:int(numqvals[file_Ipp_efw_s1_1_sample])], Sigmapp_efw_s1_1, yerr=Sigmapp_err_efw_s1_1, fmt = 'b-', label='Sigma++ errors', capsize=3.0)
            plt.errorbar(q[file_Ipm_efw_s1_1_sample,:int(numqvals[file_Ipm_efw_s1_1_sample])], Sigmapm_efw_s1_1, yerr=Sigmapm_err_efw_s1_1, fmt = 'r-', label='Sigma+- errors', capsize=3.0)
            plt.title(plotname)
            plt.xlabel('Q (inv Å)')
            plt.ylabel('Corrected data')
            plt.title(plotname)
            plt.legend()
            plt.show()            
            
            # --------------------------------------------------
            
            Xi_11 = 0.25*(np.exp(-2*opacity_depol*(1-P_He_Ipp_diff_s1_1))+np.exp(-2*opacity_depol*(1+P_He_Ipp_diff_s1_1)))
            Xi_12 = 0.5*np.exp(-opacity_depol*(1-P_He_Ipp_diff_s1_1))*np.exp(-opacity_depol*(1+P_He_Ipp_diff_s1_1))
            Xi_21 = 0.5*np.exp(-opacity_depol*(1-P_He_Ipm_diff_s1_1))*np.exp(-opacity_depol*(1+P_He_Ipm_diff_s1_1))
            Xi_22 = 0.25*(np.exp(-2*opacity_depol*(1-P_He_Ipm_diff_s1_1))+np.exp(-2*opacity_depol*(1+P_He_Ipm_diff_s1_1)))
            Xi = np.matrix([[float(Xi_11), float(Xi_12)], [float(Xi_21), float(Xi_22)]])
            invXi = inv(Xi)
            
            Sigmapp_diff_s1_1 = count[file_Ipp_diff_s1_1_sample,:int(numqvals[file_Ipp_diff_s1_1_sample])] * invXi[0,0] + count[file_Ipp_diff_s1_1_sample,:int(numqvals[file_Ipm_diff_s1_1_sample])] * invXi[0,1]
            Sigmapm_diff_s1_1 = count[file_Ipm_diff_s1_1_sample,:int(numqvals[file_Ipm_diff_s1_1_sample])] * invXi[1,0] + count[file_Ipm_diff_s1_1_sample,:int(numqvals[file_Ipp_diff_s1_1_sample])] * invXi[1,1]
            Sigma_pp_diff_s1_1_out[Sigma_pp_diff_s1_1_counter,:int(numqvals[file_Ipp_diff_s1_1_sample])] = Sigmapp_diff_s1_1
            Sigma_pm_diff_s1_1_out[Sigma_pm_diff_s1_1_counter,:int(numqvals[file_Ipm_diff_s1_1_sample])] = Sigmapm_diff_s1_1

            Sigmapp_err_diff_s1_1 = np.sqrt(count[file_Ipp_diff_s1_1_sample,:int(numqvals[file_Ipp_diff_s1_1_sample])] * np.power(invXi[0,0],2) + count[file_Ipp_diff_s1_1_sample,:int(numqvals[file_Ipm_diff_s1_1_sample])] *  np.power(invXi[0,1],2) + del_invXi[0,0]*del_P_He*count[file_Ipp_diff_s1_1_sample,:int(numqvals[file_Ipp_diff_s1_1_sample])] + del_invXi[0,1]*del_P_He*count[file_Ipm_diff_s1_1_sample,:int(numqvals[file_Ipm_diff_s1_1_sample])])
            Sigmapm_err_diff_s1_1 = np.sqrt(count[file_Ipm_diff_s1_1_sample,:int(numqvals[file_Ipm_diff_s1_1_sample])] * np.power(invXi[1,0],2) + count[file_Ipm_diff_s1_1_sample,:int(numqvals[file_Ipp_diff_s1_1_sample])] *  np.power(invXi[1,1],2) + del_invXi[0,1]*del_P_He*count[file_Ipp_diff_s1_1_sample,:int(numqvals[file_Ipp_diff_s1_1_sample])] + del_invXi[0,0]*del_P_He*count[file_Ipm_diff_s1_1_sample,:int(numqvals[file_Ipm_diff_s1_1_sample])])       
            Sigma_pp_err_diff_s1_1_out[Sigma_pp_diff_s1_1_counter,:int(numqvals[file_Ipp_diff_s1_1_sample])] = Sigmapp_err_diff_s1_1
            Sigma_pm_err_diff_s1_1_out[Sigma_pm_diff_s1_1_counter,:int(numqvals[file_Ipm_diff_s1_1_sample])] = Sigmapm_err_diff_s1_1

            plotname = "Sigma for diffraction (first S1) files %d and %d"% (file_Ipp_diff_s1_1_sample+1, file_Ipm_diff_s1_1_sample+1)
            plt.figure(j+1000)
            plt.clf()
            plt.yscale('log')
            # plt.plot(q[file_Ipp_diff_s1_1_sample,:int(numqvals[file_Ipp_diff_s1_1_sample])], Sigmapp_diff_s1_1, 'b-', label='Sigma++')
            # plt.plot(q[file_Ipp_diff_s1_1_sample,:int(numqvals[file_Ipm_diff_s1_1_sample])], Sigmapm_diff_s1_1, 'r-', label='Sigma+-')
            plt.plot(q[file_Ipp_diff_s1_1_sample,:int(numqvals[file_Ipp_diff_s1_1_sample])], count[file_Ipp_diff_s1_1_sample,:int(numqvals[file_Ipp_diff_s1_1_sample])], 'b--', label='I++ Diff S1 1')
            plt.plot(q[file_Ipp_diff_s1_1_sample,:int(numqvals[file_Ipm_diff_s1_1_sample])], count[file_Ipm_diff_s1_1_sample,:int(numqvals[file_Ipm_diff_s1_1_sample])], 'r--', label='I+- Diff S1 1')
            plt.errorbar(q[file_Ipp_diff_s1_1_sample,:int(numqvals[file_Ipp_diff_s1_1_sample])], Sigmapp_diff_s1_1, yerr=Sigmapp_err_diff_s1_1, fmt = 'b-', label='Sigma++ errors', capsize=3.0)
            plt.errorbar(q[file_Ipm_diff_s1_1_sample,:int(numqvals[file_Ipm_diff_s1_1_sample])], Sigmapm_diff_s1_1, yerr=Sigmapm_err_diff_s1_1, fmt = 'r-', label='Sigma+- errors', capsize=3.0)
            plt.title(plotname)
            plt.xlabel('Q (inv Å)')
            plt.ylabel('Corrected data')
            plt.title(plotname)
            plt.legend()
            plt.show()
            
            # --------------------------------------------------
            
            Xi_11 = 0.25*(np.exp(-2*opacity_depol*(1-P_He_Ipp_efw_s1_2))+np.exp(-2*opacity_depol*(1+P_He_Ipp_efw_s1_2)))
            Xi_12 = 0.5*np.exp(-opacity_depol*(1-P_He_Ipp_efw_s1_2))*np.exp(-opacity_depol*(1+P_He_Ipp_efw_s1_2))
            Xi_21 = 0.5*np.exp(-opacity_depol*(1-P_He_Ipm_efw_s1_2))*np.exp(-opacity_depol*(1+P_He_Ipm_efw_s1_2))
            Xi_22 = 0.25*(np.exp(-2*opacity_depol*(1-P_He_Ipm_efw_s1_2))+np.exp(-2*opacity_depol*(1+P_He_Ipm_efw_s1_2)))
            Xi = np.matrix([[float(Xi_11), float(Xi_12)], [float(Xi_21), float(Xi_22)]])
            invXi = inv(Xi)
            
            Sigmapp_efw_s1_2 = count[file_Ipp_efw_s1_2_sample,:int(numqvals[file_Ipp_efw_s1_2_sample])] * invXi[0,0] + count[file_Ipp_efw_s1_2_sample,:int(numqvals[file_Ipm_efw_s1_2_sample])] * invXi[0,1]
            Sigmapm_efw_s1_2 = count[file_Ipm_efw_s1_2_sample,:int(numqvals[file_Ipm_efw_s1_2_sample])] * invXi[1,0] + count[file_Ipm_efw_s1_2_sample,:int(numqvals[file_Ipp_efw_s1_2_sample])] * invXi[1,1]
            Sigma_pp_efw_s1_2_out[Sigma_pp_efw_s1_2_counter,:int(numqvals[file_Ipp_efw_s1_2_sample])] = Sigmapp_efw_s1_2
            Sigma_pm_efw_s1_2_out[Sigma_pm_efw_s1_2_counter,:int(numqvals[file_Ipm_efw_s1_2_sample])] = Sigmapm_efw_s1_2

            Sigmapp_err_efw_s1_2 = np.sqrt(count[file_Ipp_efw_s1_2_sample,:int(numqvals[file_Ipp_efw_s1_2_sample])] * np.power(invXi[0,0],2) + count[file_Ipp_efw_s1_2_sample,:int(numqvals[file_Ipm_efw_s1_2_sample])] *  np.power(invXi[0,1],2) + del_invXi[0,0]*del_P_He*count[file_Ipp_efw_s1_2_sample,:int(numqvals[file_Ipp_efw_s1_2_sample])] + del_invXi[0,1]*del_P_He*count[file_Ipm_efw_s1_2_sample,:int(numqvals[file_Ipm_efw_s1_2_sample])])
            Sigmapm_err_efw_s1_2 = np.sqrt(count[file_Ipm_efw_s1_2_sample,:int(numqvals[file_Ipm_efw_s1_2_sample])] * np.power(invXi[1,0],2) + count[file_Ipm_efw_s1_2_sample,:int(numqvals[file_Ipp_efw_s1_2_sample])] *  np.power(invXi[1,1],2) + del_invXi[0,1]*del_P_He*count[file_Ipp_efw_s1_2_sample,:int(numqvals[file_Ipp_efw_s1_2_sample])] + del_invXi[0,0]*del_P_He*count[file_Ipm_efw_s1_2_sample,:int(numqvals[file_Ipm_efw_s1_2_sample])])       
            Sigma_pp_err_efw_s1_2_out[Sigma_pp_efw_s1_2_counter,:int(numqvals[file_Ipp_efw_s1_2_sample])] = Sigmapp_err_efw_s1_2
            Sigma_pm_err_efw_s1_2_out[Sigma_pm_efw_s1_2_counter,:int(numqvals[file_Ipm_efw_s1_2_sample])] = Sigmapm_err_efw_s1_2


            plotname = "Sigma for diffraction (first S1) files %d and %d"% (file_Ipp_efw_s1_2_sample+1, file_Ipm_efw_s1_2_sample+1)
            plt.figure(j+1000)
            plt.clf()
            plt.yscale('log')
            # plt.plot(q[file_Ipp_efw_s1_2_sample,:int(numqvals[file_Ipp_efw_s1_2_sample])], Sigmapp_efw_s1_2, 'b-', label='Sigma++')
            # plt.plot(q[file_Ipp_efw_s1_2_sample,:int(numqvals[file_Ipm_efw_s1_2_sample])], Sigmapm_efw_s1_2, 'r-', label='Sigma+-')
            plt.plot(q[file_Ipp_efw_s1_2_sample,:int(numqvals[file_Ipp_efw_s1_2_sample])], count[file_Ipp_efw_s1_2_sample,:int(numqvals[file_Ipp_efw_s1_2_sample])], 'b--', label='I++ EFW S1 2')
            plt.plot(q[file_Ipp_efw_s1_2_sample,:int(numqvals[file_Ipm_efw_s1_2_sample])], count[file_Ipm_efw_s1_2_sample,:int(numqvals[file_Ipm_efw_s1_2_sample])], 'r--', label='I+- EFW S1 2')
            plt.errorbar(q[file_Ipp_efw_s1_2_sample,:int(numqvals[file_Ipp_efw_s1_2_sample])], Sigmapp_efw_s1_2, yerr=Sigmapp_err_efw_s1_2, fmt = 'b-', label='Sigma++ errors', capsize=3.0)
            plt.errorbar(q[file_Ipm_efw_s1_2_sample,:int(numqvals[file_Ipm_efw_s1_2_sample])], Sigmapm_efw_s1_2, yerr=Sigmapm_err_efw_s1_2, fmt = 'r-', label='Sigma+- errors', capsize=3.0)
            plt.title(plotname)
            plt.xlabel('Q (inv Å)')
            plt.ylabel('Corrected data')
            plt.title(plotname)
            plt.legend()
            plt.show()              
        
            # --------------------------------------------------
        
            Xi_11 = 0.25*(np.exp(-2*opacity_depol*(1-P_He_Ipp_diff_s1_2))+np.exp(-2*opacity_depol*(1+P_He_Ipp_diff_s1_2)))
            Xi_12 = 0.5*np.exp(-opacity_depol*(1-P_He_Ipp_diff_s1_2))*np.exp(-opacity_depol*(1+P_He_Ipp_diff_s1_2))
            Xi_21 = 0.5*np.exp(-opacity_depol*(1-P_He_Ipm_diff_s1_2))*np.exp(-opacity_depol*(1+P_He_Ipm_diff_s1_2))
            Xi_22 = 0.25*(np.exp(-2*opacity_depol*(1-P_He_Ipm_diff_s1_2))+np.exp(-2*opacity_depol*(1+P_He_Ipm_diff_s1_2)))
            Xi = np.matrix([[float(Xi_11), float(Xi_12)], [float(Xi_21), float(Xi_22)]])
            invXi = inv(Xi)            

            Sigmapp_diff_s1_2 = count[file_Ipp_diff_s1_2_sample,:int(numqvals[file_Ipp_diff_s1_2_sample])] * invXi[0,0] + count[file_Ipp_diff_s1_2_sample,:int(numqvals[file_Ipm_diff_s1_2_sample])] * invXi[0,1]
            Sigmapm_diff_s1_2 = count[file_Ipm_diff_s1_2_sample,:int(numqvals[file_Ipm_diff_s1_2_sample])] * invXi[1,0] + count[file_Ipm_diff_s1_2_sample,:int(numqvals[file_Ipp_diff_s1_2_sample])] * invXi[1,1]
            Sigma_pp_diff_s1_2_out[Sigma_pp_diff_s1_2_counter,:int(numqvals[file_Ipp_diff_s1_2_sample])] = Sigmapp_diff_s1_2
            Sigma_pm_diff_s1_2_out[Sigma_pm_diff_s1_2_counter,:int(numqvals[file_Ipm_diff_s1_2_sample])] = Sigmapm_diff_s1_2

            Sigmapp_err_diff_s1_2 = np.sqrt(count[file_Ipp_diff_s1_2_sample,:int(numqvals[file_Ipp_diff_s1_2_sample])] * np.power(invXi[0,0],2) + count[file_Ipp_diff_s1_2_sample,:int(numqvals[file_Ipm_diff_s1_2_sample])] *  np.power(invXi[0,1],2) + del_invXi[0,0]*del_P_He*count[file_Ipp_diff_s1_2_sample,:int(numqvals[file_Ipp_diff_s1_2_sample])] + del_invXi[0,1]*del_P_He*count[file_Ipm_diff_s1_2_sample,:int(numqvals[file_Ipm_diff_s1_2_sample])])
            Sigmapm_err_diff_s1_2 = np.sqrt(count[file_Ipm_diff_s1_2_sample,:int(numqvals[file_Ipm_diff_s1_2_sample])] * np.power(invXi[1,0],2) + count[file_Ipm_diff_s1_2_sample,:int(numqvals[file_Ipp_diff_s1_2_sample])] *  np.power(invXi[1,1],2) + del_invXi[0,1]*del_P_He*count[file_Ipp_diff_s1_2_sample,:int(numqvals[file_Ipp_diff_s1_2_sample])] + del_invXi[0,0]*del_P_He*count[file_Ipm_diff_s1_2_sample,:int(numqvals[file_Ipm_diff_s1_2_sample])])       
            Sigma_pp_err_diff_s1_2_out[Sigma_pp_diff_s1_2_counter,:int(numqvals[file_Ipp_diff_s1_2_sample])] = Sigmapp_err_diff_s1_2
            Sigma_pm_err_diff_s1_2_out[Sigma_pm_diff_s1_2_counter,:int(numqvals[file_Ipm_diff_s1_2_sample])] = Sigmapm_err_diff_s1_2

            plotname = "Sigma for diffraction (first S1) files %d and %d"% (file_Ipp_diff_s1_2_sample+1, file_Ipm_diff_s1_2_sample+1)
            plt.figure(j+1000)
            plt.clf()
            plt.yscale('log')
            # plt.plot(q[file_Ipp_diff_s1_2_sample,:int(numqvals[file_Ipp_diff_s1_2_sample])], Sigmapp_diff_s1_2, 'b-', label='Sigma++')
            # plt.plot(q[file_Ipp_diff_s1_2_sample,:int(numqvals[file_Ipm_diff_s1_2_sample])], Sigmapm_diff_s1_2, 'r-', label='Sigma+-')
            plt.plot(q[file_Ipp_diff_s1_2_sample,:int(numqvals[file_Ipp_diff_s1_2_sample])], count[file_Ipp_diff_s1_2_sample,:int(numqvals[file_Ipp_diff_s1_2_sample])], 'b--', label='I++ Diff S1 2')
            plt.plot(q[file_Ipp_diff_s1_2_sample,:int(numqvals[file_Ipm_diff_s1_2_sample])], count[file_Ipm_diff_s1_2_sample,:int(numqvals[file_Ipm_diff_s1_2_sample])], 'r--', label='I+- Diff S1 2')
            plt.errorbar(q[file_Ipp_diff_s1_2_sample,:int(numqvals[file_Ipp_diff_s1_2_sample])], Sigmapp_diff_s1_2, yerr=Sigmapp_err_diff_s1_2, fmt = 'b-', label='Sigma++ errors', capsize=3.0)
            plt.errorbar(q[file_Ipm_diff_s1_2_sample,:int(numqvals[file_Ipm_diff_s1_2_sample])], Sigmapm_diff_s1_2, yerr=Sigmapm_err_diff_s1_2, fmt = 'r-', label='Sigma+- errors', capsize=3.0)
            plt.title(plotname)
            plt.xlabel('Q (inv Å)')
            plt.ylabel('Corrected data')
            plt.title(plotname)
            plt.legend()
            plt.show()             
                
    return Sigma_pp_diff_s1_1_out, Sigma_pm_diff_s1_1_out, Sigma_pp_diff_s1_1_out_file, Sigma_pm_diff_s1_1_out_file, Sigma_pp_efw_s1_1_out, Sigma_pm_efw_s1_1_out, Sigma_pp_efw_s1_1_out_file, Sigma_pm_efw_s1_1_out_file, Sigma_pp_diff_s1_2_out, Sigma_pm_diff_s1_2_out, Sigma_pp_diff_s1_2_out_file, Sigma_pm_diff_s1_2_out_file, Sigma_pp_efw_s1_2_out, Sigma_pm_efw_s1_2_out, Sigma_pp_efw_s1_2_out_file, Sigma_pm_efw_s1_2_out_file, Sigma_pp_err_diff_s1_1_out, Sigma_pm_err_diff_s1_1_out, Sigma_pp_err_efw_s1_1_out, Sigma_pm_err_efw_s1_1_out, Sigma_pp_err_diff_s1_2_out, Sigma_pm_err_diff_s1_2_out, Sigma_pp_err_efw_s1_2_out, Sigma_pm_err_efw_s1_2_out




##############################################################
#
#            DATA CORRECTION PROCESS
#
##############################################################


print("\n*************************************************")
print("IMPORTANT:")
print("Run this script from the same directory as ")
print("the exp#### data folder. ")
print("*************************************************")
inputs = input_entry()

q, count, beamMon, s1, s2, date_object, wavelength, numqvals, flip_arr, analyser_arr, s1_arr, file_number = read_raw_data(inputs[1], inputs[2], inputs[3], inputs[4], inputs[7])

print("\n---------- Run #1: ----------\n")
lifetime1, opacity_theo1, opacity_depol1, time_start1, time_end1, P_He_old_empirical1, P_He_new_empirical1 = calibrate_cells(int(inputs[9]), int(inputs[35]), int(inputs[10]), int(inputs[11]), int(inputs[12]), int(inputs[13]), int(inputs[14]), q, count, numqvals, int(inputs[1]), wavelength, float(inputs[6]), float(inputs[7]))
Sigma_pp_diff_s1_1_out1, Sigma_pm_diff_s1_1_out1, Sigma_pp_diff_s1_1_out_file1, Sigma_pm_diff_s1_1_out_file1, Sigma_pp_efw_s1_1_out1, Sigma_pm_efw_s1_1_out1, Sigma_pp_efw_s1_1_out_file1, Sigma_pm_efw_s1_1_out_file1, Sigma_pp_diff_s1_2_out1, Sigma_pm_diff_s1_2_out1, Sigma_pp_diff_s1_2_out_file1, Sigma_pm_diff_s1_2_out_file1, Sigma_pp_efw_s1_2_out1, Sigma_pm_efw_s1_2_out1, Sigma_pp_efw_s1_2_out_file1, Sigma_pm_efw_s1_2_out_file1, Sigma_pp_err_diff_s1_1_out1, Sigma_pm_err_diff_s1_1_out1, Sigma_pp_err_efw_s1_1_out1, Sigma_pm_err_efw_s1_1_out1, Sigma_pp_err_diff_s1_2_out1, Sigma_pm_err_diff_s1_2_out1, Sigma_pp_err_efw_s1_2_out1, Sigma_pm_err_efw_s1_2_out1 = correct_scans(lifetime1, time_start1, time_end1, P_He_new_empirical1, inputs[36], inputs[37], date_object, opacity_depol1, count, flip_arr, analyser_arr, s1_arr, file_number)

sum_Sigma_pp_diff_s1_1_out1 = np.sum(Sigma_pp_diff_s1_1_out1, axis=0)
sum_Sigma_pm_diff_s1_1_out1 = np.sum(Sigma_pm_diff_s1_1_out1, axis=0)
sum_Sigma_err_pp_diff_s1_1_out1 = np.sqrt(np.sum(np.multiply(Sigma_pp_err_diff_s1_1_out1[:15],Sigma_pp_err_diff_s1_1_out1[:15]), axis=0))
sum_Sigma_err_pm_diff_s1_1_out1 = np.sqrt(np.sum(np.multiply(Sigma_pm_err_diff_s1_1_out1[:15],Sigma_pm_err_diff_s1_1_out1[:15]), axis=0))
plotname = "Sigma for diffraction (first S1, first run) files simple sum"
plt.figure(999)
plt.clf()
# plt.plot(q[0,:15],sum_Sigma_pp_diff_s1_1_out1[:15], 'b-', label='Sigma++')
# plt.plot(q[0,:15],sum_Sigma_pm_diff_s1_1_out1[:15], 'r-', label='Sigma+-')
plt.errorbar(q[0,:15],sum_Sigma_pp_diff_s1_1_out1[:15], yerr=sum_Sigma_err_pp_diff_s1_1_out1[:15], fmt='b-', label='Sigma++', capsize=3.0)
plt.errorbar(q[0,:15],sum_Sigma_pm_diff_s1_1_out1[:15], yerr=sum_Sigma_err_pm_diff_s1_1_out1[:15], fmt='r-', label='Sigma+-', capsize=3.0)
plt.title(plotname)
plt.xlabel('Q (inv Å)')
plt.ylabel('Corrected data')
plt.title(plotname)
plt.legend()
plt.show() 

sum_Sigma_pp_efw_s1_1_out1 = np.sum(Sigma_pp_efw_s1_1_out1, axis=0)
sum_Sigma_pm_efw_s1_1_out1 = np.sum(Sigma_pm_efw_s1_1_out1, axis=0)
sum_Sigma_err_pp_efw_s1_1_out1 = np.sqrt(np.sum(np.multiply(Sigma_pp_err_efw_s1_1_out1[:15],Sigma_pp_err_efw_s1_1_out1[:15]), axis=0))
sum_Sigma_err_pm_efw_s1_1_out1 = np.sqrt(np.sum(np.multiply(Sigma_pm_err_efw_s1_1_out1[:15],Sigma_pm_err_efw_s1_1_out1[:15]), axis=0))
plotname = "Sigma for EFW (first S1, first run) files simple sum"
plt.figure(888)
plt.clf()
# plt.plot(q[0,:15],sum_Sigma_pp_efw_s1_1_out1[:15], 'b-', label='Sigma++')
# plt.plot(q[0,:15],sum_Sigma_pm_efw_s1_1_out1[:15], 'r-', label='Sigma+-')
plt.errorbar(q[0,:15],sum_Sigma_pp_efw_s1_1_out1[:15], yerr=sum_Sigma_err_pp_efw_s1_1_out1[:15], fmt='b-', label='Sigma++', capsize=3.0)
plt.errorbar(q[0,:15],sum_Sigma_pm_efw_s1_1_out1[:15], yerr=sum_Sigma_err_pm_efw_s1_1_out1[:15], fmt='r-', label='Sigma+-', capsize=3.0)
plt.title(plotname)
plt.xlabel('Q (inv Å)')
plt.ylabel('Corrected data')
plt.title(plotname)
plt.legend()
plt.show() 

sum_Sigma_pp_diff_s1_2_out1 = np.sum(Sigma_pp_diff_s1_2_out1, axis=0)
sum_Sigma_pm_diff_s1_2_out1 = np.sum(Sigma_pm_diff_s1_2_out1, axis=0)
sum_Sigma_err_pp_diff_s1_2_out1 = np.sqrt(np.sum(np.multiply(Sigma_pp_err_diff_s1_2_out1[:9],Sigma_pp_err_diff_s1_2_out1[:9]), axis=0))
sum_Sigma_err_pm_diff_s1_2_out1 = np.sqrt(np.sum(np.multiply(Sigma_pm_err_diff_s1_2_out1[:9],Sigma_pm_err_diff_s1_2_out1[:9]), axis=0))
plotname = "Sigma for diffraction (second S1, first run) files simple sum"
plt.figure(998)
plt.clf()
# plt.plot(q[0,:9],sum_Sigma_pp_diff_s1_2_out1[:9], 'b-', label='Sigma++')
# plt.plot(q[0,:9],sum_Sigma_pm_diff_s1_2_out1[:9], 'r-', label='Sigma+-')
plt.errorbar(q[0,:9],sum_Sigma_pp_diff_s1_2_out1[:9], yerr=sum_Sigma_err_pp_diff_s1_2_out1[:9], fmt='b-', label='Sigma++', capsize=3.0)
plt.errorbar(q[0,:9],sum_Sigma_pm_diff_s1_2_out1[:9], yerr=sum_Sigma_err_pm_diff_s1_2_out1[:9], fmt='r-', label='Sigma+-', capsize=3.0)
plt.title(plotname)
plt.xlabel('Q (inv Å)')
plt.ylabel('Corrected data')
plt.title(plotname)
plt.legend()
plt.show() 

sum_Sigma_pp_efw_s1_2_out1 = np.sum(Sigma_pp_efw_s1_2_out1, axis=0)
sum_Sigma_pm_efw_s1_2_out1 = np.sum(Sigma_pm_efw_s1_2_out1, axis=0)
sum_Sigma_err_pp_efw_s1_2_out1 = np.sqrt(np.sum(np.multiply(Sigma_pp_err_efw_s1_2_out1[:9],Sigma_pp_err_efw_s1_2_out1[:9]), axis=0))
sum_Sigma_err_pm_efw_s1_2_out1 = np.sqrt(np.sum(np.multiply(Sigma_pm_err_efw_s1_2_out1[:9],Sigma_pm_err_efw_s1_2_out1[:9]), axis=0))
plotname = "Sigma for EFW (second S1, first run) files simple sum"
plt.figure(887)
plt.clf()
# plt.plot(q[0,:9],sum_Sigma_pp_efw_s1_2_out1[:9], 'b-', label='Sigma++')
# plt.plot(q[0,:9],sum_Sigma_pm_efw_s1_2_out1[:9], 'r-', label='Sigma+-')
plt.errorbar(q[0,:9],sum_Sigma_pp_efw_s1_2_out1[:9], yerr=sum_Sigma_err_pp_efw_s1_2_out1[:9], fmt='b-', label='Sigma++', capsize=3.0)
plt.errorbar(q[0,:9],sum_Sigma_pm_efw_s1_2_out1[:9], yerr=sum_Sigma_err_pm_efw_s1_2_out1[:9], fmt='r-', label='Sigma+-', capsize=3.0)
plt.title(plotname)
plt.xlabel('Q (inv Å)')
plt.ylabel('Corrected data')
plt.title(plotname)
plt.legend()
plt.show() 





if int(inputs[8]) > 1:
    print("\n---------- Run #2: ----------\n")
    lifetime2, opacity_theo2, opacity_depol2, time_start2, time_end2, P_He_old_empirical2, P_He_new_empirical2 = calibrate_cells(int(inputs[9]), int(inputs[35]), int(inputs[15]), int(inputs[16]), int(inputs[17]), int(inputs[18]), int(inputs[19]), q, count, numqvals, int(inputs[1]), wavelength, float(inputs[6]), float(inputs[7]))
    Sigma_pp_diff_s1_1_out2, Sigma_pm_diff_s1_1_out2, Sigma_pp_diff_s1_1_out_file2, Sigma_pm_diff_s1_1_out_file2, Sigma_pp_efw_s1_1_out2, Sigma_pm_efw_s1_1_out2, Sigma_pp_efw_s1_1_out_file2, Sigma_pm_efw_s1_1_out_file2, Sigma_pp_diff_s1_2_out2, Sigma_pm_diff_s1_2_out2, Sigma_pp_diff_s1_2_out_file2, Sigma_pm_diff_s1_2_out_file2, Sigma_pp_efw_s1_2_out2, Sigma_pm_efw_s1_2_out2, Sigma_pp_efw_s1_2_out_file2, Sigma_pm_efw_s1_2_out_file2, Sigma_pp_err_diff_s1_1_out2, Sigma_pm_err_diff_s1_1_out2, Sigma_pp_err_efw_s1_1_out2, Sigma_pm_err_efw_s1_1_out2, Sigma_pp_err_diff_s1_2_out2, Sigma_pm_err_diff_s1_2_out2, Sigma_pp_err_efw_s1_2_out2, Sigma_pm_err_efw_s1_2_out2 = correct_scans(lifetime2, time_start2, time_end2, P_He_new_empirical2, inputs[38], inputs[39], date_object, opacity_depol2, count, flip_arr, analyser_arr, s1_arr, file_number)

    sum_Sigma_pp_diff_s1_1_out2 = np.sum(Sigma_pp_diff_s1_1_out2, axis=0)
    sum_Sigma_pm_diff_s1_1_out2 = np.sum(Sigma_pm_diff_s1_1_out2, axis=0)
    sum_Sigma_err_pp_diff_s1_1_out2 = np.sqrt(np.sum(np.multiply(Sigma_pp_err_diff_s1_1_out2[:15],Sigma_pp_err_diff_s1_1_out2[:15]), axis=0))
    sum_Sigma_err_pm_diff_s1_1_out2 = np.sqrt(np.sum(np.multiply(Sigma_pm_err_diff_s1_1_out2[:15],Sigma_pm_err_diff_s1_1_out2[:15]), axis=0))
    plotname = "Sigma for diffraction (first S1, second run) files simple sum"
    plt.figure(9999)
    plt.clf()
    # plt.plot(q[0,:15],sum_Sigma_pp_diff_s1_1_out2[:15], 'b-', label='Sigma++')
    # plt.plot(q[0,:15],sum_Sigma_pm_diff_s1_1_out2[:15], 'r-', label='Sigma+-')
    plt.errorbar(q[0,:15],sum_Sigma_pp_diff_s1_1_out2[:15], yerr=sum_Sigma_err_pp_diff_s1_1_out2[:15], fmt='b-', label='Sigma++', capsize=3.0)
    plt.errorbar(q[0,:15],sum_Sigma_pm_diff_s1_1_out2[:15], yerr=sum_Sigma_err_pm_diff_s1_1_out2[:15], fmt='r-', label='Sigma+-', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show()     

    sum_Sigma_pp_efw_s1_1_out2 = np.sum(Sigma_pp_efw_s1_1_out2, axis=0)
    sum_Sigma_pm_efw_s1_1_out2 = np.sum(Sigma_pm_efw_s1_1_out2, axis=0)
    sum_Sigma_err_pp_efw_s1_1_out2 = np.sqrt(np.sum(np.multiply(Sigma_pp_err_efw_s1_1_out2[:15],Sigma_pp_err_efw_s1_1_out2[:15]), axis=0))
    sum_Sigma_err_pm_efw_s1_1_out2 = np.sqrt(np.sum(np.multiply(Sigma_pm_err_efw_s1_1_out2[:15],Sigma_pm_err_efw_s1_1_out2[:15]), axis=0))
    plotname = "Sigma for EFW (first S1, second run) files simple sum"
    plt.figure(8888)
    plt.clf()
    # plt.plot(q[0,:15],sum_Sigma_pp_efw_s1_1_out2[:15], 'b-', label='Sigma++')
    # plt.plot(q[0,:15],sum_Sigma_pm_efw_s1_1_out2[:15], 'r-', label='Sigma+-')
    plt.errorbar(q[0,:15],sum_Sigma_pp_efw_s1_1_out2[:15], yerr=sum_Sigma_err_pp_efw_s1_1_out2[:15], fmt='b-', label='Sigma++', capsize=3.0)
    plt.errorbar(q[0,:15],sum_Sigma_pm_efw_s1_1_out2[:15], yerr=sum_Sigma_err_pm_efw_s1_1_out2[:15], fmt='r-', label='Sigma+-', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show()

    sum_Sigma_pp_diff_s1_2_out2 = np.sum(Sigma_pp_diff_s1_2_out2, axis=0)
    sum_Sigma_pm_diff_s1_2_out2 = np.sum(Sigma_pm_diff_s1_2_out2, axis=0)
    sum_Sigma_err_pp_diff_s1_2_out2 = np.sqrt(np.sum(np.multiply(Sigma_pp_err_diff_s1_2_out2[:9],Sigma_pp_err_diff_s1_2_out2[:9]), axis=0))
    sum_Sigma_err_pm_diff_s1_2_out2 = np.sqrt(np.sum(np.multiply(Sigma_pm_err_diff_s1_2_out2[:9],Sigma_pm_err_diff_s1_2_out2[:9]), axis=0))
    plotname = "Sigma for diffraction (second S1, second run) files simple sum"
    plt.figure(9998)
    plt.clf()
    # plt.plot(q[0,:9],sum_Sigma_pp_diff_s1_2_out2[:9], 'b-', label='Sigma++')
    # plt.plot(q[0,:9],sum_Sigma_pm_diff_s1_2_out2[:9], 'r-', label='Sigma+-')
    plt.errorbar(q[0,:9],sum_Sigma_pp_diff_s1_2_out2[:9], yerr=sum_Sigma_err_pp_diff_s1_2_out2[:9], fmt='b-', label='Sigma++', capsize=3.0)
    plt.errorbar(q[0,:9],sum_Sigma_pm_diff_s1_2_out2[:9], yerr=sum_Sigma_err_pm_diff_s1_2_out2[:9], fmt='r-', label='Sigma+-', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show()     

    sum_Sigma_pp_efw_s1_2_out2 = np.sum(Sigma_pp_efw_s1_2_out2, axis=0)
    sum_Sigma_pm_efw_s1_2_out2 = np.sum(Sigma_pm_efw_s1_2_out2, axis=0)
    sum_Sigma_err_pp_efw_s1_2_out2 = np.sqrt(np.sum(np.multiply(Sigma_pp_err_efw_s1_2_out2[:9],Sigma_pp_err_efw_s1_2_out2[:9]), axis=0))
    sum_Sigma_err_pm_efw_s1_2_out2 = np.sqrt(np.sum(np.multiply(Sigma_pm_err_efw_s1_2_out2[:9],Sigma_pm_err_efw_s1_2_out2[:9]), axis=0))
    plotname = "Sigma for EFW (first S1, second run) files simple sum"
    plt.figure(8887)
    plt.clf()
    # plt.plot(q[0,:9],sum_Sigma_pp_efw_s1_2_out2[:9], 'b-', label='Sigma++')
    # plt.plot(q[0,:9],sum_Sigma_pm_efw_s1_2_out2[:9], 'r-', label='Sigma+-')
    plt.errorbar(q[0,:9],sum_Sigma_pp_efw_s1_2_out2[:9], yerr=sum_Sigma_err_pp_efw_s1_2_out2[:9], fmt='b-', label='Sigma++', capsize=3.0)
    plt.errorbar(q[0,:9],sum_Sigma_pm_efw_s1_2_out2[:9], yerr=sum_Sigma_err_pm_efw_s1_2_out2[:9], fmt='r-', label='Sigma+-', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show()    
    
    
    # BACKGROUND REMOVAL
    
    
    # assuming number of files is the same for sample (run 1) and bg (run 2)!!!
    bg_rem_Sigma_pp_diff_s1_1 = sum_Sigma_pp_diff_s1_1_out1 - sum_Sigma_pp_diff_s1_1_out2
    bg_rem_Sigma_pm_diff_s1_1 = sum_Sigma_pm_diff_s1_1_out1 - sum_Sigma_pm_diff_s1_1_out2
    bg_rem_Sigma_err_pp_diff_s1_1 = sum_Sigma_err_pp_diff_s1_1_out1 - sum_Sigma_err_pp_diff_s1_1_out2
    bg_rem_Sigma_err_pm_diff_s1_1 = sum_Sigma_err_pm_diff_s1_1_out1 - sum_Sigma_err_pm_diff_s1_1_out2
    plotname = "Sigma for Diff (first S1, bg removed) files simple sum"
    plt.figure(7777)
    plt.clf()
    # plt.plot(q[0,:15],bg_rem_Sigma_pp_diff_s1_1[:15], 'b-', label='Sigma++')
    # plt.plot(q[0,:15],bg_rem_Sigma_pm_diff_s1_1[:15], 'r-', label='Sigma+-')
    plt.errorbar(q[0,:15],bg_rem_Sigma_pp_diff_s1_1[:15], yerr=bg_rem_Sigma_err_pp_diff_s1_1[:15], fmt='b-', label='Sigma++', capsize=3.0)
    plt.errorbar(q[0,:15],bg_rem_Sigma_pm_diff_s1_1[:15], yerr=bg_rem_Sigma_err_pm_diff_s1_1[:15], fmt='r-', label='Sigma+-', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show()    

    bg_rem_Sigma_pp_efw_s1_1 = sum_Sigma_pp_efw_s1_1_out1 - sum_Sigma_pp_efw_s1_1_out2
    bg_rem_Sigma_pm_efw_s1_1 = sum_Sigma_pm_efw_s1_1_out1 - sum_Sigma_pm_efw_s1_1_out2
    bg_rem_Sigma_err_pp_efw_s1_1 = sum_Sigma_err_pp_efw_s1_1_out1 - sum_Sigma_err_pp_efw_s1_1_out2
    bg_rem_Sigma_err_pm_efw_s1_1 = sum_Sigma_err_pm_efw_s1_1_out1 - sum_Sigma_err_pm_efw_s1_1_out2
    plotname = "Sigma for EFW (first S1, bg removed) files simple sum"
    plt.figure(6666)
    plt.clf()
    # plt.plot(q[0,:15],bg_rem_Sigma_pp_efw_s1_1[:15], 'b-', label='Sigma++')
    # plt.plot(q[0,:15],bg_rem_Sigma_pm_efw_s1_1[:15], 'r-', label='Sigma+-')
    plt.errorbar(q[0,:15],bg_rem_Sigma_pp_efw_s1_1[:15], yerr=bg_rem_Sigma_err_pp_efw_s1_1[:15], fmt='b-', label='Sigma++', capsize=3.0)
    plt.errorbar(q[0,:15],bg_rem_Sigma_pm_efw_s1_1[:15], yerr=bg_rem_Sigma_err_pm_efw_s1_1[:15], fmt='r-', label='Sigma+-', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show() 
    
    bg_rem_Sigma_pp_diff_s1_2 = sum_Sigma_pp_diff_s1_2_out1 - sum_Sigma_pp_diff_s1_2_out2
    bg_rem_Sigma_pm_diff_s1_2 = sum_Sigma_pm_diff_s1_2_out1 - sum_Sigma_pm_diff_s1_2_out2
    bg_rem_Sigma_err_pp_diff_s1_2 = sum_Sigma_err_pp_diff_s1_2_out1 - sum_Sigma_err_pp_diff_s1_2_out2
    bg_rem_Sigma_err_pm_diff_s1_2 = sum_Sigma_err_pm_diff_s1_2_out1 - sum_Sigma_err_pm_diff_s1_2_out2
    plotname = "Sigma for Diff (second S1, bg removed) files simple sum"
    plt.figure(7776)
    plt.clf()
    # plt.plot(q[0,:9],bg_rem_Sigma_pp_diff_s1_2[:9], 'b-', label='Sigma++')
    # plt.plot(q[0,:9],bg_rem_Sigma_pm_diff_s1_2[:9], 'r-', label='Sigma+-')
    plt.errorbar(q[0,:9],bg_rem_Sigma_pp_diff_s1_2[:9], yerr=bg_rem_Sigma_err_pp_diff_s1_2[:9], fmt='b-', label='Sigma++', capsize=3.0)
    plt.errorbar(q[0,:9],bg_rem_Sigma_pm_diff_s1_2[:9], yerr=bg_rem_Sigma_err_pm_diff_s1_2[:9], fmt='r-', label='Sigma+-', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show()    

    bg_rem_Sigma_pp_efw_s1_2 = sum_Sigma_pp_efw_s1_2_out1 - sum_Sigma_pp_efw_s1_2_out2
    bg_rem_Sigma_pm_efw_s1_2 = sum_Sigma_pm_efw_s1_2_out1 - sum_Sigma_pm_efw_s1_2_out2
    bg_rem_Sigma_err_pp_efw_s1_2 = sum_Sigma_err_pp_efw_s1_2_out1 - sum_Sigma_err_pp_efw_s1_2_out2
    bg_rem_Sigma_err_pm_efw_s1_2 = sum_Sigma_err_pm_efw_s1_2_out1 - sum_Sigma_err_pm_efw_s1_2_out2
    plotname = "Sigma for EFW (second S1, bg removed) files simple sum"
    plt.figure(6665)
    plt.clf()
    # plt.plot(q[0,:9],bg_rem_Sigma_pp_efw_s1_2[:9], 'b-', label='Sigma++')
    # plt.plot(q[0,:9],bg_rem_Sigma_pm_efw_s1_2[:9], 'r-', label='Sigma+-')
    plt.errorbar(q[0,:9],bg_rem_Sigma_pp_efw_s1_2[:9], yerr=bg_rem_Sigma_err_pp_efw_s1_2[:9], fmt='b-', label='Sigma++', capsize=3.0)
    plt.errorbar(q[0,:9],bg_rem_Sigma_pm_efw_s1_2[:9], yerr=bg_rem_Sigma_err_pm_efw_s1_2[:9], fmt='r-', label='Sigma+-', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show()     
 
    
     # COHERENT vs INCOHERENT

    Sigma_inc_diff_s1_1 =  1.5 * bg_rem_Sigma_pm_diff_s1_1
    Sigma_coh_diff_s1_1 =  1.5 * bg_rem_Sigma_pp_diff_s1_1 - 0.5 * bg_rem_Sigma_pm_diff_s1_1
    Sigma_inc_diff_err_s1_1 =  1.5 * bg_rem_Sigma_err_pm_diff_s1_1
    Sigma_coh_diff_err_s1_1 =  1.5 * bg_rem_Sigma_err_pp_diff_s1_1 - 0.5 * bg_rem_Sigma_err_pm_diff_s1_1
    plotname = "Coh vs Incoh for Diff (first S1, bg removed) files simple sum"
    plt.figure(5555)
    plt.clf()
    # plt.plot(q[0,:15],Sigma_inc_diff_s1_1[:15], 'b-', label='Sigma-inc')
    # plt.plot(q[0,:15],Sigma_coh_diff_s1_1[:15], 'r-', label='Sigma-coh')
    plt.errorbar(q[0,:15],Sigma_inc_diff_s1_1[:15], yerr=Sigma_inc_diff_err_s1_1[:15], fmt='b-', label='Sigma-inc', capsize=3.0)
    plt.errorbar(q[0,:15],Sigma_coh_diff_s1_1[:15], yerr=Sigma_coh_diff_err_s1_1[:15], fmt='r-', label='Sigma-coh', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show() 

    Sigma_inc_efw_s1_1 =  1.5 * bg_rem_Sigma_pm_efw_s1_1
    Sigma_coh_efw_s1_1 =  1.5 * bg_rem_Sigma_pp_efw_s1_1 - 0.5 * bg_rem_Sigma_pm_efw_s1_1
    Sigma_inc_efw_err_s1_1 =  1.5 * bg_rem_Sigma_err_pm_efw_s1_1
    Sigma_coh_efw_err_s1_1 =  1.5 * bg_rem_Sigma_err_pp_efw_s1_1 - 0.5 * bg_rem_Sigma_err_pm_efw_s1_1
    plotname = "Coh vs Incoh for EFW (first S1, bg removed) files simple sum"
    plt.figure(4444)
    plt.clf()
    # plt.plot(q[0,:15],Sigma_inc_efw_s1_1[:15], 'b-', label='Sigma-inc')
    # plt.plot(q[0,:15],Sigma_coh_efw_s1_1[:15], 'r-', label='Sigma-coh')
    plt.errorbar(q[0,:15],Sigma_inc_efw_s1_1[:15], yerr=Sigma_inc_efw_err_s1_1[:15], fmt='b-', label='Sigma-inc', capsize=3.0)
    plt.errorbar(q[0,:15],Sigma_coh_efw_s1_1[:15], yerr=Sigma_coh_efw_err_s1_1[:15], fmt='r-', label='Sigma-coh', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show()       

    Sigma_inc_diff_s1_2 =  1.5 * bg_rem_Sigma_pm_diff_s1_2
    Sigma_coh_diff_s1_2 =  1.5 * bg_rem_Sigma_pp_diff_s1_2 - 0.5 * bg_rem_Sigma_pm_diff_s1_2
    Sigma_inc_diff_err_s1_2 =  1.5 * bg_rem_Sigma_err_pm_diff_s1_2
    Sigma_coh_diff_err_s1_2 =  1.5 * bg_rem_Sigma_err_pp_diff_s1_2 - 0.5 * bg_rem_Sigma_err_pm_diff_s1_2
    plotname = "Coh vs Incoh for Diff (second S1, bg removed) files simple sum"
    plt.figure(5554)
    plt.clf()
    # plt.plot(q[0,:9],Sigma_inc_diff_s1_2[:9], 'b-', label='Sigma-inc')
    # plt.plot(q[0,:9],Sigma_coh_diff_s1_2[:9], 'r-', label='Sigma-coh')
    plt.errorbar(q[0,:9],Sigma_inc_diff_s1_2[:9], yerr=Sigma_inc_diff_err_s1_2[:9], fmt='b-', label='Sigma-inc', capsize=3.0)
    plt.errorbar(q[0,:9],Sigma_coh_diff_s1_2[:9], yerr=Sigma_coh_diff_err_s1_2[:9], fmt='r-', label='Sigma-coh', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show() 

    Sigma_inc_efw_s1_2 =  1.5 * bg_rem_Sigma_pm_efw_s1_2
    Sigma_coh_efw_s1_2 =  1.5 * bg_rem_Sigma_pp_efw_s1_2 - 0.5 * bg_rem_Sigma_pm_efw_s1_2
    Sigma_inc_efw_err_s1_2 =  1.5 * bg_rem_Sigma_err_pm_efw_s1_2
    Sigma_coh_efw_err_s1_2 =  1.5 * bg_rem_Sigma_err_pp_efw_s1_2 - 0.5 * bg_rem_Sigma_err_pm_efw_s1_2
    plotname = "Coh vs Incoh for EFW (second S1, bg removed) files simple sum"
    plt.figure(4443)
    plt.clf()
    # plt.plot(q[0,:9],Sigma_inc_efw_s1_2[:9], 'b-', label='Sigma-inc')
    # plt.plot(q[0,:9],Sigma_coh_efw_s1_2[:9], 'r-', label='Sigma-coh').
    plt.errorbar(q[0,:9],Sigma_inc_efw_s1_2[:9], yerr=Sigma_inc_efw_err_s1_2[:9], fmt='b-', label='Sigma-inc', capsize=3.0)
    plt.errorbar(q[0,:9],Sigma_coh_efw_s1_2[:9], yerr=Sigma_coh_efw_err_s1_2[:9], fmt='r-', label='Sigma-coh', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show() 
    
    plotname = "Coh vs Incoh for Diff (both S1, bg removed) files simple sum"
    plt.figure(5553)
    plt.clf()
    plt.errorbar(q[0,:9],Sigma_inc_diff_s1_2[:9], yerr=Sigma_inc_diff_err_s1_2[:9], fmt='b--', label='Sigma-inc', capsize=3.0)
    plt.errorbar(q[0,:9],Sigma_coh_diff_s1_2[:9], yerr=Sigma_coh_diff_err_s1_2[:9], fmt='r--', label='Sigma-coh', capsize=3.0)
    plt.errorbar(q[0,:15],Sigma_inc_diff_s1_1[:15], yerr=Sigma_inc_diff_err_s1_1[:15], fmt='b-', label='Sigma-inc', capsize=3.0)
    plt.errorbar(q[0,:15],Sigma_coh_diff_s1_1[:15], yerr=Sigma_coh_diff_err_s1_1[:15], fmt='r-', label='Sigma-coh', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show() 

    plotname = "Coh vs Incoh for EFW (both S1, bg removed) files simple sum"
    plt.figure(4442)
    plt.clf()
    plt.errorbar(q[0,:9],Sigma_inc_efw_s1_2[:9], yerr=Sigma_inc_efw_err_s1_2[:9], fmt='b--', label='Sigma-inc', capsize=3.0)
    plt.errorbar(q[0,:9],Sigma_coh_efw_s1_2[:9], yerr=Sigma_coh_efw_err_s1_2[:9], fmt='r--', label='Sigma-coh', capsize=3.0)
    plt.errorbar(q[0,:15],Sigma_inc_efw_s1_1[:15], yerr=Sigma_inc_efw_err_s1_1[:15], fmt='b-', label='Sigma-inc', capsize=3.0)
    plt.errorbar(q[0,:15],Sigma_coh_efw_s1_1[:15], yerr=Sigma_coh_efw_err_s1_1[:15], fmt='r-', label='Sigma-coh', capsize=3.0)
    plt.title(plotname)
    plt.xlabel('Q (inv Å)')
    plt.ylabel('Corrected data')
    plt.title(plotname)
    plt.legend()
    plt.show()    
    
    
    
    
    ##############################################################
    #
    #  COMPOSITE PLOTS
    #
    ##############################################################


if int(inputs[8]) > 2:
    print("\n---------- Run #3: ----------\n")
    lifetime3, opacity_theo3, opacity_depol3, time_start3, time_end3, P_He_old_empirical3, P_He_new_empirical3 = calibrate_cells(int(inputs[9]), int(inputs[35]), int(inputs[20]), int(inputs[21]), int(inputs[22]), int(inputs[23]), int(inputs[24]), q, count, numqvals, int(inputs[1]), wavelength, float(inputs[6]), float(inputs[7]))

if int(inputs[8]) > 3:
    print("\n---------- Run #4: ----------\n")
    lifetime4, opacity_theo4, opacity_depol4, time_start4, time_end4, P_He_old_empirical4, P_He_new_empirical4 = calibrate_cells(int(inputs[9]), int(inputs[35]), int(inputs[25]), int(inputs[26]), int(inputs[27]), int(inputs[28]), int(inputs[29]), q, count, numqvals, int(inputs[1]), wavelength, float(inputs[6]), float(inputs[7]))

if int(inputs[8]) > 4:
    print("\n---------- Run #5: ----------\n")
    lifetime5, opacity_theo5, opacity_depol5, time_start5, time_end5, P_He_old_empirical5, P_He_new_empirical5 = calibrate_cells(int(inputs[9]), int(inputs[35]), int(inputs[30]), int(inputs[31]), int(inputs[32]), int(inputs[33]), int(inputs[34]), q, count, numqvals, int(inputs[1]), wavelength, float(inputs[6]), float(inputs[7]))






