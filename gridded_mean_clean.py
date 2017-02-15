#!/usr/bin/env python
"""
Created on Mon Jan 30 14:54:07 2017

@author: naus010

Takes a gridded netcdf file as input and returns global mean, based on the 
pressure distribution set up by at and bt (and p0), and local mcf and ch4 
mixing ratios.

Remaining problem: Defining a tropopause. Do I impose it externally (also
    seasonal cycle?) or do I compute it from gradients. What gradients?
    Now I have Temperature gradient, but it's not perfect.
"""

import xarray as xr
import numpy as np
from numpy import array
import pandas as pa
import netCDF4 as ncdf
import matplotlib.pylab as plt
import os
import sys

def calc_pfield(at,bt,p0):
    '''Convert at,bt and p0 to pressure in each grid box'''
    pres = np.zeros((nz+1,ny,nx))
    for i in range(nx):
        for j in range(ny):
            pres[:,j,i] = p0[j,i]*bt + at
    return pres
    
def tropheight(tgrad):
    '''
    Compute the height in each gridbox where the temperature gradient gets
    below 2 degree C, ie where the troposphere ends
    Returns an nz x ny x nx array, where boxes to be included are indicated by
    one, and the others by zero.
    '''
    trop = np.ones((nz,ny,nx))
    for iy in range(ny):
        for ix in range(nx):
            for iz in range(10,nz-1):
                if tgrad[iz,iy,ix] < 2:
                    trop[iz:,iy,ix] = np.zeros(nz-iz)
                    break
    return trop

name_files = raw_input()
grd_drp_vars = ["rn222","sf6","ch4bb","ch4wlbb","ch4ctle4"] # Variables not to read
ar = []
for name in name_files:
    data = xr.open_dataset(name,drop_variables=grd_drp_vars)
    date = data.idatei
    yr,mo = date[0],date[1]
    p0 = data['presm'].values
    temp = data['tempm'].values
    mcf = data['mcf'].values
    ch4 = data['ch4ctl'].values
    at,bt = data.at, data.bt
    
    ny,nx = p0.shape
    nz = len(at)-1
    
    p_all = calc_pfield(at,bt,p0)
    p_dif = p_all[:-1]-p_all[1:]
    temp_grad = temp[:-1]-temp[1:]
    trop = tropheight(temp_grad)
    mcf_we = np.sum((trop*p_dif*mcf))/np.sum((trop*p_dif))*1e12
    ch4_we = np.sum((trop*p_dif*ch4))/np.sum((trop*p_dif))*1e9
    ari = [yr,mo,ch4_we,mcf_we]
    ar.append(array(ari))
    
np.savetxt('Gridded_monthly_mean_ch4_mcf',ar)


























