# -*- coding: utf-8 -*-
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

cwd = os.getcwd()
grd_file = "gridded_data.nc"
grd_drp_vars = ["rn222","sf6","ch4bb","ch4wlbb","ch4ctle4"] # Variables not to read
sta_file = "station_data.nc"
data = xr.open_dataset(os.path.join(cwd,"TransCom data",grd_file),drop_variables=grd_drp_vars)
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
print 'mcf',mcf_we
print 'ch4',ch4_we


plt.figure()
cp = plt.contourf(np.linspace(-88,88,num=ny),np.linspace(0.5,24.5,num=nz),np.mean(p_dif,axis=2))
cb = plt.colorbar(cp)
cb.set_label('Pressure gradient (Pa)')

plt.figure()
cp = plt.contourf(np.linspace(-88,88,num=ny),np.linspace(0.5,24.5,num=nz),np.mean(temp,axis=2))
cb = plt.colorbar(cp)
cb.set_label('T (K)')

plt.figure()
cp2 = plt.contourf(np.linspace(-88,88,num=ny),np.linspace(1.0,24.0,num=nz-1),np.mean(temp_grad,axis=2),20)
for i in range(10):
    plt.plot(np.linspace(-88,88,num=ny),[10+i]*ny)
cb2 = plt.colorbar(cp2)
cb2.set_label('T gradient (K)')

plt.figure()
cp2 = plt.contourf(np.linspace(-88,88,num=ny),np.linspace(.5,24.5,num=nz),np.mean(trop,axis=2),20)
cb2 = plt.colorbar(cp2)
cb2.set_label('Tropospheric height')

plt.figure()
plt.plot(np.mean(p_all,axis=(1,2)), range(nz+1))
plt.plot(p_all[:,1,1], range(nz+1))

























