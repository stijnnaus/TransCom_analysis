# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 14:38:34 2017

@author: naus010
"""

import os
import xarray as xar
import numpy as np
from numpy import array
import matplotlib.pylab as plt
from netCDF4 import Dataset
import time

def grid_yav(data):
    ''' From monthly global grid data, computes yearly averages for MCF and CH4 '''
    yrs, mos = data[:,0],data[:,1]
    data = data[np.lexsort((mos,yrs))] # sort by year and then month
    ny = len(data)/12
    if ny != len(data)/12.: print 'number of months is not a multiple of 12'
    data_resh = data.reshape((ny,12,4)) # reshape so that years are grouped
    yavs = np.zeros((ny,3))
    for i,datai in enumerate(data_resh):
        yr = datai[0,0]
        yav_ch4 = np.mean(datai[:,2])
        yav_mcf = np.mean(datai[:,3])
        yavs[i] = array([yr+.5,yav_ch4,yav_mcf])
    return yavs

def grid_mav(data):
    ''' From monthly global grid data, computes monthly averages for MCF and CH4 '''
    yrs, mos = data[:,0],data[:,1]
    data = data[np.lexsort((mos,yrs))] # sort by year and then month
    nmo = len(data)
    data_resh = data.reshape((nmo,4)) # reshape so that years are grouped
    mavs = np.zeros((ny,3))
    for i,datai in enumerate(data_resh):
        yr = datai[0,0]
        mav_ch4 = np.mean(datai[:,2])
        mav_mcf = np.mean(datai[:,3])
        mavs[i] = array([mav_ch4,mav_mcf])
    return mavs

def read_grid_data(direc):
    '''
    Reads in the separate monthly grid files from the TransCom data.
    Returns the monthly weighted average of CH4 and MCF.
    '''    
    cwd = os.getcwd()
    indir = os.path.join(cwd,direc)
    ar = []
    grd_drp_vars = ["rn222","sf6","ch4bb","ch4wlbb","ch4ctle4"]
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            fi = os.path.join(root,f)
            data = xar.open_dataset(fi,drop_variables=grd_drp_vars)
            date = data.idatei
            yr,mo = date[0],date[1]
            p0 = data['presm'].values # surface pressure
            temp = data['tempm'].values # 3D temperature field
            mcf = data['mcf'].values # 3D mcf concentrations
            ch4 = data['ch4ctl'].values # 3D ch4 concentrations
            at,bt = data.at, data.bt
            
            p_all = calc_pfield(at,bt,p0) # 3D pressure field at edges of grid boxes
            p_grad = p_all[:-1]-p_all[1:] # pdif between top/bot grid box
            temp_grad = temp[:-1]-temp[1:] # temeprature gradient
            trop = tropheight(temp_grad) 
            # Weighted means
            mcf_we = np.sum((trop*p_grad*mcf))/np.sum((trop*p_grad))*1e12
            ch4_we = np.sum((trop*p_grad*ch4))/np.sum((trop*p_grad))*1e9
            ari = [yr,mo,ch4_we,mcf_we]
            ar.append(array(ari))
    return array(ar)

def read_stat_data(direc,sty,edy,stations='All'):
    '''
    Returns the station data: CH4(ctl) and MCF mixing ratios at the specified
    stations.
    '''
    if stations=='All': stations=range(395)
    cwd = os.getcwd()
    indir = os.path.join(cwd,direc)    
    data_stat = []
    for root,dirs,filenames in os.walk(indir):
        for i,f in enumerate(filenames):
            fi = os.path.join(root,f)
            data = xar.open_dataset(fi)
            yr,mo = unpack_hdf(data)
            mcf = np.swapaxes(data['mcf_grd'].values,0,1)[stations]
            ch4 = np.swapaxes(data['ch4ctl_grd'].values,0,1)[stations]
            data_stat.append([yr,mo,mcf,ch4])
    data_srt = sort_stat(data_stat) # sort data by year and month
    yrs,data_sel = select_time(data_srt,sty,edy) # select the relevant years
    data_ref = stat_reform(data_sel) # shape it in the good shape
    return yrs,data_ref

def sort_stat(data):
    '''
    Sort station data according to year and month.
    This is somewhat complicated because station data is not rectangular and
    thus not a numpy array.
    '''
    yrs = array([item[0] for item in data])
    mos = array([item[1] for item in data])
    keys_srt = np.lexsort((mos,yrs))
    data_srt = []
    for key in keys_srt:
        data_srt.append(data[key])
    return data_srt
    
def select_time(data, sty, edy):
    '''
    Select a time period for the sorted station dataset
    Returns selected data, but also divides the data in bricks of 1 year each.
    '''
    cyr = sty
    data_ful = []
    datay = []
    for item in data:
        yr,mo = item[0],item[1]
        if yr<sty: continue
        if yr>cyr:
            if len(datay)!=12: print cyr, 'does not have 12 months'
            cyr=yr
            data_ful.append(datay)
            datay = []
        datay.append(array(item[2:4]))
        if yr>edy: break
    yrs = np.arange(sty,edy+1)
    return yrs,data_ful

def stat_reform(data):
    '''
    This function takes the sorted, selected array and groups the data in groups
    of 1 year. The monthly distinction is removed.
    Output is: data[ch4(0),mcf(1)][istation][iyear][imonth][imeas]
    '''
    ch4_tot,mcf_tot = [],[] # all data from all stations
    nst = len(data[0][0][0])
    for ist in range(nst):
        ch4_st = []; mcf_st = [] # all data from 1 station
        for iy, datay in enumerate(data):
            ch4y_st = []; mcfy_st = [] # 1 year data from 1 station
            for im, datam in enumerate(datay):
                ch4i_st = datam[0][ist]; mcfi_st = datam[1][ist] # 1 month of data from 1 station
                ch4y_st.append(array(ch4i_st))
                mcfy_st.append(array(mcfi_st))
            ch4_st.append(array(ch4y_st))
            mcf_st.append(array(mcfy_st))
        ch4_tot.append(array(ch4_st))
        mcf_tot.append(array(mcf_st))
        #datar.append([ch4_st,mcf_st]) if you want [ist][mcf/ch4] instead of [mcf/ch4][ist]
    return array([array(ch4_tot),array(mcf_tot)])

def unpack_hdf(data):
    ''' 
    Uses the name of the HDF source of a nc station dataset to find out the 
    year & month the data corresponds to
    '''
    name = data.hdf_source
    yr,mo = int(name[8:12]), int(name[12:14])
    return yr,mo
    

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

def load_txt(f):
    ''' Load 1 column text files '''
    fil = open(f,'r')
    data = []
    for line in fil.readlines():
        if line[0] == '#': continue
        data.append(line.split()[0].upper())
    return data

def select_unique(l):
    ''' Select the unique elements in a list '''
    lu = []
    for e in l:
        if e not in lu:
            lu.append(e)
    return lu

read_grid = False
read_stat = True
sty,edy = 1990,2006

# ++++ Create a station dictionary
file_ex = os.getcwd()+'\\TransCom data\\Stat data\\station_file_002.nc' # example data file for station information
stat_datex = Dataset(file_ex, 'r')
stat_ids = np.array(stat_datex.station_ident.split())
stat_lon = stat_datex.station_lon
stat_lat = stat_datex.station_lat
stat_elev = stat_datex.station_height

boxes_noaa = [-90,-60,-30,0,30,60,90] # edges of the boxes
boxes_gage = [-90,-30,0,30,90]
nbox_noaa = len(boxes_noaa)-1
nbox_gage = len(boxes_gage)-1
ar1, ar2, ar3 = 0.5, 0.5*np.sqrt(3) - 0.5, 1 - 0.5*np.sqrt(3)
w_noaa = [ar3, ar2, ar1, ar1, ar2, ar3] # Weights per box based on area per box
w_gage = [.25, .25, .25, .25]

# Making a station dictionary, so that info from each station can be requested
stat_dic = {}
stat_dic['Description'] = ['Longitude', 'Latitude', 'Elevation','Box Number']
for i,sid in enumerate(stat_ids):
    if sid in stat_dic.keys(): continue # No duplicates
    boxno_noaa = None; box_gage = None
    loni = stat_lon[i]
    lati = stat_lat[i]
    elevi = stat_elev[i]
    for j in range(nbox_noaa):
        if lati >= boxes_noaa[j] and lati < boxes_noaa[j+1]: boxno_noaa = j
    for j in range(nbox_gage):
        if lati >= boxes_gage[j] and lati < boxes_gage[j+1]: boxno_gage = j
    if boxno_noaa == None: print sid, 'has no noaa box'
    if boxno_gage == None: print sid, 'has no agage box'
    stat_dic[sid] = [loni, lati, elevi, boxno_gage, boxno_noaa]

stations_agage = load_txt('Stations AGAGE.txt')
stations_noaa_mcf = load_txt('Stations NOAA MCF.txt')
stations_noaa_ch4 = load_txt('Stations NOAA CH4.txt')
stations_all = select_unique(stations_agage+stations_noaa_mcf+stations_noaa_ch4)
stat_nos = [stat_ids.tolist().index(stat) for stat in stations_all]

# Grid data
if read_grid:
    nx,ny,nz = 60,45,25
    dirc = 'TransCom data\Grid data'
    print 'Reading grid data .........'
    start = time.time()
    grid_data = read_grid_data(dirc)
    end = time.time()
    print 'Reading the grid data took', end-start, 'seconds'
    grid_y = grid_yav(grid_data)

# Station data
if read_stat:
    dirc2 = 'TransCom data\Stat data'
    print 'Reading station data ..........'
    start = time.time()
    yrs,station_data = read_stat_data(dirc2,sty,edy,stations=stat_nos) # station data
    end = time.time()
    print 'Reading the stat data took',end-start,'seconds'

grid_yrs = grid_data[:,0]+grid_data[:,1]/12.
grid_ch4 = grid_data[:,2]
grid_mcf = grid_data[:,3]
fig = plt.figure(figsize=(10,100))
ax1 = fig.add_subplot(211); ax2 = fig.add_subplot(212)
ax1.set_title('Global means from gridded data\n\nMCF')
ax2.set_title(r'CH$_4$')
ax1.set_ylabel('MCF (ppt)')
ax2.set_ylabel(r'CH$_4$ (ppb)')
ax1.plot(grid_yrs,grid_mcf,'bo', label='Monthly mean')
ax1.plot(grid_y[:,0],grid_y[:,2],'k-',linewidth=4.,label='Global mean')
ax2.plot(grid_yrs,grid_ch4,'go', label='Monthly mean')
ax2.plot(grid_y[:,0],grid_y[:,1],'k-',linewidth=4.,label='Global mean')
ax1.grid(); ax2.grid()
ax1.legend(loc='best'); ax2.legend(loc='best')

xar.open_dataset('station_file_002.nc')







