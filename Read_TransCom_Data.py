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
import pandas as pd
import seaborn as sns

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
            tmask = tropheight(temp) 
            # Weighted means
            p_grad, mcf, ch4 = p_grad[tmask],mcf[tmask],ch4[tmask]
            mcf_we = np.sum((p_grad*mcf))/np.sum((p_grad))*1e12
            ch4_we = np.sum((p_grad*ch4))/np.sum((p_grad))*1e9
            ari = [yr,mo,ch4_we,mcf_we]
            ar.append(array(ari))
    return array(ar)

def read_stat_data2(direc, stations):
    '''
    Reads in all station files. 
    Input:
     - direc: The directory containing the station files
     - stations: The stations that need to be read
    Output: 
     - A pandas DataFrame, which contains stations as its rows, and measurement
        instances as its columns. Columns are subdivided in year,month,day,hour.
    '''
    cwd        = os.getcwd()
    indir      = os.path.join(cwd,direc)
    stat_nos   = array(stat_pars.loc[stations,'Index'].astype('int'))
    id_hr      = np.arange(1,25)
    for root,dirs,filenames in os.walk(indir):
        for i,f in enumerate(filenames):
            fi    = os.path.join(root,f) # locate file
            data  = xar.open_dataset(fi) # open file
            yr,mo = unpack_hdf(data)     # year/month of file
            ch4   = np.swapaxes(data['ch4ctl_grd'].values,0,1)[stat_nos]
            mcf   = np.swapaxes(data['mcf_grd'].values,0,1)[stat_nos]
            nms   = ch4.shape[1]
            colum = make_pdindex(yr,mo,nms,id_hr)
            ch4_sti = pd.DataFrame(ch4, index=stations, columns=colum)
            mcf_sti = pd.DataFrame(mcf, index=stations, columns=colum)
            if i == 0:
                ch4_sta = ch4_sti
                mcf_sta = mcf_sti
            else:
                ch4_sta = ch4_sta.join(ch4_sti)
                mcf_sta = mcf_sta.join(ch4_sti)
    return ch4_sta.sort_index(),mcf_sta.sort_index()
    
def make_pdindex(yr,mo,nms,id_hr):
    '''
    Makes a panda index for a given month.
    Input:
    - yr : Year of interest
    - mo : Month of interest
    - nms: Number of measurements in the month of interest.
    - id_hr: The index of hours per day (simply np.arange(1,25))
    Output:
    - A pandas multi-index with a level for the year, month and hour
    '''
    id_dy   = np.arange(nms/24.)
    colum_t = pd.MultiIndex.from_product([[yr],[mo],id_dy,id_hr], names=['Year','Month','Day','Hour'])
    return colum_t
    
def unpack_hdf(data):
    ''' 
    Uses the name of the HDF source of a nc station dataset to find out the 
    year & month the data corresponds to
    '''
    name = data.hdf_source
    yr,mo = int(name[8:12]), int(name[12:14])
    return yr,mo

def calc_pfield(at,bt,p0):
    p0r = p0.reshape(1,45,60)
    atr = at.reshape(26,1,1)
    btr = bt.reshape(26,1,1)
    pfield = p0r*btr + atr
    return pfield

def tropheight(t):
    '''
    Compute the height in each gridbox where the temperature gradient gets
    below 2 degree C, ie where the troposphere ends
    Returns a mask
    '''
    tgrad = t[:-1]-t[1:]
    pheight = np.argmax(tgrad[9:]<=2,axis=0)+10
    mask = np.ones(t.shape,dtype='bool')
    for iy,py in enumerate(pheight):
        for ix,px in enumerate(py):
            mask[px:,iy,ix] = False
    return mask

def load_txt(f):
    ''' Load 1 column text files '''
    fil = open(f,'r')
    data = []
    for line in fil.readlines():
        if line[0] == '#': continue
        data.append(line.split()[0].upper())
    return data

def find_box(lat, box_edges):
    '''
    Input:
    - lati     : Latitude of a given station
    - box_edges: The latitudinal edges of the intended box distribution
    Output:
    - box_no   : The box number the station should be placed in
    '''
    box_no = None
    n_edge = len(box_edges)
    for j in range(n_edge):
        if lati >= box_edges[j] and lati < box_edges[j+1]: 
            box_no = j
    if box_no == None: 
        print 'No box number found for the station with latitude:',lat
    return box_no
    

read_grid = False
read_stat = True
sty,edy = 1990,2006

# ++++ Create a station dictionary
file_ex = os.getcwd()+'\\TransCom data\\Stat data\\station_file_002.nc' # example data file for station information
stat_datex = Dataset(file_ex, 'r')
stat_ids = np.array(stat_datex.station_ident.split()) # station abbreviations
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

# Making a station pandaframe, so that info from each station can be requested
stat_pars = ['Lon', 'Lat', 'Elev','Box_NOAA','Box_AGAGE','Index']  # station parameters
nst_pars  = len(stat_pars)
stat_uni  = np.unique(stat_ids); nst_u = len(stat_uni) # unique station ids
stat_ind = pd.Index(stat_uni, name='Station')
stat_col = pd.Index(stat_pars,name='Parameter')
stat_pars = pd.DataFrame(np.zeros((nst_u,nst_pars)), index=stat_ind, columns=stat_col)
passed = []
for i,sid in enumerate(stat_ids):
    if sid in passed: continue; passed.append(sid) # No duplicates
    index = i
    loni  = stat_lon[i]
    lati  = stat_lat[i]
    elevi = stat_elev[i]
    box_noaa = find_box(lati, boxes_noaa)
    box_gage = find_box(lati, boxes_gage)
    stat_pars.loc[sid,'Lon']       = loni     # longitude
    stat_pars.loc[sid,'Lat']       = lati     # latitude
    stat_pars.loc[sid,'Elev']      = elevi    # elevation
    stat_pars.loc[sid,'Box_NOAA']  = box_noaa # noaa box number
    stat_pars.loc[sid,'Box_AGAGE'] = box_gage # agage box number
    stat_pars.loc[sid,'Index']     = index    # index in the station data
# stations in the NOAA and AGAGE network
sid_agage    = load_txt('Stations AGAGE.txt')
sid_noaa_mcf = load_txt('Stations NOAA MCF.txt')
sid_noaa_ch4 = load_txt('Stations NOAA CH4.txt')
sid_all      = np.unique(sid_agage+sid_noaa_mcf+sid_noaa_ch4)
stat_nos     = array(stat_pars.loc[sid_all,'Index'].astype('int'))

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
    yrs,ch4_st,mcf_st = read_stat_data2(dirc2,stations=sid_all) # station data
    ch4_st*=1e9
    mcf_st*=1e12
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

m = xar.open_dataset('station_file_002.nc')
m2 = xar.open_dataset('gridded_data.nc')
