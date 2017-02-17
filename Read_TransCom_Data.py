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
import random

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
            temp_grad = temp[:-1]-temp[1:] # temperature gradient
            tmask = tropheight(temp_grad) 
            # Weighted means
            p_grad, mcf, ch4 = p_grad[tmask],mcf[tmask],ch4[tmask]
            mcf_we = np.sum((p_grad*mcf))/np.sum((p_grad))*1e12
            ch4_we = np.sum((p_grad*ch4))/np.sum((p_grad))*1e9
            ari = [yr,mo,ch4_we,mcf_we]
            ar.append(array(ari))
    return array(ar)

def read_stat_data(direc,sty,edy,stations):
    '''
    Returns the station data: CH4(ctl) and MCF mixing ratios at the specified
    stations.
    '''
    cwd = os.getcwd()
    indir = os.path.join(cwd,direc)    
    data_stat = []
    stat_nos   = array(stat_pars.loc[stations,'Index'].astype('int'))
    for root,dirs,filenames in os.walk(indir):
<<<<<<< HEAD
        for i,f in enumerate(filenames): # reading in the unsorted data files
            fi    = os.path.join(root,f)
            data  = xar.open_dataset(fi)
            yr,mo = unpack_hdf(data)
            mcf   = np.swapaxes(data['mcf_grd'].values,0,1)[stat_nos]
            ch4   = np.swapaxes(data['ch4ctl_grd'].values,0,1)[stat_nos]
            data_stat.append([yr,mo,ch4,mcf])
    data_srt = sort_stat(data_stat) # sort data by year and month
    data_sel = select_time(data_srt,sty,edy) # select the relevant years
    ch4,mcf  = stat_reshape(data_sel)
    return ch4, mcf

def make_pandas(data,sty,edy,stations):
    '''
    Makes a pandas data frame from data under very specific conditions.
    Input:
     - data    : the data. Should have dimensions (nst,(edy-sty+1),hpy)
     - sty,edy : start year and end year
     - stations: station identification tags
    Returns:
     - Pandas dataframe
    '''
    nst   = len(stations)
    dataf = data.reshape((nst,(edy-sty+1)*hpy))
    id_yr = np.arange(sty,edy+1)
    id_mo = np.arange(hpy)
    colum = pd.MultiIndex.from_product([id_yr,id_mo],names=['Year','Month'])
    index = pd.Index(stations,name='Station')
    frame = pd.DataFrame(dataf, index=index, columns=colum)
    return frame
=======
        for i,f in enumerate(filenames):
<<<<<<< HEAD
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
=======
            fi = os.path.join(root,f)
            data = xar.open_dataset(fi)
            yr,mo = unpack_hdf(data)
            mcf = np.swapaxes(data['mcf_grd'].values,0,1)[stations]
            ch4 = np.swapaxes(data['ch4ctl_grd'].values,0,1)[stations]
            data_stat.append([yr,mo,mcf,ch4])
    data_srt = sort_stat(data_stat) # sort data by year and month
    yrs,data_sel = select_time(data_srt,sty,edy) # select the relevant years
    data_ref = stat_reform2(data_sel) # shape it in the good shape
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
>>>>>>> parent of bd9f3f4... pandas is working
>>>>>>> master
    
def make_dataframe(ch4,mcf,sty,edy,stations):
    '''
    Makes a pandas data frame from data under very specific conditions.
    Input:
     - ch4,mcf : the data
     - sty,edy : start year and end year
     - stations: station identification tags
    Returns:
     - Dataframe
    '''
    ch4f,mcff = ch4.flatten(),mcf.flatten()
    nyr = edy-sty+1
    nst = len(stations)
    col_st = np.repeat(stations,nyr*hpy)
    col_yr = np.tile(np.repeat(np.arange(sty,edy+1),hpy),nst)
    col_mo = np.tile(np.repeat(np.arange(0,12),hpm), nst*nyr)
    col_hr = np.tile(np.arange(0,hpm), nst*nyr*12)
    datat = np.column_stack((col_st,col_yr,col_mo,col_hr,ch4f,mcff))
    colum = pd.Index(['station','year','month','hour','ch4','mcf'])
    df    = pd.DataFrame(datat, columns=colum)
    df[['year','month','hour','ch4','mcf']] = df[['year','month','hour','ch4','mcf']].apply(pd.to_numeric)
    return df
    
def sort_stat(data):
    '''
    Sort station data according to year and month.
    This is somewhat complicated because station data is not rectangular and
    thus not a numpy array.
    '''
<<<<<<< HEAD
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
    return data_ful

def stat_reshape(data):
    '''
    This function takes the sorted, selected array and groups the data in groups
    of 1 year. The monthly distinction is removed.
    Output is: ch4,mcf[istation][iyear][imeas]
    '''
=======
<<<<<<< HEAD
    id_dy   = np.arange(nms/24.)
    colum_t = pd.MultiIndex.from_product([[yr],[mo],id_dy,id_hr], names=['Year','Month','Day','Hour'])
    return colum_t
    
=======
>>>>>>> master
    ch4_tot,mcf_tot = [],[] # all data from all stations
    nst = len(data[0][0][0])
    for ist in range(nst):
        ch4_st = []; mcf_st = [] # all data from 1 station
        for iy, datay in enumerate(data):
            ch4_sty = []; mcf_sty = [] # 1 year from 1 station
            for im,datam in enumerate(datay):
                nd = len(datam[0][ist])
                for di in range(nd):
                    ch4_sty.append(datay[im][0][ist][di]) # 1 meas
                    mcf_sty.append(datay[im][1][ist][di])
<<<<<<< HEAD
            ch4_st.append(array(ch4_sty[:8760])) 
            mcf_st.append(array(mcf_sty[:8760]))
=======
            ch4_st.append(array(ch4_sty)) 
            mcf_st.append(array(mcf_sty))
>>>>>>> master
        ch4_tot.append(array(ch4_st))
        mcf_tot.append(array(mcf_st))
    return array(ch4_tot), array(mcf_tot)

<<<<<<< HEAD
=======
>>>>>>> parent of 3c988c8... Finished filter
>>>>>>> master
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

def tropheight(tgrad):
    '''
    Compute the height in each gridbox where the temperature gradient gets
    below 2 degree C, ie where the troposphere ends
    Returns a mask
    '''
    pheight = np.argmax(tgr[9:]<=2,axis=0)+9
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

read_grid = True
read_stat = True
sty,edy = 1990,2006
yrs = np.arange(sty,edy+1)
hpy = 365*24 # hours per year
hpm = hpy/12

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
<<<<<<< HEAD
    ch4_st,mcf_st = read_stat_data(dirc2,sty,edy,sid_all) # station data
    ch4_st*=1e9
    mcf_st*=1e12
    ch4_stp = make_pandas(ch4_st,sty,edy,sid_all)
    mcf_stp = make_pandas(mcf_st,sty,edy,sid_all)
    df_st   = make_dataframe(ch4_st,mcf_st,sty,edy,sid_all)
=======
<<<<<<< HEAD
<<<<<<< HEAD
    ch4_st,mcf_st = read_stat_data2(dirc2,stations=sid_all) # station data
    ch4_st*=1e9
    mcf_st*=1e12
=======
    yrs,ch4_st,mcf_st = read_stat_data(dirc2,sty,edy,stations=stat_nos) # station data
<<<<<<< HEAD
>>>>>>> parent of bd9f3f4... pandas is working
=======
    yrs,station_data = read_stat_data(dirc2,sty,edy,stations=stat_nos) # station data
>>>>>>> parent of 3c988c8... Finished filter
=======
>>>>>>> parent of bd9f3f4... pandas is working
>>>>>>> master
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