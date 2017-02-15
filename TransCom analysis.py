# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:45:40 2016

@author: naus010

In this file the TransCom data of MCF and CH4 of February 2000 is analysed.
Plots of the gridded, monthly mean data are given.
Also given is an analysis of how well different sets of station capture the
monthly mean.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import sys
from math import *
import os
import netCDF4 as ncdf
import random as rnd
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap

def global_mean(data, stat_nos, stat_error, box_dist, we, nruns=50, random = False):
    '''
    Calculates the global means from station data. 
    data: the station dataset
    stat_nos: contains the indices of the stations in the full dataset data. 
    stat_error: error per station
    box_dist: the distribution of the stations over the boxes
    we: weights per box
    random: if False, means are calculated per timestep. If True, means are 
    calculated by taking data from each station at a different (random) time
    instance.
    '''
    nmeas = len(data)
    nstat = len(stat_nos)
    tims = range(nmeas)
    glob = np.zeros(nmeas)
    glob_e = np.zeros(nmeas)
    for ii,tim in enumerate(tims):
        datai = np.zeros(nstat)
        if random:
            for j,no in enumerate(stat_nos):
                ind = rnd.choice(tims)
                datai[j] = data[ind][no]
        if not random:
            datai = data[tim][stat_nos]
        glob_ch4_avs = np.zeros(nruns)
        for i in range(nruns):
            glob_ch4_av = box_mean(datai, box_dist, err=stat_error, weights=we)
            glob_ch4_avs[i] = glob_ch4_av
        glob_ch4_av = np.mean(glob_ch4_avs)
        glob_ch4_sd = np.std(glob_ch4_avs)
        glob[ii] = glob_ch4_av
        glob_e[ii] = glob_ch4_sd
    return glob, glob_e

def box_distribution(stations, stat_dic, nos=False, datatype='NOAA'):
    '''
    Given the station dictionary, this function returns the distribution over 
    the boxes of a list of station IDs. The box distribution is returned with
    stations indicated by their ID (nos = False), or by their index in the \
    stations list (nos = True).
    Datatype tells which box numbers to get: those from the NOAA or from the
    AGAGE box distribution.
    '''
    if datatype == 'NOAA':
        box_nos = [ [] for i in range(nbox_noaa) ]
        box_ids = [ [] for i in range(nbox_noaa) ]
        for i,stat in enumerate(stations):
            box_no = stat_dic[stat][-1]
            box_nos[box_no].append(i)
            box_ids[box_no].append(stat)
    elif datatype == 'GAGE' or datatype == 'AGAGE':
        box_nos = [ [] for i in range(nbox_gage) ]
        box_ids = [ [] for i in range(nbox_gage) ]
        for i,stat in enumerate(stations):
            box_no = stat_dic[stat][-2]
            box_nos[box_no].append(i)
            box_ids[box_no].append(stat)
    if not nos:
        return box_ids
    elif nos:
        return box_nos
    
def box_mean(data, box_nos, err=0., weights=None):
    '''
    Calculates the average of the data (one timestep), using the given distribution
    of boxes. Optional argument err applies a random perturbation to the station
    data and weights uses a weight distribution for the boxes.
    '''
    box_avs = np.zeros(len(box_nos))
    for i,box in enumerate(box_nos):
        box_av = 0
        for j in box:
            box_av += data[j] + err*(1-2*np.random.rand())
        box_av /= len(box)
        box_avs[i] = box_av
    return np.average(box_avs, weights=weights)
    
def date_to_dec(dates):
    ''' Converts date objects of type [yr,mo,dy,hr,mn,sc] to decimal dates '''
    ndates = len(dates)
    decdates = np.zeros(ndates)
    dpm = [31,28,31,30,31,30,31,31,30,31,30,31] # days per month
    daypyr = 365.
    hrpyr = daypyr*24.
    mnpyr = hrpyr*60.
    scpyr = mnpyr*60.
    for i,date in enumerate(dates):
        yr,mo,dy,hr,mn,sc = date
        nd = sum([dpm[mon] for mon in range(mo-1)]) + dy
        decdates[i] = yr + nd/daypyr + hr/hrpyr + mn/mnpyr + sc/scpyr
    return decdates

make_plots = True

cwd = os.getcwd()
data_dir = cwd + '\\TransCom Data\\'
os.chdir(data_dir)
file1 = 'gridded_data.nc'
file2 = 'station_data.nc'

stat_data = Dataset(file2, 'r')
stat_ids = np.array(stat_data.station_ident.split())
stat_lon = stat_data.station_lon
stat_lat = stat_data.station_lat
stat_elev = stat_data.station_height

boxes_noaa = [-90,-60,-30,0,30,60,90] # edges of the boxes
boxes_gage = [-90,-30,0,30,90]
nbox_noaa = len(boxes_noaa)-1
nbox_gage = len(boxes_gage)-1
ar1, ar2, ar3 = 0.5, 0.5*sqrt(3) - 0.5, 1 - 0.5*sqrt(3)
w_noaa = [ar3, ar2, ar1, ar1, ar2, ar3] # Weights per box based on area per box
w_gage = [.25, .25, .25, .25]

# Making a station dictionary, so that info from each station can be requested
stat_dic = {}
stat_dic['Description'] = ['Longitude', 'Latitude', 'Elevation','Box Number']
for i,sid in enumerate(stat_ids):
    if sid in stat_dic.keys(): continue # No duplicates
    boxno_noaa = None
    loni = stat_lon[i]
    lati = stat_lat[i]
    elevi = stat_elev[i]
    for i in range(nbox_noaa):
        if lati >= boxes_noaa[i] and lati < boxes_noaa[i+1]: boxno_noaa = i
    for i in range(nbox_gage):
        if lati >= boxes_gage[i] and lati < boxes_gage[i+1]: boxno_gage = i
    if boxno_noaa == None: print sid, 'has no noaa box'
    if boxno_gage == None: print sid, 'has no agage box'
    stat_dic[sid] = [loni, lati, elevi, boxno_gage, boxno_noaa]

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                       Montzka station calculations
mon_stats = ['SPO','CGO', 'SMO', 'MLO', 'KUM', 'NWR', 'LEF', 'BRW', 'ALT'] # Stations used in the Montzka (2011) study
mon_nos = [stat_ids.tolist().index(stat) for stat in mon_stats] # Index no of the selected stations in the full station list
mon_lon = [stat_dic[stat][0] for stat in mon_stats]
mon_lat = [stat_dic[stat][1] for stat in mon_stats]
stat_dates = stat_data.variables['idate'][:]
nmeas = len(stat_dates)
stat_decdat = date_to_dec(stat_dates)
box_dist_mon = box_distribution(mon_stats, stat_dic, nos=True, datatype='NOAA')

stat_mcf = stat_data.variables['mcf_grd'][:]
stat_ch4 = stat_data.variables['ch4ctl_grd'][:]
stat_error_mcf = .4e-12
stat_error_ch4 = 3e-9

# Sampling from each station on the same timestep:
mcf_mon, mcf_e_mon = global_mean(stat_mcf, mon_nos, stat_error_mcf, box_dist_mon, w_noaa)
ch4_mon, ch4_e_mon = global_mean(stat_ch4, mon_nos, stat_error_ch4, box_dist_mon, w_noaa)
# Randomized sampling
mcf_mon_rnd, mcf_e_mon_rnd = global_mean(stat_mcf, mon_nos, stat_error_mcf, box_dist_mon, w_noaa, random=True)
ch4_mon_rnd, ch4_e_mon_rnd = global_mean(stat_ch4, mon_nos, stat_error_ch4, box_dist_mon, w_noaa, random=True)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                       (A)GAGE station calculations
gage_stats = ['CGO', 'SMO', 'RPB', 'THD', 'MHD'] # Stations used in the Montzka (2011) study
gage_nos = [stat_ids.tolist().index(stat) for stat in gage_stats]
gage_lon = [stat_dic[stat][0] for stat in gage_stats]
gage_lat = [stat_dic[stat][1] for stat in gage_stats]
stat_dates = stat_data.variables['idate'][:]
nmeas = len(stat_dates)
stat_decdat = date_to_dec(stat_dates)
box_dist_gage = box_distribution(gage_stats, stat_dic, nos=True, datatype='GAGE')

mcf_gage, mcf_e_gage = global_mean(stat_mcf, gage_nos, stat_error_mcf, box_dist_gage, w_gage)
ch4_gage, ch4_e_gage = global_mean(stat_ch4, gage_nos, stat_error_ch4, box_dist_gage, w_gage)
mcf_gage_rnd, mcf_e_gage_rnd = global_mean(stat_mcf, gage_nos, stat_error_mcf, box_dist_gage, w_gage, random=True)
ch4_gage_rnd, ch4_e_gage_rnd = global_mean(stat_ch4, gage_nos, stat_error_ch4, box_dist_gage, w_gage, random=True)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                       Gridded data calculations
grid_data = Dataset(file1, 'r')
mcf_head = grid_data.variables['mcf']
mcf_grid = mcf_head[:]
ch4_head = grid_data.variables['ch4ctl']
ch4_grid = ch4_head[:]
(nz, ny, nx) = mcf_grid.shape

lat = np.linspace(-90, 90, num=ny) # latitudinal grid distribution
lon = np.linspace(0, 360, num=nx) # longitudinal grid distribution
vert = np.linspace(0, 25, num=nz) # vertical layers

# Global means from the gridded data
ch4_grd_10k = round(np.mean(ch4_grid[:10])*1e9, 2)
mcf_grd_10k = round(np.mean(mcf_grid[:10])*1e12, 2)
ch4_grd_15k = round(np.mean(ch4_grid[:15])*1e9, 2)
mcf_grd_15k = round(np.mean(mcf_grid[:15])*1e12, 2)
ch4_grd_20k = round(np.mean(ch4_grid[:20])*1e9, 2)
mcf_grd_20k = round(np.mean(mcf_grid[:20])*1e12, 2)

if make_plots:
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #                               PLOTS

    # STATION DATA
    fig = plt.figure(figsize=(10,20))
    ax1 = fig.add_subplot(211)
    ax1.errorbar(stat_decdat, mcf_mon*1e12, yerr=mcf_e_mon*1e12, label='Mean from NOAA stations')
    ax1.errorbar(stat_decdat, mcf_gage*1e12, yerr=mcf_e_gage*1e12, label='Mean from (A)GAGE stations')
    ax1.plot(stat_decdat, [mcf_grd_10k]*nmeas, 'r--', label='Mean from grid (lower 10km)')
    ax1.plot(stat_decdat, [mcf_grd_15k]*nmeas, 'r-', label='Mean from grid (lower 15km)')
    #ax1.plot(stat_decdat, [mcf_grd_20k]*nmeas, 'r-.', label='Mean from grid (lower 20km)')
    ax1.set_title('Time evolution of the global mean MCF concentration, as calculated\n\
    from station data during one month. For each station we take the measurements \n\
    at the same timestep (x-axis) Both mean concentrations\n\
    from the NOAA as well as from the (A)GAGE stations are given',y=1.05)
    ax1.set_ylabel('MCF (ppt)')
    ax2 = fig.add_subplot(212)
    ax2.errorbar(stat_decdat, ch4_mon*1e9, yerr=ch4_e_mon*1e9, label='Mean from NOAA stations')
    ax2.errorbar(stat_decdat, ch4_gage*1e9, yerr=ch4_e_gage*1e9, label='Mean from (A)GAGE stations')
    ax2.plot(stat_decdat, [ch4_grd_10k]*nmeas, 'r--', label='Mean from grid (lower 10km)')
    ax2.plot(stat_decdat, [ch4_grd_15k]*nmeas, 'r-', label='Mean from grid (lower 15km)')
    ax2.set_title(r'Time evolution of the global mean CH$_4$'+' concentration,\n\
    as calculated from station data during one month',y=1.05)
    ax2.set_xlabel('Time')
    ax2.set_ylabel(r'CH$_4$ (ppb)')
    lgd = ax1.legend(loc='best')
    plt.tight_layout()
    plt.savefig('Station_CH4_MCF_means')
    
    fig = plt.figure(figsize=(10,20))
    ax1 = fig.add_subplot(211)
    ax1.errorbar(stat_decdat, mcf_mon_rnd*1e12, yerr=mcf_e_mon_rnd*1e12, label='Mean from NOAA stations')
    ax1.errorbar(stat_decdat, mcf_gage_rnd*1e12, yerr=mcf_e_gage_rnd*1e12, label='Mean from (A)GAGE stations')
    ax1.plot(stat_decdat, [mcf_grd_10k]*nmeas, 'r--', label='Mean from grid (lower 10km)')
    ax1.plot(stat_decdat, [mcf_grd_15k]*nmeas, 'r-', label='Mean from grid (lower 15km)')
    #ax1.plot(stat_decdat, [mcf_grd_20k]*nmeas, 'r-.', label='Mean from grid (lower 20km)')
    ax1.set_title('Time evolution of the global mean MCF concentration, as calculated\n\
    from station data during one month. For each station we take the measurements\n\
    at a different random timesteps. Both mean concentrations\n\
    from the NOAA as well as from the (A)GAGE stations are given',y=1.05)
    ax1.set_ylabel('MCF (ppt)')
    ax2 = fig.add_subplot(212)
    ax2.errorbar(stat_decdat, ch4_mon_rnd*1e9, yerr=ch4_e_mon_rnd*1e9, label='Mean from NOAA stations')
    ax2.errorbar(stat_decdat, ch4_gage_rnd*1e9, yerr=ch4_e_gage_rnd*1e9, label='Mean from (A)GAGE stations')
    ax2.plot(stat_decdat, [ch4_grd_10k]*nmeas, 'r--', label='Mean from grid (lower 10km)')
    ax2.plot(stat_decdat, [ch4_grd_15k]*nmeas, 'r-', label='Mean from grid (lower 15km)')
    ax2.set_title(r'Time evolution of the global mean CH$_4$'+' concentration,\n\
    as calculated from station data during one month',y=1.05)
    ax2.set_xlabel('Time')
    ax2.set_ylabel(r'CH$_4$ (ppb)')
    lgd = ax1.legend(loc='best')
    plt.tight_layout()
    plt.savefig('Station_CH4_MCF_means_rnd')
    
    mcf_vert = np.mean(mcf_grid, axis=(1,2))
    mcf_lat = np.mean(mcf_grid, axis=(0,2))
    mcf_lat_l10 = np.mean(mcf_grid[:10], axis=(0,2))
    mcf_lon = np.mean(mcf_grid, axis=(0,1))
    mcf_zy = np.mean(mcf_grid, axis=2)
    mcf_xy = np.mean(mcf_grid, axis=0)
    mcf_xz = np.mean(mcf_grid, axis=1)
    mcf_xytrop = np.mean(mcf_grid[:10], axis=0)
    ch4_vert = np.mean(ch4_grid, axis=(1,2))
    ch4_lat = np.mean(ch4_grid, axis=(0,2))
    ch4_lat_l10 = np.mean(ch4_grid[:10], axis=(0,2))
    ch4_lon = np.mean(ch4_grid, axis=(0,1))
    ch4_zy = np.mean(ch4_grid, axis=2)
    ch4_xy = np.mean(ch4_grid, axis=0)
    ch4_xz = np.mean(ch4_grid, axis=1)
    ch4_xytrop = np.mean(ch4_grid[:10,:,:], axis=0)
    
    fig = plt.figure(figsize=(20,20))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(223)
    ax1.set_title('Globally averaged vertical profile of MCF')
    ax1.set_xlabel('MCF (ppt)')
    ax1.set_ylabel('Vertical height (km)')
    ax1.plot(mcf_vert*1e12, vert, 'o-')
    ax2.set_title('Zonally averaged distribution of MCF, \n\
    (lower 10 km)')
    ax2.set_xlabel('Latitude')
    ax2.set_ylabel('MCF (ppt)')
    ax2.plot(lat, mcf_lat_l10*1e12, 'o-') 
    ax3 = fig.add_subplot(222)
    ax4 = fig.add_subplot(224)
    ax3.set_title('Globally averaged vertical profile of '+r'CH$_4$')
    ax3.set_xlabel(r'CH$_4$ (ppb)')
    ax3.set_ylabel('Vertical height (km)')
    ax3.plot(ch4_vert*1e9, vert, 'o-')
    ax4.set_title('Zonally averaged distribution of '+r'CH$_4$'+',\n\
    (lower 10 km)')
    ax4.set_xlabel('Latitude')
    ax4.set_ylabel(r'CH$_4$ (ppb)')
    ax4.plot(lat, ch4_lat_l10*1e9, 'o-')
    fig.savefig('Global_MCF_CH4_distribution_1D')
    plt.close()
    
    fig = plt.figure(figsize=(10,30))
    ax1 = fig.add_subplot(311)
    ax1.set_title('Zonally averaged MCF concentration')
    ax1.set_xlabel('Latitude (deg)')
    ax1.set_ylabel('Height (km)')
    cp_yz = ax1.contourf(lat, vert, mcf_zy*1e12, 400)
    cb_yz = plt.colorbar(cp_yz)
    cb_yz.set_label('MCF (ppt)')
    ax2 = fig.add_subplot(312)
    ax2.set_title('Latidunally averaged MCF concentration')
    ax2.set_xlabel('Longitude (deg)')
    ax2.set_ylabel('Height (km)')
    cp_xz = ax2.contourf(lon, vert, mcf_xz*1e12, 400)
    cb_xz = plt.colorbar(cp_xz)
    cb_xz.set_label('MCF (ppt)')
    ax3 = fig.add_subplot(313)
    ax3.set_title('Vertically averaged MCF concentration (lower 10 km)')
    ax3.set_xlabel('Longitude (deg)')
    ax3.set_ylabel('Latitude (deg)')
    cp_xy = ax3.contourf(lon, lat, mcf_xytrop*1e12, 400)
    cb_xy = plt.colorbar(cp_xy)
    cb_xy.set_label('MCF (ppt)')
    fig.tight_layout()
    fig.savefig('Global_MCF_distribution_2D')
    plt.close()
    
    fig = plt.figure(figsize=(10,30))
    ax1 = fig.add_subplot(311)
    ax1.set_title('Zonally averaged CH4 concentration')
    ax1.set_xlabel('Latitude (deg)')
    ax1.set_ylabel('Height (km)')
    cp_yz = ax1.contourf(lat, vert, ch4_zy*1e9, 400)
    cb_yz = plt.colorbar(cp_yz)
    cb_yz.set_label('CH4 (ppb)')
    ax2 = fig.add_subplot(312)
    ax2.set_title('Latidunally averaged CH4 concentration')
    ax2.set_xlabel('Longitude (deg)')
    ax2.set_ylabel('Height (km)')
    cp_xz = ax2.contourf(lon, vert, ch4_xz*1e9, 400)
    cb_xz = plt.colorbar(cp_xz)
    cb_xz.set_label('CH4 (ppb)')
    ax3 = fig.add_subplot(313)
    ax3.set_title('Vertically averaged CH4 concentration (lower 10 km)')
    ax3.set_xlabel('Longitude (deg)')
    ax3.set_ylabel('Latitude (deg)')
    cp_xy = ax3.contourf(lon, lat, ch4_xytrop*1e9, 400)
    cb_xy = plt.colorbar(cp_xy)
    cb_xy.set_label('CH4 (ppb)')
    fig.tight_layout()
    fig.savefig('Global_CH4_distribution_2D')    
    
    fig = plt.figure(figsize=(10,20))
    ax1 = fig.add_subplot(211)
    ax1.set_title(r'Global CH$_4$ distribution, February 2000\n\
        The (A)GAGE stations are also indicated.')
    m_ch4 = Basemap(projection='cyl',lon_0=0,resolution='c')
    m_ch4.drawparallels(np.arange(-90.,91.,30.), labels=[True, False, False, False])
    m_ch4.drawmeridians(np.arange(-180.,181.,60.), labels=[False, False, False, True])
    m_ch4.drawcoastlines()
    lons, lats = m_ch4.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
    x, y = m_ch4(lons, lats) # compute map proj coordinates.
    cs = m_ch4.contourf(x, y, ch4_xytrop*1e9, 20)
    cb = m_ch4.colorbar(cs)
    cb.set_label('CH4 (ppb)')
    xstat, ystat = m_ch4(gage_lon, gage_lat)
    for i,stat in enumerate(gage_stats):
        xi,yi = xstat[i],ystat[i]
        m_ch4.plot(xi, yi, 'ko')
        plt.text(xi-8,yi+2.,stat)
    ax2 = fig.add_subplot(212)
    ax2.set_title('Global MCF distribution, February 2000\n\
        The NOAA stations are also indicated.')
    m_mcf = Basemap(projection='cyl',lon_0=0,resolution='c')
    m_mcf.drawparallels(np.arange(-90.,91.,30.), labels=[True, False, False, False])
    m_mcf.drawmeridians(np.arange(-180.,181.,60.), labels=[False, False, False, True])
    m_mcf.drawcoastlines()
    x, y = m_mcf(lons, lats) # compute map proj coordinates.
    cs = m_mcf.contourf(x, y, mcf_xytrop*1e12, 20)
    cb = m_mcf.colorbar(cs)
    cb.set_label('MCF (ppt)')
    xstat, ystat = m_mcf(mon_lon, mon_lat)
    for i,stat in enumerate(mon_stats):
        xi,yi = xstat[i],ystat[i]
        m_mcf.plot(xi, yi, 'ko')
        if stat == 'KUM':
            plt.text(xi-8,yi-8.,stat)
        else:
            plt.text(xi-8,yi+2.,stat)
    plt.savefig('Global_CH4_MCF_distribution_map')





























