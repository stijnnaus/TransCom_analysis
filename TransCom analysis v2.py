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
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import curve_fit

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

def pertData(data,ch4e,mcfe):
    '''
    Inpurt is data of the format: 
    data[istation][ch4(0),mcf(1)][iyear][imonth][imeas]
    '''
    data_prt = data[:]
    ny = len(data_prt)
    freq = len(data_prt[0][0][0][0])
    nst = len(data_prt[0][0])
    for iy in range(ny):
        for ist in range(nst):
            ch4es = np.random.normal(0.0,ch4e,size=freq)
            mcfes = np.random.normal(0.0,mcfe,size=freq)
            data_prt[iy][0][ist][1] += ch4es
            data_prt[iy][1][ist][1] += mcfes
    return data_prt

def genMeas_NOAA(data,freq,ch4_ids,mcf_ids):
    '''
    Generates station data, such that it resembles NOAA data.
    That means 9/46 stations for MCF/CH4;
    With a pre-defined measurement frequency per station;
    Initially no data is thrown away
    
    Data format of output is for each year: [yr, ch4, mcf]
    Where ch4 (and mcf) have the format: 
    [[meas_nos,ch4]_station1, [meas_nos,ch4s]_station2, ...]
    '''
    ch4,mcf = data[0],data[1]
    ch4sel,mcfsel = ch4[ch4_ids],mcf[mcf_ids]
    ch4gen = genMCF_NOAA(mcfsel,freq)
    mcfgen = genCH4_NOAA(ch4sel,freq)
    return array([ch4gen,mcfgen])
    
    
    freq = float(freq)
    ifreq = int(freq)
    data_gen = []
    for iyr,datai in enumerate(data):
        ch4 = datai[0] # Methane data from iyr
        mcf = datai[1] # MCF data from iyr
        ch4_st,mcf_st = [],[]
        for sid in station_ids: # loop over stations
            ch4_s = ch4[sid] # sid measurement series of ch4
            mcf_s = mcf[sid] # sid measurement series of mcf
            nd = len(ch4_s)
            ti = nd/freq # average time interval between measurements
            meas_no = np.zeros(ifreq,dtype=int) # random measurements
            for i in range(ifreq):
                rmeas = np.random.randint(i*ti,(i+1)*ti) # a random measurement number in the measurement interval
                meas_no[i] = rmeas
            ch4_sl = array([meas_no, ch4_s[meas_no]])
            mcf_sl = array([meas_no, mcf_s[meas_no]])
            ch4_st.append(ch4_sl)
            mcf_st.append(mcf_sl)
        data_gen.append(array([ch4_st,mcf_st]))
    return data_gen

def genMeas_AGAGE(data,freq,stat_ids):
    '''
    Generates station data, such that it resembles AGAGE data.
    That means 5/5 stations for MCF/CH4;
    Measurement frequency is 1.5/hr, so I need the full temporal data;
    Pollution data is filtered statistically    
    '''
    ch4,mcf = data[0],data[1]
    ch4sel,mcfsel = ch4[stat_ids], mcf[stat_ids]
    mcfgen = genMCF_AGAGE(mcfsel,freq)
    ch4gen = genCH4_AGAGE(ch4sel,freq)
    return array([ch4gen,mcfgen])
            
def filterStationData_ar(data,nsd=2.5,ws=100,curvefit=False,flr=0):
    '''
    This routine is designed to filter out the polluted data from the full 
    station data.
    The filtering method is adopted from AGAGE. It selects a window with a
    width of 4 months. Then it iteratively removes all values more than nsd
    standard deviations above the median of the window, until no more data
    is removed.
    
    data: Either MCF or CH4 data.
    nsd : Number of STD above which a value is considered polluted.
    ws  : Window spacing, i.e. how many data points the window jumps after 
        finishing each filter.
    flr : The floor of the filtering treshold: the minimum of flr and 2.5 SD is
        the threshold (prevents incorrect filtering when pollution is low)
            
    Returns a boolean mask.
    '''
    nst  = len(data)
    nyr  = len(data[0]) # number of years
    nms  = hpy*nyr # total number of meas
    nw   = 4*hpy/12 # width of the filtering window in hours
    mask_tot = np.ones((nst,nms),dtype='bool')
    for ist,data_st in enumerate(data):
        data_fl = data_st.flatten() # flattened data for all years
        mask_st = mask_tot[ist]
        for strt in range(0,nms,ws):
            end = strt+nw # end of the window
            mask_w = mask_st[strt:end]
            data_w = data_fl[strt:end]
            nwi = len(data_w)
            if curvefit:
                [a,b,c],_ = curve_fit(fitfunc, np.arange(nwi), data_w)
                fit = fitfunc(np.arange(nwi), a, b, c)
                data_w = data_w - fit
            ch = True
            while ch: # continue iterating as long as there are changes
                med  = np.median(data_w[mask_w])                  # median
                crit = (nsd*np.std(data_w[mask_w])).clip(min=flr) # standard deviation
                cond = data_w[mask_w] < med+crit                  # check filtering condition
                if np.all(mask_w[mask_w]==cond): # no changes since the last iteration
                    ch = False
                mask_w[mask_w] = cond # implement changes
            mask_st[strt:end] = mask_w
        print 'For station',sid_all[ist],',',17*hpy-np.sum(mask_st),'points are filtered'
    return mask_tot.reshape((nst,nyr,hpy))
    
def fitfunc(x,a,b,c):
    '''Second order polynomial fit function'''
    return a*x**2 + b*x + c

def df_to_ar(df, nyr=17):
    '''
    Converts a pandas dataframe to a numpy array.
    For this the array is made even, ie leap years are cut of.
    '''
    stations = df.columns
    nst = len(stations)
    ar = np.zeros((nst,nyr,hpy))
    for ist,st in enumerate(stations):
        df_st = df[st]
        for iyr,yr in enumerate(range(sty,edy+1)):
            df_y = df_st[str(yr)]
            ar[ist][iyr] = array(df_y)[:hpy]
    return ar
    
def filterStationData_df(df,nsd=2.5,ww=2900,stp=0,flr=0):
    '''
    This routine is designed to filter out the polluted data from the full 
    station data.
    The filtering method is adopted from AGAGE. It selects a window with a
    width of 4 months. Then it iteratively removes all values more than nsd
    standard deviations above the median of the window, until no more data
    is removed.
    
    df  : Either MCF or CH4 pandas dataframe.
    nsd : Number of STD above which a value is considered polluted.
    ww  : Width of window [hours]
    ws  : Spacing between subsequent windows
    stp : Stop filtering when iteration filters less than stp points
    flr : The floor of the filtering treshold: the minimum of flr and 2.5 SD is
        the threshold (prevents incorrect filtering when pollution is low)
            
    Returns a boolean mask.
    '''
    nst, nms = df.shape
    dr = df.index
    dfc = df.copy()
    for stat in df.columns:
        print 'Filtering',stat
        df_st  = dfc[stat]    # Select a station
        # removing trend and seasonal cycle:
        desea  = sm.tsa.seasonal_decompose(df_st.values, freq=hpy, model='additive')
        df_res = pd.Series(desea.resid, index=dr) # residuals
        df_res  = (df_res.fillna(method='bfill')).fillna(method='ffill')
        n_nan = stp+1; n_nan0 = 0; it = 0 # initialization
        while (n_nan-n_nan0)>stp:    # Stop if little data is removed
            it+=1
            n_nan0  = n_nan
            rol     = df_res.rolling(ww,min_periods=int(ww*.8),center=True) # rolling window over data
            crit    = (nsd*rol.std()).clip(lower=flr) # Minimum threshold is floor
            limit   = (rol.median() + crit) # pollution limit per window
            rol_lim = limit.rolling(ww,min_periods=0,center=True) # rolling window over limits
            min_lim = rol_lim.min() # for each point, select the most stringent condition
            df_res[df_res>min_lim] = np.nan
            n_nan   = df_res.isnull().sum()
        df_st[df_res.isnull()] = np.nan # impose filter on original station data
        print n_nan, 'measurements filtered (or',100.*n_nan/len(df_st),'% of all measurements)'
    return dfc
    
def deseasonalize(series):
    '''
    Deseasonalizes a pandas dataseries
    !!! NOT IMPLEMENTED !!!
    '''
    pass

def glob_mean_df(df,sids,w,network='NOAA',ip=True):
    '''
    Calculates the global mean from station data
    ip: If true, interpolates the nan values
    '''
    plt.figure()
    df_sel = df[sids] # station selection
    df_y = pd.DataFrame(index=df_sel.columns).interpolate(method='time',axis=0)
    for y in range(sty,edy+1):
        df_y[str(y)] = df_sel[str(y)].mean() # Yearly means per station
    plt.plot(df_y.T,'--')
    box_no = {sid:stat_pars.loc[sid]['Box_'+network] for sid in sids} # box assignment
    df_y = df_y.groupby(box_no).mean()       # Yearly means per box
    plt.plot(df_y.T)
    df_yw = df_y.multiply(array(w), axis=0)  # Multiply by relative weight per box
    df_glob = df_yw.sum(axis=0)/np.sum(w)    # Global yearly means
    plt.plot(df_glob,'ko')
    return df_glob
ch4_glob_df = glob_mean_df(ch4_stf_df, sid_agage, w_gage, network='AGAGE')
ch4_glob_df = glob_mean_df(ch4_stf_df, sid_noaa_mcf,  w_noaa, network='NOAA')
    
def glob_mean_ar(ar,sids,network='NOAA'):
    pass
    
def moving_average(a, n=200):
    '''
    Moving average of a numpy array a, with a window width of n.
    '''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] 

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                       FILTERING STATION DATA
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
start = time.time()
ch4_stf_df = filterStationData_df(ch4_st,ww=800)
end = time.time()
print 'Dataframe filter takes',end-start
#mcf_stf = filterStationData_df(mcf_st,ww=300)
dens_df = [ch4_stf_df[sid_all[i]].isnull().rolling(2000,center=True).mean() for i in range(12)]

densw = 2000 # width of the window that computes density of polluted data
start = time.time()
ch4_st_ar = df_to_ar(ch4_st)
mcf_st_ar = df_to_ar(mcf_st)
ch4_mask_simple = filterStationData_ar(ch4_st_ar)
dens_simple = array([moving_average(ch4_mask_simple[i].flatten(),densw) for i in range(12)]) # pollution density
ch4_stf_ar_simple = ch4_st_ar.copy()
ch4_stf_ar_simple[ch4_mask_simple==False] = np.nan
print 'Simple array filter takes', time.time()-start
start = time.time()

ch4_mask_compl = filterStationData_ar(ch4_st_ar,curvefit=True,flr=0)
nan1 = ch4_mask_compl.sum(axis=1).sum(axis=1)
ch4_mask_compl = filterStationData_ar(ch4_st_ar,curvefit=True,flr=.5)
nan2 = ch4_mask_compl.sum(axis=1).sum(axis=1)
ch4_mask_compl = filterStationData_ar(ch4_st_ar,curvefit=True,flr=2.)
nan3 = ch4_mask_compl.sum(axis=1).sum(axis=1)
ch4_mask_compl = filterStationData_ar(ch4_st_ar,curvefit=True,flr=5.)
nan4 = ch4_mask_compl.sum(axis=1).sum(axis=1)
df_nans = pd.DataFrame(np.transpose(array([nan1,nan2,nan3,nan4])), index=sid_all, columns=[0,0.5,2.,5.]) # the effect of flooring

mcf_mask_compl = filterStationData_ar(mcf_st_ar,curvefit=True)
dens_compl = array([moving_average(ch4_mask_compl[i].flatten(),densw) for i in range(12)])
dens_mcf = array([moving_average(ch4_mask_compl[i].flatten(),densw) for i in range(12)])
ch4_stf_ar_compl = ch4_st_ar.copy()
ch4_stf_ar_compl[ch4_mask_compl==False] = np.nan
mcf_stf_ar_compl = mcf_st_ar.copy()
mcf_stf_ar_compl[mcf_mask_compl==False] = np.nan
print '2nd order removal array filter takes', time.time()-start

# plot of complete filtered v unfiltered data per station
for i in range(12):
    st = sid_all[i]
    xyr = np.linspace(1990,2007,num=17*hpy)
    fig,ax = plt.subplots(3,2)
    ax[0,0].set_title('Simple filtering')
    ax[1,0].set_title('Decomposed filtering')
    ax[2,0].set_title('Second order filtering')
    ax[0,1].set_ylabel('Pollution density(%)')
    ax[1,1].set_ylabel('Pollution density(%)')
    ax[2,1].set_ylabel('Pollution density(%)')
    ax[0,0].plot(xyr, ch4_st_ar[i].flatten(), color='maroon')
    ax[0,0].plot(xyr, ch4_stf_ar_simple[i].flatten(), color='darkgreen')
    ax[0,1].plot(xyr[densw/2:-densw/2+1], 100-100*dens_simple[i])
    ax[0,1].set_ylim([0,100])
    ax[0,0].set_xlim(1990,2007)
    ax[1,0].plot(ch4_st[st], color='maroon')
    ax[1,0].plot(ch4_stf_df[st], color='darkgreen')
    ax[1,1].plot(100*dens_df[i])
    ax[1,1].set_ylim([0,100])
    ax[2,0].plot(xyr, ch4_st_ar[i].flatten(), color='maroon')
    ax[2,0].plot(xyr, ch4_stf_ar_compl[i].flatten(), color='darkgreen')
    ax[2,1].plot(xyr[densw/2:-densw/2+1], 100-100*dens_compl[i])
    ax[2,1].set_ylim([0,100])
    plt.tight_layout()
    plt.savefig('Figures/'+st+' CH4 filtering.png')
    plt.close()
    
for i in range(12):
    st = sid_all[i]
    xyr = np.linspace(1990,2007,num=17*hpy)
    fig,ax = plt.subplots(1,2)
    ax[0].set_title('Second order filtering')
    ax[1].set_ylabel('Pollution density(%)')
    ax[1].set_ylim([0,100])
    ax[0].plot(xyr, mcf_st_ar[i].flatten(), color='maroon')
    ax[0].plot(xyr, mcf_stf_ar_compl[i].flatten(), color='darkgreen')
    ax[1].plot(xyr[densw/2:-densw/2+1], 100-100*dens_mcf[i])
    ax[1].set_ylim([0,100])
    plt.tight_layout()
    plt.savefig('Figures/'+st+' MCF filtering.png')
    plt.close()
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                   COMPUTING GLOBAL MEANS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

colors = ['steelblue','maroon','blue','red',      'cyan',   'magenta',\
            'black',  'green', 'lime','peachpuff','fuchsia','silver']    

yrs_gr,ch4_gr,mcf_gr = grid_y[1:-1,0],grid_y[1:-1,1],grid_y[1:-1,2]
ch4_glob_df = glob_mean_df(ch4_stf_df, sid_agage, w_gage, network='AGAGE')
ch4_glob_uf = glob_mean_df(ch4_st    , sid_agage, w_gage, network='AGAGE')

# plot of the yearly global mean from grid, filtered and unfiltered (3 methods) data
fig, ax = plt.subplots(1,1)
ax.set_title('Global mean CH4 from different methods')
ax.plot(grid_y[1:-1,0],grid_y[1:-1,1],label='From grid (True)')
ax.plot(grid_y[1:-1,0],ch4_glob_df.values,label='From decomposition')
ax.plot(grid_y[1:-1,0],ch4_glob_uf.values,label='Unfiltered')
ax.legend(loc='best')

# Plot of station yearly means versus gridded yearly means
yrs_df = np.linspace(1990,2007,num=len(ch4_st['CGO'].values))
rol_st = ch4_st.rolling(hpy,min_periods=100,center=True).mean()
rol_stf = ch4_stf_df.interpolate(method='time',axis=0).rolling(hpy,min_periods=100,center=True).mean()
fig,ax = plt.subplots(1,1)
for i,st in enumerate(rol_st.columns):
    ax.plot(yrs_df,rol_st[st].values,'--',color=colors[i],label=st)
    ax.plot(yrs_df,rol_stf[st].values,'-',color=colors[i],label=st)
ax.plot(grid_y[1:-1,0],grid_y[1:-1,1],'ko-',label='gridded mean')
ax.plot(grid_y[1:-1,0],ch4_glob_df,'k-',label='station mean filt')
ax.plot(grid_y[1:-1,0],ch4_glob_df,'k--',label='station mean unfilt')
ax.legend(loc='best')
ax.set_ylim([1650,1950])
#plt.plot(ch4_glob_df,'ko',linewidth=2)

# Plot of the errors in the global yearly means wrt gridded means
fig,ax = plt.subplots(1,2)
# from unfiltered data
dif_uf = ch4_gr - ch4_glob_uf
mer_uf = np.abs(dif_uf - np.mean(dif_uf))
ax[0].set_title('From unfiltered data')
ax[0].plot(grid_y[1:-1,0],mer_uf,'o',color='maroon',label='From decomp')
ax[0].plot(np.linspace(1990,2007),[np.mean(mer_uf)]*50,'-',color='maroon',label='From decomp')
ax[0].fill_between(np.linspace(1990,2007), [np.mean(mer_uf)+np.std(mer_uf)]*50, 
                 [np.mean(mer_uf)-np.std(mer_uf)]*50, color='maroon',alpha=.5) # STD
ax[0].fill_between(np.linspace(1990,2007), 
                    [np.mean(mer_uf)+np.std(mer_uf)/np.sqrt(17)]*50, 
                    [np.mean(mer_uf)-np.std(mer_uf)/np.sqrt(17)]*50, 
                     color='maroon',alpha=.3) # SDOM
                     
# From seasonal decomposition
dif_dec = (grid_y[1:-1,1]-ch4_glob_df.values) # difference between the two
mer_dec = np.abs(dif_dec-np.mean(dif_dec))    # absolute error in growth rate
ax[1].set_title('From seasonal decomposition')
ax[1].plot(grid_y[1:-1,0],mer_dec,'o',color='maroon',label='From decomp')
ax[1].plot(np.linspace(1990,2007),[np.mean(mer_dec)]*50,'-',color='maroon',label='From decomp')
ax[1].fill_between(np.linspace(1990,2007), [np.mean(mer_dec)+np.std(mer_dec)]*50, 
                 [np.mean(mer_dec)-np.std(mer_dec)]*50, color='maroon',alpha=.5) # STD
ax[1].fill_between(np.linspace(1990,2007), 
                    [np.mean(mer_dec)+np.std(mer_dec)/np.sqrt(17)]*50, 
                    [np.mean(mer_dec)-np.std(mer_dec)/np.sqrt(17)]*50, 
                     color='maroon',alpha=.3) # SDOM

#ax[0].plot(label='From grid (True)')
#ax[0].plot(label='From grid (True)')




'''
colors = ['steelblue','maroon','blue','red',      'cyan',   'magenta',\
            'black',  'green', 'lime','peachpuff','fuchsia','silver']

# ++++++++++++++++++++++++++++++++++
#               AGAGE
# FILTERED: 
ch4_st_agage     = ch4_st_mn.loc[sid_agage]
ch4_st_gm_agage  = (ch4_st_agage.loc['CGO']+(ch4_st_agage.loc['MHD']+ch4_st_agage.loc['THD'])/2.\
                    +ch4_st_agage.loc['RPB']+ch4_st_agage.loc['SMO'])/4.
# UNFILTERED:
sno_gage         = array([np.where(sid_all==sid_g)[0][0] for sid_g in sid_agage]) # agage indices in sid_all
ch4_agage_uf     = ch4_st[sno_gage] # unfiltered agage data
ch4_agage_uf_mn  = np.mean(ch4_agage_uf, axis=2) # unfiltered agage yearly means per station
ch4_agage_uf_gmn = (ch4_agage_uf_mn[0] + (ch4_agage_uf_mn[1] + ch4_agage_uf_mn[4])/2.\
                    + ch4_agage_uf_mn[2] + ch4_agage_uf_mn[3])/4. # unfiltered agage global yearly means
# COMPARISON TO GRIDDED
stav_ch4 = ch4_st_gm_agage.values
grav_ch4 = grid_y[1:-1,1]
dif_fl = grav_ch4 - stav_ch4
dif_uf = grav_ch4 - ch4_agage_uf_gmn

# +++++++++++++++++++++++++++++++++++
#               NOAA


yrs = np.arange(1990,2007)+.5
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_title('Global means CH4')
ax2.set_title('Difference between station mean and gridded mean')
ax1.set_ylabel('CH4 (ppb)')
ax2.set_ylabel('CH4 error (ppb)')
ax1.plot(yrs,grav_ch4,         'go', label='True')
ax1.plot(yrs,stav_ch4,         'bo', label='AGAGE filt')
ax1.plot(yrs,ch4_agage_uf_gmn, 'ro', label='AGAGE unfilt')
ax2.plot(yrs,dif_fl,           'bo-',label='AGAGE filt')
ax2.plot(yrs,dif_uf,           'ro-',label='AGAGE unfilt')
ax2.fill_between(np.linspace(1990,2007), [np.mean(dif_fl)+np.std(dif_fl)]*50, 
                 [np.mean(dif_fl)-np.std(dif_fl)]*50, color='b',alpha=.3)
ax2.fill_between(np.linspace(1990,2007), [np.mean(dif_uf)+np.std(dif_uf)]*50, 
                 [np.mean(dif_uf)-np.std(dif_uf)]*50, color='r',alpha=.3)
ax2.fill_between(np.linspace(1990,2007), [np.mean(dif_fl)+np.std(dif_fl)/np.sqrt(17.)]*50, 
                 [np.mean(dif_fl)-np.std(dif_fl)/np.sqrt(17.)]*50, color='b',alpha=.5)
ax2.fill_between(np.linspace(1990,2007), [np.mean(dif_uf)+np.std(dif_uf)/np.sqrt(17.)]*50, 
                 [np.mean(dif_uf)-np.std(dif_uf)/np.sqrt(17.)]*50, color='r',alpha=.5)
#plt.plot(ch4_st_gm_noaa, label='NOAA')
ax2.plot(np.linspace(1990,2007),[0]*50,'k-',linewidth=2.)
ax1.legend(loc='best')
ax2.legend(loc='best')

load_grd_data = False
load_station_data = False
make_plots = True

cwd = os.getcwd()
data_dir = cwd + '\\TransCom Data\\'
os.chdir(data_dir)
fmeas = 30. # measurement frequency yr-1
ch4e = 2.
mcfe = 2.

if load_grd_data:
    nx,ny,nz = 60,45,25
    dirc = 'TransCom data\Grid data'
    print('Reading grid data .........')
    start = time.time()
    grid_data = read_grid_data(dirc)
    end = time.time()
    print('Reading the grid data took', end-start, 'seconds')
    grid_y = grid_yav(grid_data)

if load_station_data:
    dirc2 = 'TransCom data\Stat data'
    print('Reading station data ..........')
    start = time.time()
    stat_data = read_stat_data(dirc2) # raw station data
    yrs,stat_sel = select_time(stat_data,sty,edy) # data for selected years
    station_data = stat_reform(stat_sel) # final station data
    end = time.time()
    print('Reading the stat data took',end-start,'seconds')



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                       Montzka station calculations
mon_stats = ['SPO','CGO', 'SMO', 'MLO', 'KUM', 'NWR', 'LEF', 'BRW', 'ALT'] # Stations used in the Montzka (2011) study
mon_nos = [stat_ids.tolist().index(stat) for stat in mon_stats] # Index no of the selected stations in the full station list

colors = ['red','blue','green','maroon','steelblue','indigo','cyan','pink','black']


fig=plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_ylabel('CH4 (ppb)')
ax2.set_ylabel('MCF (ppt)')
for n,no in enumerate(mon_nos):
    ch4i = station_data[0][no].flatten()
    mcfi = station_data[0][no].flatten()
    ch4fl,mcffl = [],[]
    for i in range(len(ch4i)):
        for j in range(len(ch4i[i])):
            ch4fl.append(ch4i[i][j])
            mcffl.append(mcfi[i][j])
    yearfl = np.linspace(sty, edy+1, num=len(ch4fl))
    ax1.plot(yearfl,ch4fl,'o',color=colors[n], label=stat_ids[no])
    ax2.plot(yearfl,mcffl,'o',color=colors[n], label=stat_ids[no])
ax1.legend(loc='best')

plt.savefig('TransCom_Montzka_stations.png')



real_data = genMeas(stat_ref, fmeas, mon_nos) # select relevant station data
prt_data = pertData(real_data, 2., 2.) # perturb the data

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

'''












