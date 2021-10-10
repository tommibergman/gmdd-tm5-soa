from scipy.special import erf
from settings import *
import matplotlib.pyplot as plt 
import datetime
import glob
import netCDF4 as nc
import re
from mpl_toolkits.basemap import Basemap
from general_toolbox import get_gridboxarea,lonlat,write_netcdf_file,NMB
import os
from scipy.stats import pearsonr
import numpy as np
import logging
import matplotlib as mpl
from mass_budget import mapit_boundary
OH_sim={}
for i in EXPs:
	OH_sim[i]=nc.Dataset(rawoutput+i+'/general_TM5_'+i+'_2010.mm.nc')['GAS_OH'][:]


tdata=(OH_sim[EXPs[0]].mean(axis=0)-OH_sim[EXPs[1]].mean(axis=0))/OH_sim[EXPs[1]].mean(axis=0)
print tdata.shape
gbarea=get_gridboxarea('TM53x2')
for i in range(0,34):
	print i,(tdata[i,:,:]*gbarea[:,:]).sum()/gbarea[:,:].sum()
	data=tdata[i,:,:].squeeze()
	#print data[20:40,20:40],OH_sim[EXPs[0]].mean(axis=0)[0,20:40,20:40]
	diverging=True
	clevs=[0.9,0.95,0.97,0.975,0.98,0.985,0.99,0.995,1.00,1.005,1.01,1.015,1.02,1.025,1.03,1.05,1.1]
	clevs=[-0.1,-0.05,-0.03,-0.025,-0.02,-0.015,-0.01,-0.005,0.00,0.005,0.01,0.015,0.02,0.025,0.03,0.05,0.1]
	f,ax=plt.subplots(1)
	mapit_boundary(data,clevs,ax,diverging=diverging)
plt.show()
