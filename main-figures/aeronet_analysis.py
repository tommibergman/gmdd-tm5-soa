import numpy as np 
import matplotlib.pyplot as plt 
import netCDF4 as nc 
from mpl_toolkits.basemap import Basemap
#import matplotlib as mpl
import sys
import os
#sys.path.append("/Users/bergmant/Documents/Project/ifs+tm5-validation/scripts")
#from colocate_aeronet import do_colocate
import matplotlib as mpl
#import read_colocation_aeronet as ra
import datetime
import scipy.stats as stats
import scipy
import glob
import subprocess
#from bivariate_fit import bivariate_fit
from matplotlib.colors import LogNorm
import xarray as xr
import re
import cis
from matplotlib.patches import Polygon
from settings import *
from general_toolbox import str_months,RMSE,MFE,MFB,NMB,NME
import string
SMALL_SIZE=16
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels# def site_type():

# def RMSE(obs,model):
# 	N=0
# 	RMSE=0.0
# 	for o,m in zip(obs,model):	
# 		if not (np.isnan(o) or np.isnan(m)):
# 			#print m,o
# 			RMSE+=(m-o)**2
# 			N+=1
# 	return np.sqrt(RMSE/N)
# def MFE(obs,model):
# 	N=0
# 	MFE=0.0
# 	for o,m in zip(obs,model):
# 		if not (np.isnan(o) or np.isnan(m)):
# 			MFE+=np.abs(m-o)/((o+m)/2)
# 			N+=1
# 	return MFE/N
# def MFB(obs,model):
# 	N=0
# 	MFB=0.0
# 	#print obs,model
# 	for o,m in zip(obs,model):
# 		if not (np.isnan(o) or np.isnan(m)):
# 			MFB+=(m-o)/((o+m)/2)
# 			N+=1
# 	return MFB/N
# def NMB(obs,model):
# 	N=0
# 	nom=0.0
# 	denom=0.0
# 	#print obs,model
# 	for o,m in zip(obs,model):
# 		#print o,m,type(o),type(m)
# 		if not (np.isnan(o) or np.isnan(m)):
# 			nom+=(m-o)
# 			denom+=o
# 			N+=1
# 	NMB=nom/denom
# 	return NMB
# def NME(obs,model):
# 	N=0
# 	nom=0.0
# 	denom=0.0
# 	#print obs,model
# 	for o,m in zip(obs,model):
# 		if not (np.isnan(o) or np.isnan(m)):
# 			nom+=np.abs(m-o)
# 			denom+=o
# 			N+=1
# 	NME=nom/denom
# 	return NME

def concatenate_sites(data1,data2,data3):
	TM5=[]
	TM5_OLD=[]
	Aeronet=[]
	count1=0
	std=[]
	stderr=[]
	relstderr=[]

	for i in data1:

		if type(data1[i][4]) != int:

			std.append(np.std(data1[i][4]))
			stderr.append(np.std(data1[i][4])/np.sqrt(float(len(data1[i][4]))))
			relstderr.append((np.std(data1[i][4])/np.sqrt(float(len(data1[i][4]))))/np.mean(data1[i][4]))

			if all(data1[i][4]) is not np.ma.masked:
				for j in range(len(data1[i][4])):
					#if 'Nauru' in i:
					#	print len(data1[i][4])
					#	print j,data1[i][4][j],data2[i][4][j],data3[i][4][j]
					#print i,j, len(data1[i])
					if data1[i][4][j] is not np.ma.masked and data2[i][4][j] is not np.ma.masked and data3[i][4][j] is not np.ma.masked and data1[i][4][j] >= 0 and data2[i][4][j] >= 0 and data3[i][4][j]>=0 :			
						TM5.append(data1[i][4][j])
						TM5_OLD.append(data2[i][4][j])
						Aeronet.append(data3[i][4][j])
					else:
						if data3[i][4][j] is not np.ma.masked: 	
							print 'ERROR'
							exit()
							pass
						count1+=1
			else:
				print data1[i][4][j],data2[i][4][j],data3[i][4][j]
				
		else: #Yearly data, only 1 data point in data[i][4], 
			#print data1[i][4] ,data2[i][4] ,data3[i][4] 
			if data1[i][4] is not np.ma.masked:
				#for j in range(len(data1[i][4])):
				if data1[i][4] is not np.ma.masked and data2[i][4] is not np.ma.masked and data3[i][4] is not np.ma.masked and data1[i][4] >= 0 and data2[i][4] >= 0 and data3[i][4]>=0 :			
					TM5.append(data1[i][4])
					TM5_OLD.append(data2[i][4])
					Aeronet.append(data3[i][4])
				else:
					if data3[i][4] is not np.ma.masked: 	
						print 'aeronet data is not masked'
						exit()
						pass
					count1+=1

	return TM5,TM5_OLD,Aeronet
def concatenate_sites2(data):
	outdata=[]
	count1=0
	std=[]
	stderr=[]
	relstderr=[]

	count2=0
	for i in data:

		std.append(np.std(data[i][4]))
		stderr.append(np.std(data[i][4])/np.sqrt(float(len(data[i][4]))))
		relstderr.append((np.std(data[i][4])/np.sqrt(float(len(data[i][4]))))/np.mean(data[i][4]))

		count1=0
		for j in range(len(data[i][4])):
			#if data[i][4][j] is not np.ma.masked and data[i][4][j] > 0 :			
			if not np.ma.is_masked(data[i][4][j]):			
				outdata.append(data[i][4][j])
				count1+=1

			else:
				pass
				#count1+=1
		count2+=count1
	return outdata
# def sites2numpy(data):
# 	outdata=[]
# 	count1=0
# 	std=[]
# 	stderr=[]
# 	relstderr=[]
# 	sitesnumpy=np.zeros(np.shape(data))
# 	for i in range(len(data)):
		
# 		print data[i][:]
# 		sitesnumpy[i,:]=data[i][:]
# 	print sitesnumpy
# 	asdfa
# 	return outdata


def scatter_heat(obs,model,obsname=None,modelname=None):
	plt.figure()
	cm=plt.get_cmap('BuPu')
	if obsname==None:
		obsname='Aeronet'
	if modelname==None:
		modelname='TM5'
	#print np.shape(obs),np.shape(model)
	plt.hist2d(obs,model,bins=100,norm=LogNorm(),cmap=cm)
	XX=np.max([np.max(obs),np.max(model)])
	XX=1.0
	plt.xlim(0,XX)
	plt.ylim(0,XX)
	l2,=plt.plot([0,XX],[0,XX],'--k',lw=3)
	fit = np.polyfit(obs, model, 1)
	YY=np.poly1d(fit)
	#aa,bb,SS=bivariate_fit(Aeronetb,TM5b,0.01,0.003+0.001*TM5b,0.0,1.0)
	#print fit
	slope,intercept,r,p,stderr=stats.linregress(obs, model)
	print slope,intercept,r,p,stderr
	xxx=np.logspace(-3,1,1000)
	l3,=plt.plot(xxx,YY(xxx),'-b',lw=4)
	#l55,=plt.plot(xxx,bb*xxx+aa,'-g',lw=4)
	plt.title('Individual observations',fontsize=24)
	plt.xlabel(obsname+' 2010 per point',fontsize=24)
	plt.ylabel(modelname+' collocated 2010 per point',fontsize=24)
	plt.legend([l2,l3],['1:1 line','Fitted'],loc=2,numpoints=1,fontsize=24)
	#plt.legend([l1,l2,l3],['Colocated annual mean','1:1 line','Fitted'],loc=2,numpoints=1)
	plt.annotate('Slope:         %6.2f\nintercept:   %6.2f\nR:               %6.2f'%(slope,intercept,r), 
				xy=(0.02, 0.67), xycoords='axes fraction'
				,fontsize=24)
	#plt.annotate('Slope:         %6.2f\nintercept:   %6.2f\nR:               %6.2f'%(bb,aa,-1), 
	#            xy=(0.02, 0.27), xycoords='axes fraction'
	#            ,fontsize=24)
	cb=plt.colorbar()
	cb.set_label('Number of observations')
	#plt.figure()
def scatter_dot(obs,model,**kwargs):
	if 'ax' in kwargs:
		ax=kwargs['ax']
	else:
		fig,ax=plt.subplots(1)
	if 'ms' in kwargs:
		msize=kwargs['ms']
	else:
		msize=5
	if 'col' in kwargs:
		col=kwargs['col']
	else:
		col='b'
	if 'obsname' in kwargs:
		obsname=kwargs['obsname']
	else:
		obsname='Aeronet'
	if 'modelname' in kwargs:
		modelname=kwargs['modelname']
	else:
		modelname='TM5'

	if 'label' in kwargs:
		label=kwargs['label']
	else:
		label=None
	if 'o_sem':
		o_sem=kwargs['o_sem']
	else:
		o_sem=None
	if 'm_sem':
		m_sem=kwargs['m_sem']
	else:
		m_sem=None
	#print label
	#print obs,model
	mask = ~np.isnan(obs) & ~np.isnan(model)
	#xerror=scipy.std(obs[mask])
	#yerror= scipy.std(model[mask])
	xerror=stats.sem(obs[mask])
	yerror= stats.sem(model[mask])
	
	#ax.scatter(obs,model,c=col,s=msize, label=label)
	ax.errorbar(obs,model,xerr=o_sem,yerr=m_sem,c=col, label=label,fmt='none',ls='none')
	XX=np.max([np.max(obs),np.max(model)])
	XX=1.0
	ax.set_xlim(0.01,XX)
	ax.set_ylim(0.01,XX)
	ax.set_xlim(0.01,1.0)
	ax.set_ylim(0.01,1.0)
	l2,=ax.plot([0,XX],[0,XX],'--k',lw=1.5)
	mask = ~np.isnan(obs) & ~np.isnan(model)
	fit = np.polyfit(obs[mask], model[mask], 1)
	YY=np.poly1d(fit)
	#aa,bb,SS=bivariate_fit(Aeronetb,TM5b,0.01,0.003+0.001*TM5b,0.0,1.0)
	#print fit
	# mask out nans in obs and model for both arrays
	mask = ~np.isnan(obs) & ~np.isnan(model)
	slope,intercept,r,p,stderr=stats.linregress(obs[mask], model[mask])
	#raw_input()
	#print slope,intercept,r,p,stderr
	xxx=np.logspace(-3,1,1000)
	#l3,=plt.plot(xxx,YY(xxx),'-b',lw=4)
	#l55,=plt.plot(xxx,bb*xxx+aa,'-g',lw=4)
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	#ax.set_title('Annual means',fontsize=24)
	ax.set_xlabel(obsname+' 2010 ',fontsize=24)
	ax.set_ylabel(modelname+' collocated 2010 ',fontsize=24)
	
	return slope,intercept,r,p,stderr

def get_regions():
	regions={}
	#regions['US']=[(49,-129),(24,-65)]
	regions['NA']=[(75,-155),(24,-55)]
	regions['EU']=[(70,-16),(39,45)]
	regions['SA']=[(0,-86),(-35,-36)]
	regions['AUS']=[(-12,110),(-40,156)]
	regions['SEAS']=[(45,70),(10,122)]
	regions['SIB']=[(70,60),(50,166)]
	regions['SAH']=[(34,-18),(13,61)]
	regions['AFR']=[(13,-18),(-34,51)]
	return regions
def get_region_names():
	regions={}
	regions['US']='USA'
	regions['BONA']='Boreal North America'
	regions['NA']='North America'
	regions['EU']='Eurasia'
	regions['SA']='South America'
	regions['AUS']='Australia'
	regions['SEAS']='South Eastern Asia'
	regions['SIB']='Boreal Russia'
	regions['SAH']='Norhern Africa/Middle East'
	regions['AFR']='Sub-Saharan Africa'
	regions['ALL']='All Stations'
	return regions
def group_by_region(data):
	regions=get_regions()
	outdata={}
	#outdata['US']={}
	#outdata['BONA']={}
	outdata['NA']={}
	outdata['EU']={}
	outdata['SA']={}
	outdata['AUS']={}
	outdata['SEAS']={}
	outdata['SIB']={}
	outdata['SAH']={}
	outdata['AFR']={}
	#print data
	for i in data:
		#print data[i]
		if not np.ma.is_masked(data[i]):
			#print data[i]
			for reg in regions:
				if regions[reg][0][1]<  data[i][0]<regions[reg][1][1] and regions[reg][1][0]< data[i][1]<regions[reg][0][0]: 
					outdata[reg][i]=data[i]	
		else:
			print 'data is not masked'

	return outdata
def plot_rectangle(bmap, lonmin,lonmax,latmin,latmax,color):
	# Andrew Straw in stackOverflow: 
	# https://stackoverflow.com/questions/12251189/how-to-draw-rectangles-on-a-basemap
    xs = [lonmin,lonmax,lonmax,lonmin,lonmin]
    ys = [latmin,latmin,latmax,latmax,latmin]
    x,y=bmap([lonmin,lonmin,lonmax,lonmax],[latmin,latmax,latmax,latmin])
    xy=zip(x,y)
    #bmap.plot(xs, ys,latlon = True)
    poly = Polygon( xy, facecolor=color)
    plt.gca().add_patch(poly)
def biasmap(data,aerodata,bounds=None,**kwargs):
	if not 'ax' in kwargs:
		f,ax=plt.subplots(1,figsize=(10,7))
	else:
		ax=kwargs['ax']

	m = Basemap(projection='robin',lon_0=0)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,120.,30.))
	m.drawmeridians(np.arange(0.,3060.,60.))
	mycmap=plt.get_cmap('coolwarm',11) 
	# define the bins and normalize
	if bounds==None:
		bounds = [-1.1,-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.1]
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	count=0
	for i in range(len(aerodata)):
		#print i[0][0],i[1][0]
		x1,y1=m(aerodata[i][0][0],aerodata[i][1][0])
		
		diff=(np.ma.mean(data[i][4][:])-np.ma.mean(aerodata[i][4][:]))/np.ma.mean(aerodata[i][4][:])
		if np.ma.min(data[i][4]) is np.ma.masked:
			count+=1
			continue
		m.scatter(x1,y1,marker='o',c=diff,s=300*(np.mean(aerodata[i][4][:])),norm=norm,cmap = mycmap )
		xx,yy=m(-160,-70)
		m.scatter(xx,yy,marker='o',c=0,s=300,norm=norm,cmap = mycmap)
		xx,yy=m(-150,-70)
		plt.text(xx,yy,'AOD 1.0')
		xx,yy=m(-160,-60)
		m.scatter(xx,yy,marker='o',c=0,s=150,norm=norm,cmap = mycmap)
		xx,yy=m(-150,-60)
		plt.text(xx,yy,'AOD 0.5')
	
	c = plt.colorbar(orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7)
	c.ax.tick_params(labelsize=10)
	c.set_label('AOD relative bias [(TM5-AERONET)/AERONET]')
	return f,ax,c
def map_bias(obs,model,**kwargs):
	if 'ax'not in kwargs:
		f,axs=plt.subplots(1,figsize=(10,7))
	else:
		axs=kwargs['ax']
	if 'boundmax'in kwargs:
		boundmax=kwargs['boundmax']
	#print 'axs ',axs
	m = Basemap(projection='robin',lon_0=0,ax=axs)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,91.,30.))
	m.drawmeridians(np.arange(-180.,181.,60.))
	mycmap=plt.get_cmap('coolwarm',11) 
	# define the bins and normalize

	bounds = [-0.45,-0.35,-0.25,-0.15,-0.05,-0.02,0.02,0.05,0.15,0.25,0.35,0.45]
	bounds = [-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]
	#print obs
	if 'boundmax' not in kwargs:
		temp=0.0
		boundmax=0.0
		for i in obs:
			temp=max(np.amax(np.mean(model[i][4][:])-np.mean(obs[i][4][:])),np.abs(np.amin(np.mean(model[i][4][:])-np.mean(obs[i][4][:]))))
			if temp >boundmax:
				boundmax=temp
		if boundmax< 0.375:
			bounds=np.linspace(-boundmax,boundmax,12)
		#boundmax=1.1
	if 'bounds' not in kwargs:
		bounds=np.linspace(-boundmax,boundmax,12)
	else:
		bounds=kwargs['bounds']
	norm = mpl.colors.BoundaryNorm(bounds, len(bounds)-1)
	##
	#intitialise
	if 'name1' in kwargs:
		name1=kwargs['name1']
	if 'name2' in kwargs:
		name2=kwargs['name2']
	annual_obs=[]
	annual_model=[]
	if 'ref' in kwargs:
		ref=kwargs['ref']
	else:
		ref=obs
	for i in obs:
		x1,y1=m(obs[i][0][0],obs[i][1][0])

		diff=np.mean(model[i][4][:])-np.mean(obs[i][4][:])	
		reldiff=(np.mean(model[i][4][:])-np.mean(obs[i][4][:]))/np.mean(obs[i][4][:])	
		if reldiff>boundmax:
			print 'station over bounds on figure [station, relati difference, bound]',i,reldiff,boundmax
		days_data=np.ma.count(model[i][4][:])
		if np.ma.min(model[i][4]) is np.ma.masked:
			continue
		annual_model.append(np.mean(model[i][4][:]))
		annual_obs.append(np.mean(obs[i][4][:]))
		cs=m.scatter(x1,y1,marker='o',c=reldiff,s=300*(np.mean(ref[i][4][:])),norm=norm,cmap = mycmap )
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.85,extend='both',ax=axs)
	c.ax.tick_params(labelsize=14)
	c.set_label('AOD relative change [('+name2+'-'+name1+')/'+name1+']',fontsize=18)
	xx,yy=m(-160,-70)
	m.scatter(xx,yy,marker='o',c=0,s=300,norm=norm,cmap = mycmap)
	xx,yy=m(-150,-70)
	axs.annotate('AOD 1.0',xy=(xx,yy),xycoords='data',xytext=(xx,yy),textcoords='data')
	#print xx,yy
	xx,yy=m(-160,-60)
	m.scatter(xx,yy,marker='o',c=0,s=150,norm=norm,cmap = mycmap)
	xx2,yy2=m(-150,-60)
	axs.annotate('AOD 0.5',xy=(xx2,yy2),xycoords='data',xytext=(xx2,yy2),textcoords='data')
	#plt.text(xx2,yy2,'AOD 0.5')
	return axs

def categorize(obs,model):
	overestimatemodel=[]
	overestimateobs=[]
	underestimatemodel=[]
	underestimateobs=[]
	goodmodel=[]
	goodobs=[]
	#print diff
	for i in obs:
		#print i, model[i][4].data
		diff=np.mean(model[i][4].data)-np.mean(obs[i][4].data)
			
		if diff >0.1:
			overestimateobs.append(obs[i])
			overestimatemodel.append(model[i])
		if diff >-0.025 and diff <0.025:
			goodobs.append(obs[i])
			goodmodel.append(model[i])
		if diff <-0.1:
			underestimateobs.append(obs[i])
			underestimatemodel.append(model[i])
	return underestimatemodel,goodmodel,overestimatemodel,underestimateobs,goodobs,overestimateobs

def read_aggregated(inpath=None):
	import numpy as np 
	import netCDF4 as nc
	#from netCDF4 import netcdftime
	import glob
	#print inpath
	if inpath==None:
		print 'No input file'
		quit()
	outputdata={}
	print 'reading aggregated data from '+inpath
	for ifile,datafile in enumerate(glob.iglob(inpath+'*.nc')):
		site=datafile.split('/')[-1].split('.')[0]
		site= re.sub('[0-9]{6}_[0-9]{6}_','',site)
		if 'South_Pole_Obs_NOAA' in datafile:
			print datafile
			print 'skipping South Pole'
			continue
		ncdata=nc.Dataset(datafile,'r')
		#print ncdata.variables
		if 'AOT_500' in ncdata.variables:
			aotname='AOT_500'
		else:
			aotname='od550aer'
		stdname=aotname+'_std_dev'
		numname=aotname+'_num_points'
		lon=ncdata.variables['longitude'][:]
		lat=ncdata.variables['latitude'][:]
		alt=ncdata.variables['altitude'][:]
		aot=ncdata.variables[aotname][:]
		if stdname in ncdata.variables:
			aotstd=ncdata.variables[stdname][:]
		else:
			aotstd=None
		if numname in ncdata.variables:
			aotnum=ncdata.variables[numname][:]
		else:
			aotnum=None
		time=ncdata.variables['time'][:]
		unit_temps = ncdata.variables['time'].units 
		#print ncdata.variables['time'].ncattrs()

		if 'calendar' in ncdata.variables['time'].ncattrs():
			cal_temps = ncdata.variables['time'].calendar
		else:
			cal_temps='gregorian'
		datevar = []
		datevar.append(nc.num2date(time,units = unit_temps,calendar = cal_temps))
		#if aot[0] is not np.ma.masked: 
		outputdata[site]=lon,lat,alt,time,np.squeeze(aot),datafile,aotstd,aotnum,np.squeeze(datevar)
	return outputdata

def read_aeronet_all(inpath=None):
	import numpy as np 
	import netCDF4 as nc
	#from netCDF4 import netcdftime
	import glob
	if inpath==None:
		print 'no inpath'
		quit()
	TM5data=[]
	TM5_collocated=inpath
	for TM5 in  glob.iglob(TM5_collocated+'*'):
		#print TM5
		ncTM5=nc.Dataset(TM5,'r')
		if 'longitude' in ncTM5.variables:
			lon=ncTM5.variables['longitude'][:]
		else:	
			lon=ncTM5.variables['lon'][:]
		if 'latitude' in ncTM5.variables:
			lat=ncTM5.variables['latitude'][:]
		else:
			lat=ncTM5.variables['lat'][:]
		if 'altitude' in ncTM5.variables:
			alt=ncTM5.variables['altitude'][:]
		aot=ncTM5.variables['od550aer'][:]
		time=ncTM5.variables['time'][:]
		#if aot[0] is not np.ma.masked: 
		TM5data.append([lon,lat,alt,time,aot,TM5])
	return TM5data


def do_colocate(AODmodel = 'od550aer',AODdata = 'AOT_500',input_TM5_data = '/Users/bergmant/Documents/codes/anaconda/data/newdata/aerocom3_TM5_AP3-CTRL2016_global_2010_hourly.od550aer.nc',output= 'Collocated_aeronet_ctrl2016_lin/',aeronet='aeronet_2010/',year=2010):
	from datetime import timedelta
	model=cis.read_data(input_TM5_data,AODmodel)
	#for outputdata in glob.iglob(aeronet+'*'):
	wavelength=np.float(AODdata[0].split('_')[1])
	aeronet_out_path=aeronet_path+str(year)+'/AOT_550test/'
	for aeronetdata in glob.iglob(aeronet_path+'/AOT_Level2_All_Points/AOT/LEV20/ALL_POINTS/*'):
		
			#outputname=os.path.basename(outputdata)[14:]
		# Strip the beginning code away.
		outputname=os.path.basename(aeronetdata)[14:]+'.nc'

		#aeronetinname=os.path.basename(aeronetdata)[14:]
		#print os.path.isfile(output+outputname)
		# read in the aeronet data for a given station
		# from existing NC-file if it is available
		#print 'testing aeronet '+outputname
		aeronet_aot=None

		# using nc-files does not work, cis will collocate also for missing data
		#if os.path.isfile('/Users/bergmant/Documents/obs-data/aeronet/'+str(year)+'/all/'+outputname):
		#	aeronet_aot=cis.read_data('/Users/bergmant/Documents/obs-data/aeronet/'+str(year)+'/all/'+outputname,AODdata)
		# or from original files 
		# needs a switch not to redo everytime
		#else:
			#print aeronet_aot
		print aeronet_out_path+'/all/'
		if not os.path.isdir(aeronet_out_path+'/all/'):
			os.mkdir(aeronet_out_path)
			os.mkdir(aeronet_out_path+'/all/')
		aeronet_aot=cis.read_data(aeronetdata,AODdata)
		#Subset wanted year
		aeronet_aot=aeronet_aot.subset(t=cis.time_util.PartialDateTime(year))
		# needed for interpolation of 550
		aeronet_ang=cis.read_data(aeronetdata,'440-675Angstrom')
		aeronet_ang=aeronet_ang.subset(t=cis.time_util.PartialDateTime(year))
		if aeronet_aot !=None:
			#print wavelength
			#print aeronet_aot.data
			print #aeronet_ang.data
			print #(550.0/500.0)**(-aeronet_ang.data)
			#print aeronet_aot.data*(550.0/wavelength)**(-aeronet_ang.data)
			print #(aeronet_aot.data*(550.0/500.0)**(-aeronet_ang.data))/aeronet_aot.data
 			aeronet_aot.data=aeronet_aot.data*(550.0/wavelength)**(-aeronet_ang.data)
			tempdata=cis.utils.create_masked_array_for_missing_data(aeronet_aot.data,aeronet_aot.metadata.missing_value)
			aeronet_aot.data=tempdata
 			aeronet_aot.save_data(aeronet_out_path+'/all/'+outputname)
		#if aeronet_ang !=None:
			#print aeronet_ang.data
		#	tempdata=cis.utils.create_masked_array_for_missing_data(aeronet_ang.data,aeronet_ang.metadata.missing_value)
		#	aeronet_ang.data=tempdata
 			#aeronet_ang.save_data('/Users/bergmant/Documents/obs-data/aeronet/'+str(year)+'/all/'+outputname+'ang')

		# write aeronet subset out for comparisons

		#if not os.path.isfile(output+outputname):
		if aeronet_aot !=None:		

			#if not os.path.isfile(output+outputname):
			# make sure subset exists
				# collocate
			print 'processing:', aeronetdata
			print 'to:',output+'/all/',' as: '+outputname

			if not os.path.isdir(output+'/all/'):
				os.mkdir(output+'/all')
			collocated=model.collocated_onto(aeronet_aot)
			# save collocated data (could be also saved into dataholder...) 
			# although file preserves speeds up future use
			collocated.save_data(output+'/all/'+outputname)

			periods=[1,365]
			for period in periods:
				if period==1:
					pname='daily'
				elif period==365:
					pname='yearly'
				elif period==30:
					pname='monthly'
				print 'aggregating: ', outputname, ' with period ',period#, ' to: ',outputname
				if not os.path.isdir(output+'/'+pname):
					os.mkdir(output+'/'+pname)

				#print 'aggregating: ', input_data, ' with period ',period
				print 'to: ',output+'/'+pname+'/'
				print 'station:',outputname
				#subprocess.call("cis aggregate "+AODmodel+":"+input_TM5_data+" t=[2010-01-01,2010-12-31,"+period+"] -o "+output+outputname,shell=True)
				#test=subprocess.call(command,shell=True)
				#data=cis.read_data(input_data,AODvariable)
				#print data
				aggregated_data=collocated.aggregate(t=[cis.time_util.PartialDateTime(2010),timedelta(days=period)] )
				aggregated_data.save_data(output+'/'+pname+'/'+outputname)
				if not os.path.isdir(aeronet_out_path+'/'+pname):
					os.mkdir(aeronet_out_path+'/'+pname)
				print os.path.exists(aeronet_out_path+'/'+pname+'/')

				if os.path.exists(aeronet_out_path+'/'+pname+'/'):
					# aggregate aeronetdata as well
					aggregated_data=aeronet_aot.aggregate(t=[cis.time_util.PartialDateTime(2010),timedelta(days=period)] )
					aggregated_data.save_data(aeronet_out_path+'/'+pname+'/'+outputname)
		else:
			print 'no aeronet data for year: '+str(year)

def do_aggregate(AODvariable = 'od550aer',input_data="" ,output="",period=1):
	from datetime import timedelta
	#print input_data
	for input_data in glob.iglob(input_data+'/*.nc'):
		
		outputname=os.path.basename(input_data)
		
		if not os.path.isfile(output+"/"+outputname):
			print 'aggregating: ', input_data, ' with period ',period, ' to: ',outputname
			#print 'aggregating: ', input_data, ' with period ',period
			print 'to: ',output
			print 'station:',outputname
			#subprocess.call("cis aggregate "+AODmodel+":"+input_TM5_data+" t=[2010-01-01,2010-12-31,"+period+"] -o "+output+outputname,shell=True)
			#test=subprocess.call(command,shell=True)
			data=cis.read_data(input_data,AODvariable)
			#print data
			#print period
			aggregated_data=data.aggregate(t=[cis.time_util.PartialDateTime(2010),timedelta(days=period)] )
			aggregated_data.save_data(output+outputname)
def month_aggregation(indata):
	jindex=0
	monthly_data_sum=np.zeros((12))
	monthly_data_sum_SH=np.zeros((12))
	monthly_data_sum_NH=np.zeros((12))
	monthly_data_std=np.zeros((12))
	monthly_data=np.zeros((12,1000000))
	monthly_data_SH=np.zeros((12,1000000))
	monthly_data_NH=np.zeros((12,1000000))
	monthly_data[:,:]=np.nan
	monthly_data_SH[:,:]=np.nan
	monthly_data_NH[:,:]=np.nan
	monthly_numsites=np.zeros((12))
	monthly_numsites_SH=np.zeros((12))
	monthly_numsites_NH=np.zeros((12))
	for i in indata:
		monthly_mean_data=np.zeros((12))
		monthly_mean_data_NH=np.zeros((12))
		monthly_mean_data_SH=np.zeros((12))
		for kk in range(12):
			data=[]
			data_NH=[]
			data_SH=[]
			for jj,itime in enumerate(indata[i][8][:]):
				if itime.month==kk+1 and not np.ma.is_masked(indata[i][4][jj]):
					data.append(indata[i][4][jj])
					monthly_data[kk,jindex]=indata[i][4][jj]
					if indata[i][1][0]>0:
						data_NH.append(indata[i][4][jj])
						monthly_data_NH[kk,jindex]=indata[i][4][jj]
					else:		
						data_SH.append(indata[i][4][jj])
						monthly_data_SH[kk,jindex]=indata[i][4][jj]
					
					jindex+=1
				else:
					continue
			monthly_mean_data[kk]=np.nanmean(np.array(data))
			monthly_mean_data_NH[kk]=np.nanmean(np.array(data_NH))
			monthly_mean_data_SH[kk]=np.nanmean(np.array(data_SH))
			if not np.isnan(monthly_mean_data[kk]):
				monthly_data_sum[kk]+=monthly_mean_data[kk]
				monthly_numsites[kk]+=1
			if not np.isnan(monthly_mean_data_NH[kk]):
				monthly_data_sum_NH[kk]+=monthly_mean_data_NH[kk]
				monthly_numsites_NH[kk]+=1
			if not np.isnan(monthly_mean_data_SH[kk]):
				monthly_data_sum_SH[kk]+=monthly_mean_data_SH[kk]
				monthly_numsites_SH[kk]+=1
	mean_monthly_data=monthly_data_sum/monthly_numsites
	mean_monthly_data_NH=monthly_data_sum_NH/monthly_numsites_NH
	mean_monthly_data_SH=monthly_data_sum_SH/monthly_numsites_SH
	for i in range(12):
		monthly_data_std[i]=np.nanstd(monthly_data[i,:])
	return mean_monthly_data,monthly_numsites,monthly_data_std,mean_monthly_data_NH,mean_monthly_data_SH

#if __name__="__main__":
AODmodel = 'od550aer'
aero_wavelength=550
AODdata = ['AOT_500']
year=2010
aeronet_out_path='/Users/bergmant/Documents/obs-data/aeronet/'+str(year)+'/AOT_'+str(aero_wavelength)+'/'
aeronet='/Users/bergmant/Documents/obs-data/aeronet/aeronet_2010-550_all/'
experiment_paths={}
experiment_paths['newsoa-ri']=None
experiment_paths['oldsoa-bhn']=None
l_do_aggregate=False
#l_do_aggregate=True
l_do_collocate=False
#l_collocate=True
for i in experiment_paths:
	input_TM5_data = '/Users/bergmant/Documents/tm5-soa/output/raw/'+i+'/general_TM5_'+i+'_2010.lev0.od550aer.nc'
	output_col= '/Users/bergmant/Documents/tm5-soa/output/processed/col_aeronet_'+i+'-2010-550/'

	experiment_paths[i]={'inputdata':input_TM5_data,'collocated':output_col}
	if l_do_collocate:
		print 'collocating', i
		if not os.path.isdir(output_col+'/all/'):
			print "creating output directory: "+ output_col + '/all/'
			if not os.path.isdir(output_col):
				os.mkdir(output_col)
			os.mkdir(output_col+'/all')

		do_colocate(AODmodel,AODdata,input_TM5_data,output_col,aeronet,2010)
	if l_do_aggregate:
		output_aggre=output_col+'/daily/'
		if not os.path.isdir(output_aggre):
			print "creating output directory: "+ output_aggre
			os.mkdir(output_aggre)
		do_aggregate('od550aer',output_col+'/all/',output_aggre,1)
	
		# output_aggre=output_col+'/monthly'
		# if not os.path.isdir(output_aggre):
		# 	print "creating output directory: "+ output_aggre
		# 	os.mkdir(output_aggre)
		# do_aggregate('od550aer',output_col+'/all/',output_aggre,'P1M')
		
		output_aggre=output_col+'/yearly/'
		if not os.path.isdir(output_aggre):
			print "creating output directory: "+ output_aggre
			os.mkdir(output_aggre)
		do_aggregate('od550aer',output_col+'/all/',output_aggre,365)
		#subprocess.call("cis col "+AODmodel+":"+input_TM5_data+" "+outputdata+':variable='+AODdata+',collocator=lin -o '+output+outputname,shell=True)
		   #aggregate AOT_500:${i} t=[2010-01-01,2010-12-31,P1M]  -o aeronet_2010_monthly/$(basename $i .nc).monthly.nc
	output_aggre_aeronet='/Users/bergmant/Documents/obs-data/aeronet/2010/daily/'
	input_aeronet_all='/Users/bergmant/Documents/obs-data/aeronet/2010/all/'
	if not os.path.isdir(output_aggre_aeronet):
		print "creating output directory: "+ output_aggre_aeronet
		os.mkdir(output_aggre_aeronet)
	print 'aggregating aeronet'
	do_aggregate('AOT_500',input_aeronet_all,output_aggre_aeronet,1)

	if l_do_aggregate and l_do_collocate:
		print ("using previously aggregated files")
	else:
		print ("collocation and aggregation done!")



aeronet_data_dict={}
TM5NEWdatadict={}
TM5OLDdatadict={}
dataperiods=['yearly','monthly','daily','all']
for period in dataperiods:
	aeronet_data_dict[period]=read_aggregated(aeronet_out_path+'/'+period+'/')
	TM5NEWdatadict[period]=read_aggregated(experiment_paths['newsoa-ri']['collocated']+'/'+period+'/')
	TM5OLDdatadict[period]=read_aggregated(experiment_paths['oldsoa-bhn']['collocated']+'/'+period+'/')
tm5new_regions=group_by_region(TM5NEWdatadict['yearly'])
tm5old_regions=group_by_region(TM5OLDdatadict['yearly'])
aeronet_regions=group_by_region(aeronet_data_dict['yearly'])


aeronet_table=open(paper+"aeronet_table.txt","w")
aeronet_table_latex=open(paper+"aeronet_table.tex","w")


f_scat_comb,ax_scat_comb=plt.subplots(nrows=2,ncols=2,figsize=(16,16))#,tight_layout=True)
colordict={'NA':'r','EU':'b','SA':'k','SIB':'g','SEAS':'y','AUS':'m','AFR':'brown','SAH':'c'}
m = Basemap(projection='cyl',lon_0=0,ax=ax_scat_comb[1,1])
m.drawcoastlines()
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,3060.,60.))
regions=get_regions()
for index,reg in enumerate(colordict):
	aero_concat=[]#np.empty(len(aeronet[reg]))
	tm5old_concat=[]#np.empty(len(tm5old[reg]))
	tm5new_concat=[]#np.empty(len(tm5new[reg]))
	aero_concat_sem=[]#np.empty(len(aeronet[reg]))
	tm5old_concat_sem=[]#np.empty(len(tm5old[reg]))
	tm5new_concat_sem=[]#np.empty(len(tm5new[reg]))
	
	if len(aeronet_regions[reg])==0:
			continue
	
	lonmin=min(regions[reg][0][1],regions[reg][1][1])
	lonmax=max(regions[reg][0][1],regions[reg][1][1])
	latmin=min(regions[reg][0][0],regions[reg][1][0])
	latmax=max(regions[reg][0][0],regions[reg][1][0])

	plot_rectangle(m,lonmin,lonmax,latmin,latmax,color=colordict[reg])
	x, y = m(lonmin,latmin)
	ax_scat_comb[1,1].annotate(reg, xy=(x, y), xycoords='data', xytext=(x, y), textcoords='data',color='w',fontsize=14)
	for i in aeronet_regions[reg]:
		if not np.ma.is_masked(aeronet_regions[reg][i][4]):
			print 'test', i,aeronet_regions[reg][i][4],aeronet_regions[reg][i][6],aeronet_regions[reg][i][7],aeronet_regions[reg][i][6][:]/np.sqrt(aeronet_regions[reg][i][7][:])
			aero_concat.append(aeronet_regions[reg][i][4])
			tm5old_concat.append(tm5old_regions[reg][i][4])
			tm5new_concat.append(tm5new_regions[reg][i][4])
			aero_concat_sem.append(aeronet_regions[reg][i][6]/np.sqrt(aeronet_regions[reg][i][7]))
			tm5old_concat_sem.append(tm5old_regions[reg][i][6][:]/np.sqrt(tm5old_regions[reg][i][7][:]))
			tm5new_concat_sem.append(tm5new_regions[reg][i][6][:]/np.sqrt(tm5new_regions[reg][i][7][:]))
		else: # everything is masked, do not add masked data
			if debug:
				print 'all data is masked, probably no data available for the current year' 
				print 'site,aeronet, oldsoa, newsoa',i,aeronet_regions[reg][i][4],tm5old_regions[reg][i][4],tm5new_regions[reg][i][4]
			continue
		if np.isnan(tm5new_regions[reg][i][4]):
			print 'NAN',tm5new_regions[reg][i][5],aeronet_regions[reg][i][5]
			print 'NAN',tm5new_regions[reg][i][4],aeronet_regions[reg][i][4]
			
	aero_concat=np.array(aero_concat)

	tm5old_concat=np.array(tm5old_concat)
	tm5new_concat=np.array(tm5new_concat)

	aero_concat_sem=np.array(aero_concat_sem)
	tm5old_concat_sem=np.array(tm5old_concat_sem)
	tm5new_concat_sem=np.array(tm5new_concat_sem)
	
	if len(aero_concat)==0:
		continue
	NMB_new=NMB(aero_concat,tm5new_concat)*100
	NMB_old=NMB(aero_concat,tm5old_concat)*100
	print stats.sem(aero_concat)
	print stats.sem(tm5new_concat)
	print stats.sem(tm5old_concat)

	slope,interp,rnew,p,stderrnew=scatter_dot(aero_concat,tm5new_concat,col=colordict[reg],ax=ax_scat_comb[0,0],modelname='NEWSOA',label=reg,ms=10,o_sem=aero_concat_sem,m_sem=tm5new_concat_sem)

	slope,interp,rold,p,stderrold=scatter_dot(aero_concat,tm5old_concat,col=colordict[reg],ax=ax_scat_comb[0,1],modelname='OLDSOA',label=reg,ms=10,o_sem=aero_concat_sem,m_sem=tm5old_concat_sem)
	rmse_old=RMSE(aero_concat,tm5old_concat)
	rmse_new=RMSE(aero_concat,tm5new_concat)
	aeronet_table.write('%-6.6s\t%2d\t%6.3f\t%6.3f\t%6.3f\t%5.1f\t%5.1f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\\\\ \n'%(reg,len(tm5old_concat),np.ma.mean(np.array(aero_concat)),np.ma.mean(np.array(tm5new_concat)),np.ma.mean(np.array(tm5old_concat)),NMB_new,NMB_old,rnew,rold,rmse_new,rmse_old))
	aeronet_table_latex.write('%-6.6s&%2d&%6.3f&%6.3f&%6.3f&%5.1f&%5.1f&%6.3f&%6.3f&%6.3f&%6.3f\\\\ \n'%(reg,len(tm5old_concat),np.ma.mean(np.array(aero_concat) ),np.ma.mean(np.array(tm5new_concat)),np.ma.mean(np.array(tm5old_concat)),NMB_new,NMB_old,rnew,rold,rmse_new,rmse_old))
if debug:
	print 'table done'	
for i in aeronet_data_dict['yearly']:
	px,py=m(aeronet_data_dict['yearly'][i][0],aeronet_data_dict['yearly'][i][1])
	m.scatter(px,py,5,marker='o',color='w',edgecolor='k',linewidth=0.5,zorder=100)
n_month,n_month_sites,astd,n_month_NH,n_month_SH=month_aggregation(aeronet_data_dict['all'])
n_month_new,n_month_sites_new,nstd,n_month_new_NH,n_month_new_SH=month_aggregation(TM5NEWdatadict['all'])
n_month_old,n_month_sites_old,ostd,n_month_old_NH,n_month_old_SH=month_aggregation(TM5OLDdatadict['all'])
#ax_scat_comb[1,0].plot(np.linspace(0,11,12),n_month,'k',label='AERONET')
#ax_scat_comb[1,0].plot(np.linspace(0,11,12),n_month_new,'r',label=EXP_NAMEs[0])
#ax_scat_comb[1,0].plot(np.linspace(0,11,12),n_month_old,'b',label=EXP_NAMEs[1])
ax_scat_comb[1,0].plot(np.linspace(0,11,12),n_month_NH,'-k',label='AERONET NH')
ax_scat_comb[1,0].plot(np.linspace(0,11,12),n_month_new_NH,'-r',label=EXP_NAMEs[0]+' NH')
ax_scat_comb[1,0].plot(np.linspace(0,11,12),n_month_old_NH,'-b',label=EXP_NAMEs[1]+' NH')
ax_scat_comb[1,0].plot(np.linspace(0,11,12),n_month_SH,'--k',label='AERONET SH')
ax_scat_comb[1,0].plot(np.linspace(0,11,12),n_month_new_SH,'--r',label=EXP_NAMEs[0]+' SH')
ax_scat_comb[1,0].plot(np.linspace(0,11,12),n_month_old_SH,'--b',label=EXP_NAMEs[1]+' SH')
ax_scat_comb[1,0].set_xticks(np.linspace(0,11,12))
ax_scat_comb[1,0].set_xticklabels(str_months(),fontsize=16)
ax_scat_comb[1,0].set_xlabel('Month',fontsize=24)
ax_scat_comb[1,0].set_ylabel('AOD [1]',fontsize=24)

ax_scat_comb[0,0].legend(loc=4,fontsize=14)
ax_scat_comb[1,0].legend(loc=3,fontsize=14)
ax_scat_comb[0,1].legend(loc=4,fontsize=14)
#print (aeronet_data_dict['all'].keys())
aero_R=[]
TM5n_R=[]
TM5o_R=[]
for site in aeronet_data_dict['yearly']:
	if not np.ma.is_masked(aeronet_data_dict['yearly'][site][4]):			
		aero_R.append(aeronet_data_dict['yearly'][site][4])
		TM5n_R.append(TM5NEWdatadict['yearly'][site][4])
		TM5o_R.append(TM5OLDdatadict['yearly'][site][4])


ax_scat_comb[0,0].annotate('R : %5.2f '%(stats.pearsonr(np.array(TM5n_R),np.array(aero_R))[0]),xy=(0.05,0.95),xycoords='axes fraction',fontsize=14)
ax_scat_comb[0,0].annotate('MB : %5.3f '%(np.mean(np.array(TM5n_R))-np.mean(np.array(aero_R))),xy=(0.05,0.90),xycoords='axes fraction',fontsize=14)
ax_scat_comb[0,1].annotate('R : %5.2f '%(stats.pearsonr(np.array(TM5o_R),np.array(aero_R))[0]),xy=(0.05,0.95),xycoords='axes fraction',fontsize=14)
ax_scat_comb[0,1].annotate('MB : %5.3f'%(np.mean(np.array(TM5o_R))-np.mean(np.array(aero_R))),xy=(0.05,0.90),xycoords='axes fraction',fontsize=14)
k=0
for i in range(2):
	for j in range(2):
		if k==3:
			ax_scat_comb[i,j].annotate(string.ascii_lowercase[k]+')',xy=(0.01,1.51),xycoords='axes fraction',fontsize=20)
		else:
			ax_scat_comb[i,j].annotate(string.ascii_lowercase[k]+')',xy=(0.01,1.01),xycoords='axes fraction',fontsize=20)
		k+=1
f_scat_comb.savefig(output_png_path+'article/fig14_revised_scatter_categorized_log_seasonal_map_hemispheric.png',dpi=600)
f_scat_comb.savefig(output_pdf_path+'article/fig14_revised_scatter_categorized_log_seasonal_map_hemispheric.pdf')
f_scat_comb.savefig('test-fig14_scatter_categorized_log_seasonal_map.pdf')

TM5n2=concatenate_sites2(TM5NEWdatadict['daily'])
TM5o2=concatenate_sites2(TM5OLDdatadict['daily'])
aero2=concatenate_sites2(aeronet_data_dict['daily'])


aero=[]
TM5n=[]
TM5o=[]
for i in TM5NEWdatadict['yearly']:
	#aa=outputdataym[i][4]
	aa=aeronet_data_dict['yearly'][i][4]
	
	#print np.ma.is_masked(aa)
	#print np.isnan(aa)
	if np.isnan(aa) or np.isnan(TM5NEWdatadict['yearly'][i][4]):
		raw_input('nnannnanann')
	#print aa
	if not np.ma.is_masked(aa):
		aero.append(aa)
		TM5o.append(TM5OLDdatadict['yearly'][i][4])
		TM5n.append(TM5NEWdatadict['yearly'][i][4])
	else:
		if debug:
			print 'all data is masked due to missing data or other reasons'

modelu,modelg,modelo,obsu,obsg,obso=categorize(aeronet_data_dict['yearly'],TM5NEWdatadict['yearly'])
print 'OLDSOA MB: %6.3f'%(np.ma.mean(np.array(TM5o)-np.array(aero)))
print 'OLDSOA NMB: %6.3f'%(NMB(TM5o,aero))
print 'NEWSOA MB: %6.3f'%(np.ma.mean(np.array(TM5n)-np.array(aero)))
print 'NEWSOA NMB: %6.3f'%(NMB(TM5n,aero))
TM5n=np.array(TM5n)
TM5o=np.array(TM5o)
aero=np.array(aero)
NMB_new=NMB(TM5n,aero)
NMB_old=NMB(TM5o,aero)
mask = ~np.isnan(aero) & ~np.isnan(TM5o)
slope,intercept,rold,p,stderr=stats.linregress(aero[mask], TM5o[mask])	
slope,intercept,rnew,p,stderr=stats.linregress(aero[mask], TM5n[mask])	
rmse_old=RMSE(aero,TM5o)
rmse_new=RMSE(aero,TM5n)
print '%-6.6s\t%2d\t%6.4f\t%6.4f\t%6.4f\t%5.1f\t%5.1f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\\\\ \n'%('all',len(TM5o),np.ma.mean(np.array(aero)),np.ma.mean(np.array(TM5n)),np.ma.mean(np.array(TM5o)),NMB_new,NMB_old,rnew,rold,rmse_new,rmse_old)
aeronet_table.write('%-6.6s\t%2d\t%6.3f\t%6.3f\t%6.3f\t%5.1f\t%5.1f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\\\\ \n'%('All',len(TM5o),np.ma.mean(np.array(aero)),np.ma.mean(np.array(TM5n)),np.ma.mean(np.array(TM5o)),NMB_new,NMB_old,rnew,rold,rmse_new,rmse_old))
aeronet_table_latex.write('%-6.6s&%2d&%6.3f&%6.3f&%6.3f&%5.1f&%5.1f&%6.3f&%6.3f&%6.3f&%6.3f\\\\ \n'%('All',len(TM5o),np.ma.mean(np.array(aero) ),np.ma.mean(np.array(TM5n)),np.ma.mean(np.array(TM5o)),NMB_new,NMB_old,rnew,rold,rmse_new,rmse_old))

aeronet_table.close()
aeronet_table_latex.close()


f_2panel_diff_old_diff_aeronet,axit=plt.subplots(nrows=1,ncols=2,figsize=(20,6))
map_bias(TM5OLDdatadict['all'],TM5NEWdatadict['all'],ax=axit[0],boundmax=0.5,name1='OLDSOA',name2='NEWSOA',bounds=[-0.5,-0.35,-0.15,-0.05,-0.03,-0.01,0.01,0.03,0.05,0.15,0.35,0.5],ref=aeronet_data_dict['all'])
axit[0].annotate('a)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=20)
map_bias(aeronet_data_dict['all'],TM5NEWdatadict['all'],ax=axit[1],boundmax=1.1,name1='AERONET',name2='NEWSOA')
axit[1].annotate('b)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=20)
plt.tight_layout()
f_2panel_diff_old_diff_aeronet.savefig(output_png_path+'/article/fig15_revised_map-2panel-aeronet-oldsoa.png',dpi=600)
f_2panel_diff_old_diff_aeronet.savefig(output_pdf_path+'/article/fig15_revised_map-2panel-aeronet-oldsoa.pdf')

N_station=0
for i in aeronet_data_dict['all']:
	if debug:
		print 'no. datapoints for station: ',i, np.ma.count(TM5NEWdatadict['all'][i][4][:])
	if np.ma.count(TM5NEWdatadict['all'][i][4][:])>0:
		N_station+=1
print 'N stations with data for 20101, total number of stations',N_station,len(aeronet_data_dict['all'])
plt.show()


