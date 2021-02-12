from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import datetime, timedelta
#import sys
#sys.path.append(r'/Users/bergmant/Documents/python/tm5/')
#from lonlat import lonlat
from general_toolbox import get_gridboxarea,lonlat,write_netcdf_file
import matplotlib.gridspec as gridspec
#from netcdfutils import write_netcdf_file
import glob
import xarray as xr
from scipy.stats import pearsonr
import os
import pandas as pd
from settings import *
class observation(object):
	def __init__(self,name=None,ele=None,lon=None,lat=None):
		self.name=name
		self.ele=ele
		self.lon=lon
		self.lat=lat
		self.data={}

class obs_set(observation):
	def __init__(self):
		observation.__init__(self)


def RMSE(obs,model):
	N=0
	RMSE=0.0
	for o,m in zip(obs,model):
		if not (np.isnan(o) or np.isnan(m)):
			#print m,o
			RMSE+=(m-o)**2
			N+=1
	return np.sqrt(RMSE/N)
def MFE(obs,model):
	N=0
	MFE=0.0
	for o,m in zip(obs,model):
		if not (np.isnan(o) or np.isnan(m)):
			MFE+=np.abs(m-o)/((o+m)/2)
			N+=1
	return MFE/N
def MFB(obs,model):
	N=0
	MFB=0.0
	#print obs,model
	for o,m in zip(obs,model):
		if not (np.isnan(o) or np.isnan(m)):
			MFB+=(m-o)/((o+m)/2)
			N+=1
	return MFB/N
def NMB(obs,model):
	N=0
	nom=0.0
	denom=0.0
	#print obs,model
	for o,m in zip(obs,model):
		if not (np.isnan(o) or np.isnan(m)):
			nom+=(m-o)
			denom+=o
			N+=1
	NMB=nom/denom
	return NMB
def MB(obs,model):
	N=0
	nom=0.0
	denom=0.0
	#print obs,model
	for o,m in zip(obs,model):
		if not (np.isnan(o) or np.isnan(m)):
			nom+=(m-o)
			N+=1
	MB=nom/N
	return MB
def NME(obs,model):
	N=0
	nom=0.0
	denom=0.0
	#print obs,model
	for o,m in zip(obs,model):
		if not (np.isnan(o) or np.isnan(m)):
			nom+=np.abs(m-o)
			denom+=o
			N+=1
	NME=nom/denom
	return NME
def select_gp(data,mlat,mlon):
	lon,lat=lonlat('TM53x2')
	lonidx = (np.abs(lon-mlon)).argmin()
	latidx = (np.abs(lat-mlat)).argmin()
	if len(np.shape(data))==4:      
		return np.squeeze(data[:,:,latidx,lonidx])
	elif len(np.shape(data))==3:
		return np.squeeze(data[:,latidx,lonidx])
	else:
		print 'Data not compatible to select gridpoint timeseries'
		return None
	
def pm_sel(data,rad=1.25):
	for i in data:
		pass
def read_radius(inputdata):
	modes=['NUS','AIS','ACS','COS','AII','ACI','COI']
	data=nc.Dataset(inputdata,'r')
	output={}
	for imode in modes:
		#print 'RWET_'+imode
		output['RWET_'+imode]=data.variables['RWET_'+imode]
	return output
def calc_dens(inputdata):
	data=nc.Dataset(inputdata,'r')
	modes=['NUS','AIS','ACS','COS','AII','ACI','COI']
	comps=['SOA','POM','SO4','BC','DU','SS']
	dry = ['AII','ACI','COI']
	densities={}
	densities['SOA']=1.300
	densities['POM']=1.300
	densities['SO4']=1.841
	densities['BC']=1.800
	densities['SS']=2.165
	densities['DU']=2.65
	roo_a={}
	#h2o=['NUS','AIS','ACS','COS']
	for m in modes:
		mass=np.zeros_like(data.variables['M_SOAAIS'])
		roo=np.zeros_like(data.variables['M_SOAAIS'])

		for c in comps:
			indexi='M_'+c+m 
			if indexi in data.variables:
				#print  indexi
				mass+=data.variables[indexi][:]
				roo+=data.variables[indexi][:]*densities[c]
		#if m in h2o:
		if m not in dry:
			indexi='aerh2o3d_'+m
			mass+=data.variables[indexi][:]
			roo+=data.variables[indexi][:]*1.0 #denisty of water
		rindex='RWET_'+m
		dens=mass/(4/3*np.pi*(data.variables[rindex][:]**3)*data.variables['N_'+m][:])
		roo=roo/mass
		roo_a[m]=roo
		print m, roo
		#raw_input()
	#roo_a['AII']=np.ones_like(data.variables['M_SOAAIS'])
	#roo_a['ACI']=np.ones_like(data.variables['M_SOAAIS'])
	#roo_a['COI']=np.ones_like(data.variables['M_SOAAIS'])
	return roo_a

def read_mass(comp,inputdata,rad=1.25):
	modesigma={}
	modesigma['NUS']=1.59
	modesigma['AIS']=1.59
	modesigma['ACS']=1.59
	modesigma['COS']=2.00
	modesigma['AII']=1.59
	modesigma['ACI']=1.59
	modesigma['COI']=2.00
	roo_a=calc_dens(inputdata)
	# select mass below threshold
	d_a_correction_factor={}
	for m in roo_a:
		d_a_correction_factor[m]=np.sqrt(roo_a[m]/1.0) #sqrt(roo_p/roo_0), roo_0=1.00g/cm3 (Water)
	 	print np.mean(d_a_correction_factor[m])	
	raw_input('correction factors')
	if rad < 0.1:
		print 'WARNING: You are trying to check mass of particless smaller than 0.2um. Mass check only for AIS, ACS and COS at the moment!!!'
	completedata={}
	data=nc.Dataset(inputdata,'r')
	outputdataset=None
	rwetdata=read_radius(inputdata)
	modfrac={}
	hr2=(0.5*np.sqrt(2.0))
	for mode in modesigma:
		cmedr2mmedr= np.exp(3.0*(np.log(modesigma[mode])**2))
		# correct for aerodynamic diameter(radius)
		rdata=np.where(rwetdata['RWET_'+mode][:]<1e-20,1e-10,rwetdata['RWET_'+mode][:])
		rdata=rdata*d_a_correction_factor[mode]
		z=(np.log(rad)-np.log(rdata*1e6*cmedr2mmedr)/np.log(modesigma[mode]))
		print z, rad,rdata,cmedr2mmedr
		modfrac[mode]=0.5+0.5*erf(z*hr2)

	modes=['NUS','AIS','ACS','COS','AII','ACI','COI']
	#modes=['NUS','AIS']#,'ACS','COS','AII','ACI','COI']
	outputmodes={}
	# for i in data.variables:
	# 	for t in modes:
	# 		print comp+t,(comp+t) in i
	# 	if comp in i:
	for imode in modes:
			#completedata[comp+'']

		i=comp+imode
		if i  in data.variables:

			print comp,imode,i

			#outputmodes[i]=np.zeros(np.shape(data.variables[i][:]))
			outputmodes[i]=data.variables[i][:]
			if outputdataset==None:
				outputdata=np.zeros(np.shape(data.variables[i][:]))
				outpudataset=1
			print 'modfrac'
			#outputdata+=data.variables[i][:]*modfrac[modes] 
			if i[-3:]=='COS':
				outputdata+=data.variables[i][:]*modfrac['COS']#_cos 
			elif i[-3:]=='ACS':
				outputdata+=data.variables[i][:]*modfrac['ACS']#_acs
			elif i[-3:]=='AIS':
				outputdata+=data.variables[i][:]*modfrac['AIS']#modfrac_ais
			elif i[-3:]=='AII':
				outputdata+=data.variables[i][:]*modfrac['AII']#modfrac_aii
			elif i[-3:]=='ACI' or i[-3:]=='COI':
				#no soa/oa
				pass
			elif i[-3:]=='NUS':	
				outputdata+=data.variables[i][:]*modfrac['NUS']
			else:
				print('something wrong, mode unknown',imode)
				break

	return outputdata,outputmodes
def improve(filein='/Users/bergmant/Documents/obs-data/improve2010/improvedata.20170626.txt'):
	
	#f=open('/Users/bergmant/Documents/obs-data/improve2010/201761261120541INx0LI.txt','rU')
	f=open(filein,'rU')
	#line=f.readline()
	j=0
	header=[]
	hline=f.readline().rstrip()
	while hline !='Data':
		header.append(hline)
		hline=f.readline().rstrip()
	sites=False
	sitedata={}
	sitedata2={}
	siteline=[]
	j=0
	while j<len(header):
		element=header[j]
		#print element
		if element=='Sites':
			j+=2
			#create sitedata from header information
			while header[j]!='':
				siteline=header[j].split(',')
				#code: name, lat,lon,ele,startdate,enddate,data (empty)
				if len(siteline)>4 and siteline[0]!='Site':
					sitedict={}
					sitedict['name']=siteline[0]
					sitedict['lat']=float(siteline[7])
					sitedict['lon']=float(siteline[8])
					sitedict['ele']=float(siteline[9])
					sitedict['startdate']=siteline[10]
					sitedict['enddate']=siteline[11]
					sitedata[siteline[1]]=[siteline[0],float(siteline[7]),float(siteline[8]),float(siteline[9]),siteline[10],siteline[11],[]]
					sitedata2[siteline[1]]=sitedict
				j+=1
		elif element=='Parameters':	
			j+=2
			while header[j]!='':
				j+=1

		elif element=='Overview':	
			j+=1
		elif element=='Site History':	
			j+=1
		elif element=='Datasets':
			j+=1
		else:
			j+=1

	# read data from file
	j=0
	data=[]
	for line in f: 
		dcells=line.rstrip().split(',')
		j+=1
		# Dataset header row:
		if dcells[0]=='Dataset':
			#check the index for values
			i_ocf=dcells.index('OCf:Value')
			i_ecf=dcells.index('ECf:Value')
			i_ocf_u=dcells.index('OCf:Unc')
			i_ecf_u=dcells.index('ECf:Unc')
			i_lon=dcells.index('Longitude')
			i_lat=dcells.index('Latitude')
		#check the data projec
		if dcells[0]=='IMPFSPED':
			# gather all data to variable
			data.append(dcells)

	#select date, OC and EC data based on sites
	for site in sitedata.keys():
		print site
		j=0
		N=0
		ocec=[]
		#data={}
		while j< len(data):
			if data[j][1]==site:
				N+=1
				#print j,site
				# set timestamp to middle of the day
				ocec.append([datetime.strptime(data[j][3], '%Y%m%d')+timedelta(hours=12),float(data[j][i_ecf]),float(data[j][i_ocf]),float(data[j][i_ecf_u]),float(data[j][i_ocf_u])])
				#sitedata[site]['time']=datetime.strptime(data[j][3], '%Y%m%d')+timedelta(hours=12)
				# some sites have the data twice, do only the first one
				#print j+1,len(data),len(ocec)
				if j+1>len(data)-1:	
					break
				elif ocec[-1][0]>datetime.strptime(data[j+1][3], '%Y%m%d')+timedelta(hours=12):
					break				
			j+=1
		#print 'N',N
		#print len(ocec)
		temp=np.zeros((len(ocec),2))
		tempoc=[]
		tempec=[]
		tempoc_u=[]
		tempec_u=[]
		datetemp=[]
		datetempoc=[]
		datetempec=[]

		for i in range(len(ocec)):
			datetemp.append(ocec[i][0])	
			# clean up missing data (-999)
			# NaN instead of -999, makes plotting nicer
			if ocec[i][1]<-990:
				temp[i,0]=np.nan
				#print ocec[i][1],ocec[i][2]
			else:
				tempoc.append(float(ocec[i][1]))
				tempoc_u.append(float(ocec[i][3]))
				datetempoc.append(ocec[i][0])	
				temp[i,0]=float(ocec[i][1])	
			if ocec[i][2]<-990:
				temp[i,1]=np.nan
			else:
				temp[i,1]=float(ocec[i][2])
				datetempec.append(ocec[i][0])	
				tempec.append(float(ocec[i][1]))
				tempec_u.append(float(ocec[i][4]))
		if len(datetempoc)>0:
			dataout=np.zeros((len(datetempoc),4))
			for i in range(len(datetempoc)):
				dataout[i,0]=tempoc[i]
				dataout[i,1]=tempec[i]
				dataout[i,2]=tempoc_u[i]
				dataout[i,3]=tempec_u[i]
			sitedata2[site]['oc']=dataout[:,0]
			sitedata2[site]['bc']=dataout[:,1]
			sitedata2[site]['oc_unc']=dataout[:,2]
			sitedata2[site]['bc_unc']=dataout[:,3]
			sitedata2[site]['time_oc']=np.array(datetempoc)
			sitedata2[site]['time_bc']=np.array(datetempec)
			
			sitedata[site][6]=[datetempoc,dataout]
		else:
			#no data so remove the key
			sitedata.pop(site,None)
	return sitedata

def col_improve(modeldata,timeaxis,sitedata):
		##
	# IMPROVE data sampling happens twice a week
	# wed and sat from 0000 to 2400 local time
	yearmean_model=[]
	yearmean_obs=[]
	NN=0
	ncmodel=[]
	ncmodel_std=[]
	ncnames=[]
	nctime=[]
	model_dict={}
	#ncnames=[]
	#time=[]
	print "collocating..."
	for i in sitedata:
		model_dict[i]={}
		NN+=1
		lat=sitedata[i][1]
		lon=sitedata[i][2]
		model_dict[i]['name']=sitedata[i][0]
		model_dict[i]['lon']=lon
		model_dict[i]['lat']=lat
		#print 'c',lat,lon
		gpdata=select_gp(modeldata,lat,lon)
		model_mean=[]
		model_std=[]
		time1=[]
		if len(sitedata[i][6])>0:#[0],sitedata[i][6][1][:,1]
			#for date,d in zip(sitedata[i][6][0],sitedata[i][6][1][:,1]):
			timestep=0
			for date in sitedata[i][6][0]:
				#initilaize counting
				data=[]
				# timesteps in one obs period
				N=0
				# from obs timestamp tp day of year
				curday=date.timetuple().tm_yday
				# Approximate the timezone differences in US
				# timezone shift
				timeshift=np.floor(lat/15)
				### find the first data point that is within obs
				while np.floor(timeaxis[timestep]+timeshift/24.0)<curday:
					timestep+=1
					if timestep==np.size(timeaxis):
						break 
				## if at the end of model data stop
				if timestep== np.size(timeaxis):
					break
				## go through model data and add data that is within the same day
				## and advance to next time step (a) , and keep track of datapoints (N)
				while np.floor(timeaxis[timestep]+timeshift/24.0)==curday:
					data.append(gpdata[timestep])
					timestep+=1
					N+=1
					if timestep==np.size(timeaxis):
						break 
				
				## if we found more htan 1 data point calculate mean
				if N>0:
					#mean of the day
					model_mean.append(np.mean(data)*1e9)
					model_std.append(np.std(data))
					# time of midpoint of current day
					time1.append((timeaxis[timestep-N]+timeaxis[timestep-1])/2)
					#print time1
				## check if we are at the end of model data
				if timestep==np.size(timeaxis):
					break 
				##otherwise continue to next obs data point (date)


				
			yearmean_model.append(np.mean(model_mean))
			yearmean_obs.append(np.mean(sitedata[i][6][1][:,1]))
			ncmodel.append(model_mean)
			ncmodel_std.append(model_std)
			model_dict[i]['N_OM']=len(model_mean)
			model_dict[i]['OM']=model_mean # list of daily means
			model_dict[i]['OM_std']=model_std #list of daily std
			model_dict[i]['OM_ymean']=np.mean(model_mean) # yearly mean
			model_dict[i]['OM_ymean_std']=np.std(model_mean) #std of year
			model_dict[i]['time']=time1 #timestamps for daily means
			model_dict[i]['unit']='ug/m3'
			
			ncnames.append(i)
			nctime.append(time1)
			#print sitedata[i][0]
	return (ncmodel,ncnames,nctime,model_dict)
def read_model_data(exp,sitedata,logger,basepathprocessed='/Users/bergmant/Documents/tm5-SOA/output/processed/',basepathraw='/Users/bergmant/Documents/tm5-SOA/output/raw/'):
	logger.info('Processing experiment %s ',exp )
	#print len(glob.glob(path_improve_col+exp+'*'))
	#print len(glob.glob(path_improve_col+'*'))
	logger.debug(path_improve_col+exp+'*')
	if not os.path.isdir(path_improve_col):
		logger.debug('Processing experiment %s ',exp )
		os.mkdir(path_improve_col)
	if len(glob.glob(path_improve_col+exp+'*'))==0:# and len(glob.glob(basepath+'emep_col/'+exp+'*'))==0:
		if  os.path.exists(basepathraw+exp+'/general_TM5_'+exp+'_2010.lev1.nc'):
			filepath=basepathraw+exp+'/general_TM5_'+exp+'_2010.lev1.nc'
		else:	
			filepath='/Volumes/Utrecht/'+exp+'/general_TM5_'+exp+'_2010.lev1.nc'			
		####READ
		print 'reading simulated radii'
		RW=improve_tools.read_radius(filepath)

		radius_pm25=1.25

		print 'read soa'
		logger.info('Reading SOA for experiment %s from %s',exp, filepath )
		soamassexp,soamodesexp=improve_tools.read_mass('M_SOA',filepath,radius_pm25) 
		print 'read oc'
		pommassexp,pommodesexp=improve_tools.read_mass('M_POM',filepath,radius_pm25)
		print 'sum'	
		poamass=soamassexp+pommassexp
		print soamodesexp
		print 'time'
		oamass=[]
		oamass.append(soamassexp+pommassexp)

		timeaxis_o=nc.Dataset(filepath,'r').variables['time'][:]
		lon_o=nc.Dataset(filepath,'r').variables['lon'][:]
		lat_o=nc.Dataset(filepath,'r').variables['lat'][:]
		t_unit=nc.Dataset(filepath,'r').variables['time'].units
		## day of year, model data starts from day 0 so add 1
		timeaxis=timeaxis_o-np.floor(timeaxis_o[0])+1
		dt_axis=nc.num2date(timeaxis_o[:],units = t_unit,calendar = 'gregorian')[:]

		#dt_ax2=[]
		#dt_ax2=[nc.num2date(x,units = t_unit,calendar = 'gregorian') for x in timeaxis_o]
		logger.debug('time axes, length %s, first value %s, type %s',len(dt_axis),dt_axis[0],type(dt_axis[0]))
		#raw_input()
		#Collocate total oa	
		ncmodel,ncname,nctime,model_site=improve_tools.col_improve(poamass,timeaxis,sitedata)
		modetest={}
		for i in soamodesexp:
			test,nn,nt,ms=improve_tools.col_improve(soamodesexp[i],timeaxis,sitedata)
			print 'fsdfasdf',i,test,nn,nt 
			print i,nn,nt
			modetest[i]=test
			#raw_input()
		#ncmodel_pom,ncname_pom,nctime,model_site=improve_tools.col_improve(pommassexp,timeaxis,sitedata)
			for data,name,time,sdata in zip(test,nn,nt,ms):
				print 'time',time
				write_netcdf_file([data],[name],path_improve_col+exp+'_'+i+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)
							#### select!
		for i in RW:
			RW2,nn,nt,ms=improve_tools.col_improve(RW[i],timeaxis,sitedata)
			for data,name,time,sdata in zip(RW2,nn,nt,ms):
				write_netcdf_file([data],[name],path_improve_col+exp+'_'+i+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)
		
		for data,name,time,sdata in zip(ncmodel,ncname,nctime,model_site):
			write_netcdf_file([data],[name],path_improve_col+exp+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)
		
		#collocate primary oa
		ncmodel_pom,ncname_pom,nctime,model_site=improve_tools.col_improve(pommassexp,timeaxis,sitedata)
		for data,name,time,sdata in zip(ncmodel_pom,ncname,nctime,model_site):
			write_netcdf_file([data],[name],path_improve_col+exp+'_pom_'+name+'.nc',None,None,np.array(time))#,lat,lon)
		#collocate secondary oa
		ncmodel_soa,ncname_soa,nctime,model_site=improve_tools.col_improve(soamassexp,timeaxis,sitedata)
		for data,name,time,sdata in zip(ncmodel_soa,ncname,nctime,model_site):
			write_netcdf_file([data],[name],path_improve_col+exp+'_soa_'+name+'.nc',None,None,np.array(time))#,lat,lon)
			


