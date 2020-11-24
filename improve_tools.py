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
		print 'RWET_'+imode
		output['RWET_'+imode]=data.variables['RWET_'+imode]
	print output	
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
				print  indexi
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
	print rwetdata
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
		print 'modfrac',mode, np.mean(modfrac[mode])
		#print mode,np.where(rwetdata['RWET_'+mode][:]<1e-20)
		#data=np.where(rwetdata['RWET_'+mode][:]<1e-20,1e-10,rwetdata['RWET_'+mode][:])
		print mode,np.where(rdata<1e-20)
		#print mode,np.where(rwetdata['RWET_'+mode][:]<1e-20)
		#raw_input()
	# select mass below threshold
	#ACS


	# hr2=(0.5*np.sqrt(2.0))
	# modfrac={}
	# rACS=data.variables['RWET_ACS'][:]
	# completedata['RWET_ACS']=rACS
 # 	cmedr2mmedr= np.exp(3.0*(np.log(1.59)**2))	
	# z=(np.log(rad)-np.log(rACS*1e6*cmedr2mmedr)/np.log(1.59))
	# print 0.5+0.5*erf(z*hr2)
	# raw_input()
	# modfrac['ACS']=0.5+0.5*erf(z*hr2)
	# #free memory
	# del rACS,z
	# #COS
	# rCOS=data.variables['RWET_COS'][:]
	# completedata['RWET_COS']=rCOS
	# cmedr2mmedr= np.exp(3.0*(np.log(2.00)**2))	
	# z=(np.log(rad)-np.log(rCOS*1e6*cmedr2mmedr)/np.log(2.00))
	# modfrac['COS']=0.5+0.5*erf(z*hr2)
	# del rCOS,z

	# rAIS=data.variables['RWET_AIS'][:]
	# completedata['RWET_AIS']=rAIS
	# cmedr2mmedr= np.exp(3.0*(np.log(1.59)**2))	
	# z=(np.log(rad)-np.log(rAIS*1e6*cmedr2mmedr)/np.log(1.59))
	# modfrac['AIS']=0.5+0.5*erf(z*hr2)
	# #free memory
	# del rAIS,z
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
	#raw_input()
	print outputmodes.keys()
	print outputmodes

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

			# if len(sitedata[i][6][0])>1:
			# 	obs=sitedata[i][6][1][:len(model_mean),1]
			# 	if len(model_mean)>0:
			# 		MFBm= MFB(obs,model_mean)
			# 		MFEm= MFE(obs,model_mean)

				
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


# def main():
# 	#output_pdf_path='/Users/bergmant/Documents/tm5-SOA/figures/pdf/IMPROVE/'
# 	#output_png_path='/Users/bergmant/Documents/tm5-SOA/figures/png/IMPROVE/'
# 	#output_jpg_path='/Users/bergmant/Documents/tm5-SOA/figures/jpg/IMPROVE/'
# 	## get IMPROVE observations
# 	#basepath='/Users/bergmant/Documents/tm5-SOA/output/processed/'
# 	sitedata=improve()
# 	#for exp in ['soa-riccobono','oldsoa-final','nosoa']:
# 	EXPS=['soa-riccobono','oldsoa-final','nosoa']
# 	EXPS=['newsoa-ri','oldsoa-bhn']#,'nosoa']
# 	#EXPS=['test']#,'nosoa']
# 	col_path=basepathprocessed+'improve_col2/'	
# 	for exp in EXPS:
# 		print exp
# 		print len(glob.glob(col_path+exp+'*'))
# 		if not os.path.isdir(col_path):
# 			os.mkdir(col_path)
# 		if len(glob.glob(col_path+exp+'*'))==0:# and len(glob.glob(basepath+'emep_col/'+exp+'*'))==0:
# 			filepath='/Volumes/Utrecht/'+exp+'/general_TM5_'+exp+'_2010.lev1.nc'
# 			#testidata:
# 			#filepath='general_TM5_test_2010.lev1.nc'
# 			# 
# 			print 'read radius'
# 			test=read_radius(filepath)
# 			r_pm25=1.25
# 			print 'read soa'
# 			soamassexp=read_mass('M_SOA',filepath,r_pm25) 
# 			print 'read oc'
# 			pommassexp=read_mass('M_POM',filepath,r_pm25)
# 			print 'sum'	
# 			poamass=soamassexp+pommassexp
			
# 			print 'time'
# 			oamass=[]
# 			oamass.append(soamassexp+pommassexp)
# 			timeaxis_o=nc.Dataset(filepath,'r').variables['time'][:]
# 			lon_o=nc.Dataset(filepath,'r').variables['lon'][:]
# 			lat_o=nc.Dataset(filepath,'r').variables['lat'][:]
# 			t_unit=nc.Dataset(filepath,'r').variables['time'].units
# 			## day of year, model data starts from day 0 so add 1
# 			timeaxis=timeaxis_o-np.floor(timeaxis_o[0])+1
# 			dt_axis=nc.num2date(timeaxis_o[:],units = t_unit,calendar = 'gregorian')[:]
# 			dt_ax2=[]
# 			dt_ax2=[nc.num2date(x,units = t_unit,calendar = 'gregorian') for x in timeaxis_o]


# 			ncmodel,ncname,nctime,model_site=col_improve(poamass,timeaxis,sitedata)

# 			## total oa
# 			for data,name,time,sdata in zip(ncmodel,ncname,nctime,model_site):
# 				print 'time',time
# 				write_netcdf_file([data],[name],col_path+exp+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)

# 			## primary oa
# 			ncmodel_pom,ncname_pom,nctime,model_site=col_improve(pommassexp,timeaxis,sitedata)
# 			for data,name,time,sdata in zip(ncmodel_pom,ncname,nctime,model_site):
# 				print 'time',time
# 				write_netcdf_file([data],[name],col_path+exp+'_pom_'+name+'.nc',None,None,np.array(time))#,lat,lon)
# 			## secondary oa
# 			ncmodel_soa,ncname_soa,nctime,model_site=col_improve(soamassexp,timeaxis,sitedata)
# 			for data,name,time,sdata in zip(ncmodel_soa,ncname,nctime,model_site):
# 				print 'time',time
# 				write_netcdf_file([data],[name],col_path+exp+'_soa_'+name+'.nc',None,None,np.array(time))#,lat,lon)
# 			for i in range(len(ncname)):
# 				print ncname[i],ncname_pom[i]
# 	meanmodelmon={}
# 	meanmodelmon_pom={}
# 	print sitedata
# 	#sitepanda=pd.DataFrame.from_dict(sitedata)
# 	#print sitepanda.index

# 	#raw_input()
# 	for exp in EXPS:
# 		meanmodelmon[exp]=np.zeros((len(sitedata.keys()),12))
# 		meanmodelmon_pom[exp]=np.zeros((len(sitedata.keys()),12))
# 	meanmodelmon['obs']=np.zeros((len(sitedata.keys()),12))
# 	#meanobsmon=np.zeros((sitedata.keys().len,12))
# 	kk=0
# 	yearmean_model={}
# 	yearmean_modelpom={}
	
# 	for i in sitedata:
# 		print i
# 		f,a=plt.subplots(1)
# 		fmon,amon=plt.subplots(1)
# 		obs=sitedata[i][6][1][:,1]
# 		timedata=sitedata[i][6][0][:]
# 		monthindices={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[]}
# 		monthdata=np.zeros([12])
# 		monthdata[:]=np.NAN
# 		modelmonthdata=np.zeros([12])
# 		modelmonthdata[:]=np.NAN
# 		modelmonthdata_pom=np.zeros([12])
# 		modelmonthdata_pom[:]=np.NAN
# 		for j in timedata:
# 			print j.month
# 			monthindices[j.month].append(timedata.index(j))
# 		for k in monthindices:
# 			print monthdata.shape,monthindices[k],(sitedata[i][6][1][monthindices[k],1]),np.mean(sitedata[i][6][1][monthindices[k],1])
# 			monthdata[k-1]=np.mean(sitedata[i][6][1][monthindices[k],1])
# 		print monthindices[1]
# 		print monthdata
# 		print monthdata.shape
# 		print meanmodelmon['obs']
# 		print type(meanmodelmon['obs'])
# 		meanmodelmon['obs'][kk,:]=monthdata[:]
			
# 		amon.plot(np.linspace(1,12,12),monthdata)
# 		a.plot(timedata,obs,'k')
# 		for exp in EXPS:
# 			print i
# 			model=nc.Dataset(col_path+exp+'_'+i+'.nc','r').variables[i][:]
# 			model_pom=nc.Dataset(col_path+exp+'_pom_'+i+'.nc','r').variables[i][:]
# 			print model,col_path+exp+'_'+i+'.nc'
# 			for kmod in monthindices:
# 				print kmod
# 				print model
# 				#raw_input()
# 				print monthindices
# 				print monthdata.shape,monthindices[kmod],(model[monthindices[kmod]]),np.mean(model[monthindices[kmod]])

# 				modelmonthdata[kmod-1]=np.mean(model[monthindices[kmod]])
# 				modelmonthdata_pom[kmod-1]=np.mean(model_pom[monthindices[kmod]])
# 			print modelmonthdata
# 			meanmodelmon[exp][kk,:]=modelmonthdata
# 			meanmodelmon_pom[exp][kk,:]=modelmonthdata_pom
# 			amon.plot(np.linspace(1,12,12),modelmonthdata,'r',ls='-')		
# 			amon.plot(np.linspace(1,12,12),modelmonthdata_pom,c='r',ls='--')		
# 			print len(obs)
# 			mfb=MFB(obs,model)
# 			mfe=MFE(obs,model)
# 			nmb=NMB(obs,model)
# 			nme=NME(obs,model)
# 			rmse=RMSE(obs,model)
# 			r=pearsonr(monthdata,modelmonthdata)
# 			mfbmon=MFB(monthdata,modelmonthdata)
# 			mfemon=MFE(monthdata,modelmonthdata)
# 			nmbmon=NMB(monthdata,modelmonthdata)
# 			nmemon=NME(monthdata,modelmonthdata)
# 			rmsemon=RMSE(monthdata,modelmonthdata)
# 			rmon=pearsonr(monthdata,modelmonthdata)
# 			print r
# 			#model=model_site[i]['OM']		
# 			if exp==EXPS[0]:
# 				X=0.05
# 				colori='r'
# 			elif exp==EXPS[1]:
# 				X=0.8
# 				colori='b'
# 			else:# exp==EXPS[1]:
# 				X=0.4
# 				colori='g'
# 			amon.plot(np.linspace(1,12,12),modelmonthdata,c=colori,ls='-')		
# 			amon.plot(np.linspace(1,12,12),modelmonthdata_pom,c=colori,ls='--')		
# 			amon.set_title(i)
# 			a.plot(timedata,model,colori)
# 			a.annotate('EXP: '+exp,xy=(X,0.95),xycoords='axes fraction')
# 			a.annotate(('MFB: %6.2f')%mfb,xy=(X,0.9),xycoords='axes fraction')
# 			a.annotate(('MFE: %6.2f')%mfe,xy=(X,0.85),xycoords='axes fraction')
# 			a.annotate(('NMB: %6.2f')%nmb,xy=(X,0.8),xycoords='axes fraction')
# 			a.annotate(('NME: %6.2f')%nme,xy=(X,0.75),xycoords='axes fraction')
# 			a.annotate(('RMSE: %6.2f')%rmse,xy=(X,0.7),xycoords='axes fraction')
# 			a.annotate(('R: %6.2f')%r[0],xy=(X,0.65),xycoords='axes fraction')
# 			amon.annotate('EXP: '+exp,xy=(X,0.95),xycoords='axes fraction')
# 			amon.annotate(('MFB: %6.2f')%mfbmon,xy=(X,0.9),xycoords='axes fraction')
# 			amon.annotate(('MFE: %6.2f')%mfemon,xy=(X,0.85),xycoords='axes fraction')
# 			amon.annotate(('NMB: %6.2f')%nmbmon,xy=(X,0.8),xycoords='axes fraction')
# 			amon.annotate(('NME: %6.2f')%nmemon,xy=(X,0.75),xycoords='axes fraction')
# 			amon.annotate(('RMSE: %6.2f')%rmsemon,xy=(X,0.7),xycoords='axes fraction')
# 			amon.annotate(('R: %6.2f')%rmon[0],xy=(X,0.65),xycoords='axes fraction')
# 		a.set_title(i)
# 		f.savefig(output_pdf_path+'/IMPROVE/siteplots/timeseries-IMPROVE'+i+'.pdf',dpi=400)
# 		f.savefig(output_png_path+'/IMPROVE/siteplots/timeseries-IMPROVE'+i+'.png',dpi=400)
# 		f.savefig(output_jpg_path+'/IMPROVE/siteplots/timeseries-IMPROVE'+i+'.jpg',dpi=400)
# 		fmon.savefig(output_pdf_path+'/IMPROVE/siteplots/monthly-IMPROVE'+i+'.pdf',dpi=400)
# 		fmon.savefig(output_png_path+'/IMPROVE/siteplots/monthly-IMPROVE'+i+'.png',dpi=400)
# 		fmon.savefig(output_jpg_path+'/IMPROVE/siteplots/monthly-IMPROVE'+i+'.jpg',dpi=400)	
# 		print 
# 		kk+=1
# 	for i in range(15):
# 		print np.count_nonzero(~np.isnan(meanmodelmon['obs'][i,:]))
# 		print (meanmodelmon['obs'][i,:])
# 		print '----',meanmodelmon['obs'][i,:]
# 		print '----',np.count_nonzero(~np.isnan(meanmodelmon['obs'][i,:]))
# 	#raw_input()
# 	fmean,amean=plt.subplots(ncols=4,figsize=(12,4))
# 	colors=['red', 'green','blue','black']
# 	shadingcolors=['#ff000033', '#00ff0033','#0000ff33','#55555533']
# 	for n,exp in enumerate(EXPS,0):	
# 		std=np.nanstd(meanmodelmon[exp],axis=0)
# 		maxi=np.nanmax(meanmodelmon[exp],axis=0)
# 		mini=np.nanmin(meanmodelmon[exp],axis=0)
# 		#amean.fill_between(np.linspace(0,12,12), mini, maxi, facecolor=shadingcolors[n], alpha=0.3,interpolate=True)
# 		#amean.plot(np.linspace(0,12,12), mini, color=shadingcolors[n])
# 		#amean.plot(np.linspace(0,12,12), maxi, color=shadingcolors[n])
# 		amean[n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)+std, color=shadingcolors[n])
# 		amean[n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)-std, color=shadingcolors[n])
# 		amean[n].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon[exp],axis=0),color=colors[n])
# 		amean[n].set_ylim([0,2.0])
# 		amean[n].set_title(exp)
# 		nmbmean=NMB(np.nanmean(meanmodelmon['obs'],axis=0),np.nanmean(meanmodelmon[exp],axis=0))
# 		rmean=pearsonr(np.nanmean(meanmodelmon[exp],axis=0),np.nanmean(meanmodelmon['obs'],axis=0))
# 		amean[n].annotate(('NMB: %6.2f')%nmbmean,xy=(X,0.8),xycoords='axes fraction')
# 		amean[n].annotate(('R: %6.2f,%6.2e')%(rmean[0],rmean[1]),xy=(X,0.75),xycoords='axes fraction')
# 		#ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
# 		print np.nanmean(meanmodelmon[exp],axis=0)
# 	std=np.nanstd(meanmodelmon['obs'],axis=0)
# 	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
# 	mini=np.nanmin(meanmodelmon['obs'],axis=0)
# 	#amean.fill_between(np.linspace(0,12,12), mini, maxi,facecolor=shadingcolors[3], alpha=0.3,interpolate=True)
# 	#amean.plot(np.linspace(0,12,12), mini,color=shadingcolors[3])
# 	#amean.plot(np.linspace(0,12,12), maxi,color=shadingcolors[3])
# 	amean[3].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)+std,color=shadingcolors[3])
# 	amean[3].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)-std,color=shadingcolors[3])
# 	amean[3].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0),color=colors[3])
# 	amean[3].set_title('obs')
# 	amean[3].set_ylim([0,2.0])
# 	fmean.suptitle('IMPROVE')
# 	#amean.set_yscale("log", nonposy='clip')
# 	fmean.savefig(output_png_path+'/IMPROVE/siteplots/monthly-IMPROVE-allmean.png',dpi=400)

# 	fmean,amean=plt.subplots(ncols=3,figsize=(12,4))
# 	colors=['red', 'blue','black']
# 	shadingcolors=['#ff000033', '#00ff0033','#0000ff33','#55555533']
# 	for n,exp in enumerate(EXPS[:],0):	
# 		if n==0:
# 			amean[n].set_title('NEWSOA')
# 		elif n==1:
# 			amean[n].set_title('OLDSOA')
# 		std=np.nanstd(meanmodelmon[exp],axis=0)
# 		maxi=np.nanmax(meanmodelmon[exp],axis=0)
# 		mini=np.nanmin(meanmodelmon[exp],axis=0)
# 		#amean.fill_between(np.linspace(0,12,12), mini, maxi, facecolor=shadingcolors[n], alpha=0.3,interpolate=True)
# 		#amean.plot(np.linspace(0,12,12), mini, color=shadingcolors[n])
# 		#amean.plot(np.linspace(0,12,12), maxi, color=shadingcolors[n])
# 		amean[n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)+std, color=shadingcolors[n])
# 		amean[n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)-std, color=shadingcolors[n])
# 		amean[n].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon[exp],axis=0),color=colors[n])
# 		amean[n].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
# 		amean[n].set_ylim([0,2.0])
# 		#amean[n].set_title(exp)
# 		nmbmean=NMB(np.nanmean(meanmodelmon['obs'],axis=0),np.nanmean(meanmodelmon[exp],axis=0))
# 		rmean=pearsonr(np.nanmean(meanmodelmon[exp],axis=0),np.nanmean(meanmodelmon['obs'],axis=0))
# 		amean[n].annotate(('NMB: %6.2f')%nmbmean,xy=(0.2,0.8),xycoords='axes fraction',fontsize=16)
# 		amean[n].annotate(('R: %6.2f,%6.2e')%(rmean[0],rmean[1]),xy=(0.2,0.7),xycoords='axes fraction',fontsize=16)
# 		#ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
# 		print np.nanmean(meanmodelmon[exp],axis=0)
# 	std=np.nanstd(meanmodelmon['obs'],axis=0)
# 	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
# 	mini=np.nanmin(meanmodelmon['obs'],axis=0)
# 	#amean.fill_between(np.linspace(0,12,12), mini, maxi,facecolor=shadingcolors[3], alpha=0.3,interpolate=True)
# 	#amean.plot(np.linspace(0,12,12), mini,color=shadingcolors[3])
# 	#amean.plot(np.linspace(0,12,12), maxi,color=shadingcolors[3])
# 	amean[2].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)+std,color=shadingcolors[2])
# 	amean[2].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)-std,color=shadingcolors[2])
# 	amean[2].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0),color=colors[2])
# 	amean[2].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])

# 	amean[2].set_title('Observations')
# 	amean[2].set_ylim([0,2.0])
# 	fmean.suptitle('IMPROVE')
# 	#amean.set_yscale("log", nonposy='clip')
# 	fmean.savefig(output_png_path+'/IMPROVE/monthly-IMPROVE-SOAmean.png',dpi=400)
	
# 	#raw_input()
# 	fmean,amean=plt.subplots(ncols=2,figsize=(12,4))
# 	colors=['red', 'blue','black']
# 	shadingcolors=['#ff000033', '#00ff0033','#0000ff33','#55555533']
# 	amean[0].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
# 	#amean[0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)+std,color=shadingcolors[2])
# 	#amean[0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)-std,color=shadingcolors[2])
# 	#amean[0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0),color=colors[2])
# 	amean[0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])

# 	#amean[0].set_title('Observations')
# 	amean[0].set_ylim([0,2.0])
# 	amean[1].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
# 	#plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)+std,'',color=shadingcolors[2])
# 	#amean[1].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)-std,color=shadingcolors[2])
# 	#amean[1].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0),color=colors[2])
# 	amean[1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])

# 	#amean[1].set_title('Observations')
# 	amean[1].set_ylim([0,2.0])
# 	for n,exp in enumerate(EXPS[:],0):	
# 		if n==0:
# 			amean[n].set_title('NEWSOA')
# 			labeli='NEWSOA'
# 		elif n==1:
# 			amean[n].set_title('OLDSOA')
# 			labeli='OLDSOA'
# 		std=np.nanstd(meanmodelmon[exp],axis=0)
# 		maxi=np.nanmax(meanmodelmon[exp],axis=0)
# 		mini=np.nanmin(meanmodelmon[exp],axis=0)
# 		#amean.fill_between(np.linspace(0,12,12), mini, maxi, facecolor=shadingcolors[n], alpha=0.3,interpolate=True)
# 		#amean.plot(np.linspace(0,12,12), mini, color=shadingcolors[n])
# 		#amean.plot(np.linspace(0,12,12), maxi, color=shadingcolors[n])
# 		amean[n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)+std, color=shadingcolors[n])
# 		amean[n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)-std, color=shadingcolors[n])
# 		amean[n].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon[exp],axis=0),color=colors[n],label=labeli)
# 		amean[n].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
# 		if n==0:
# 			amean[n].set_ylim([0,0.5])
# 		else:
# 			amean[n].set_ylim([0,2.0])
# 		#amean[n].set_title(exp)
# 		nmbmean=NMB(np.nanmean(meanmodelmon['obs'],axis=0),np.nanmean(meanmodelmon[exp],axis=0))
# 		rmean=pearsonr(np.nanmean(meanmodelmon[exp],axis=0),np.nanmean(meanmodelmon['obs'],axis=0))
# 		amean[n].annotate(('NMB: %6.2f')%nmbmean,xy=(0.2,0.8),xycoords='axes fraction',fontsize=16)
# 		amean[n].annotate(('R: %6.2f,%6.2e')%(rmean[0],rmean[1]),xy=(0.2,0.7),xycoords='axes fraction',fontsize=16)
# 		#ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
# 		print np.nanmean(meanmodelmon[exp],axis=0)
# 	std=np.nanstd(meanmodelmon['obs'],axis=0)
# 	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
# 	mini=np.nanmin(meanmodelmon['obs'],axis=0)
# 	#amean.fill_between(np.linspace(0,12,12), mini, maxi,facecolor=shadingcolors[3], alpha=0.3,interpolate=True)
# 	#amean.plot(np.linspace(0,12,12), mini,color=shadingcolors[3])
# 	#amean.plot(np.linspace(0,12,12), maxi,color=shadingcolors[3])
# 	fmean.suptitle('IMPROVE')
# 	#amean.set_yscale("log", nonposy='clip')
# 	fmean.savefig(output_png_path+'/IMPROVE/monthly-IMPROVE-SOAmean_2panels.png',dpi=400)




# 	f2,ax2=plt.subplots(ncols=3,figsize=(12,4))
# 	f2b,ax2b=plt.subplots(ncols=2,figsize=(10,4))
# 	fb,axb=plt.subplots(ncols=3,figsize=(12,4))

# 	k=-1
# 	yearmean_model={}
# 	yearmean_model_pom={}
# 	for exp in EXPS:
# 		k+=1
# 		yearmean_obs=[]
# 		#all_model=np.array()
# 		#all_obs=np.array()
# 		print exp
# 		f,ax=plt.subplots(1)
# 		#temp=None
# 		for i in sitedata:
# 			#print i
# 			model=nc.Dataset(col_path+exp+'_'+i+'.nc','r').variables[i][:]
# 			model_pom=nc.Dataset(col_path+exp+'_pom_'+i+'.nc','r').variables[i][:]
# 			#model=model_site[i]['OM']		
# 			##### trying to get yearly means in dict
# 			if exp not in yearmean_model.keys():
# 				yearmean_model[exp]=[]
# 				yearmean_model[exp].append(np.mean(model))
# 				yearmean_model_pom[exp]=[]
# 				yearmean_model_pom[exp].append(np.mean(model))
# 			else:
# 				yearmean_model[exp].append(np.mean(model))
# 				yearmean_model_pom[exp].append(np.mean(model))
			
# 			####
# 			#yearmean_model[exp].append(np.mean(model))
# 			yearmean_obs.append(np.mean(sitedata[i][6][1][:,1]))
# 			print type(model)
# 			if not 'temp_mod' in locals():
# 				temp_mod=model.copy()
# 			else:			
# 				temp_mod=np.concatenate((all_model,model))
# 				print np.shape(temp_mod)
# 			all_model=temp_mod.copy()
# 			if not 'temp_obs' in locals():
# 				temp_obs=sitedata[i][6][1][:,1].copy()
# 			else:			
# 				temp_obs=np.concatenate((all_obs,sitedata[i][6][1][:,1]))
# 				print np.shape(temp_obs)
# 			all_obs=temp_obs.copy()
# 			#print type(sitedata[i][6][1][:,1])
# 			#all_obs+=list(sitedata[i][6][1][:,1])
# 			ax.loglog(sitedata[i][6][1][:,1],model,'or',ms=2)
# 			axb[k].loglog(sitedata[i][6][1][:,1],model,'or',ms=2)
# 			xmax=max(max(model),max(sitedata[i][6][1][:,1]))
# 			xmin=min(min(model),min(sitedata[i][6][1][:,1]))

# 			ymax=xmax
# 			ymin=xmin
# 			ax.set_xlabel('IMPROVE OM[pm25 1.8*C][ug m-3]')
# 			ax.set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
# 			ax.set_ylabel('TM5 OM[pm25][ug m-3]')
# 			ax.set_title(sitedata[i][0])
# 			ax.set_title(exp+': all sites')
# 			axb[k].set_xlabel('IMPROVE OM[pm25 1.8*C][ug m-3]')
# 			axb[k].set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
# 			axb[k].set_ylabel('TM5 OM[pm25][ug m-3]')
# 			axb[k].set_title(sitedata[i][0])
# 			axb[k].set_title(exp+': all sites')
# 		#plt.show()
# 		ax.plot([0.0001,1000],[0.0001,1000])
# 		ax.plot([0.0001,1000],[0.001,10000],'g--')
# 		ax.plot([0.001,10000],[0.0001,1000],'g--')
# 		ax.set_ylim([.9e-4,2e1])
# 		ax.set_xlim([.9e-4,2e1])
# 		axb[k].plot([0.0001,1000],[0.0001,1000])
# 		axb[k].plot([0.0001,1000],[0.001,10000],'g--')
# 		axb[k].plot([0.001,10000],[0.0001,1000],'g--')
# 		axb[k].set_ylim([.9e-4,2e1])
# 		axb[k].set_xlim([.9e-4,2e1])
# 				#raw_input()
# 		print 'N '+exp+':',len(yearmean_model[exp]),len(yearmean_obs)
# 		if len(yearmean_model[exp])>1 and len(yearmean_obs)>1 and False:

# 			f,ax=plt.subplots(1)
# 			print yearmean_model.keys,len(yearmean_model[exp]),len(yearmean_obs)
# 			ax.loglog(yearmean_obs,yearmean_model[exp],'ob')
# 			ax2[k].loglog(yearmean_obs,yearmean_model[exp],'or',ms=2)
# 			if k<2:
# 				ax2b[k].loglog(yearmean_obs,yearmean_model[exp],'or',ms=2)
# 			xmax=max(max(yearmean_model[exp]),max(yearmean_obs))
# 			xmin=min(min(yearmean_model[exp]),min(yearmean_obs))
# 			mfb=MFB(yearmean_obs,yearmean_model[exp])
# 			mfe=MFE(yearmean_obs,yearmean_model[exp])
# 			nmb=NMB(yearmean_obs,yearmean_model[exp])
# 			nme=NME(yearmean_obs,yearmean_model[exp])
# 			rmse=RMSE(yearmean_obs,yearmean_model[exp])
# 			r=pearsonr(yearmean_obs,yearmean_model[exp])
# 			r_all=pearsonr(all_obs,all_model)
# 			mfb_all=MFB(all_obs,all_model)
# 			mfe_all=MFE(all_obs,all_model)
# 			nmb_all=NMB(all_obs,all_model)
# 			nme_all=NME(all_obs,all_model)
# 			rmse_all=RMSE(all_obs,all_model)
# 			ymax=xmax
# 			ymin=xmin
# 			#ax.plot([xmin,xmax],[ymin,ymax])
# 			#ax.plot([xmin,xmax],[10*ymin,10*ymax],'g--')
# 			#ax.plot([xmin,xmax],[0.1*ymin,0.10*ymax],'g--')
# 			#ax.set_ylim([0.9*ymin,1.1*ymax])
# 			#ax.set_xlim([0.9*xmin,1.1*xmax])
# 			ax.set_ylim([5E-3,5e0])
# 			ax.set_xlim([5E-3,5e0])
# 			ax.set_xlabel('IMPROVE OM[pm25 1.8*C][ug m-3]')
# 			ax.set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
# 			ax.set_ylabel('TM5 OM[pm25][ug m-3]')
# 			ax2[k].set_ylim([5E-3,5e0])
# 			ax2[k].set_xlim([5E-3,5e0])
# 			ax2[k].set_xlabel('IMPROVE OM[pm25 1.8*C][ug m-3]')
# 			ax2[k].set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
# 			ax2[k].set_title('Run: '+exp)
# 			#if exp=='soa-riccobono':
# 			if exp=='newsoa-ri':
# 				ax2[k].set_title('NEWSOA')
# 			elif exp=='oldsoa-final':
# 				ax2[k].set_title('OLDSOA')
# 			else:
# 				ax2[k].set_title('NOSOA')
# 			if k<2:
# 				ax2b[k].set_ylim([5E-3,5e0])
# 				ax2b[k].set_xlim([5E-3,5e0])
# 				ax2b[k].set_xlabel('IMPROVE OM[pm25 1.8*C][ug m-3]')
# 				ax2b[k].set_ylabel('TM5 OM[pm25][ug m-3]')
# 				ax2b[k].set_title('Run: '+exp)
# 				#if exp=='soa-riccobono':
# 				if exp=='newsoa-ri':
# 					ax2b[k].set_title('NEWSOA')
	
# 				elif exp=='oldsoa-final':
# 					ax2b[k].set_title('OLDSOA')
# 			#print yearmean_obs
# 			print exp,'rmse: ',rmse,rmse_all
# 			print exp,'mfb: ',mfb,mfb_all
# 			print exp,'mfe: ',mfe,mfe_all
# 			print exp,'nmb: ',nmb,nmb_all
# 			print exp,'nme: ',nme,nme_all
# 			ax.plot([0.001,1000],[0.001,1000])
# 			ax.plot([0.001,1000],[0.01,10000],'g--')
# 			ax.plot([0.01,10000],[0.001,1000],'g--')
# 			ax2[k].plot([0.001,1000],[0.001,1000])
# 			ax2[k].plot([0.001,1000],[0.01,10000],'g--')
# 			ax2[k].plot([0.01,10000],[0.001,1000],'g--')
# 			ax2[k].annotate(('MFB: %6.2f')%mfb,xy=(0.05,0.9),xycoords='axes fraction')
# 			ax2[k].annotate(('MFE: %6.2f')%mfe,xy=(0.05,0.85),xycoords='axes fraction')
# 			ax2[k].annotate(('NMB: %6.2f')%nmb,xy=(0.05,0.8),xycoords='axes fraction')
# 			ax2[k].annotate(('NME: %6.2f')%nme,xy=(0.05,0.75),xycoords='axes fraction')
# 			ax2[k].annotate(('RMSE: %6.2f')%rmse,xy=(0.05,0.7),xycoords='axes fraction')
# 			ax2[k].annotate(('R: %6.2f')%r[0],xy=(0.05,0.65),xycoords='axes fraction')
# 			axb[k].annotate(('MFB: %6.2f')%mfb_all,xy=(0.05,0.95),xycoords='axes fraction')
# 			axb[k].annotate(('MFE: %6.2f')%mfe_all,xy=(0.05,0.9),xycoords='axes fraction')
# 			axb[k].annotate(('NMB: %6.2f')%nmb_all,xy=(0.05,0.85),xycoords='axes fraction')
# 			axb[k].annotate(('NME: %6.2f')%nme_all,xy=(0.05,0.8),xycoords='axes fraction')
# 			axb[k].annotate(('RMSE: %6.2f')%rmse_all,xy=(0.05,0.75),xycoords='axes fraction')
# 			axb[k].annotate(('R: %6.2f')%r_all[0],xy=(0.05,0.7),xycoords='axes fraction')
			
# 			if k<2:
# 				ax2b[k].plot([0.001,1000],[0.001,1000])
# 				ax2b[k].plot([0.001,1000],[0.01,10000],'g--')
# 				ax2b[k].plot([0.01,10000],[0.001,1000],'g--')
# 				ax2b[k].annotate(('MFB: %6.2f')%mfb,xy=(0.05,0.9),xycoords='axes fraction')
# 				ax2b[k].annotate(('MFE: %6.2f')%mfe,xy=(0.05,0.85),xycoords='axes fraction')
# 				ax2b[k].annotate(('NMB: %6.2f')%nmb,xy=(0.05,0.8),xycoords='axes fraction')
# 				ax2b[k].annotate(('NME: %6.2f')%nme,xy=(0.05,0.75),xycoords='axes fraction')
# 				ax2b[k].annotate(('RMSE: %6.2f')%rmse,xy=(0.05,0.7),xycoords='axes fraction')
# 				ax2b[k].annotate(('R: %6.2f')%r[0],xy=(0.05,0.65),xycoords='axes fraction')
# 		print 	
# 	f2b.savefig(output_png_path+'scatter-IMPROVE-1x2.png',dpi=400)
# 	f2b.savefig(output_jpg_path+'scatter-IMPROVE-1x2.jpg',dpi=400)
# 	f2b.savefig(output_pdf_path+'scatter-IMPROVE-1x2.pdf',dpi=400)
# 	f2.savefig(output_pdf_path+'scatter-IMPROVE-1x3.pdf',dpi=400)
# 	f2.savefig(output_png_path+'scatter-IMPROVE-1x3.png',dpi=400)
# 	f2.savefig(output_jpg_path+'scatter-IMPROVE-1x3.jpg',dpi=400)
# 	fb.savefig(output_pdf_path+'scatter-all-IMPROVE-1x3.pdf',dpi=400)
# 	fb.savefig(output_png_path+'scatter-all-IMPROVE-1x3.png',dpi=400)
# 	fb.savefig(output_jpg_path+'scatter-all-IMPROVE-1x3.jpg',dpi=400)
# 	#plt.show()	

# 	fmean,amean=plt.subplots(ncols=2,nrows=2,figsize=(12,8))
# 	colors=['red', 'blue','black']
# 	shadingcolors=['#ff000033', '#0000ff33','#00ff0033','#55555533']
# 	#amean[0,0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)+std,color=shadingcolors[2])
# 	#amean[0,0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)-std,color=shadingcolors[2])
# 	#amean[0,0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0),color=colors[2])
# 	amean[0,0].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0), yerr=[std, std], fmt='--o',color=shadingcolors[3],label='observations')
# 	amean[0,0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
# 	amean[0,0].set_ylim([0,2.0])

# 	amean[0,1].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0), yerr=[std, std], fmt='--o',color=shadingcolors[3],label='observations')
# 		#amean[2].set_title('Observations')
# 	amean[0,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
# 	amean[0,1].set_ylim([0,2.0])
	
# 	for n,exp in enumerate(EXPS[:],0):	
# 		if n==0:
# 			amean[0,n].set_title('NEWSOA')
# 			colori='r'
# 		elif n==1:
# 			amean[0,n].set_title('OLDSOA')
# 			colori='b'
# 		print n, exp
# 		std=np.nanstd(meanmodelmon[exp],axis=0)
# 		maxi=np.nanmax(meanmodelmon[exp],axis=0)
# 		mini=np.nanmin(meanmodelmon[exp],axis=0)
# 		#amean.fill_between(np.linspace(0,12,12), mini, maxi, facecolor=shadingcolors[n], alpha=0.3,interpolate=True)
# 		#amean.plot(np.linspace(0,12,12), mini, color=shadingcolors[n])
# 		#amean.plot(np.linspace(0,12,12), maxi, color=shadingcolors[n])
# 		amean[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)+std, color=shadingcolors[n])
# 		amean[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)-std, color=shadingcolors[n])
# 		amean[0,n].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon[exp],axis=0),color=colors[n],label=labeli)
# 		amean[0,n].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon_pom[exp],axis=0),color=colors[n],ls='--',label=labeli)
# 		amean[0,n].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
# 		if n==0:
# 			amean[0,n].set_ylim([0,1.0])
# 		else:
# 			amean[0,n].set_ylim([0,2.0])
# 		#amean[n].set_title(exp)
# 		nmbmean=NMB(np.nanmean(meanmodelmon['obs'],axis=0),np.nanmean(meanmodelmon[exp],axis=0))
# 		rmean=pearsonr(np.nanmean(meanmodelmon[exp],axis=0),np.nanmean(meanmodelmon['obs'],axis=0))
# 		amean[0,n].annotate(('NMB: %6.2f')%nmbmean,xy=(0.2,0.8),xycoords='axes fraction',fontsize=16)
# 		amean[0,n].annotate(('R: %6.2f,%6.2e')%(rmean[0],rmean[1]),xy=(0.2,0.7),xycoords='axes fraction',fontsize=16)
# 		amean[0,n].set_xlabel('Month')
# 		#amean[0,n].set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
# 		amean[0,n].set_ylabel('OM[pm25][ug m-3]')
# 		#ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
# 		print np.nanmean(meanmodelmon[exp],axis=0)
# 		print 'teststste',n,exp
# 		amean[1,n].loglog(yearmean_obs,yearmean_model[exp],'o',c=colori,ms=3)
# 		amean[1,n].plot([0.0001,1000],[0.0001,1000],'k')
# 		amean[1,n].plot([0.0001,1000],[0.001,10000],'k--')
# 		amean[1,n].plot([0.001,10000],[0.0001,1000],'k--')
# 		amean[1,n].set_ylim([.5e-2,3e0])
# 		amean[1,n].set_xlim([.5e-2,3e0])
# 		amean[1,n].set_xlabel('IMPROVE OM[pm25][ug m-3]')
# 		#amean[0,n].set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
# 		amean[1,n].set_ylabel('TM5 OM[pm25][ug m-3]')
# 		amean[1,n].set_aspect('equal')			

# 	std=np.nanstd(meanmodelmon['obs'],axis=0)
# 	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
# 	mini=np.nanmin(meanmodelmon['obs'],axis=0)
# 	#amean.fill_between(np.linspace(0,12,12), mini, maxi,facecolor=shadingcolors[3], alpha=0.3,interpolate=True)
# 	#amean.plot(np.linspace(0,12,12), mini,color=shadingcolors[3])
# 	#amean.plot(np.linspace(0,12,12), maxi,color=shadingcolors[3])
# 	fmean.suptitle('IMPROVE')
# 	#amean.set_yscale("log", nonposy='clip')
# 	fmean.savefig(output_png_path+'/IMPROVE/scatter-seasonal-IMPROVE-2x2.png',dpi=400)
# 	fmean.savefig(output_pdf_path+'/IMPROVE/scatter-seasonal-IMPROVE-2x2.pdf')
# 	fmean.savefig(output_jpg_path+'/IMPROVE/scatter-seasonal-IMPROVE-2x2.jpg',dpi=400)
# 	plt.show()


	
# if __name__=='__main__':
# 	main()
# plt.figure(figsize=(10,2))
# gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1],)
# ax0 = plt.subplot(gs[0, 0])
# ax1 = plt.subplot(gs[0, 1])
# #plt.plot(sitedata[i][6][0],sitedata[i][6][1][:,0])
# ax0.plot(sitedata[i][6][0],sitedata[i][6][1][:,1])
# #print len(sitedata[i][6][0]),len(model)
# ax0.plot(sitedata[i][6][0],model)
# print dt_axis
# ax0.plot(dt_ax2,gpdata*1e9)
# ax1.loglog(sitedata[i][6][1][:,1],model,'o')
# ax0.set_title((i+' Bias: %6.2f, Error: %6.2f')%(MFBm,MFEm))
# #plt.plot(sitedata['AREN1'][6][0],sitedata['AREN1'][6][1][:,0])
# #plt.plot(sitedata['AREN1'][6][0],sitedata['AREN1'][6][1][:,1])
# xmax=1.1*max([max(model),max(obs)])
# xmin=0.9*min([min(model),min(obs)])

# ymax=xmax
# ymin=xmin
# #plt.plot([xmin,xmax],[ymin,ymax])
# #plt.plot([xmin,xmax],[10*ymin,10*ymax],'--')
# #plt.plot([xmin,xmax],[0.1*ymin,0.10*ymax],'--')
# ax1.set_ylim([0.9*ymin,1.1*ymax])
# ax1.set_xlim([0.9*xmin,1.1*xmax])

# if NN>20:
# 	plt.show()
# 	NN=0

