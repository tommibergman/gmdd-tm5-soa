from scipy.special import erf
import matplotlib.pyplot as plt 
import nappy
import datetime
import glob
import netCDF4 as nc
import re
from lonlat import lonlat
from mpl_toolkits.basemap import Basemap
from general_toolbox import write_netcdf_file
import os

from  cdo import *
from scipy.stats import pearsonr

import numpy as np
from settings import *
def list_stations(indict):
	tabletex=open('../paper/sitelist_emep.tex','w')
	tabletex.write("%Name & Station code&Longitude& Latitude &Height\\\\\n")
	sortdict={}
	for i in sorted(indict.keys()):
		print i,indict[i]
		#print len(indict[i][0])
		print ('%6s,%35s,%4s,%4s,%4s')%(i,indict[i]['name'],indict[i]['lon'],indict[i]['lat'],indict[i]['ele'])
		stationname=indict[i]['name']
		print stationname
		stationname=stationname.replace("#","\\#")
		print stationname
		sortdict[stationname]=[i,indict[i]['lon'],indict[i]['lat'],indict[i]['ele']]
	for j in sorted(sortdict):
		print ('%-35s,%6s,%4s,%4s,%4s')%(j,sortdict[j][0],sortdict[j][1],sortdict[j][2],sortdict[j][3])
		tabletex.write("%s&%6s&%4s&%4s&%4s\\\\\n"%(j,sortdict[j][0],sortdict[j][1],sortdict[j][2],sortdict[j][3]))
	tabletex.close()

def str_months():
	return ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
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
			denom+=o
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
def parse_nas_normalcomments(comments):
	for comm in comments:
		#print comm
		if 'Station' in comm:
			if 'longitude' in comm:
				lon=float(comm.split(':')[1])
			elif 'latitude' in comm:
				lat=float(comm.split(':')[1])
			elif 'name' in comm:
				stationname=comm.split(':')[1].strip()
			elif 'code' in comm:
				site=comm.split(':')[1].strip()
			elif 'altitude' in comm:
				elev=comm.split(':')[1].strip()
				
		if 'Startdate' in comm:

			sdate=comm.split(':')[1].strip()
		if 'Matrix' in comm:
			pm=comm.split(':')[1].strip()
		if 'Unit' in comm:
			#if comm.split(':')[1].strip()=='ug/m3':
			#	unit=''
			#else:
			unit= comm.split(':')[1].strip()
	return lon,lat,stationname,site,elev,sdate,pm,unit
def emep(filein='/Users/bergmant/Documents/obs-data/Ebas_170727_1044_EMEP/*.nas'):
	# reader for NASA Ames format used in EUSAAR
	# full description http://www.eusaar.net/files/data/nasaames/index.html
	sitedata={}
	sitedata2={}
	#for i in glob.iglob('/Users/bergmant/Documents/obs-data/EMEP-SOA/*.nas'):
	for i in glob.iglob(filein):
		print 'WARNING!!! Works only for files with reference date 2010-01-01 00:00'
		varindex=None
		#print i
		#f = nappy.openNAFile('/Users/bergmant/Documents/obs-data/EMEP-Ebas_170608_1156/SE0011R.20100101110000.20111018000000.lvs_denuder_tandem.organic_carbon.pm10.1y.1w.SE04L_LU_DRI-1_1.SE04L_EUSAAR-2..nas')
		f = nappy.openNAFile(i)
		n_lines=f.getNumHeaderLines()
		#print n_lines
		# Get Organisation from header
		org = f.getOrg()
		# Get the Normal Comments (SCOM) lines.
		norm_comms = f.getNormalComments()
		#print norm_comms
		lon,lat,stationname,site,elev,sdate,pm,unit=parse_nas_normalcomments(norm_comms)
		sitedict={}
		sitedict['name']=stationname
		sitedict['lat']=lat
		sitedict['lon']=lon
		sitedict['ele']=elev
		sitedict['unit']=unit
		sitedict['pm']=pm
		# analyse 2010 only
		if sdate[:4]!='2010':
			continue
		print stationname,lon,lat,pm,unit
		# Get the Special Comments (SCOM) lines.
		###???###
		spec_comms = f.getSpecialComments()

		varindex=-1
		# Get a list of metadata for all main (non-auxiliary or independent) variables
		var_list = f.getVariables()
		possible_vars=['organic_carbon']
		for i in var_list:
			#print i[0]
			if 'starttime'in i[0]:
				starttime_i=var_list.index(i)
			if 'endtime'in i[0]:
				endtime_i=var_list.index(i)
			if 'organic_carbon' in i[0]:# and 'arithmetic mean' in i[0] and 'Artifact' not in i[0] and 'numflag' not in i[0] and 'Fraction' not in i[0]:
				#print i, var_list.index(i)
				if varindex==-1:
					varindex=var_list.index(i)
			if 'organic_carbon' in i[0] and 'arithmetic mean' in i[0] and 'Artifact' not in i[0] and 'numflag' not in i[0] and 'Fraction' not in i[0]:
				#print i, var_list.index(i)
				if varindex==-1:
					varindex=var_list.index(i)

		#sitedict['startdate']=starttime
		#sitedict['enddate']=endtime		

		if varindex==-1:
			continue
			#raw_input()
		# Get Auxiliary variable metadata for auxiliary variable number 2
		#(variable, units, miss, scale) = f.getAuxVariable(2)

		# Get scale factor for primary variable number 3
		#scale_factor = f.getScaleFactor(3)

		# Get missing value for primary variable number 1
		#missing = f.getMissingValue(1)

		# Let's get the contents dictionary of the whole file
		#print sdate
		sd = datetime.datetime.strptime(sdate, '%Y%m%d%H%M%S')
		#print sd

		a=f.getNADict()
		#print 'tasta'
		data=f.readData()
		#print f.getIndependentVariables()
		#print f.getAuxVariables()
		#print f.getVariable(-1)
		#print data
		#print f["V"]
		#print 'a',f["A"]
		#print 'x',f["X"]

		#print f["V"][varindex]
		#print f["V"][varindex+1]
		stime=[]
		etime=[]
		sdate=[]
		edate=[]
		stamp=[]
		#for j,k in zip(f["V"][0],f["X"]):
			#print j,k#sdatetime.datetime.strptime(j, '%d%H%M%S')
		#	stime.append(k)
		#	etime.append(j)

		# X = "time", first row in data
		# V = the rest, readable with index

		OC=[]
		#loop over datapoints
		for j,k,data in zip(f["V"][0],f["X"],f["V"][varindex]):
			#print data
			if float(data)<90:
				stime.append(k)
				etime.append(j)
				sdate.append(k)
				edate.append(j)
				stamp.append(nc.num2date((j+k)/2,units='days since 2010-01-01 00:00:00',calendar='standard'))
				#print 'ff',float(data)
				OC.append(float(data))
			# do not save erroenous or non available data even as nan
			else:	
				pass
				#stime.append(k)
				#etime.append(j)
				#OC.append(np.nan)
		if len(OC)>0:
			sitedata2[site]=sitedict
			sitedata2[site]['oc']=np.array(OC)
			sitedata2[site]['etime']=np.array(etime)
			sitedata2[site]['stime']=np.array(stime)
			sitedata2[site]['edate']=np.array(edate)
			sitedata2[site]['sdate']=np.array(sdate)
			print sdate
			sitedata2[site]['time']=stamp

			sitedata[site]=[stime,etime,OC,pm,unit,stationname,lon,lat,stamp]
		else:
			sitedata.pop(site,None)
	return sitedata,sitedata2

def lonlat_index(mlon,mlat):
	lon,lat=lonlat('TM53x2')
	lonidx = (np.abs(lon-mlon)).argmin()
	latidx = (np.abs(lat-mlat)).argmin()
	return lonidx,latidx	
def select_gp(data,mlat,mlon):
	lon,lat=lonlat('TM53x2')
	#lonidx = (np.abs(lon-mlon)).argmin()
	#latidx = (np.abs(lat-mlat)).argmin()
	lonidx,latidx=lonlat_index(mlon,mlat)
	#print 'gp',np.shape(data)
	if len(np.shape(data))==4:      
		return np.squeeze(data[:,:,latidx,lonidx]),lat[latidx],lon[lonidx]
	elif len(np.shape(data))==3:
		return np.squeeze(data[:,latidx,lonidx]),lat[latidx],lon[lonidx]
	else:
		print 'Data not compatible to select gridpoint timeseries'
		return None,None,None
def select_gp_cdo(data,mlat,mlon):
	lon,lat=lonlat('TM53x2')
	lonidx = (np.abs(lon-mlon)).argmin()
	latidx = (np.abs(lat-mlat)).argmin()
	print 'gp',np.shape(data)
	if len(np.shape(data))==4:      
		return np.squeeze(data[:,:,latidx,lonidx]),lat[latidx],lon[lonidx]
	elif len(np.shape(data))==3:
		return np.squeeze(data[:,latidx,lonidx]),lat[latidx],lon[lonidx]
	else:
		print 'Data not compatible to select gridpoint timeseries'
		return None,None,None
		
def calc_dens(inputdata):
	data=nc.Dataset(inputdata,'r')
	modes=['NUS','AIS','ACS','COS']
	comps=['SOA','POM','SO4','BC','DU','SS']
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
		indexi='aerh2o3d_'+m
		mass+=data.variables[indexi][:]
		roo+=data.variables[indexi][:]*1.0 #denisty of water
		rindex='RWET_'+m
		dens=mass/(4/3*np.pi*(data.variables[rindex][:]**3)*data.variables['N_'+m][:])
		roo=roo/mass
		roo_a[m]=roo
		print m, roo
		#raw_input()
	return roo_a
def read_mass(comp,inputdata,rad=1.25):
	if rad < 1.0:
		print 'WARNING: Mass check only for ACS and COS at the moment!!!'
	data=nc.Dataset(inputdata,'r')
	outputdataset=None
	roo_a=calc_dens(inputdata)
	# select mass below threshold
	#ACS
	print 'radii read'	 	
	print inputdata, data.variables
	d_a_correction_factor={}
	for m in roo_a:
		d_a_correction_factor[m]=np.sqrt(roo_a[m]/1.0) #sqrt(roo_p/roo_0), roo_0=1.00g/cm3 (Water)
	rACS=data.variables['RWET_ACS'][:]*d_a_correction_factor['ACS']
	rCOS=data.variables['RWET_COS'][:]*d_a_correction_factor['COS']
	
	print 'modfrac_acs'
	cmedr2mmedr= np.exp(3.0*(np.log(1.59)**2))	
	hr2=(0.5*np.sqrt(2.0))
	
	rad=1.25
	print 'modfrac_acs z'
	z=(np.log(rad)-np.log(rACS*1e6*cmedr2mmedr)/np.log(1.59))
	print 'modfrac_acs mf'
	modfrac_acs1=0.5+0.5*erf(z*hr2)
	rad=5.0
	print 'modfrac_acs z'
	z=(np.log(rad)-np.log(rACS*1e6*cmedr2mmedr)/np.log(1.59))
	print 'modfrac_acs mf'
	modfrac_acs2=0.5+0.5*erf(z*hr2)
	#free memory
	del z
	print 'modfrac_cos'
	#COS
	cmedr2mmedr= np.exp(3.0*(np.log(2.00)**2))	

	rad=1.25
	z=(np.log(rad)-np.log(rCOS*1e6*cmedr2mmedr)/np.log(2.00))
	modfrac_cos1=0.5+0.5*erf(z*hr2)
	rad=5.0
	z=(np.log(rad)-np.log(rCOS*1e6*cmedr2mmedr)/np.log(2.00))
	modfrac_cos2=0.5+0.5*erf(z*hr2)
	#free memory

	del z
	for i in data.variables:
		if comp in i:
			print comp,i
			if outputdataset==None:
				outputdata=np.zeros(np.shape(data.variables[i][:]))
				outputdata2=np.zeros(np.shape(data.variables[i][:]))
				outputdataset=1
			if i[-3:]=='COS':
				temp=data.variables[i][:]
				outputdata+=temp*modfrac_cos1 
				outputdata2+=temp*modfrac_cos2 
			elif i[-3:]=='ACS':
				temp=data.variables[i][:]
				outputdata+=temp*modfrac_acs1
				outputdata2+=temp*modfrac_acs2
			else:	
				temp=data.variables[i][:]
				outputdata+=temp
				outputdata2+=temp
	return outputdata,outputdata2
# def read_mass(comp,inputdata):
# 	data=nc.Dataset(inputdata,'r')
# 	outputdataset=None
# 	for i in data.variables:
# 		if comp in i:
# 			print comp,i
# 			if outputdataset==None:

# 				outputdata=np.zeros(np.shape(data.variables[i][:]))
# 				outpudataset=1
# 			print np.shape(data.variables[i][:])
# 			outputdata+=data.variables[i][:] 

# 	return outputdata

def monmean(timedata,data):
	outdata=np.zeros((12))
	outstd=np.zeros((12))
	for i in range(12):
		if i <11:
			mask=datetime.datetime(2010,i+1,1,0,0)<=timedata<datetime.datetime(2010,i+2,1,0,0)
		else:
			mask=datetime.datetime(2010,i+1,1,0,0)<=timedata<datetime.datetime(2011,1,1,0,0)
		
		outdata[i]=np.mean(data[mask])
		outstd[i]=np.std(data[mask])
	return outdata,outstd

def colocate_emep(modeldata25,modeldata10,timeaxis,sdata,exp):
	model=[]
	obs=[]
	mlats=[]
	mlons=[]
	names=[]
	model_dict={}
	ddd=0
	model_mean=[]
	model_std=[]
	times=[]
	print np.shape(modeldata25)
	for site in sdata:
		#Do not colocate again
		if os.path.isfile(basepath+'output/processed/emep_col/'+exp+'_'+site+'.nc'):
			print 'site already collocated'
			continue
		if not os.path.isdir(basepath+'output/processed/emep_col/'):
			os.mkdir(basepath+'output/processed/emep_col/')
		yearmean_model=[]
		yearmean_obs=[]
		time=[]
		ddd+=1
		if sdata[site][3]=='pm25':
			modeldata=modeldata25
		elif sdata[site][3]=='pm10':
			modeldata=modeldata10
		else:
			print 'choice of pm not found: '+sdata[site][3],site
			
		# f,ax=plt.subplots(2)
		print 'processing ',site,sdata[site][5]
		model_dict[site]={}
		stime=sdata[site][0]
		etime=sdata[site][1]
		OC=sdata[site][2]
		pm=sdata[site][3]
		lat=sdata[site][7]
		lon=sdata[site][6]
		model_dict[site]['name']=sdata[site][5]
		model_dict[site]['lon']=lon
		model_dict[site]['lat']=lat

		if sdata[site][4]=='ug C/m3':
			factor=1.4
			print 'site ',site
		else:
			print 'factor1 site ',site
			factor =1.0

		#print np.shape(modeldata)
		gpdata,mlat,mlon=select_gp(modeldata,lat,lon)
		#print np.shape(gpdata)
		mlats.append(mlat)
		mlons.append(mlon)
		a=0
		model1_mean=[]
		model1_std=[]
		time1=[]
		if len(stime)<1:
			print 'nodata'
			continue
		for i in range(len(stime)):
			#obsstep_mean=[]
			#obsstep_std=[]
			N=0
			temp=[]
			#check for not going out of time eaxis
			if a>(np.size(timeaxis)-1):
				continue
			#print '1',np.size(timeaxis),a
			# increase a until model data corresponds to obsdata
			#print '2',timeaxis[a],stime[i]
			while(timeaxis[a]<stime[i]):
				a+=1
				if  a>(np.size(timeaxis)-1):
					break
			#check that we are still within model data
			if a>(np.size(timeaxis)-1):
				continue
			#print timeaxis
			#print '3',np.size(timeaxis),a,timeaxis[a],etime[i]
			# amodel_mean data until model time exceeds current 
			# obs period (etime)
			while(timeaxis[a]>=stime[i] and timeaxis[a]<etime[i]):
				#print 'range',stime[i],etime[i],timeaxis[a],N
				#print '4',stime[i],timeaxis[a],etime[i],OC[i],gpdata[a]
				if OC[i]<90:
					N+=1
					temp.append(gpdata[a])
					a+=1
					if a==np.size(timeaxis):
						break
				else:
					a+=1
					#continue
				# again check taht we are within model data
				if  a>(np.size(timeaxis)-1):
					break
			if N>0:
				obsstep_mean=np.mean(temp)*1e9
				obsstep_std=np.std(temp)*1e9
				time1=(timeaxis[a-N]+timeaxis[a-1])/2

			else:
				print 'ERROR: '+site+' no data points found in modeldata, '
				print 'times',stime[i],etime[i],timeaxis[a],
				continue	
				#model_mean.append(np.nan)
			#print model_mean
			#print N
			#print timeaxis[a-1]
			#print stime[i],etime[i]
			#raw_input()
			model1_mean.append(obsstep_mean)
			model1_std.append(obsstep_std)
			#print model_mean
			#print N
			#print time1
			
			time.append(nc.num2date(time1,'days since 2010-01-01 00:00:00',calendar='standard'))

		print time
		#raw_input()

	
		times.append(time)
		model_mean.append(model1_mean)
		model_std.append(model1_std)
		print site,len(time),ddd
		names.append(site)
		model_dict[site]['N_OM']=len(model1_mean)
		model_dict[site]['OM']=np.array(model1_mean) # list of daily means
		#print model_std,
		model_dict[site]['OM_std']=np.array(model1_std) #list of daily std
		model_dict[site]['OM_ymean']=np.mean(model1_mean) # yearly mean
		model_dict[site]['OM_ymean_std']=np.std(model1_mean) #std of year
		model_dict[site]['time']=np.array(times) #timestamps for daily means
		model_dict[site]['unit']='ug/m3'
		yearmean_model.append(np.mean(model1_mean))
		yearmean_obs.append(np.mean(sdata[site][2]))
	print len(names), len(model_mean),len(times)
	#raw_input()
	#print len(model_mean),'model_mean'
	#print model_mean
	#print site,len(OC),len(stime),len(model_dict[site]['OM']),len(model_mean)
	#raw_input()
	return model_mean,names,times,model_dict

def scatter_exp(o,m,ax=None):
	obs=[]
	for i in sdata:
		obs.append(sdata[i][2])
	oo=[]
	mm1=[]
	oall=[]
	mall1=[]
	
	print len(obs),len(model)
	print obs
	#print model
	if len(obs)>0:
		for i in sdata:
			o=sdata[i][2]
			m1=mdict_exp[0][i]['OM']
			RMSEemep1=RMSE(o,m1)
			MFBemep1=MFB(o,m1)
			MFEemep1=MFE(o,m1)
			NMBemep1=NMB(o,m1)
			NMEemep1=NME(o,m1)
			oo.append(np.mean(np.array(o)))
			mm1.append(np.mean(np.array(m1)))
			if sdata[i][4]=='ug C/m3':
				factor=1.4
				print 'site ',i
			else:
				print 'factor1 site ',i
				factor =1.0
			if sdata[i][6]=='pm25':
				mark='o'
			elif sdata[i][6]=='pm10' :
				mark ='+'
			ax.loglog(np.array(o*factor)*factor,np.array(m1),'b',marker=mark,ms=3)
			ax.loglog(np.array(o*factor)*factor,np.array(m2),'r',marker=mark,ms=3)
			xmax=max([max(o),max(m1)])
			xmin=min([min(o),min(m1)])
			ax.annotate(('MFB: %6.2f')%MFBemep,xy=(.1,0.9),xycoords='axes fraction')
			ax.annotate(('MFE: %6.2f')%MFEemep,xy=(.1,0.85),xycoords='axes fraction')
			ax.annotate(('NMB: %6.2f')%NMBemep,xy=(.1,0.8),xycoords='axes fraction')
			ax.annotate(('NME: %6.2f')%NMEemep,xy=(.1,0.75),xycoords='axes fraction')
			ax.annotate(('RMSE: %6.2f')%RMSEemep,xy=(.1,0.7),xycoords='axes fraction')
			ax.annotate(('MFB: %6.2f')%MFBemep2,xy=(.1,0.65),xycoords='axes fraction')
			ax.annotate(('MFE: %6.2f')%MFEemep2,xy=(.1,0.6),xycoords='axes fraction')
			ax.annotate(('NMB: %6.2f')%NMBemep2,xy=(.1,0.55),xycoords='axes fraction')
			ax.annotate(('NME: %6.2f')%NMEemep2,xy=(.1,0.5),xycoords='axes fraction')
			ax.annotate(('RMSE: %6.2f')%RMSEemep2,xy=(.1,0.45),xycoords='axes fraction')
			ax.loglog([0.001,1000],[0.001,1000],'k-')
			ax.loglog([0.1,1000],[0.01,100],'r--')
			ax.loglog([0.01,100],[0.1,1000],'r--')
			ax.set_xlim([0.01,1000])
			ax.set_ylim([0.01,1000])
			for ok,mk1,mk2 in zip(o,m1,m2):
				oall.append(ok*factor)
				mall1.append(mk1)
				mall2.append(mk2)

def main():
	lon,lat=lonlat('TM53x2')
	sdata,sitedata=emep()
	for i in sdata:
		print sdata[i][0],sdata[i][1],sdata[i][-1]
		print sdata[i][-1][0].month
	#raw_input()
	list_stations(sitedata)
	#raw_input()
	org10=[]
	org25=[]
	org10d={}
	org25d={}
	pom10=[]
	pom25=[]
	pom10d={}
	pom25d={}
	timeaxis_day=[]
	EXPS=['soa-riccobono','oldsoa-bhn','nosoa']
	EXPS=['newsoa-ri','oldsoa-bhn','nosoa']
	EXPS=['newsoa-ri','oldsoa-bhn']#,'nosoa']

	#for i in range(3):
	#	exp=EXPS[i]
	i=0
	model_exp=[]
	names_exp=[]
	time_exp=[]
	mdict_exp=[]
	mdictpom_exp=[]
	exp_dict={}
	for exp in EXPS:
		exp_dict[exp]={}
	basepathraw='/Users/bergmant/Documents/tm5-SOA/output/raw/'

	for exp in EXPS:
		#if len(glob.glob(basepath+'emep_col/'+exp+'*'))==0:
		#read data if already processed
		if os.path.isfile('/Volumes/Utrecht/'+exp+'/'+exp+'_org101.nc') and os.path.isfile('/Volumes/Utrecht/'+exp+'/'+exp+'_org251.nc'):
			org10.append(nc.Dataset('/Volumes/Utrecht/'+exp+'/'+exp+'_org10.nc','r').variables['org10'][:])
			org25.append(nc.Dataset('/Volumes/Utrecht/'+exp+'/'+exp+'_org25.nc','r').variables['org25'][:])
			print '/Volumes/Utrecht/'+exp+'/'+exp+'_poa10.nc'
			pom10.append(nc.Dataset('/Volumes/Utrecht/'+exp+'/'+exp+'_poa10.nc','r').variables['poa10'][:])
			pom25.append(nc.Dataset('/Volumes/Utrecht/'+exp+'/'+exp+'_poa25.nc','r').variables['poa25'][:])
			# put just added data to dictionary
			org10d[exp]=org10[-1].copy()
			org25d[exp]=org25[-1].copy()
			pom10d[exp]=pom10[-1].copy()
			pom25d[exp]=pom25[-1].copy()
			#time
			timeaxis=nc.Dataset('/Volumes/Utrecht/'+exp+'/general_TM5_'+exp+'_2010.lev1.nc','r').variables['time'][:]
			timeaxis_day.append(timeaxis-np.floor(timeaxis[0]))
			print 'load',timeaxis-np.floor(timeaxis[0]),timeaxis,np.floor(timeaxis[0])
			print np.shape(timeaxis)
			#raw_input()

		else:
			# process data
			
			if  os.path.exists(basepathraw+exp+'/general_TM5_'+exp+'_2010.lev1.nc'):
				filepath=basepathraw+exp+'/general_TM5_'+exp+'_2010.lev1.nc'
			else:	
				filepath='/Volumes/Utrecht/'+exp+'/general_TM5_'+exp+'_2010.lev1.nc'			
			print 'read soa'

			soa25,soa10=read_mass('M_SOA','/Volumes/Utrecht/'+exp+'/general_TM5_'+exp+'_2010.lev1.nc')

			print 'read oc'
			poa25,poa10=read_mass('M_POM','/Volumes/Utrecht/'+exp+'/general_TM5_'+exp+'_2010.lev1.nc')
			print 'sum'
			org25.append(soa25+poa25)
			org10.append(soa10+poa10)
			pom25.append(poa25)
			pom10.append(poa10)
			print 'time'
			timeaxis=nc.Dataset('/Volumes/Utrecht/'+exp+'/general_TM5_'+exp+'_2010.lev1.nc','r').variables['time'][:]
			print timeaxis-np.floor(timeaxis[0]),timeaxis,np.floor(timeaxis[0])
			timeaxis_day.append(timeaxis-np.floor(timeaxis[0]))
			print np.ndim(org10[i]),np.shape(org10[i])
			#write_netcdf_file([np.squeeze(org10[i])],['org10'],'/Volumes/Utrecht/'+exp+'/'+exp+'_org10.nc',lat,lon,timeaxis)#,lat,lon)
			#write_netcdf_file([np.squeeze(org25[i])],['org25'],'/Volumes/Utrecht/'+exp+'/'+exp+'_org25.nc',lat,lon,timeaxis)#,lat,lon)
			
			#write_netcdf_file([np.squeeze(pom10[i])],['poa10'],'/Volumes/Utrecht/'+exp+'/'+exp+'_poa10.nc',lat,lon,timeaxis)#,lat,lon)
			#write_netcdf_file([np.squeeze(pom25[i])],['poa25'],'/Volumes/Utrecht/'+exp+'/'+exp+'_poa25.nc',lat,lon,timeaxis)#,lat,lon)

			del poa10,soa10,poa25,soa25	
		#print 'tim',timeaxis_day[0][1]
		#exp_dict['model']
		#for i in range(3):
		print  i,len(org25),len(org10),len(timeaxis_day),len(sdata)
		model,names,times,mdict=colocate_emep(org25[i],org10[i],timeaxis_day[i],sdata,exp)
		model_pom,names_pom,times_pom,mdictpom=colocate_emep(pom25[i],pom10[i],timeaxis_day[i],sdata,exp)
		print names
		model_exp.append(model)
		names_exp.append(names)
		time_exp.append(times)
		mdict_exp.append(mdict)
		mdictpom_exp.append(mdictpom)
		for ii in mdict:
			print ii
			for tt in mdict[ii]['time'][0]:
				print tt
				print tt.month
		#raw_input()
		print exp_dict
		print EXPS[i]
		#print EXPS[i]
		exp_dict[EXPS[i]]['model']=model
		exp_dict[EXPS[i]]['sitenames']=names
		exp_dict[EXPS[i]]['time']=times
		exp_dict[EXPS[i]]['mdict']=mdict
		exp_dict[EXPS[i]]['mdictpom']=mdictpom
		print len(model),len(names),len(times)
		for data,data_pom,name,time in zip(model,model_pom,names,times):
			#raw_input()
			print 'hep'
			print basepath+'output/processed/emep_col/'+exp+'_'+name+'.nc'
			print len(data),len(name),len(time)
			#print time
			if not os.path.isfile(basepath+'output/processed/emep_col/'+exp+'_'+name+'.nc'):
				print 'write data for ' + name 
				#write_netcdf_file([data],[name],basepath+'emep_col/'+exp+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)
				write_netcdf_file([data],[name],basepath+'output/processed/emep_col/'+exp+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)
				write_netcdf_file([data_pom],[name],basepath+'output/processed/emep_col/'+exp+'_'+name+'_poa.nc',None,None,np.array(time))#,lat,lon)
		i+=1
		
	# free memory
	del org10,org25
	
	#write to files
	#print monmean(np.array(time_exp[0][0]),np.array(model_exp[0][0]))	

	#print datetime.datetime(2010,2,1,0,0)<=mdict['CH0002R']['time']<datetime.datetime(2010,3,1,0,0)
	#print datetime.datetime(2010,2,1,0,0),mdict['CH0002R']['time'],datetime.datetime(2010,3,1,0,0)
	

	print model
	obs=[]
	for i in sdata:
		obs.append(sdata[i][2])
	oo=[]
	mm1=[]
	mm2=[]
	oall=[]
	mall1=[]
	mall2=[]
	print len(obs),len(model)
	print obs
	#print model
	obs_all=[]
	model_all={}
	yearmean_obs=[]
	yearmean_model={}
	yearmean_modelpom={}
	yearmean_pm25_obs=[]
	yearmean_pm25_model={}
	yearmean_pm25_modelpom={}
	yearmean_pm10_obs=[]
	yearmean_pm10_model={}
	yearmean_pm10_modelpom={}
	meanmodelmon={}
	meanmodelmonpom={}

	for exp in EXPS:
		meanmodelmon[exp]=np.zeros((len(sdata),12))
		meanmodelmonpom[exp]=np.zeros((len(sdata),12))
	meanmodelmon['obs']=np.zeros((len(sdata),12))
	kk=0
	#Loop over sites
	for i in sdata:
		fmon,amon=plt.subplots(1)
		# create monthly mean dataholders
		timedata=sdata[i][-1]		
		timedata=sitedata[i]['time']		
		monthdata=np.zeros([12])
		monthdata[:]=np.NAN
		monthindices={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[]}
		# find indices of data for different months
		for j in timedata:
			print j.month
			monthindices[j.month].append(timedata.index(j))
		# read monthly measurements
		for k in monthindices:
			print k,type(sdata[i][2])
			#print monthdata.shape,monthindices[k],(sdata[i][2][monthindices[k]]),np.mean(sdata[i][2][monthindices[k]])
			#monthdata[k-1]=np.mean(sdata[i][2][monthindices[k]])
			print monthdata.shape,monthindices[k],(sitedata[i]['oc'][monthindices[k]]),np.mean(sitedata[i]['oc'][monthindices[k]])
			monthdata[k-1]=np.mean(sitedata[i]['oc'][monthindices[k]])


		if sdata[i][4]=='ug C/m3':
			factor=1.8
			print 'f1.8 site ',i
		else:
			factor =1.0
			print 'f1 site ',i
		meanmodelmon['obs'][kk,:]=monthdata[:]*factor
		amon.plot(np.linspace(1,12,12),monthdata)

		print '----',meanmodelmon['obs'][kk,:]
		print '----',np.count_nonzero(~np.isnan(meanmodelmon['obs'][kk,:]))
		o=sdata[i][2]
		yearmean_obs.append(np.mean(o)*factor)
		print sdata[i][3]
		#raw_input()
		if sdata[i][3]=='pm25':
			yearmean_pm25_obs.append(np.mean(o)*factor)
		else:
			yearmean_pm10_obs.append(np.mean(o)*factor)

		
		for ok in o:
			obs_all.append(ok*factor)

		for exp in EXPS:
			modelmonthdata=np.zeros([12])
			modelmonthdata[:]=np.NAN
			modelmonthdatapom=np.zeros([12])
			modelmonthdatapom[:]=np.NAN
			print exp 
			model=nc.Dataset(basepath+'output/processed/emep_col/'+exp+'_'+i+'.nc','r').variables[i][:]
			modelpom=nc.Dataset(basepath+'output/processed/emep_col/'+exp+'_'+i+'_poa.nc','r').variables[i][:]
			if exp not in model_all.keys():
				model_all[exp]=model
			else:
				model_all[exp]=np.concatenate((model_all[exp],model))
				
			for kmod in monthindices:
				print kmod
				print model
				print monthdata.shape,monthindices[kmod],(model[monthindices[kmod]]),np.mean(model[monthindices[kmod]])
				modelmonthdata[kmod-1]=np.mean(model[monthindices[kmod]])
				modelmonthdatapom[kmod-1]=np.mean(modelpom[monthindices[kmod]])
			print 'hhhhhhhh',modelmonthdata
			#raw_input()
			meanmodelmon[exp][kk,:]=modelmonthdata
			meanmodelmonpom[exp][kk,:]=modelmonthdatapom
			if exp==EXPS[0]:
				X=0.05
				colori='r'
			elif exp==EXPS[1]:
				X=0.8
				colori='b'
			else:# exp==EXPS[1]:
				X=0.4
				colori='g'
			amon.plot(np.linspace(1,12,12),modelmonthdata,colori)		
			amon.plot(np.linspace(1,12,12),modelmonthdatapom,'--'+colori)		
			amon.set_title(i)
			print exp,i
			if exp not in yearmean_model.keys():
				yearmean_model[exp]=[]
				yearmean_model[exp].append(np.mean(model))
			else:
				yearmean_model[exp].append(np.mean(model))
			


			if sdata[i][3]=='pm25':
				if exp not in yearmean_pm25_model.keys():
					yearmean_pm25_model[exp]=[]
					yearmean_pm25_model[exp].append(np.mean(model))
				else:
					yearmean_pm25_model[exp].append(np.mean(model))
			else:
				if exp not in yearmean_pm10_model.keys():
					yearmean_pm10_model[exp]=[]
					yearmean_pm10_model[exp].append(np.mean(model))
				else:
					yearmean_pm10_model[exp].append(np.mean(model))
		fmon.savefig(output_pdf_path+'/siteplots/monthly-EMEP-'+i+'.pdf',dpi=400)
		fmon.savefig(output_png_path+'/siteplots/monthly-EMEP-'+i+'.png',dpi=400)
		fmon.savefig(output_jpg_path+'/siteplots/monthly-EMEP-'+i+'.jpg',dpi=400)	
		kk+=1
	for i in range(15):
		print np.count_nonzero(~np.isnan(meanmodelmon['obs'][i,:]))
		print (meanmodelmon['obs'][i,:])
		print '----',meanmodelmon['obs'][i,:]
		print '----',np.count_nonzero(~np.isnan(meanmodelmon['obs'][i,:]))
	#raw_input()
		
	fmean,amean=plt.subplots(ncols=3,figsize=(12,5))
	colors=['red','blue','green','black']
	shadingcolors=['#ff000033','#0000ff33', '#00ff0033','#99999966']
	for n,exp in enumerate(EXPS,0):	
		if exp=='newsoa-ri':
			amean[n].set_title('NEWSOA')
		elif exp=='oldsoa-bhn':
			amean[n].set_title('OLDSOA')
		print meanmodelmon[exp]
		print np.nanstd(meanmodelmon[exp],axis=0)
		
		std=np.nanstd(meanmodelmon[exp],axis=0)
		maxi=np.nanmax(meanmodelmon[exp],axis=0)
		mini=np.nanmin(meanmodelmon[exp],axis=0)
		#amean.fill_between(np.linspace(0,12,12), mini, maxi, facecolor=shadingcolors[n], alpha=0.3,interpolate=True)
		#amean.plot(np.linspace(0,12,12), mini, color=shadingcolors[n])
		#amean.plot(np.linspace(0,12,12), maxi, color=shadingcolors[n])
		amean[n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)+std, color=shadingcolors[n])
		amean[n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)-std, color=shadingcolors[n])
		amean[n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0),color=colors[n],label=exp)
		amean[n].set_ylim([0,7.5])
		#amean[n].set_title(exp)
		amean[n].legend()
		nmbmean=NMB(np.nanmean(meanmodelmon['obs'],axis=0),np.nanmean(meanmodelmon[exp],axis=0))
		rmean=pearsonr(np.nanmean(meanmodelmon[exp],axis=0),np.nanmean(meanmodelmon['obs'],axis=0))
		amean[n].annotate(('NMB: %6.2f')%nmbmean,xy=(0.2,0.8),xycoords='axes fraction',fontsize=16)
		amean[n].annotate(('R: %6.2f,%6.2e')%(rmean[0],rmean[1]),xy=(0.2,0.7),xycoords='axes fraction',fontsize=16)
		amean[n].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	
		#ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
		print np.nanmean(meanmodelmon[exp],axis=0)
	std=np.nanstd(meanmodelmon['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
	mini=np.nanmin(meanmodelmon['obs'],axis=0)
	#amean.fill_between(np.linspace(0,12,12), mini, maxi,facecolor=shadingcolors[3], alpha=0.3,interpolate=True)
	#amean.plot(np.linspace(0,12,12), mini,color=shadingcolors[3])
	#amean.plot(np.linspace(0,12,12), maxi,color=shadingcolors[3])
	amean[2].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)+std,color=shadingcolors[3])
	amean[2].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)-std,color=shadingcolors[3])
	amean[2].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0),color=colors[3],label='obs')
	amean[2].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean[2].set_ylim([0,7.5])
	amean[2].set_title('Observations')
	amean[2].legend()
	fmean.suptitle('EMEP')
	#amean.set_yscale("log", nonposy='clip')
	fmean.savefig(output_png_path+'EMEP//monthly-EMEP-allmean.png',dpi=400)

	fmean,amean=plt.subplots(nrows=2,ncols=2,figsize=(12,10))#,tight_layout=True)
	
	'''
	pos = ax3.get_position()
	print pos
	pos.x0 = 0.2+pos.x0       # for example 0.2, choose your value
	pos.x1 = 0.2+pos.x1       # for example 0.2, choose your value
	print pos
	ax3.set_position(pos)
	'''

	#colors=['red', 'green','blue','black']
	#shadingcolors=['#ff000033', '#00ff0033','#0000ff33','#55555533']
	amean[0,0].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	amean[0,0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean[0,0].set_ylim([0,7.5])
	amean[0,1].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	amean[0,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean[0,1].set_ylim([0,7.5])
	letters=[['a','b'],['c','d']]

	for n,exp in enumerate(EXPS[:],0):	
		if exp=='newsoa-ri':
			amean[0,n].set_title('NEWSOA')
			labeli='NEWSOA'
		elif exp=='oldsoa-bhn':
			amean[0,n].set_title('OLDSOA')
			labeli='OLDSOA'
		print meanmodelmon[exp]
		print np.nanstd(meanmodelmon[exp],axis=0)
		
		std=np.nanstd(meanmodelmon[exp],axis=0)
		maxi=np.nanmax(meanmodelmon[exp],axis=0)
		mini=np.nanmin(meanmodelmon[exp],axis=0)
		#amean.fill_between(np.linspace(0,12,12), mini, maxi, facecolor=shadingcolors[n], alpha=0.3,interpolate=True)
		#amean.plot(np.linspace(0,12,12), mini, color=shadingcolors[n])
		#amean.plot(np.linspace(0,12,12), maxi, color=shadingcolors[n])
		#amean[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)+std, color=shadingcolors[n])
		#amean[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)-std, color=shadingcolors[n])
		amean[0,n].fill_between(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)-std, np.nanmean(meanmodelmon[exp],axis=0)+std, color=shadingcolors[n],alpha=0.3)
		
		amean[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0),color=colors[n],label=labeli)

		amean[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmonpom[exp],axis=0),ls='--',color=colors[n],label=labeli+' primary')

		amean[0,n].set_ylim([0,8.5])
		#amean[0,n].set_title(exp)
		amean[0,n].legend(loc=1)
		nmbmean=NMB(np.nanmean(meanmodelmon['obs'],axis=0),np.nanmean(meanmodelmon[exp],axis=0))
		mbmean=MB(np.nanmean(meanmodelmon['obs'],axis=0),np.nanmean(meanmodelmon[exp],axis=0))
		rmean=pearsonr(np.nanmean(meanmodelmon[exp],axis=0),np.nanmean(meanmodelmon['obs'],axis=0))
		amean[0,n].annotate(('NMB (MB): %6.1f %% (%5.2f)')%(nmbmean*100,mbmean),xy=(0.05,0.9),xycoords='axes fraction',fontsize=10)
		amean[0,n].annotate(('R: %6.2f,%6.2e')%(rmean[0],rmean[1]),xy=(0.05,0.85),xycoords='axes fraction',fontsize=10)
		amean[0,n].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
		amean[0,n].set_xticklabels(str_months())
		amean[0,n].annotate(('%s)')%(letters[0][n]),xy=(0.0,1.02),xycoords='axes fraction',fontsize=16)
		amean[0,n].set_xlabel('Month')
		amean[0,n].set_ylabel('OM [ug m-3]')
		
		#ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
		print np.nanmean(meanmodelmon[exp],axis=0)

		d1=amean[1,n].scatter(yearmean_pm10_obs,yearmean_pm10_model[exp],marker='+',color=colors[n],s=25,label='PM10')
		d2=amean[1,n].scatter(yearmean_pm25_obs,yearmean_pm25_model[exp],marker='d',color=colors[n],s=25,label='PM25')
		amean[1,n].legend(loc=4)
		#amean[1,n].scatter(yearmean_obs,yearmean_model[exp],marker='o',color=colors[n],s=3)
		amean[1,n].loglog([0.001,1000],[0.001,1000],'k-')
		amean[1,n].loglog([0.1,1000],[0.01,100],'k--')
		amean[1,n].loglog([0.01,100],[0.1,1000],'k--')
		amean[1,n].set_xlim([0.5,10])
		amean[1,n].set_ylim([0.5,10])
		MFBemep1=MFB(yearmean_obs,yearmean_model[exp])
		MFEemep1=MFE(yearmean_obs,yearmean_model[exp])
		NMBemep1=NMB(yearmean_obs,yearmean_model[exp])
		NMEemep1=NME(yearmean_obs,yearmean_model[exp])
		RMSEemep1=RMSE(yearmean_obs,yearmean_model[exp])
		#amean[1,n].annotate(('MFB: %6.2f')%MFBemep1,xy=(.01,0.9),xycoords='axes fraction')
		#amean[1,n].annotate(('MFE: %6.2f')%MFEemep1,xy=(.01,0.85),xycoords='axes fraction')
		amean[1,n].annotate(('NMB: %7.1f %%')%(NMBemep1*100),xy=(.01,0.95),xycoords='axes fraction')
		#amean[1,n].annotate(('NME: %6.2f')%NMEemep1,xy=(.01,0.75),xycoords='axes fraction')
		amean[1,n].annotate(('RMSE: %6.2f')%RMSEemep1,xy=(.01,0.9),xycoords='axes fraction')
		amean[1,n].set_xlabel('EMEP OM [ug m-3]')
		amean[1,n].set_ylabel(EXP_NAMEs[n]+' OM [ug m-3]')
		amean[1,n].set_aspect('equal')
		amean[1,n].annotate(('%s)')%(letters[1][n]),xy=(0.0,1.02),xycoords='axes fraction',fontsize=16)
		#if n==0:
		#	amean[1,n].set_title('NEWSOA')
		#else:	
		#	amean[1,n].set_title('OLDSOA')
	
	std=np.nanstd(meanmodelmon['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
	mini=np.nanmin(meanmodelmon['obs'],axis=0)
	#amean.fill_between(np.linspace(0,12,12), mini, maxi,facecolor=shadingcolors[3], alpha=0.3,interpolate=True)
	#amean.plot(np.linspace(0,12,12), mini,color=shadingcolors[3])
	#amean.plot(np.linspace(0,12,12), maxi,color=shadingcolors[3])
	#amean[2].set_title('Observations')
	#amean[2].legend()
	fmean.suptitle('EMEP')
	#amean.set_yscale("log", nonposy='clip')
	fmean.savefig(output_png_path+'EMEP//monthly-EMEP-allmean_2panels.png',dpi=400)
	fmean.savefig(output_pdf_path+'EMEP//monthly-EMEP-allmean_2panels.pdf',dpi=400)

			#yearmean_model
	#print np.shape(yearmean_model)

	#print NMB(np.array(obs_all),model_all['soa-riccobono'])
	print NMB(np.array(obs_all),model_all['newsoa-ri'])
	print NMB(np.array(obs_all),model_all['oldsoa-bhn'])
	#print NMB(np.array(obs_all),model_all['nosoa'])
	print np.mean(np.array(obs_all)),np.mean(model_all['newsoa-ri']),np.mean(model_all['oldsoa-bhn'])
	#print obs_all
	raw_input() 
	'''
	if len(obs)>0:
		for i in sdata:
			o=sdata[i][2]
			if sdata[i][4]=='ug C/m3':
				factor=1.8
			else:
				factor =1.0
			m1=mdict_exp[0][i]['OM']
			m2=mdict_exp[1][i]['OM']
			#print 'hep'
			#print o
			print type(o)
			print type(m1)
			RMSEemep1=RMSE(o,m1)
			MFBemep1=MFB(o,m1)
			MFEemep1=MFE(o,m1)
			NMBemep1=NMB(np.array(o)*factor,np.array(m1))
			NMEemep1=NME(o,m1)
			RMSEemep2=RMSE(o,m2)
			MFBemep2=MFB(o,m2)
			MFEemep2=MFE(o,m2)
			NMBemep2=NMB(np.array(o)*factor,np.array(m2))
			NMEemep2=NME(o,m1)
			yearmean_obs.append(np.mean(np.array(o)))
			yearmean_model.append(np.mean(np.array(m1)))
			mm2.append(np.mean(np.array(m2)))
			plt.figure()
			print i,RMSEemep1
			#print o
			#print m1
			#print i
			plt.loglog(np.array(o)*factor,np.array(m1),'ob')
			plt.loglog(np.array(o)*factor,np.array(m2),'or')
			xmax=max([max(o),max(m1)])
			xmin=min([min(o),min(m1)])
			plt.annotate(('MFB: %6.2f')%MFBemep1,xy=(.1,0.9),xycoords='axes fraction')
			plt.annotate(('MFE: %6.2f')%MFEemep1,xy=(.1,0.85),xycoords='axes fraction')
			plt.annotate(('NMB: %6.2f')%NMBemep1,xy=(.1,0.8),xycoords='axes fraction')
			plt.annotate(('NME: %6.2f')%NMEemep1,xy=(.1,0.75),xycoords='axes fraction')
			plt.annotate(('RMSE: %6.2f')%RMSEemep1,xy=(.1,0.7),xycoords='axes fraction')
			plt.annotate(('MFB: %6.2f')%MFBemep2,xy=(.1,0.65),xycoords='axes fraction')
			plt.annotate(('MFE: %6.2f')%MFEemep2,xy=(.1,0.6),xycoords='axes fraction')
			plt.annotate(('NMB: %6.2f')%NMBemep2,xy=(.1,0.55),xycoords='axes fraction')
			plt.annotate(('NME: %6.2f')%NMEemep2,xy=(.1,0.5),xycoords='axes fraction')
			plt.annotate(('RMSE: %6.2f')%RMSEemep2,xy=(.1,0.45),xycoords='axes fraction')
			plt.loglog([0.001,1000],[0.001,1000],'k-')
			plt.loglog([0.1,1000],[0.01,100],'r--')
			plt.loglog([0.01,100],[0.1,1000],'r--')
			plt.xlim([0.01,1000])
			plt.ylim([0.01,1000])
			print 'new,old',NMBemep1,NMBemep2
			for ok,mk1,mk2 in zip(o,m1,m2):
				oall.append(ok*factor)
				mall1.append(mk1)
				mall2.append(mk2)
'''
	print 'obs loop'
	f2,ax=plt.subplots(ncols=3,figsize=(12,4))
	k=-1
	#print oall
	#print np.shape(oall)
	for exp in EXPS:
		k+=1
		ax[k].loglog(obs_all,model_all[exp],'ob',ms=3)
		#plt.loglog(oall,mall1,'ob',ms=2)
		#plt.loglog(oall,mall2,'or',ms=1)
		xmax=max([max(obs_all),max(model_all[exp])])
		xmin=min([min(obs_all),min(model_all[exp])])
		ax[k].loglog([0.001,1000],[0.001,1000],'k-')
		ax[k].loglog([0.1,1000],[0.01,100],'r--')
		ax[k].loglog([0.01,100],[0.1,1000],'r--')
		ax[k].set_xlim([0.1,100])
		ax[k].set_ylim([0.1,100])
		MFBemep1=MFB(obs_all,model_all[exp])
		MFEemep1=MFE(obs_all,model_all[exp])
		NMBemep1=NMB(obs_all,model_all[exp])
		NMEemep1=NME(obs_all,model_all[exp])
		RMSEemep1=RMSE(obs_all,model_all[exp])
		ax[k].annotate(('MFB: %6.2f')%MFBemep1,xy=(.1,0.9),xycoords='axes fraction')
		ax[k].annotate(('MFE: %6.2f')%MFEemep1,xy=(.1,0.85),xycoords='axes fraction')
		ax[k].annotate(('NMB: %6.2f')%NMBemep1,xy=(.1,0.8),xycoords='axes fraction')
		ax[k].annotate(('NME: %6.2f')%NMEemep1,xy=(.1,0.75),xycoords='axes fraction')
		ax[k].annotate(('RMSE: %6.2f')%RMSEemep1,xy=(.1,0.7),xycoords='axes fraction')
		#print np.mean(oall),np.mean(mall1)
		ax[k].loglog(np.mean(obs_all),np.mean(model_all[exp]),'sb',ms=3)
		ax[k].set_xlabel('EMEP OM [ug m-3]')
		ax[k].set_ylabel('TM5 OM [ug m-3]')
		#plt.loglog(np.mean(oall),np.mean(mall1),'sb',ms=10)
		#plt.loglog(np.mean(oall),np.mean(mall2),'sr',ms=10)
	
	#plt.show()
	f3,ax=plt.subplots(ncols=3,figsize=(12,4))
	f3b,axb=plt.subplots(ncols=2,figsize=(8,4))
	k=-1
	for exp in EXPS:
		k+=1
		print 'N '+exp+': ',len(yearmean_model[exp]),len(yearmean_obs)
		#for i in len(yearmean_obs):
		#ax[k].loglog(yearmean_obs[i],yearmean_model[exp][i],'ob',ms=3)
		ax[k].loglog(yearmean_pm25_obs,yearmean_pm25_model[exp],'ob',ms=3)
		ax[k].loglog(yearmean_pm10_obs,yearmean_pm10_model[exp],'+b',ms=3)
		#ax[k].loglog(oo,mm2,'or')
		#print oo
		#print mm1
		#xmax=max([max(oo),max(mm1)])
		#xmin=min([min(oo),min(mm1)])
		ax[k].loglog([0.001,1000],[0.001,1000],'k-')
		ax[k].loglog([0.1,1000],[0.01,100],'r--')
		ax[k].loglog([0.01,100],[0.1,1000],'r--')
		ax[k].set_xlim([0.1,10])
		ax[k].set_ylim([0.1,10])
		ax[k].set_xlabel('EMEP OM [ug m-3]')
		ax[0].set_ylabel('TM5 OM [ug m-3]')
		#if exp=='soa-riccobono':
		if exp=='newsoa-ri':
			ax[k].set_title('NEWSOA')
		elif exp=='oldsoa-bhn':
			ax[k].set_title('OLDSOA')
		else:
			ax[k].set_title('NOSOA')
		MFBemep1=MFB(yearmean_obs,yearmean_model[exp])
		MFEemep1=MFE(yearmean_obs,yearmean_model[exp])
		NMBemep1=NMB(yearmean_obs,yearmean_model[exp])
		NMEemep1=NME(yearmean_obs,yearmean_model[exp])
		RMSEemep1=RMSE(yearmean_obs,yearmean_model[exp])
		
		MFBemeppm25=MFB(yearmean_pm25_obs,yearmean_pm25_model[exp])
		MFEemeppm25=MFE(yearmean_pm25_obs,yearmean_pm25_model[exp])
		NMBemeppm25=NMB(yearmean_pm25_obs,yearmean_pm25_model[exp])
		NMEemeppm25=NME(yearmean_pm25_obs,yearmean_pm25_model[exp])
		RMSEemeppm25=RMSE(yearmean_pm25_obs,yearmean_pm25_model[exp])

		MFBemeppm10=MFB(yearmean_pm10_obs,yearmean_pm10_model[exp])
		MFEemeppm10=MFE(yearmean_pm10_obs,yearmean_pm10_model[exp])
		NMBemeppm10=NMB(yearmean_pm10_obs,yearmean_pm10_model[exp])
		NMEemeppm10=NME(yearmean_pm10_obs,yearmean_pm10_model[exp])
		RMSEemeppm10=RMSE(yearmean_pm10_obs,yearmean_pm10_model[exp])
		#ax[k].annotate(('MFB: %6.2f')%MFBemeppm25,xy=(.1,0.9),xycoords='axes fraction')
		#ax[k].annotate(('MFE: %6.2f')%MFEemeppm25,xy=(.1,0.85),xycoords='axes fraction')
		ax[k].annotate(('NMB 25: %6.2f')%NMBemeppm25,xy=(.1,0.9),xycoords='axes fraction')
		ax[k].annotate(('NME 25: %6.2f')%NMEemeppm25,xy=(.1,0.85),xycoords='axes fraction')
		ax[k].annotate(('RMSE 25: %6.2f')%RMSEemeppm25,xy=(.1,0.8),xycoords='axes fraction')
		#ax[k].annotate(('MFB: %6.2f')%MFBemeppm25,xy=(.1,0.9),xycoords='axes fraction')
		#ax[k].annotate(('MFE: %6.2f')%MFEemeppm25,xy=(.1,0.85),xycoords='axes fraction')
		ax[k].annotate(('NMB 10: %6.2f')%NMBemeppm10,xy=(.1,0.75),xycoords='axes fraction')
		ax[k].annotate(('NME 10: %6.2f')%NMEemeppm10,xy=(.1,0.70),xycoords='axes fraction')
		ax[k].annotate(('RMSE 10: %6.2f')%RMSEemeppm25,xy=(.1,0.65),xycoords='axes fraction')
		if k<2:
			axb[k].loglog(yearmean_obs,yearmean_model[exp],'ob',ms=3)
			axb[k].loglog([0.001,1000],[0.001,1000],'k-')
			axb[k].loglog([0.1,1000],[0.01,100],'r--')
			axb[k].loglog([0.01,100],[0.1,1000],'r--')
			axb[k].set_xlim([0.1,10])
			axb[k].set_ylim([0.1,10])
			axb[k].annotate(('MFB: %6.2f')%MFBemep1,xy=(.1,0.9),xycoords='axes fraction')
			axb[k].annotate(('MFE: %6.2f')%MFEemep1,xy=(.1,0.85),xycoords='axes fraction')
			axb[k].annotate(('NMB: %6.2f')%NMBemep1,xy=(.1,0.8),xycoords='axes fraction')
			axb[k].annotate(('NME: %6.2f')%NMEemep1,xy=(.1,0.75),xycoords='axes fraction')
			axb[k].annotate(('RMSE: %6.2f')%RMSEemep1,xy=(.1,0.7),xycoords='axes fraction')
			axb[k].set_xlabel('EMEP OM [ug m-3]')
			if k==0:
				axb[k].set_title('NEWSOA')
			else:	
				axb[k].set_title('OLDSOA')
			# amean[1,k].loglog(yearmean_obs,yearmean_model[exp],'ob',ms=3)
			# amean[1,k].loglog([0.001,1000],[0.001,1000],'k-')
			# amean[1,k].loglog([0.1,1000],[0.01,100],'k--')
			# amean[1,k].loglog([0.01,100],[0.1,1000],'k--')
			# amean[1,k].set_xlim([0.1,10])
			# amean[1,k].set_ylim([0.1,10])
			# amean[1,k].annotate(('MFB: %6.2f')%MFBemep1,xy=(.1,0.9),xycoords='axes fraction')
			# amean[1,k].annotate(('MFE: %6.2f')%MFEemep1,xy=(.1,0.85),xycoords='axes fraction')
			# amean[1,k].annotate(('NMB: %6.2f')%NMBemep1,xy=(.1,0.8),xycoords='axes fraction')
			# amean[1,k].annotate(('NME: %6.2f')%NMEemep1,xy=(.1,0.75),xycoords='axes fraction')
			# amean[1,k].annotate(('RMSE: %6.2f')%RMSEemep1,xy=(.1,0.7),xycoords='axes fraction')
			# amean[1,k].set_xlabel('EMEP OM [ug m-3]')
			# amean[1,k].set_aspect('equal')
			# if k==0:
			# 	amean[1,k].set_title('NEWSOA')
			# else:	
			# 	amean[1,k].set_title('OLDSOA')

		axb[0].set_ylabel('TM5 OM [ug m-3]')
		#amean[0,0].set_ylabel('TM5 OM [ug m-3]')
		'''MFBemep2=MFB(oo,mm2)
		MFEemep2=MFE(oo,mm2)
		NMBemep2=NMB(oo,mm2)
		NMEemep2=NME(oo,mm2)
		RMSEemep2=RMSE(oo,mm2)
		plt.annotate(('MFB: %6.2f')%MFBemep2,xy=(.1,0.65),xycoords='axes fraction')
		plt.annotate(('MFE: %6.2f')%MFEemep2,xy=(.1,0.6),xycoords='axes fraction')
		plt.annotate(('NMB: %6.2f')%NMBemep2,xy=(.1,0.55),xycoords='axes fraction')
		plt.annotate(('NME: %6.2f')%NMEemep2,xy=(.1,0.5),xycoords='axes fraction')
		plt.annotate(('RMSE: %6.2f')%RMSEemep2,xy=(.1,0.45),xycoords='axes fraction')
		'''
		'''plt.figure(figsize=(10,7))
		m = Basemap(projection='robin',lon_0=0)
		m.drawcoastlines()
		m.drawparallels(np.arange(-90.,120.,30.))
		m.drawmeridians(np.arange(0.,360,60.))
		mycmap=plt.get_cmap('coolwarm',11) 
		count=0
		for x,y in zip(mlons,mlats):
			#print i[0][0],i[1][0]
			print x,y
			x1,y1=m(x,y)
			m.scatter(x1,y1,marker='o',s=10 )
		'''
	f2.savefig(output_png_path+'EMEP/scatter-all-EMEP-1x3.png',dpi=200)
	f3.savefig(output_png_path+'EMEP/scatter-yearmean-EMEP-1x3.png',dpi=200)
	f3b.savefig(output_png_path+'EMEP/scatter-yearmean-EMEP-1x2.png',dpi=200)
	f2.savefig(output_jpg_path+'EMEP/scatter-all-EMEP-1x3.jpg',dpi=200)
	f3.savefig(output_jpg_path+'EMEP/scatter-yearmean-EMEP-1x3.jpg',dpi=200)
	f3b.savefig(output_jpg_path+'EMEP/scatter-yearmean-EMEP-1x2.jpg',dpi=200)
	f2.savefig(output_pdf_path+'EMEP/scatter-all-EMEP-1x3.pdf',dpi=200)
	f3.savefig(output_pdf_path+'EMEP/scatter-yearmean-EMEP-1x3.pdf',dpi=200)
	f3b.savefig(output_pdf_path+'EMEP/scatter-yearmean-EMEP-1x2.pdf',dpi=200)
	
	fmean.savefig(output_png_path+'EMEP/scatter-seasonal-EMEP-2x2.png',dpi=200)
	fmean.savefig(output_jpg_path+'EMEP/scatter-seasonal-EMEP-2x2.jpg',dpi=200)
	fmean.savefig(output_pdf_path+'EMEP/scatter-seasonal-EMEP-2x2.pdf',dpi=200)
	
	#c = plt.colorbar(orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7)
	#c.ax.tick_params(labelsize=10)
	#c.set_label('AOD relative bias [(TM5-AERONET)/AERONET]')

	plt.show()
if __name__=='__main__':
	main()

