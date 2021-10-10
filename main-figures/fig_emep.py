from scipy.special import erf
import matplotlib.pyplot as plt 
import nappy
import datetime
import glob
import netCDF4 as nc
import h5py
from mpl_toolkits.basemap import Basemap
from general_toolbox import write_netcdf_file,str_months,parse_nas_normalcomments,MB,NMB,MFB,MFE,NME,RMSE,lonlat
import general_toolbox as gt
import os
#from  cdo import *
from scipy.stats import pearsonr
import scipy.stats as stats
import numpy as np
from settings import *

SMALL_SIZE=12
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels# def site_type():

def list_stations(indict):
	tabletex=open(paper+'sitelist_emep.tex','w')
	tabletex.write("%Name & Station code&Longitude& Latitude &Height\\\\\n")
	sortdict={}
	for i in sorted(indict.keys()):
		print ('%6s,%35s,%4s,%4s,%4s,%4s')%(i,indict[i]['name'],indict[i]['lon'],indict[i]['lat'],indict[i]['ele'],indict[i]['pm'])
		stationname=indict[i]['name']
		stationname=stationname.replace("#","\\#")
		sortdict[stationname]=[i,indict[i]['lon'],indict[i]['lat'],indict[i]['ele']]
	for j in sorted(sortdict):
		print ('%-35s,%6s,%4s,%4s,%4s')%(j,sortdict[j][0],sortdict[j][1],sortdict[j][2],sortdict[j][3])
		tabletex.write("%s&%6s&%4s&%4s&%4s\\\\\n"%(j,sortdict[j][0],sortdict[j][1],sortdict[j][2],sortdict[j][3]))
	tabletex.close()

def read_emep_observations(filein='/Users/bergmant/Documents/obs-data/Ebas_170727_1044_EMEP/*.nas'):
	# reader for NASA Ames format used in EUSAAR
	# full description http://www.eusaar.net/files/data/nasaames/index.html
	sitedata={}
	sitedata2={}
	#for i in glob.iglob('/Users/bergmant/Documents/obs-data/EMEP-SOA/*.nas'):
	for i in glob.iglob(filein):
		print 'WARNING!!! Works only for files with reference date 2010-01-01 00:00'
		possible_vars=['organic_carbon']
		a=None
		varindex=None
		#print i
		#f = nappy.openNAFile('/Users/bergmant/Documents/obs-data/EMEP-Ebas_170608_1156/SE0011R.20100101110000.20111018000000.lvs_denuder_tandem.organic_carbon.pm10.1y.1w.SE04L_LU_DRI-1_1.SE04L_EUSAAR-2..nas')
		f = nappy.openNAFile(i)
		n_lines=f.getNumHeaderLines()
		#print f["VNAME"]
		for i in f["VNAME"]:
			#print i
			if  possible_vars[0] in i:
				if 'Matrix' in i:
					a=i.rsplit('Matrix=')#find('Matrix=')
					print a[-1]
		#print dir(f)
		#print n_lines
		# Get Organisation from header
		org = f.getOrg()
		# Get the Normal Comments (SCOM) lines.
		norm_comms = f.getNormalComments()
		#print norm_comms
		#print '---'
		lon,lat,stationname,site,elev,sdate,pm,unit=parse_nas_normalcomments(norm_comms)



		sitedict={}
		sitedict['name']=stationname
		sitedict['lat']=lat
		sitedict['lon']=lon
		sitedict['ele']=elev
		sitedict['unit']=unit
		sitedict['pm']=pm
		#print 'PM:',a,pm
		# analyse 2010 only
		if sdate[:4]!='2010':
			continue
		#print stationname,lon,lat,pm,unit
		# Get the Special Comments (SCOM) lines.
		###???###
		spec_comms = f.getSpecialComments()
		#print spec_comms
		varindex=-1
		# Get a list of metadata for all main (non-auxiliary or independent) variables
		var_list = f.getVariables()
		#print f["VNAME"]
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
				#print stationname,i, var_list.index(i)
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
		stime=[]
		etime=[]
		sdate=[]
		edate=[]
		stamp=[]

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
			#print sdate
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
		
def calculate_density(inputdata):
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
				#print  indexi
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
		#print m, roo
		#raw_input()
	return roo_a
def read_mass_pmX(comp,inputdata,rad=1.25):
	if rad < 1.0:
		print 'WARNING: Mass check only for ACS and COS at the moment!!!'
	data=nc.Dataset(inputdata,'r')
	outputdataset=None
	roo_a=calculate_density(inputdata)
	# select mass below threshold
	#ACS
	d_a_correction_factor={}
	for m in roo_a:
		d_a_correction_factor[m]=np.sqrt(roo_a[m]/1.0) #sqrt(roo_p/roo_0), roo_0=1.00g/cm3 (Water)
	rACS=data.variables['RWET_ACS'][:]*d_a_correction_factor['ACS']
	rCOS=data.variables['RWET_COS'][:]*d_a_correction_factor['COS']
	
	cmedr2mmedr= np.exp(3.0*(np.log(1.59)**2))	
	hr2=(0.5*np.sqrt(2.0))
	
	rad=1.25
	z=(np.log(rad)-np.log(rACS*1e6*cmedr2mmedr)/np.log(1.59))
	modfrac_acs1=0.5+0.5*erf(z*hr2)
	rad=5.0
	z=(np.log(rad)-np.log(rACS*1e6*cmedr2mmedr)/np.log(1.59))
	modfrac_acs2=0.5+0.5*erf(z*hr2)
	#free memory
	del z
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
			#print comp,i
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
		print  sdata[site][3]
		if sdata[site][3]=='pm25':
			modeldata=modeldata25
		elif sdata[site][3]=='pm10':
			modeldata=modeldata10
		else:
			print 'choice of pm not found: '+sdata[site][3],site
			modeldata=modeldata10
			
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
			factor=1.8
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
			model1_mean.append(obsstep_mean)
			model1_std.append(obsstep_std)
			
			time.append(nc.num2date(time1,'days since 2010-01-01 00:00:00',calendar='standard'))

		#print time
		#raw_input()

	
		times.append(time)
		model_mean.append(model1_mean)
		model_std.append(model1_std)
		#print site,len(time),ddd
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
	return model_mean,names,times,model_dict

def main():
	lon,lat=lonlat('TM53x2')
	sdata,sitedata=read_emep_observations()
	list_stations(sitedata)
	org10=[]
	org25=[]
	org10d={}
	org25d={}
	pom10=[]
	pom25=[]
	pom10d={}
	pom25d={}
	timeaxis_day=[]
	#EXPS=['soa-riccobono','oldsoa-bhn','nosoa']
	#EXPS=['newsoa-ri','oldsoa-bhn','nosoa']
	#EXPS=['newsoa-ri','oldsoa-bhn']#,'nosoa']
	EXPS=['newsoa-ri','oldsoa-bhn','oldsoa-bhn-megan2']#,'nosoa']

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

	for exp in EXPS:
		#if len(glob.glob(basepath+'emep_col/'+exp+'*'))==0:
		#read data if already processed
		if os.path.isfile(raw_store+exp+'/'+exp+'_org10.nc') and os.path.isfile(raw_store+exp+'/'+exp+'_org25.nc'):
			org10.append(nc.Dataset(raw_store+exp+'/'+exp+'_org10.nc','r').variables['org10'][:])
			org25.append(nc.Dataset(raw_store+exp+'/'+exp+'_org25.nc','r').variables['org25'][:])
			print raw_store+exp+'/'+exp+'_poa10.nc'
			pom10.append(nc.Dataset(raw_store+exp+'/'+exp+'_poa10.nc','r').variables['poa10'][:])
			pom25.append(nc.Dataset(raw_store+exp+'/'+exp+'_poa25.nc','r').variables['poa25'][:])
			# put just added data to dictionary
			org10d[exp]=org10[-1].copy()
			org25d[exp]=org25[-1].copy()
			pom10d[exp]=pom10[-1].copy()
			pom25d[exp]=pom25[-1].copy()
			#time
			timeaxis=nc.Dataset(raw_store+exp+'/general_TM5_'+exp+'_2010.lev1.nc','r').variables['time'][:]
			timeaxis_day.append(timeaxis-np.floor(timeaxis[0]))
			print 'load',timeaxis-np.floor(timeaxis[0]),timeaxis,np.floor(timeaxis[0])
			#raw_input()

		else:
			# process data
			
			if  os.path.exists(basepathraw+exp+'/general_TM5_'+exp+'_2010.lev1.nc'):
				filepath=basepathraw+exp+'/general_TM5_'+exp+'_2010.lev1.nc'
			else:	
				filepath=raw_store+exp+'/general_TM5_'+exp+'_2010.lev1.nc'
			print 'read soa'

			soa25,soa10=read_mass_pmX('M_SOA',raw_store+exp+'/general_TM5_'+exp+'_2010.lev1.nc')

			print 'read oc'
			poa25,poa10=read_mass_pmX('M_POM',raw_store+exp+'/general_TM5_'+exp+'_2010.lev1.nc')
			print 'sum'
			org25.append(soa25+poa25)
			org10.append(soa10+poa10)
			pom25.append(poa25)
			pom10.append(poa10)
			print 'time'
			timeaxis=nc.Dataset(raw_store+exp+'/general_TM5_'+exp+'_2010.lev1.nc','r').variables['time'][:]
			timeaxis_day.append(timeaxis-np.floor(timeaxis[0]))
			#write_netcdf_file([np.squeeze(org10[i])],['org10'],raw_store+exp+'/'+exp+'_org10.nc',lat,lon,timeaxis)#,lat,lon)
			#write_netcdf_file([np.squeeze(org25[i])],['org25'],raw_store+exp+'/'+exp+'_org25.nc',lat,lon,timeaxis)#,lat,lon)
			
			#write_netcdf_file([np.squeeze(pom10[i])],['poa10'],raw_store+exp+'/'+exp+'_poa10.nc',lat,lon,timeaxis)#,lat,lon)
			#write_netcdf_file([np.squeeze(pom25[i])],['poa25'],raw_store+exp+'/'+exp+'_poa25.nc',lat,lon,timeaxis)#,lat,lon)

			del poa10,soa10,poa25,soa25	
		#print 'tim',timeaxis_day[0][1]
		#exp_dict['model']
		#for i in range(3):
		#print  i,len(org25),len(org10),len(timeaxis_day),len(sdata)
		model,names,times,mdict=colocate_emep(org25[i],org10[i],timeaxis_day[i],sdata,exp)
		model_pom,names_pom,times_pom,mdictpom=colocate_emep(pom25[i],pom10[i],timeaxis_day[i],sdata,exp)
		#print names
		model_exp.append(model)
		names_exp.append(names)
		time_exp.append(times)
		mdict_exp.append(mdict)
		mdictpom_exp.append(mdictpom)
		exp_dict[EXPS[i]]['sitenames']=names
		exp_dict[EXPS[i]]['time']=times
		exp_dict[EXPS[i]]['mdict']=mdict
		exp_dict[EXPS[i]]['mdictpom']=mdictpom
		#print len(model),len(names),len(times)
		for data,data_pom,name,time in zip(model,model_pom,names,times):
			#raw_input()
			#print 'hep'
			#print basepath+'output/processed/emep_col/'+exp+'_'+name+'.nc'
			#print len(data),len(name),len(time)
			#print time
			if not os.path.isfile(basepath+'output/processed/emep_col/'+exp+'_'+name+'.nc'):
				print 'write data for ' + name 
				#write_netcdf_file([data],[name],basepath+'emep_col/'+exp+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)
				write_netcdf_file([data],[name],basepath+'output/processed/emep_col/'+exp+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)
				write_netcdf_file([data_pom],[name],basepath+'output/processed/emep_col/'+exp+'_'+name+'_poa.nc',None,None,np.array(time))#,lat,lon)
		i+=1
		
	# free memory
	del org10,org25
	#print model
	obs=[]
	for i in sdata:
		obs.append(sdata[i][2])
	oo=[]
	mm1=[]
	mm2=[]
	oall=[]
	mall1=[]
	mall2=[]
	#print len(obs),len(model)
	#print obs
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

	yearstderr_pm10_obs=[]
	yearstderr_pm10_model={}
	yearstderr_pm25_model={}
	yearstderr_pm25_obs=[]

	meanmodelmon_pm25={}
	meanmodelmonpom_pm25={}
	meanmodelmon_pm10={}
	meanmodelmonpom_pm10={}

	for exp in EXPS:
		meanmodelmon[exp]=np.zeros((len(sdata),12))
		meanmodelmonpom[exp]=np.zeros((len(sdata),12))
		meanmodelmon_pm25[exp]=np.zeros((len(sdata),12))
		meanmodelmonpom_pm25[exp]=np.zeros((len(sdata),12))
		meanmodelmon_pm10[exp]=np.zeros((len(sdata),12))
		meanmodelmonpom_pm10[exp]=np.zeros((len(sdata),12))

	meanmodelmon['obs']=np.zeros((len(sdata),12))
	meanmodelmon_pm25['obs']=np.zeros((len(sdata),12))
	meanmodelmon_pm10['obs']=np.zeros((len(sdata),12))
	k_site=0
	N_pm25=0
	#Loop over sites
	for i in sdata:
		# create monthly mean dataholders
		timedata=sdata[i][-1]		
		timedata=sitedata[i]['time']		
		monthdata=np.zeros([12])
		monthdata[:]=np.NAN
		monthindices={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[]}
		# find indices of data for different months
		for j in timedata:
			#print j.month
			monthindices[j.month].append(timedata.index(j))
		# read monthly measurements
		for k in monthindices:
			#print k,type(sdata[i][2])
			#print monthdata.shape,monthindices[k],(sdata[i][2][monthindices[k]]),np.mean(sdata[i][2][monthindices[k]])
			#monthdata[k-1]=np.mean(sdata[i][2][monthindices[k]])
			#print monthdata.shape,monthindices[k],(sitedata[i]['oc'][monthindices[k]]),np.mean(sitedata[i]['oc'][monthindices[k]])
			monthdata[k-1]=np.mean(sitedata[i]['oc'][monthindices[k]])


		if sdata[i][4]=='ug C/m3':
			factor=1.8
			print 'f1.8 site ',i
		else:
			factor =1.0
			print 'f1 site ',i
		meanmodelmon['obs'][k_site,:]=monthdata[:]*factor
		if sdata[i][3]=='pm25':
			meanmodelmon_pm25['obs'][k_site,:]=monthdata[:]*factor
		else:
			meanmodelmon_pm10['obs'][k_site,:]=monthdata[:]*factor

		o=sdata[i][2]
		print type(o)
		print type([float(iii) for iii in o])
		print type([float(iii) for iii in o][0])
		yearmean_obs.append(np.mean(o)*factor)
		print sdata[i][:]
		if sdata[i][3]=='pm25':
			yearmean_pm25_obs.append(np.mean(o)*factor)
			yearstderr_pm25_obs.append(stats.sem([float(iii)*factor for iii in o]))
			N_pm25+=1
			#print N_pm25
		else:
			yearmean_pm10_obs.append(np.mean(o)*factor)
			yearstderr_pm10_obs.append(stats.sem([float(iii)*factor for iii in o]))

		
		for ok in o:
			obs_all.append(ok*factor)

		for exp in EXPS:
			modelmonthdata=np.zeros([12])
			modelmonthdata[:]=np.NAN
			modelmonthdatapom=np.zeros([12])
			modelmonthdatapom[:]=np.NAN
			model=nc.Dataset(basepath+'output/processed/emep_col/'+exp+'_'+i+'.nc','r').variables[i][:]
			modelpom=nc.Dataset(basepath+'output/processed/emep_col/'+exp+'_'+i+'_poa.nc','r').variables[i][:]
			if exp not in model_all.keys():
				model_all[exp]=model
			else:
				model_all[exp]=np.concatenate((model_all[exp],model))
				
			for kmod in monthindices:
				modelmonthdata[kmod-1]=np.mean(model[monthindices[kmod]])
				modelmonthdatapom[kmod-1]=np.mean(modelpom[monthindices[kmod]])	

			meanmodelmon[exp][k_site,:]=modelmonthdata
			meanmodelmonpom[exp][k_site,:]=modelmonthdatapom

			if sdata[i][3]=='pm25':
				meanmodelmon_pm25[exp][k_site,:]=modelmonthdata
				meanmodelmonpom_pm25[exp][k_site,:]=modelmonthdatapom
			else:
				meanmodelmon_pm10[exp][k_site,:]=modelmonthdata
				meanmodelmonpom_pm10[exp][k_site,:]=modelmonthdatapom

			if exp==EXPS[0]:
				X=0.05
				colori='r'
			elif exp==EXPS[1]:
				X=0.8
				colori='b'
			else:# exp==EXPS[1]:
				X=0.4
				colori='g'
			if exp not in yearmean_model.keys():
				yearmean_model[exp]=[]
				yearmean_model[exp].append(np.mean(model))
			else:
				yearmean_model[exp].append(np.mean(model))
			


			if sdata[i][3]=='pm25':
				print sdata[i][:]
				if exp not in yearmean_pm25_model.keys():
					yearmean_pm25_model[exp]=[]
					yearmean_pm25_model[exp].append(np.mean(model))
				else:
					yearmean_pm25_model[exp].append(np.mean(model))
				if exp not in yearstderr_pm25_model.keys():
					yearstderr_pm25_model[exp]=[]
					yearstderr_pm25_model[exp].append(stats.sem(model))
				else:
					yearstderr_pm25_model[exp].append(stats.sem(model))
			else:
				if exp not in yearmean_pm10_model.keys():
					yearmean_pm10_model[exp]=[]
					yearmean_pm10_model[exp].append(np.mean(model))
				else:
					yearmean_pm10_model[exp].append(np.mean(model))
				if exp not in yearstderr_pm10_model.keys():
					yearstderr_pm10_model[exp]=[]
					yearstderr_pm10_model[exp].append(stats.sem(model))
				else:
					yearstderr_pm10_model[exp].append(stats.sem(model))
		k_site+=1
	colors=['red','blue','green','black']
	shadingcolors=['#ff000033','#0000ff33', '#00ff0033','#99999966']

	letters=[['a','b'],['c','d'],['e','f']]
	
	fmean,amean=plt.subplots(nrows=3,ncols=3,figsize=(19,13))#,tight_layout=True)
	std=np.nanstd(meanmodelmon_pm25['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon_pm25['obs'],axis=0)
	mini=np.nanmin(meanmodelmon_pm25['obs'],axis=0)
	amean[0,0].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon_pm25['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	amean[0,0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean[0,0].set_ylim([0,7.5])
	amean[0,1].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon_pm25['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	amean[0,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean[0,1].set_ylim([0,7.5])
	amean[0,2].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon_pm25['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	amean[0,2].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean[0,2].set_ylim([0,7.5])
	std=np.nanstd(meanmodelmon_pm10['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon_pm10['obs'],axis=0)
	mini=np.nanmin(meanmodelmon_pm10['obs'],axis=0)
	amean[1,0].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon_pm10['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	amean[1,0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean[1,0].set_ylim([0,7.5])
	amean[1,1].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon_pm10['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	amean[1,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean[1,1].set_ylim([0,7.5])
	amean[1,2].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon_pm10['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	amean[1,2].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean[1,2].set_ylim([0,7.5])
	letters=[['a','b'],['c','d']]
	letters=[['a','b'],['c','d'],['e','f']]
	letters=[['a','b','c'],['d','e','f'],['g','h','i']]

	for n,exp in enumerate(EXPS[:],0):
		if exp=='newsoa-ri':
			amean[0,n].set_title('NEWSOA')
			labeli='NEWSOA'
		elif exp=='oldsoa-bhn':
			amean[0,n].set_title('OLDSOA')
			labeli='OLDSOA'
		elif exp=='oldsoa-bhn-megan2':
			amean[0,n].set_title('OLDSOA-MEGAN2')
			labeli='OLDSOA-MEGAN2'
		#print meanmodelmon[exp]
		#print np.nanstd(meanmodelmon[exp],axis=0)

		std=np.nanstd(meanmodelmon_pm25[exp],axis=0)
		maxi=np.nanmax(meanmodelmon_pm25[exp],axis=0)
		mini=np.nanmin(meanmodelmon_pm25[exp],axis=0)
		amean[0,n].fill_between(np.linspace(1,12,12), np.nanmean(meanmodelmon_pm25[exp],axis=0)-std, np.nanmean(meanmodelmon_pm25[exp],axis=0)+std, color=shadingcolors[n],alpha=0.3)
		amean[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon_pm25[exp],axis=0),color=colors[n],label=labeli+' POA+SOA')
		# if n==0:
		# 	amean[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon_pm25['oldsoa-bhn-megan2'],axis=0),color=colors[2],label='oldsoamegan2'+' POA+SOA')

		amean[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmonpom_pm25[exp],axis=0),ls='--',color=colors[n],label=labeli+' POA')

		amean[0,n].set_ylim([0,8.5])
		amean[0,n].legend(loc='upper right',fontsize=12)
		nmbmean=NMB(np.nanmean(meanmodelmon_pm25['obs'],axis=0),np.nanmean(meanmodelmon_pm25[exp],axis=0))
		mbmean=MB(np.nanmean(meanmodelmon_pm25['obs'],axis=0),np.nanmean(meanmodelmon_pm25[exp],axis=0))
		rmean=pearsonr(np.nanmean(meanmodelmon_pm25[exp],axis=0),np.nanmean(meanmodelmon_pm25['obs'],axis=0))
		amean[0,n].annotate(('NMB (MB): %6.1f %% (%5.2f)')%(nmbmean*100,mbmean),xy=(0.01,0.9),xycoords='axes fraction',fontsize=12)
		amean[0,n].annotate(('R: %6.2f,%6.2e')%(rmean[0],rmean[1]),xy=(0.01,0.82),xycoords='axes fraction',fontsize=12)
		amean[0,n].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
		amean[0,n].set_xticklabels(str_months())
		amean[0,n].annotate(('%s)')%(letters[0][n]),xy=(0.0,1.02),xycoords='axes fraction',fontsize=18)
		amean[0,n].set_xlabel('Month',fontsize=12)
		amean[0,n].set_ylabel('OM in PM2.5 [$\mu$g m$^{-3}$]',fontsize=12)
		#middle row
		std=np.nanstd(meanmodelmon[exp],axis=0)
		maxi=np.nanmax(meanmodelmon[exp],axis=0)
		mini=np.nanmin(meanmodelmon[exp],axis=0)
		amean[1,n].fill_between(np.linspace(1,12,12), np.nanmean(meanmodelmon_pm10[exp],axis=0)-std, np.nanmean(meanmodelmon[exp],axis=0)+std, color=shadingcolors[n],alpha=0.3)
		
		amean[1,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon_pm10[exp],axis=0),color=colors[n],label=labeli+' POA+SOA')

		amean[1,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmonpom_pm10[exp],axis=0),ls='--',color=colors[n],label=labeli+' POA')

		amean[1,n].set_ylim([0,8.5])
		amean[1,n].legend(loc='upper right',fontsize=12)
		nmbmean=NMB(np.nanmean(meanmodelmon_pm10['obs'],axis=0),np.nanmean(meanmodelmon_pm10[exp],axis=0))
		mbmean=MB(np.nanmean(meanmodelmon_pm10['obs'],axis=0),np.nanmean(meanmodelmon_pm10[exp],axis=0))
		rmean=pearsonr(np.nanmean(meanmodelmon_pm10[exp],axis=0),np.nanmean(meanmodelmon_pm10['obs'],axis=0))
		amean[1,n].annotate(('NMB (MB): %6.1f %% (%5.2f)')%(nmbmean*100,mbmean),xy=(0.01,0.9),xycoords='axes fraction',fontsize=12)
		amean[1,n].annotate(('R: %6.2f,%6.2e')%(rmean[0],rmean[1]),xy=(0.01,0.82),xycoords='axes fraction',fontsize=12)
		amean[1,n].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
		amean[1,n].set_xticklabels(str_months())
		amean[1,n].annotate(('%s)')%(letters[1][n]),xy=(0.0,1.02),xycoords='axes fraction',fontsize=18)
		amean[1,n].set_xlabel('Month',fontsize=12)
		amean[1,n].set_ylabel('OM in PM10 [$\mu$g m$^{-3}$]',fontsize=12)
		#print np.nanmean(meanmodelmon[exp],axis=0)
		# bottom
		print yearmean_pm10_model.keys()
		#d1=amean[2,n].scatter(yearmean_pm10_obs,yearmean_pm10_model[exp],marker='+',color=colors[n],s=25,label='PM10')
		#d2=amean[2,n].scatter(yearmean_pm25_obs,yearmean_pm25_model[exp],marker='d',color=colors[n],s=25,label='PM25')
		print yearstderr_pm10_obs,
		#d1=amean[2,n].errorbar(yearmean_pm10_obs,yearmean_pm10_model[exp],yerror=yearstderr_pm10_model[exp],xerror=yearstderr_pm10_obs,fmt='none',color=colors[n],ls='none',label='PM10')
		#d2=amean[2,n].errorbar(yearmean_pm25_obs,yearmean_pm25_model[exp],yerror=yearstderr_pm25_model[exp],xerror=yearstderr_pm25_obs,fmt='none',color=colors[n],ls='none',label='PM25')
		d1=amean[2,n].errorbar(yearmean_pm10_obs,yearmean_pm10_model[exp],yerr=yearstderr_pm10_model[exp],xerr=yearstderr_pm10_obs,fmt='none',color=colors[n],ls='none',label='PM10')
		d2=amean[2,n].errorbar(yearmean_pm25_obs,yearmean_pm25_model[exp],yerr=yearstderr_pm25_model[exp],xerr=yearstderr_pm25_obs,fmt='o',color=colors[n],ls='none',label='PM25')

		amean[2,n].legend(loc=3,fontsize=12)
		amean[2,n].loglog([0.001,1000],[0.001,1000],'k-')
		amean[2,n].loglog([0.1,1000],[0.01,100],'k--')
		amean[2,n].loglog([0.01,100],[0.1,1000],'k--')
		amean[2,n].set_xlim([0.5,10])
		amean[2,n].set_ylim([0.5,10])
		MFBemep1=MFB(yearmean_obs,yearmean_model[exp])
		MFEemep1=MFE(yearmean_obs,yearmean_model[exp])
		NMBemep1=NMB(yearmean_obs,yearmean_model[exp])
		NMEemep1=NME(yearmean_obs,yearmean_model[exp])
		RMSEemep1=RMSE(yearmean_obs,yearmean_model[exp])
		NMBemep_pm25=NMB(yearmean_pm25_obs,yearmean_pm25_model[exp])
		NMBemep_pm10=NMB(yearmean_pm10_obs,yearmean_pm10_model[exp])
		RMSEemep_pm25=RMSE(yearmean_pm25_obs,yearmean_pm25_model[exp])
		RMSEemep_pm10=RMSE(yearmean_pm10_obs,yearmean_pm10_model[exp])
		#amean[1,n].annotate(('MFB: %6.2f')%MFBemep1,xy=(.01,0.9),xycoords='axes fraction')
		#amean[1,n].annotate(('MFE: %6.2f')%MFEemep1,xy=(.01,0.85),xycoords='axes fraction')
		amean[2,n].annotate(('PM2.5 NMB: %7.1f %%')%(NMBemep_pm25*100),xy=(.01,0.93),xycoords='axes fraction',fontsize=12)
		#amean[1,n].annotate(('NME: %6.2f')%NMEemep1,xy=(.01,0.75),xycoords='axes fraction')
		amean[2,n].annotate(('PM2.5 RMSE: %6.2f')%RMSEemep_pm25,xy=(.01,0.86),xycoords='axes fraction',fontsize=12)
		amean[2,n].annotate(('PM10 NMB: %7.1f %%')%(NMBemep_pm10*100),xy=(.01,0.79),xycoords='axes fraction',fontsize=12)
		#amean[1,n].annotate(('NME: %6.2f')%NMEemep1,xy=(.01,0.75),xycoords='axes fraction')
		amean[2,n].annotate(('PM10 RMSE: %6.2f')%RMSEemep_pm10,xy=(.01,0.72),xycoords='axes fraction',fontsize=12)
		amean[2,n].set_xlabel('EMEP OM [$\mu$g m$^{-3}$]',fontsize=12)
		amean[2,n].set_ylabel(EXP_NAMEs[n]+' OM [$\mu$g m$^{-3}$]',fontsize=12)
		amean[2,n].set_aspect('equal')
		amean[2,n].annotate(('%s)')%(letters[2][n]),xy=(0.0,1.02),xycoords='axes fraction',fontsize=18)
	
	std=np.nanstd(meanmodelmon['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
	mini=np.nanmin(meanmodelmon['obs'],axis=0)
	plt.tight_layout()
	#fmean.suptitle('EMEP')
		
	fmean.savefig('test-fig11_monthly-EMEP-allmean_9panels.png',dpi=600)
	fmean.savefig('test-fig11_monthly-EMEP-allmean_9panels.pdf',dpi=600)
	fmean.savefig(output_png_path+'article/fig11_revised_monthly-EMEP-allmean_9panels.png',dpi=600)
	print output_png_path+'article/fig11_revised_monthly-EMEP-allmean_9panels.png'
	fmean.savefig(output_pdf_path+'article/fig11_revised_monthly-EMEP-allmean_9panels.pdf',dpi=600)
	letters=[['a','b'],['c','d'],['e','f']]
	letters=[['a','b'],['c','d'],['e','f']]

	fmean2,amean2=plt.subplots(nrows=3,ncols=2,figsize=(18,10))#,tight_layout=True)
	std=np.nanstd(meanmodelmon_pm25['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon_pm25['obs'],axis=0)
	mini=np.nanmin(meanmodelmon_pm25['obs'],axis=0)
	amean2[0,0].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon_pm25['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	amean2[0,0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean2[0,0].set_ylim([0,7.5])
	amean2[0,1].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon_pm25['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	amean2[0,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean2[0,1].set_ylim([0,7.5])
	std=np.nanstd(meanmodelmon_pm10['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon_pm10['obs'],axis=0)
	mini=np.nanmin(meanmodelmon_pm10['obs'],axis=0)
	amean2[1,0].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon_pm10['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	amean2[1,0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean2[1,0].set_ylim([0,7.5])
	amean2[1,1].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon_pm10['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	amean2[1,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean2[1,1].set_ylim([0,7.5])
	letters=[['a','b'],['c','d']]
	letters=[['a','b'],['c','d'],['e','f']]
	letters=[['a','b','c'],['d','e','f'],['g','h','i']]
	extralabeli='OLDSOA-MEGAN2'
	for n,exp in enumerate(EXPS[:-1],0):
		if exp=='newsoa-ri':
			amean2[0,n].set_title('NEWSOA')
			labeli='NEWSOA'
		elif exp=='oldsoa-bhn':
			amean2[0,n].set_title('OLDSOA')
			labeli='OLDSOA'
		elif exp=='oldsoa-bhn-megan2':
			amean2[0,n].set_title('OLDSOA-MEGAN2')
			labeli='OLDSOA-MEGAN2'
		#print meanmodelmon[exp]
		#print np.nanstd(meanmodelmon[exp],axis=0)

		std=np.nanstd(meanmodelmon_pm25[exp],axis=0)
		maxi=np.nanmax(meanmodelmon_pm25[exp],axis=0)
		mini=np.nanmin(meanmodelmon_pm25[exp],axis=0)
		amean2[0,n].fill_between(np.linspace(1,12,12), np.nanmean(meanmodelmon_pm25[exp],axis=0)-std, np.nanmean(meanmodelmon_pm25[exp],axis=0)+std, color=shadingcolors[n],alpha=0.3)
		amean2[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon_pm25[exp],axis=0),color=colors[n],label=labeli+' POA+SOA')
		amean2[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmonpom_pm25[exp],axis=0),ls='--',color=colors[n],label=labeli+' POA')
		if n==1:
			amean2[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon_pm25['oldsoa-bhn-megan2'],axis=0),color=colors[2],label=extralabeli+' POA+SOA')
			#amean2[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmonpom_pm25['oldsoa-bhn-megan2'],axis=0),color=colors[2],label=extralabeli+' POA')

		amean2[0,n].set_ylim([0,8.5])
		amean2[0,n].legend(loc='upper right',fontsize=12)
		nmbmean=NMB(np.nanmean(meanmodelmon_pm25['obs'],axis=0),np.nanmean(meanmodelmon_pm25[exp],axis=0))
		mbmean=MB(np.nanmean(meanmodelmon_pm25['obs'],axis=0),np.nanmean(meanmodelmon_pm25[exp],axis=0))
		rmean=pearsonr(np.nanmean(meanmodelmon_pm25[exp],axis=0),np.nanmean(meanmodelmon_pm25['obs'],axis=0))
		amean2[0,n].annotate(('NMB (MB): %6.1f %% (%5.2f)')%(nmbmean*100,mbmean),xy=(0.01,0.9),xycoords='axes fraction',fontsize=12)
		amean2[0,n].annotate(('R: %6.2f,%6.2e')%(rmean[0],rmean[1]),xy=(0.01,0.82),xycoords='axes fraction',fontsize=12)
		amean2[0,n].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
		amean2[0,n].set_xticklabels(str_months())
		amean2[0,n].annotate(('%s)')%(letters[0][n]),xy=(0.0,1.02),xycoords='axes fraction',fontsize=18)
		amean2[0,n].set_xlabel('Month',fontsize=12)
		amean2[0,n].set_ylabel('OM in PM2.5 [ug m-3]',fontsize=12)
		#middle row
		std=np.nanstd(meanmodelmon[exp],axis=0)
		maxi=np.nanmax(meanmodelmon[exp],axis=0)
		mini=np.nanmin(meanmodelmon[exp],axis=0)
		amean2[1,n].fill_between(np.linspace(1,12,12), np.nanmean(meanmodelmon_pm10[exp],axis=0)-std, np.nanmean(meanmodelmon[exp],axis=0)+std, color=shadingcolors[n],alpha=0.3)
		
		amean2[1,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon_pm10[exp],axis=0),color=colors[n],label=labeli+' POA+SOA')

		amean2[1,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmonpom_pm10[exp],axis=0),ls='--',color=colors[n],label=labeli+' POA')
		if n==1:
			amean2[1,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon_pm10['oldsoa-bhn-megan2'],axis=0),color=colors[2],label=extralabeli+' POA+SOA')

			#amean2[1,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmonpom_pm10['oldsoa-bhn-megan2'],axis=0),ls='--',color=colors[2],label=extralabeli+' POA')

		amean2[1,n].set_ylim([0,8.5])
		amean2[1,n].legend(loc='upper right',fontsize=12)
		nmbmean=NMB(np.nanmean(meanmodelmon_pm10['obs'],axis=0),np.nanmean(meanmodelmon_pm10[exp],axis=0))
		mbmean=MB(np.nanmean(meanmodelmon_pm10['obs'],axis=0),np.nanmean(meanmodelmon_pm10[exp],axis=0))
		rmean=pearsonr(np.nanmean(meanmodelmon_pm10[exp],axis=0),np.nanmean(meanmodelmon_pm10['obs'],axis=0))
		amean2[1,n].annotate(('NMB (MB): %6.1f %% (%5.2f)')%(nmbmean*100,mbmean),xy=(0.01,0.9),xycoords='axes fraction',fontsize=12)
		amean2[1,n].annotate(('R: %6.2f,%6.2e')%(rmean[0],rmean[1]),xy=(0.01,0.82),xycoords='axes fraction',fontsize=12)
		amean2[1,n].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
		amean2[1,n].set_xticklabels(str_months())
		amean2[1,n].annotate(('%s)')%(letters[1][n]),xy=(0.0,1.02),xycoords='axes fraction',fontsize=18)
		amean2[1,n].set_xlabel('Month',fontsize=12)
		amean2[1,n].set_ylabel('OM in PM10 [ug m-3]',fontsize=12)
		#print np.nanmean(meanmodelmon[exp],axis=0)
		# bottom
		print yearmean_pm10_model.keys()
		#d1=amean2[2,n].scatter(yearmean_pm10_obs,yearmean_pm10_model[exp],marker='+',color=colors[n],s=25,label='PM10')
		#d2=amean2[2,n].scatter(yearmean_pm25_obs,yearmean_pm25_model[exp],marker='d',color=colors[n],s=25,label='PM25')
		print yearstderr_pm10_obs,
		#d1=amean2[2,n].errorbar(yearmean_pm10_obs,yearmean_pm10_model[exp],yerror=yearstderr_pm10_model[exp],xerror=yearstderr_pm10_obs,fmt='none',color=colors[n],ls='none',label='PM10')
		#d2=amean2[2,n].errorbar(yearmean_pm25_obs,yearmean_pm25_model[exp],yerror=yearstderr_pm25_model[exp],xerror=yearstderr_pm25_obs,fmt='none',color=colors[n],ls='none',label='PM25')
		d1=amean2[2,n].errorbar(yearmean_pm10_obs,yearmean_pm10_model[exp],yerr=yearstderr_pm10_model[exp],xerr=yearstderr_pm10_obs,fmt='.',color=colors[n],ls='none',label='PM10')
		d2=amean2[2,n].errorbar(yearmean_pm25_obs,yearmean_pm25_model[exp],yerr=yearstderr_pm25_model[exp],xerr=yearstderr_pm25_obs,fmt='.',color=colors[n],ls='none',label='PM25')
		if n==1:
			d1=amean2[2,n].errorbar(yearmean_pm10_obs,yearmean_pm10_model['oldsoa-bhn-megan2'],yerr=yearstderr_pm10_model['oldsoa-bhn-megan2'],xerr=yearstderr_pm10_obs,fmt='.',color=colors[2],ls='none',label='PM10')
			d2=amean2[2,n].errorbar(yearmean_pm25_obs,yearmean_pm25_model['oldsoa-bhn-megan2'],yerr=yearstderr_pm25_model['oldsoa-bhn-megan2'],xerr=yearstderr_pm25_obs,fmt='.',color=colors[2],ls='none',label='PM25')

		amean2[2,n].legend(loc=3,fontsize=12)
		amean2[2,n].loglog([0.001,1000],[0.001,1000],'k-')
		amean2[2,n].loglog([0.1,1000],[0.01,100],'k--')
		amean2[2,n].loglog([0.01,100],[0.1,1000],'k--')
		amean2[2,n].set_xlim([0.5,10])
		amean2[2,n].set_ylim([0.5,10])
		MFBemep1=MFB(yearmean_obs,yearmean_model[exp])
		MFEemep1=MFE(yearmean_obs,yearmean_model[exp])
		NMBemep1=NMB(yearmean_obs,yearmean_model[exp])
		NMEemep1=NME(yearmean_obs,yearmean_model[exp])
		RMSEemep1=RMSE(yearmean_obs,yearmean_model[exp])
		NMBemep_pm25=NMB(yearmean_pm25_obs,yearmean_pm25_model[exp])
		NMBemep_pm10=NMB(yearmean_pm10_obs,yearmean_pm10_model[exp])
		RMSEemep_pm25=RMSE(yearmean_pm25_obs,yearmean_pm25_model[exp])
		RMSEemep_pm10=RMSE(yearmean_pm10_obs,yearmean_pm10_model[exp])
		#amean2[1,n].annotate(('MFB: %6.2f')%MFBemep1,xy=(.01,0.9),xycoords='axes fraction')
		#amean2[1,n].annotate(('MFE: %6.2f')%MFEemep1,xy=(.01,0.85),xycoords='axes fraction')
		amean2[2,n].annotate(('PM2.5 NMB: %7.1f %%')%(NMBemep_pm25*100),xy=(.01,0.93),xycoords='axes fraction',fontsize=12)
		#amean2[1,n].annotate(('NME: %6.2f')%NMEemep1,xy=(.01,0.75),xycoords='axes fraction')
		amean2[2,n].annotate(('PM2.5 RMSE: %6.2f')%RMSEemep_pm25,xy=(.01,0.86),xycoords='axes fraction',fontsize=12)
		amean2[2,n].annotate(('PM10 NMB: %7.1f %%')%(NMBemep_pm10*100),xy=(.01,0.79),xycoords='axes fraction',fontsize=12)
		#amean2[1,n].annotate(('NME: %6.2f')%NMEemep1,xy=(.01,0.75),xycoords='axes fraction')
		amean2[2,n].annotate(('PM10 RMSE: %6.2f')%RMSEemep_pm10,xy=(.01,0.72),xycoords='axes fraction',fontsize=12)
		amean2[2,n].set_xlabel('EMEP OM [ug m-3]',fontsize=12)
		amean2[2,n].set_ylabel(EXP_NAMEs[n]+' OM [ug m-3]',fontsize=12)
		amean2[2,n].set_aspect('equal')
		amean2[2,n].annotate(('%s)')%(letters[2][n]),xy=(0.0,1.02),xycoords='axes fraction',fontsize=18)
	
	std=np.nanstd(meanmodelmon['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
	mini=np.nanmin(meanmodelmon['obs'],axis=0)
	plt.tight_layout()
	#fmean.suptitle('EMEP')
	fmean2.savefig('test-2panel-fig11_monthly-EMEP-allmean_6panels.png',dpi=600)
	fmean2.savefig('test-2panel-fig11_monthly-EMEP-allmean_6panels.pdf',dpi=600)
	fmean2.savefig(output_png_path+'article/fig11_monthly-EMEP-allmean_6panels.png',dpi=600)
	fmean2.savefig(output_pdf_path+'article/fig11_monthly-EMEP-allmean_6panels.pdf',dpi=600)
	
	print np.nanmean(yearmean_pm10_obs)
	print np.nanmean(yearmean_pm25_obs)
	print np.nanmean(yearmean_pm10_model['newsoa-ri'])
	print np.nanmean(yearmean_pm25_model['newsoa-ri'])
	print np.nanmean(yearmean_pm10_model['oldsoa-bhn'])
	print np.nanmean(yearmean_pm25_model['oldsoa-bhn'])

	print 'Npm25',N_pm25
	plt.show()
if __name__=='__main__':
	main()

