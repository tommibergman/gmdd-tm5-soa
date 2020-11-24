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
import improve_tools as ir
import logging
#import plot_m7
from settings import *
# def site_type():
# 	# categorization from http://vista.cira.colostate.edu/improve/wp-content/uploads/2016/08/IMPROVE_V_FullReport.pdf
# 	types={}
# 	types['urban']=[10730023]
# 	types['suburban']
# 	types['rurla']
def list_stations(indict):
	tabletex=open(basepath+'/paper/sitelist_improve.tex','w')
	tabletex.write("%Name & Station code&Longitude& Latitude &Height\\\\\n")
	sortdict={}
	for i in sorted(indict.keys()):
		print len(indict[i][0])
		print ('%6s,%35s,%4s,%4s,%4s')%(i,indict[i][0],indict[i][1],indict[i][2],indict[i][3])
		stationname=indict[i][0]
		print stationname
		stationname=stationname.replace("#","\\#")
		print stationname
		sortdict[stationname]=[i,indict[i][1],indict[i][2],indict[i][3]]
	for j in sorted(sortdict):
		print ('%-35s,%6s,%4s,%4s,%4s')%(j,sortdict[j][0],sortdict[j][2],sortdict[j][1],sortdict[j][3])
		tabletex.write("%s&%6s&%4s&%4s&%4s\\\\\n"%(j,sortdict[j][0],sortdict[j][2],sortdict[j][1],sortdict[j][3]))
	tabletex.close()

def str_months():
	return ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
def monthly_aggregation(sitedata):
	meanmodelmon={}
	meanmodelmon_pom={}
	meanmodelmon_C={}
	for exp in EXPS:
		meanmodelmon[exp]=np.zeros((len(sitedata.keys()),12))
		meanmodelmon_pom[exp]=np.zeros((len(sitedata.keys()),12))
		meanmodelmon_C[exp]=np.zeros((len(sitedata.keys()),12))
	meanmodelmon['obs']=np.zeros((len(sitedata.keys()),12))
	kk=0
	for i in sitedata:
		#print i
		#sitedata		
		obs=sitedata[i][6][1][:,1]
		timedata=sitedata[i][6][0][:]
		monthindices={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[]}
		monthdata=np.full(12,np.NAN)
		modelmonthdata=np.full(12,np.NAN)
		modelmonthdata_pom=np.full(12,np.NAN)
		modelmonthdata_soa=np.full(12,np.NAN)
		for j in timedata:
			monthindices[j.month].append(timedata.index(j))
		for k in monthindices:
			monthdata[k-1]=np.mean(sitedata[i][6][1][monthindices[k],1])
		meanmodelmon['obs'][kk,:]=monthdata[:]
		
		for exp in EXPS:
			model=nc.Dataset(basepathprocessed+'/improve_col/'+exp+'_'+i+'.nc','r').variables[i][:]
			model_pom=nc.Dataset(basepathprocessed+'/improve_col/'+exp+'_pom_'+i+'.nc','r').variables[i][:]
			for kmod in monthindices:
				print 
				modelmonthdata[kmod-1]=np.mean(model[monthindices[kmod]])
				modelmonthdata_pom[kmod-1]=np.mean(model_pom[monthindices[kmod]])
			meanmodelmon[exp][kk,:]=modelmonthdata
			meanmodelmon_pom[exp][kk,:]=modelmonthdata_pom
			if len(obs)!=len(model):
				logger.debug('Site %s number of obs: %i model: %i',i,len(obs),len(model))
				exit()
			mfb=ir.MFB(obs,model)
			mfe=ir.MFE(obs,model)
			nmb=ir.NMB(obs,model)
			nme=ir.NME(obs,model)
			rmse=ir.RMSE(obs,model)
			r=pearsonr(monthdata,modelmonthdata)
			mfbmon=ir.MFB(monthdata,modelmonthdata)
			mfemon=ir.MFE(monthdata,modelmonthdata)
			nmbmon=ir.NMB(monthdata,modelmonthdata)
			nmemon=ir.NME(monthdata,modelmonthdata)
			rmsemon=ir.RMSE(monthdata,modelmonthdata)
			rmon=pearsonr(monthdata,modelmonthdata)
			print i,exp,'R: ',r
			#model=model_site[i]['OM']		
		kk+=1

	return meanmodelmon,meanmodelmon_pom
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
def read_model_data(exp,sitedata,logger,basepathprocessed='/Users/bergmant/Documents/tm5-SOA/output/processed/',basepathraw='/Users/bergmant/Documents/tm5-SOA/output/raw/'):
	logger.info('Processing experiment %s ',exp )
	print len(glob.glob(path_improve_col+exp+'*'))
	print len(glob.glob(path_improve_col+'*'))
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
		print 'read radius'
		RW=ir.read_radius(filepath)
		r_pm25=1.25

		print 'read soa'
		logger.info('Reading SOA for experiment %s from %s',exp, filepath )
		soamassexp,soamodesexp=ir.read_mass('M_SOA',filepath,r_pm25) 
		print 'read oc'
		pommassexp,pommodesexp=ir.read_mass('M_POM',filepath,r_pm25)
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
		ncmodel,ncname,nctime,model_site=ir.col_improve(poamass,timeaxis,sitedata)
		modetest={}
		for i in soamodesexp:
			test,nn,nt,ms=ir.col_improve(soamodesexp[i],timeaxis,sitedata)
			print i,test,nn,nt 
			print i,nn,nt
			modetest[i]=test
			#raw_input()
		#ncmodel_pom,ncname_pom,nctime,model_site=ir.col_improve(pommassexp,timeaxis,sitedata)
			for data,name,time,sdata in zip(test,nn,nt,ms):
				print 'time',time
				write_netcdf_file([data],[name],path_improve_col+exp+'_'+i+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)
							#### select!
		for i in RW:
			RW2,nn,nt,ms=ir.col_improve(RW[i],timeaxis,sitedata)
			print i,test,nn,nt 
			print i,nn,nt
			for data,name,time,sdata in zip(RW2,nn,nt,ms):
				print 'time',time
				write_netcdf_file([data],[name],path_improve_col+exp+'_'+i+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)
		
		print model_site
		print nctime
		for data,name,time,sdata in zip(ncmodel,ncname,nctime,model_site):
			print 'time',time
			write_netcdf_file([data],[name],path_improve_col+exp+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)
		
		#collocate primary oa
		ncmodel_pom,ncname_pom,nctime,model_site=ir.col_improve(pommassexp,timeaxis,sitedata)
		for data,name,time,sdata in zip(ncmodel_pom,ncname,nctime,model_site):
			print 'time',time
			write_netcdf_file([data],[name],path_improve_col+exp+'_pom_'+name+'.nc',None,None,np.array(time))#,lat,lon)
		#collocate secondary oa
		ncmodel_soa,ncname_soa,nctime,model_site=ir.col_improve(soamassexp,timeaxis,sitedata)
		for data,name,time,sdata in zip(ncmodel_soa,ncname,nctime,model_site):
			print 'time',time
			write_netcdf_file([data],[name],path_improve_col+exp+'_soa_'+name+'.nc',None,None,np.array(time))#,lat,lon)
			

def main():
	logger=logging.getLogger('improve_plots_main')
	logger.setLevel(logging.DEBUG)
	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	# create formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	
	# add formatter to ch
	ch.setFormatter(formatter)

	# add ch to logger
	logger.addHandler(ch)
	#output_pdf_path='/Users/bergmant/Documents/tm5-SOA/figures/pdf/IMPROVE/'
	#output_png_path='/Users/bergmant/Documents/tm5-SOA/figures/png/IMPROVE/'
	#output_jpg_path='/Users/bergmant/Documents/tm5-SOA/figures/jpg/IMPROVE/'
	## get IMPROVE observations
	#basepathprocessed='/Users/bergmant/Documents/tm5-SOA/output/processed/'
	#basepathraw='/Users/bergmant/Documents/tm5-SOA/output/raw/'
	# improve network data reading
	# sitedata[site][name,lat,lon,ele,startdate,enddate,[date,[oc,ec,oc_unc,ec_unc]]]
	sitedata=ir.improve()
	list_stations(sitedata)
	EXPS=['newsoa-ri','oldsoa-bhn']#,'nosoa']
	for exp in EXPS:
		read_model_data(exp,sitedata,logger)
		# exit()
		# logger.info('Processing experiment %s ',exp )
		# print len(glob.glob(basepath+'improve_col/'+exp+'*'))
		# print len(glob.glob(basepath+'improve_col/*'))
		# logger.debug(basepath+'improve_col/'+exp+'*')
		# if not os.path.isdir(basepath+'improve_col/'):
		# 	logger.debug('Processing experiment %s ',exp )
		# 	os.mkdir(basepath+'improve_col/')
		# if len(glob.glob(basepath+'improve_col/'+exp+'*'))==0:# and len(glob.glob(basepath+'emep_col/'+exp+'*'))==0:
		# 	if  os.path.exists(basepathraw+exp+'/general_TM5_'+exp+'_2010.lev1.nc'):
		# 		filepath=basepathraw+exp+'/general_TM5_'+exp+'_2010.lev1.nc'
		# 	else:	
		# 		filepath='/Volumes/Utrecht/'+exp+'/general_TM5_'+exp+'_2010.lev1.nc'			
		# 	####READ
		# 	print 'read radius'
		# 	RW=ir.read_radius(filepath)
		# 	r_pm25=1.25

		# 	print 'read soa'
		# 	logger.info('Reading SOA for experiment %s from %s',exp, filepath )
		# 	soamassexp,soamodesexp=ir.read_mass('M_SOA',filepath,r_pm25) 
		# 	print 'read oc'
		# 	pommassexp,pommodesexp=ir.read_mass('M_POM',filepath,r_pm25)
		# 	print 'sum'	
		# 	poamass=soamassexp+pommassexp
		# 	print soamodesexp
		# 	print 'time'
		# 	oamass=[]
		# 	oamass.append(soamassexp+pommassexp)

		# 	timeaxis_o=nc.Dataset(filepath,'r').variables['time'][:]
		# 	lon_o=nc.Dataset(filepath,'r').variables['lon'][:]
		# 	lat_o=nc.Dataset(filepath,'r').variables['lat'][:]
		# 	t_unit=nc.Dataset(filepath,'r').variables['time'].units
		# 	## day of year, model data starts from day 0 so add 1
		# 	timeaxis=timeaxis_o-np.floor(timeaxis_o[0])+1
		# 	dt_axis=nc.num2date(timeaxis_o[:],units = t_unit,calendar = 'gregorian')[:]

		# 	#dt_ax2=[]
		# 	#dt_ax2=[nc.num2date(x,units = t_unit,calendar = 'gregorian') for x in timeaxis_o]
		# 	logger.debug('time axes, length %s, first value %s, type %s',len(dt_axis),dt_axis[0],type(dt_axis[0]))
		# 	#raw_input()
		# 	#Collocate total oa	
		# 	ncmodel,ncname,nctime,model_site=ir.col_improve(poamass,timeaxis,sitedata)
		# 	modetest={}
		# 	for i in soamodesexp:
		# 		test,nn,nt,ms=ir.col_improve(soamodesexp[i],timeaxis,sitedata)
		# 		print i,test,nn,nt 
		# 		print i,nn,nt
		# 		modetest[i]=test
		# 		#raw_input()
		# 	#ncmodel_pom,ncname_pom,nctime,model_site=ir.col_improve(pommassexp,timeaxis,sitedata)
		# 		for data,name,time,sdata in zip(test,nn,nt,ms):
		# 			print 'time',time
		# 			write_netcdf_file([data],[name],basepath+'improve_col/'+exp+'_'+i+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)
		# 						#### select!
		# 	for i in RW:
		# 		RW2,nn,nt,ms=ir.col_improve(RW[i],timeaxis,sitedata)
		# 		print i,test,nn,nt 
		# 		print i,nn,nt
		# 		for data,name,time,sdata in zip(RW2,nn,nt,ms):
		# 			print 'time',time
		# 			write_netcdf_file([data],[name],basepath+'improve_col/'+exp+'_'+i+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)
			
		# 	print model_site
		# 	print nctime
		# 	for data,name,time,sdata in zip(ncmodel,ncname,nctime,model_site):
		# 		print 'time',time
		# 		write_netcdf_file([data],[name],basepath+'improve_col/'+exp+'_'+name+'.nc',None,None,np.array(time))#,lat,lon)
			
		# 	#collocate primary oa
		# 	ncmodel_pom,ncname_pom,nctime,model_site=ir.col_improve(pommassexp,timeaxis,sitedata)
		# 	for data,name,time,sdata in zip(ncmodel_pom,ncname,nctime,model_site):
		# 		print 'time',time
		# 		write_netcdf_file([data],[name],basepath+'improve_col/'+exp+'_pom_'+name+'.nc',None,None,np.array(time))#,lat,lon)
		# 	#collocate secondary oa
		# 	ncmodel_soa,ncname_soa,nctime,model_site=ir.col_improve(soamassexp,timeaxis,sitedata)
		# 	for data,name,time,sdata in zip(ncmodel_soa,ncname,nctime,model_site):
		# 		print 'time',time
		# 		write_netcdf_file([data],[name],basepath+'improve_col/'+exp+'_soa_'+name+'.nc',None,None,np.array(time))#,lat,lon)
			
	meanmodelmon={}
	meanmodelmon_pom={}
	meanmodelmon_C={}
	for exp in EXPS:
		meanmodelmon[exp]=np.zeros((len(sitedata.keys()),12))
		meanmodelmon_pom[exp]=np.zeros((len(sitedata.keys()),12))
		meanmodelmon_C[exp]=np.zeros((len(sitedata.keys()),12))
	meanmodelmon['obs']=np.zeros((len(sitedata.keys()),12))
	kk=0
	paperpath=""
	tablex=open(paperpath+'stats_improve.txt','w')
	tablex.write("Name, R, NMB\n")
	tabletex=open(paperpath+'stats_improve.tex','w')
	tabletex.write("%Name & Obs&Model& R &NMB (\%)\n")
	for i in sitedata:
		#sitedata		
		obs=sitedata[i][6][1][:,1]
		timedata=sitedata[i][6][0][:]
		monthindices={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[]}
		monthdata=np.zeros([12])
		monthdata[:]=np.NAN
		modelmonthdata=np.zeros([12])
		modelmonthdata[:]=np.NAN
		modelmonthdata_pom=np.zeros([12])
		modelmonthdata_pom[:]=np.NAN
		modelmonthdata_soa=np.zeros([12])
		modelmonthdata_soa[:]=np.NAN
		# save indices for a given (jth) month indices in list
		for j in timedata:
			monthindices[j.month].append(timedata.index(j))
		# get monthly data from sitedata based on inidices
		for k in monthindices:
			monthdata[k-1]=np.mean(sitedata[i][6][1][monthindices[k],1])
		#kk=index(site)
		meanmodelmon['obs'][kk,:]=monthdata[:]

		for exp in EXPS:
			model=nc.Dataset(path_improve_col+exp+'_'+i+'.nc','r').variables[i][:]
			model_pom=nc.Dataset(path_improve_col+exp+'_pom_'+i+'.nc','r').variables[i][:]
			for kmod in monthindices:
				#print kmod
				modelmonthdata[kmod-1]=np.mean(model[monthindices[kmod]])
				modelmonthdata_pom[kmod-1]=np.mean(model_pom[monthindices[kmod]])
			meanmodelmon[exp][kk,:]=modelmonthdata
			meanmodelmon_pom[exp][kk,:]=modelmonthdata_pom
			if len(obs)!=len(model):
				logger.debug('Site %s number of obs: %i model: %i',i,len(obs),len(model))
				exit()
			mfb=ir.MFB(obs,model)
			mfe=ir.MFE(obs,model)
			nmb=ir.NMB(obs,model)
			nme=ir.NME(obs,model)
			rmse=ir.RMSE(obs,model)
			r=pearsonr(monthdata,modelmonthdata)
			mfbmon=ir.MFB(monthdata,modelmonthdata)
			mfemon=ir.MFE(monthdata,modelmonthdata)
			nmbmon=ir.NMB(monthdata,modelmonthdata)
			nmemon=ir.NME(monthdata,modelmonthdata)
			rmsemon=ir.RMSE(monthdata,modelmonthdata)
			rmon=pearsonr(monthdata,modelmonthdata)
			print i,exp,'R: ',r
			#model=model_site[i]['OM']		
		kk+=1
		plot_sites=False
		if plot_sites:
			pass
			f,a=plt.subplots(1)
			fmon,amon=plt.subplots(1)
			for exp in EXPS:	
				if exp==EXPS[0]:
					X=0.05
					colori='r'
				elif exp==EXPS[1]:
					X=0.8
					colori='b'
				else:# exp==EXPS[1]:
					X=0.4
					colori='g'

				amon.plot(np.linspace(1,12,12),modelmonthdata,'r',ls='-')		
				amon.plot(np.linspace(1,12,12),modelmonthdata_pom,c='r',ls='--')		
				amon.plot(np.linspace(1,12,12),modelmonthdata,c=colori,ls='-')		
				amon.plot(np.linspace(1,12,12),modelmonthdata_pom,c=colori,ls='--')		
				amon.set_title(i)
				a.plot(timedata,model,colori)
				a.annotate('EXP: '+exp,xy=(X,0.95),xycoords='axes fraction')
				a.annotate(('MFB: %6.2f')%mfb,xy=(X,0.9),xycoords='axes fraction')
				a.annotate(('MFE: %6.2f')%mfe,xy=(X,0.85),xycoords='axes fraction')
				a.annotate(('NMB: %6.2f')%nmb,xy=(X,0.8),xycoords='axes fraction')
				a.annotate(('NME: %6.2f')%nme,xy=(X,0.75),xycoords='axes fraction')
				a.annotate(('RMSE: %6.2f')%rmse,xy=(X,0.7),xycoords='axes fraction')
				a.annotate(('R: %6.2f')%r[0],xy=(X,0.65),xycoords='axes fraction')
				amon.annotate('EXP: '+exp,xy=(X,0.95),xycoords='axes fraction')
				amon.annotate(('MFB: %6.2f')%mfbmon,xy=(X,0.9),xycoords='axes fraction')
				amon.annotate(('MFE: %6.2f')%mfemon,xy=(X,0.85),xycoords='axes fraction')
				amon.annotate(('NMB: %6.2f')%nmbmon,xy=(X,0.8),xycoords='axes fraction')
				amon.annotate(('NME: %6.2f')%nmemon,xy=(X,0.75),xycoords='axes fraction')
				amon.annotate(('RMSE: %6.2f')%rmsemon,xy=(X,0.7),xycoords='axes fraction')
				amon.annotate(('R: %6.2f')%rmon[0],xy=(X,0.65),xycoords='axes fraction')
			amon.plot(np.linspace(1,12,12),monthdata)
			a.plot(timedata,obs,'k')
			a.set_title(i)
			f.savefig(output_pdf_path+'/IMPROVE/siteplots/timeseries-IMPROVE'+i+'.pdf',dpi=400)
			f.savefig(output_png_path+'/IMPROVE/siteplots/timeseries-IMPROVE'+i+'.png',dpi=400)
			f.savefig(output_jpg_path+'/IMPROVE/siteplots/timeseries-IMPROVE'+i+'.jpg',dpi=400)
			fmon.savefig(output_pdf_path+'/IMPROVE/siteplots/monthly-IMPROVE'+i+'.pdf',dpi=400)
			fmon.savefig(output_png_path+'/IMPROVE/siteplots/monthly-IMPROVE'+i+'.png',dpi=400)
			fmon.savefig(output_jpg_path+'/IMPROVE/siteplots/monthly-IMPROVE'+i+'.jpg',dpi=400)	

	
	#raw_input()
	std=np.nanstd(meanmodelmon['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
	mini=np.nanmin(meanmodelmon['obs'],axis=0)
	# fmean,amean=plt.subplots(ncols=2,figsize=(12,4))
	colors=['red', 'blue','black']
	shadingcolors=['#ff000033', '#00ff0033','#0000ff33','#55555533']
	# amean[0].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	# #amean[0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)+std,color=shadingcolors[2])
	# #amean[0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)-std,color=shadingcolors[2])
	# #amean[0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0),color=colors[2])
	# amean[0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])

	# #amean[0].set_title('Observations')
	# amean[0].set_ylim([0,2.0])
	# amean[1].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0), yerr=[std, std], fmt='o',color=shadingcolors[3],label='observations')
	# #plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)+std,'',color=shadingcolors[2])
	# #amean[1].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)-std,color=shadingcolors[2])
	# #amean[1].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0),color=colors[2])
	# amean[1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])

	# #amean[1].set_title('Observations')
	# amean[1].set_ylim([0,2.0])
	for n,exp in enumerate(EXPS[:],0):	
		if n==0:
			# amean[n].set_title('NEWSOA')
			labeli='NEWSOA'
		elif n==1:
			# amean[n].set_title('OLDSOA')
			labeli='OLDSOA'
		std=np.nanstd(meanmodelmon[exp],axis=0)
		maxi=np.nanmax(meanmodelmon[exp],axis=0)
		mini=np.nanmin(meanmodelmon[exp],axis=0)
		#amean.fill_between(np.linspace(0,12,12), mini, maxi, facecolor=shadingcolors[n], alpha=0.3,interpolate=True)
		#amean.plot(np.linspace(0,12,12), mini, color=shadingcolors[n])
		#amean.plot(np.linspace(0,12,12), maxi, color=shadingcolors[n])
		# amean[n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)+std, color=shadingcolors[n])
		# amean[n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)-std, color=shadingcolors[n])
		# amean[n].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon[exp],axis=0),color=colors[n],label=labeli)
		# amean[n].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
		# if n==0:
		# 	amean[n].set_ylim([0,0.5])
		# else:
		# 	amean[n].set_ylim([0,2.0])
		#amean[n].set_title(exp)
		nmbmean=ir.NMB(np.nanmean(meanmodelmon['obs'],axis=0),np.nanmean(meanmodelmon[exp],axis=0))
		rmean=pearsonr(np.nanmean(meanmodelmon[exp],axis=0),np.nanmean(meanmodelmon['obs'],axis=0))
		# amean[n].annotate(('NMB: %6.2f')%nmbmean,xy=(0.2,0.8),xycoords='axes fraction',fontsize=16)
		# amean[n].annotate(('R: %6.2f,%6.2e')%(rmean[0],rmean[1]),xy=(0.2,0.7),xycoords='axes fraction',fontsize=16)
		#ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
		#print np.nanmean(meanmodelmon[exp],axis=0)
		obsmean=np.nanmean(meanmodelmon['obs'])
		expmean=np.nanmean(meanmodelmon[exp])

		tablex.write(" %6s , %6.2f, %6.2f, %6.0f%% , %6.2f\n"%(labeli, obsmean, expmean, rmean[0],nmbmean*100))
		tabletex.write("& %6s & %6.2f & %6.2f& %6.0f\\%% & %6.2f\\\\\n"%(labeli, obsmean, expmean,nmbmean*100, rmean[0]))
		#\\unit{\\mu gm^{-3}}
		print exp,n,obsmean,expmean
	std=np.nanstd(meanmodelmon['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
	mini=np.nanmin(meanmodelmon['obs'],axis=0)
	#amean.fill_between(np.linspace(0,12,12), mini, maxi,facecolor=shadingcolors[3], alpha=0.3,interpolate=True)
	#amean.plot(np.linspace(0,12,12), mini,color=shadingcolors[3])
	#amean.plot(np.linspace(0,12,12), maxi,color=shadingcolors[3])
	# fmean.suptitle('IMPROVE')
	#amean.set_yscale("log", nonposy='clip')
	# fmean.savefig(output_png_path+'/IMPROVE/monthly-IMPROVE-SOAmean_2panels.png',dpi=400)




	# f2,ax2=plt.subplots(ncols=3,figsize=(12,4))
	# f2b,ax2b=plt.subplots(ncols=2,figsize=(10,4))
	# fb,axb=plt.subplots(ncols=3,figsize=(12,4))

	k=-1
	yearmean_model={}
	yearmean_model_pom={}
	for exp in EXPS:
		k+=1
		yearmean_obs=[]
		# f,ax=plt.subplots(1)
		for i in sitedata:
			#print i
			model=nc.Dataset(path_improve_col+exp+'_'+i+'.nc','r').variables[i][:]
			model_pom=nc.Dataset(path_improve_col+exp+'_pom_'+i+'.nc','r').variables[i][:]
			#model=model_site[i]['OM']		
			##### trying to get yearly means in dict
			if exp not in yearmean_model.keys():
				yearmean_model[exp]=[]
				yearmean_model[exp].append(np.mean(model))
				yearmean_model_pom[exp]=[]
				yearmean_model_pom[exp].append(np.mean(model))
			else:
				yearmean_model[exp].append(np.mean(model))
				yearmean_model_pom[exp].append(np.mean(model))
			
			####
			#yearmean_model[exp].append(np.mean(model))
			yearmean_obs.append(np.mean(sitedata[i][6][1][:,1]))
			#print type(model)
			if not 'temp_mod' in locals():
				temp_mod=model.copy()
			else:			
				temp_mod=np.concatenate((all_model,model))
			N_mod=np.shape(temp_mod)
			all_model=temp_mod.copy()
			if not 'temp_obs' in locals():
				temp_obs=sitedata[i][6][1][:,1].copy()
			else:			
				temp_obs=np.concatenate((all_obs,sitedata[i][6][1][:,1]))
			N_obs=np.shape(temp_obs)
			if N_obs!=N_mod:
				print N_mod,N_obs
				exit()
			all_obs=temp_obs.copy()
			# ax.loglog(sitedata[i][6][1][:,1],model,'or',ms=2)
			# axb[k].loglog(sitedata[i][6][1][:,1],model,'or',ms=2)
			
			xmax=max(max(model),max(sitedata[i][6][1][:,1]))
			xmin=min(min(model),min(sitedata[i][6][1][:,1]))
			ymax=xmax
			ymin=xmin
			# ax.set_xlabel('IMPROVE OM[pm25 1.8*C][ug m-3]')
			# ax.set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
			# ax.set_ylabel('TM5 OM[pm25][ug m-3]')
			# ax.set_title(sitedata[i][0])
			# ax.set_title(exp+': all sites')
			# axb[k].set_xlabel('IMPROVE OM[pm25 1.8*C][ug m-3]')
			# axb[k].set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
			# axb[k].set_ylabel('TM5 OM[pm25][ug m-3]')
			# axb[k].set_title(sitedata[i][0])
			# axb[k].set_title(exp+': all sites')
		#plt.show()
		# ax.plot([0.0001,1000],[0.0001,1000])
		# ax.plot([0.0001,1000],[0.001,10000],'g--')
		# ax.plot([0.001,10000],[0.0001,1000],'g--')
		# ax.set_ylim([.9e-4,2e1])
		# ax.set_xlim([.9e-4,2e1])
		# axb[k].plot([0.0001,1000],[0.0001,1000])
		# axb[k].plot([0.0001,1000],[0.001,10000],'g--')
		# axb[k].plot([0.001,10000],[0.0001,1000],'g--')
		# axb[k].set_ylim([.9e-4,2e1])
		# axb[k].set_xlim([.9e-4,2e1])
		print 'N '+exp+':',len(yearmean_model[exp]),len(yearmean_obs)
		if len(yearmean_model[exp])>1 and len(yearmean_obs)>1 and False:

			# f,ax=plt.subplots(1)
			print yearmean_model.keys,len(yearmean_model[exp]),len(yearmean_obs)
			# ax.loglog(yearmean_obs,yearmean_model[exp],'ob')
			# ax2[k].loglog(yearmean_obs,yearmean_model[exp],'or',ms=2)
			# if k<2:
			# 	ax2b[k].loglog(yearmean_obs,yearmean_model[exp],'or',ms=2)
			xmax=max(max(yearmean_model[exp]),max(yearmean_obs))
			xmin=min(min(yearmean_model[exp]),min(yearmean_obs))
			mfb=ir.MFB(yearmean_obs,yearmean_model[exp])
			mfe=ir.MFE(yearmean_obs,yearmean_model[exp])
			nmb=ir.NMB(yearmean_obs,yearmean_model[exp])
			nme=ir.NME(yearmean_obs,yearmean_model[exp])
			rmse=ir.RMSE(yearmean_obs,yearmean_model[exp])
			r=pearsonr(yearmean_obs,yearmean_model[exp])
			r_all=pearsonr(all_obs,all_model)
			mfb_all=ir.MFB(all_obs,all_model)
			mfe_all=ir.MFE(all_obs,all_model)
			nmb_all=ir.NMB(all_obs,all_model)
			nme_all=ir.NME(all_obs,all_model)
			rmse_all=ir.RMSE(all_obs,all_model)
			ymax=xmax
			ymin=xmin
			# ax.set_ylim([5E-3,5e0])
			# ax.set_xlim([5E-3,5e0])
			# ax.set_xlabel('IMPROVE OM[pm25 1.8*C][ug m-3]')
			# ax.set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
			# ax.set_ylabel('TM5 OM[pm25][ug m-3]')
			# ax2[k].set_ylim([5E-3,5e0])
			# ax2[k].set_xlim([5E-3,5e0])
			# ax2[k].set_xlabel('IMPROVE OM[pm25 1.8*C][ug m-3]')
			# ax2[k].set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
			# ax2[k].set_title('Run: '+exp)
			print 'year',exp,nmb,rmse,r
			# if exp=='newsoa-ri':
			# 	ax2[k].set_title('NEWSOA')
			# elif exp=='oldsoa-final':
			# 	ax2[k].set_title('OLDSOA')
			# else:
			# 	ax2[k].set_title('NOSOA')
			# if k<2:
			# 	ax2b[k].set_ylim([5E-3,5e0])
			# 	ax2b[k].set_xlim([5E-3,5e0])
			# 	ax2b[k].set_xlabel('IMPROVE OM[pm25 1.8*C][ug m-3]')
			# 	ax2b[k].set_ylabel('TM5 OM[pm25][ug m-3]')
			# 	ax2b[k].set_title('Run: '+exp)
			# 	#if exp=='soa-riccobono':
			# 	if exp=='newsoa-ri':
			# 		ax2b[k].set_title('NEWSOA')
	
			# 	elif exp=='oldsoa-final':
			# 		ax2b[k].set_title('OLDSOA')
			#print yearmean_obs
			print exp,'rmse: ',rmse,rmse_all
			print exp,'mfb: ',mfb,mfb_all
			print exp,'mfe: ',mfe,mfe_all
			print exp,'nmb: ',nmb,nmb_all
			print exp,'nme: ',nme,nme_all
			# ax.plot([0.001,1000],[0.001,1000])
			# ax.plot([0.001,1000],[0.01,10000],'g--')
			# ax.plot([0.01,10000],[0.001,1000],'g--')
			# ax2[k].plot([0.001,1000],[0.001,1000])
			# ax2[k].plot([0.001,1000],[0.01,10000],'g--')
			# ax2[k].plot([0.01,10000],[0.001,1000],'g--')
			# ax2[k].annotate(('MFB: %6.2f')%mfb,xy=(0.05,0.9),xycoords='axes fraction')
			# ax2[k].annotate(('MFE: %6.2f')%mfe,xy=(0.05,0.85),xycoords='axes fraction')
			# ax2[k].annotate(('NMB: %6.2f')%nmb,xy=(0.05,0.8),xycoords='axes fraction')
			# ax2[k].annotate(('NME: %6.2f')%nme,xy=(0.05,0.75),xycoords='axes fraction')
			# ax2[k].annotate(('RMSE: %6.2f')%rmse,xy=(0.05,0.7),xycoords='axes fraction')
			# ax2[k].annotate(('R: %6.2f')%r[0],xy=(0.05,0.65),xycoords='axes fraction')
			# axb[k].annotate(('MFB: %6.2f')%mfb_all,xy=(0.05,0.95),xycoords='axes fraction')
			# axb[k].annotate(('MFE: %6.2f')%mfe_all,xy=(0.05,0.9),xycoords='axes fraction')
			# axb[k].annotate(('NMB: %6.2f')%nmb_all,xy=(0.05,0.85),xycoords='axes fraction')
			# axb[k].annotate(('NME: %6.2f')%nme_all,xy=(0.05,0.8),xycoords='axes fraction')
			# axb[k].annotate(('RMSE: %6.2f')%rmse_all,xy=(0.05,0.75),xycoords='axes fraction')
			# axb[k].annotate(('R: %6.2f')%r_all[0],xy=(0.05,0.7),xycoords='axes fraction')
			
			# if k<2:
			# 	ax2b[k].plot([0.001,1000],[0.001,1000])
			# 	ax2b[k].plot([0.001,1000],[0.01,10000],'g--')
			# 	ax2b[k].plot([0.01,10000],[0.001,1000],'g--')
			# 	ax2b[k].annotate(('MFB: %6.2f')%mfb,xy=(0.05,0.9),xycoords='axes fraction')
			# 	ax2b[k].annotate(('MFE: %6.2f')%mfe,xy=(0.05,0.85),xycoords='axes fraction')
			# 	ax2b[k].annotate(('NMB: %6.2f')%nmb,xy=(0.05,0.8),xycoords='axes fraction')
			# 	ax2b[k].annotate(('NME: %6.2f')%nme,xy=(0.05,0.75),xycoords='axes fraction')
			# 	ax2b[k].annotate(('RMSE: %6.2f')%rmse,xy=(0.05,0.7),xycoords='axes fraction')
			# 	ax2b[k].annotate(('R: %6.2f')%r[0],xy=(0.05,0.65),xycoords='axes fraction')
		print 	
	# f2b.savefig(output_png_path+'/IMPROVE/scatter-IMPROVE-1x2.png',dpi=400)
	# f2b.savefig(output_jpg_path+'/IMPROVE/scatter-IMPROVE-1x2.jpg',dpi=400)
	# f2b.savefig(output_pdf_path+'/IMPROVE/scatter-IMPROVE-1x2.pdf',dpi=400)
	# f2.savefig(output_pdf_path+'/IMPROVE/scatter-IMPROVE-1x3.pdf',dpi=400)
	# f2.savefig(output_png_path+'/IMPROVE/scatter-IMPROVE-1x3.png',dpi=400)
	# f2.savefig(output_jpg_path+'/IMPROVE/scatter-IMPROVE-1x3.jpg',dpi=400)
	# fb.savefig(output_pdf_path+'/IMPROVE/scatter-all-IMPROVE-1x3.pdf',dpi=400)
	# fb.savefig(output_png_path+'/IMPROVE/scatter-all-IMPROVE-1x3.png',dpi=400)
	# fb.savefig(output_jpg_path+'/IMPROVE/scatter-all-IMPROVE-1x3.jpg',dpi=400)
	#plt.show()	

	fmean,amean=plt.subplots(ncols=2,nrows=2,figsize=(12,8))
	colors=['red', 'blue','black']
	shadingcolors=['#ff000033', '#0000ff33','#00ff0033','#55555533']
	#amean[0,0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)+std,color=shadingcolors[2])
	#amean[0,0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)-std,color=shadingcolors[2])
	#amean[0,0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0),color=colors[2])
	amean[0,0].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0), yerr=[std, std], fmt='--o',color=shadingcolors[3],label='observations')
	amean[0,0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean[0,0].set_ylim([0,2.0])

	amean[0,1].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0), yerr=[std, std], fmt='--o',color=shadingcolors[3],label='observations')
		#amean[2].set_title('Observations')
	amean[0,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean[0,1].set_ylim([0,2.0])
	letters=[['a','b'],['c','d']]
	for n,exp in enumerate(EXPS[:],0):	
		if n==0:
			labeli='NEWSOA'
			amean[0,n].set_title('NEWSOA')
			colori='r'
		elif n==1:
			labeli='OLDSOA'
			amean[0,n].set_title('OLDSOA')
			colori='b'
		print n, exp
		std=np.nanstd(meanmodelmon[exp],axis=0)
		maxi=np.nanmax(meanmodelmon[exp],axis=0)
		mini=np.nanmin(meanmodelmon[exp],axis=0)
		#amean.fill_between(np.linspace(0,12,12), mini, maxi, facecolor=shadingcolors[n], alpha=0.3,interpolate=True)
		#amean.plot(np.linspace(0,12,12), mini, color=shadingcolors[n])
		#amean.plot(np.linspace(0,12,12), maxi, color=shadingcolors[n])
		#amean[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)+std, color=shadingcolors[n])
		#amean[0,n].plot(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)-std, color=shadingcolors[n])
		amean[0,n].fill_between(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)-std, np.nanmean(meanmodelmon[exp],axis=0)+std, color=shadingcolors[n],alpha=0.3)
		
		amean[0,n].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon[exp],axis=0),color=colors[n],label=labeli+' POA+SOA')
		amean[0,n].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon_pom[exp],axis=0),color=colors[n],ls='--',label=labeli+' POA')
		amean[0,n].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
		amean[0,n].set_xticklabels(str_months())
		if n==0:
			amean[0,n].set_ylim([0,2.0])
		else:
			amean[0,n].set_ylim([0,2.0])
		#amean[n].set_title(exp)
		print np.shape(meanmodelmon['obs'])
		print np.shape(yearmean_obs)

		nmbmean=ir.NMB(np.nanmean(meanmodelmon['obs'],axis=0),np.nanmean(meanmodelmon[exp],axis=0))
		mbmean=ir.MB(np.nanmean(meanmodelmon['obs'],axis=0),np.nanmean(meanmodelmon[exp],axis=0))
		rmean=pearsonr(np.nanmean(meanmodelmon[exp],axis=0),np.nanmean(meanmodelmon['obs'],axis=0))
		amean[0,n].annotate(('NMB (MB): %6.1f %% (%4.2f)')%(nmbmean*100,mbmean),xy=(0.03,0.95),xycoords='axes fraction',fontsize=10)
		amean[0,n].annotate(('R: %6.2f')%(rmean[0]),xy=(0.03,0.9),xycoords='axes fraction',fontsize=10)
		amean[0,n].set_xlabel('Month')
		#amean[0,n].set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
		amean[0,n].set_ylabel('OM[pm25][ug m-3]')
		amean[0,n].annotate(('%s)')%(letters[0][n]),xy=(0.0,1.02),xycoords='axes fraction',fontsize=14)
		amean[0,n].legend(loc='upper right')
		#ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
		print np.nanmean(meanmodelmon[exp],axis=0)
		print 'teststste',n,exp
		amean[1,n].loglog(yearmean_obs,yearmean_model[exp],'o',c=colori,ms=3)
		amean[1,n].plot([0.0001,1000],[0.0001,1000],'k')
		amean[1,n].plot([0.0001,1000],[0.001,10000],'k--')
		amean[1,n].plot([0.001,10000],[0.0001,1000],'k--')
		amean[1,n].set_ylim([.5e-2,3e0])
		amean[1,n].set_xlim([.5e-2,3e0])
		amean[1,n].set_xlabel('IMPROVE OM[pm25][ug m-3]')
		#amean[0,n].set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
		amean[1,n].set_ylabel(EXP_NAMEs[n] + ' OM[pm25][ug m-3]')
		amean[1,n].set_aspect('equal')			
		#amean[1,n].legend(loc=4)
		nmbmean=ir.NMB(yearmean_obs,yearmean_model[exp])
		mbmean=ir.MB(yearmean_obs,yearmean_model[exp])
		rmean=pearsonr(yearmean_model[exp],yearmean_obs)
		rlogmean=pearsonr(np.log(yearmean_model[exp]),np.log(yearmean_obs))
		amean[1,n].annotate(('NMB (MB): %5.1f %% (%4.2f)')%(nmbmean*100,mbmean),xy=(0.45,0.06),xycoords='axes fraction',fontsize=10)
		amean[1,n].annotate(('R (R log): %6.2f, (%4.2f) ')%(rmean[0],rlogmean[0]),xy=(0.45,0.01),xycoords='axes fraction',fontsize=10)
		amean[1,n].annotate(('%s)')%(letters[1][n]),xy=(0.0,1.02),xycoords='axes fraction',fontsize=14)
			

	
	std=np.nanstd(meanmodelmon['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
	mini=np.nanmin(meanmodelmon['obs'],axis=0)
	#amean.fill_between(np.linspace(0,12,12), mini, maxi,facecolor=shadingcolors[3], alpha=0.3,interpolate=True)
	#amean.plot(np.linspace(0,12,12), mini,color=shadingcolors[3])
	#amean.plot(np.linspace(0,12,12), maxi,color=shadingcolors[3])
	fmean.suptitle('IMPROVE')
	#amean.set_yscale("log", nonposy='clip')
	fmean.savefig(output_png_path+'/IMPROVE/scatter-seasonal-IMPROVE-2x2.png',dpi=400)
	fmean.savefig(output_pdf_path+'/IMPROVE/scatter-seasonal-IMPROVE-2x2.pdf')
	fmean.savefig(output_jpg_path+'/IMPROVE/scatter-seasonal-IMPROVE-2x2.jpg',dpi=400)
	plt.show()


	
if __name__=='__main__':
	main()
