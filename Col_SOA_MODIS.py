from __future__ import print_function
import netCDF4  as nc
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from general_toolbox import lonlat, get_gridboxarea
from pylab import *
#from aerocom_sites import aerocom_sites
import Colocate_MODIS as MODIS
import os
from settings import *

#datagroup = 'od550aer:newdata/aerocom3_TM5_AP3-ctrl2016_global_2010_hourly.od550aer.nc',dir_out='TM5_daily_subset_new/', file_out1 = 'aerocom3_TM5_AP3-ctrl2016_global_2010_hourly.od550aer.'


class modis_compare(object):
	"""docstring for modis_compare"""
	def __init__(self, arg):
		super(modis_compare, self).__init__()
		self.paths={}
		self.paths['Basepath']='/Users/bergmant/Documents/tm5-soa/'
		self.paths['TM5_input']='/Volumes/Utrecht/'
		self.paths['outdir']='/Users/bergmant/Documents/tm5-soa/colocated/'
		self.paths['subset']='/Users/bergmant/Documents/tm5-soa/subset/'
		self.paths['output_pdf']='/Users/bergmant/Documents/tm5-soa/FIGURES/pdf/MODIS/'
		self.paths['output_jpg']='/Users/bergmant/Documents/tm5-soa/FIGURES/jpg/MODIS/'
		self.paths['output_png']='/Users/bergmant/Documents/tm5-soa/FIGURES/png/MODIS/'
		self.EXPs=[]
		self.year=2010


def process_modis():
	paths={}
	paths['Basepath']='/Users/bergmant/Documents/tm5-soa/'
	paths['TM5_input']='/Volumes/Utrecht/'
	paths['outdir']='tm5-SOA/colocated/'
	paths['subset']='tm5-SOA/subset/'
	#MODIS.subset_TM5('od550aer:'+paths['TM5_input']+'general_TM5_newsoa-ri_2010.lev0.od550aer.nc',paths['subset'],'general_TM5_newsoa-ri_2010.lev0.od550aer.' )
	#MODIS.colocate_MODIS(paths['Basepath'],paths['outdir'],paths['subset'],'general_TM5_newsoa-ri_2010.lev0.od550aer.','tm5-SOA/')
	#MODIS.bias('/Volumes/clouds/Masked_data_',paths['outdir']+'Masked_collocated_TM5_ctrl2016_MODIS_test/tm5-SOA/','/Volumes/Utrecht/tm5-SOA/BIAS/')
	#MODIS.aggregate_TM5('tm5-SOA/colocated/Masked_collocated_TM5_ctrl2016_MODIS_test/tm5-SOA/','/Volumes/Utrecht/tm5-SOA/aggregated_TM5')

	EXPs=['newsoa-ri','oldsoa-final']#,'NOSOA']
	EXPs=['newsoa-ri','oldsoa-bhn']#,'NOSOA']
	#EXPs=['newsoa-ri']#,'NOSOA']
	#EXPs=['oldsoa-final']#,'NOSOA']
	year=2010
	for EXP in EXPs:
		od550aer_filename='general_TM5_{}_{}.lev0.od550aer.nc'.format(EXP,year)
		paths['basedir']='/Volumes/Utrecht/'
		paths['out_subset']='/Volumes/Utrecht/'+EXP+'_subset/'
		paths['out_col']='/Volumes/Utrecht/'+EXP+'_col/'
		paths['out_aggre']='/Volumes/Utrecht/'+EXP+'_aggre/'
		paths['out_bias']='/Volumes/Utrecht/'+EXP+'_bias/'
		print(od550aer_filename)
		print(od550aer_filename)
		'''if os.path.isdir(paths['basedir']):
			if not os.path.isdir(paths['out_subset']):
				os.makedirs(paths['out_subset'])
			MODIS.subset_TM5('od550aer:'+paths['TM5_input']+od550aer_filename,paths['out_subset'],od550aer_filename[:-2] )
		if os.path.isdir(paths['basedir']):
			if not os.path.isdir(paths['out_col']):
				os.makedirs(paths['out_col'])
			MODIS.colocate_MODIS(paths['out_col'],paths['out_subset'],od550aer_filename[:-2],EXP)
		#MODIS.bias('/Volumes/clouds/Masked_data_',paths['outdir']+'Masked_collocated_TM5_'+EXP+'_MODIS/',paths['out_bias'])
		'''
		MODIS.aggregate_TM5(paths['out_col'],paths['out_aggre'])
		MODIS.varorder(paths['out_aggre'],paths['out_aggre']+'/ncpdq/')
def plot_modis_diff():
	#output_pdf_path='/Users/bergmant/Documents/tm5-soa/FIGURES/pdf/MODIS/'
	#output_png_path='/Users/bergmant/Documents/tm5-soa/FIGURES/png/MODIS/'
	#output_jpg_path='/Users/bergmant/Documents/tm5-soa/FIGURES/jpg/MODIS/'
	EXPs=['newsoa-ri','oldsoa-final']#,'NOSOA']
	EXPs=['newsoa-ri','oldsoa-bhn']#,'NOSOA']
	#EXPs=['newsoa-ri','oldsoa-final','nosoa']
	cellarea_file='/Users/bergmant/Documents/tm5-soa/output/ec-ei-an0tr6-sfc-glb100x100-0000-oro.nc'
	cellarea=nc.Dataset(cellarea_file,'r').variables['cell_area'][:]
	lsm_file='/Users/bergmant/Documents/tm5-soa/output/ec-ei-an0tr6-sfc-glb100x100-lsm.nc'
	lsm=nc.Dataset(lsm_file,'r').variables['lsm'][:]
	model='/Users/bergmant/Documents/tm5-soa/output/col_MOD04_MYD04_QA2_L2_TM5_'+EXPs[0]+'_2010_1x1_yearmean.nc'  #Aggregated_lin_Col_TM5_'+EXPs[0]+'_MYD04_MOD04_L2_2010_1x1_yearmean.nc'
	model2='/Users/bergmant/Documents/tm5-soa/output/col_MOD04_MYD04_QA2_L2_TM5_'+EXPs[1]+'_2010_1x1_yearmean.nc'
	#modis='/Users/bergmant/Documents/tm5-soa/output/processed/TM5-MODIS//Aggregated_MODIS_L2_2010_1x1_yearmean.nc'
	modis='/Users/bergmant/Documents/tm5-soa/output/processed/TM5-MODIS/MOD04_MYD04_L2_QA2_aggregated_2010_1x1_yearmean.nc'
	modisdata=np.squeeze(nc.Dataset(modis,'r').variables['AOD_550_Dark_Target_Deep_Blue_Combined'][:]).transpose()
	print (model)
	modeldata=np.squeeze(nc.Dataset(model,'r').variables['od550aer'][:].transpose())
	modeldata2=np.squeeze(nc.Dataset(model2,'r').variables['od550aer'][:].transpose())
	modislon=nc.Dataset(modis,'r').variables['longitude'][:]
	modislat=nc.Dataset(modis,'r').variables['latitude'][:]
	#print(np.shape(modisdata))
	lsm2=lsm.copy()
	lsm[lsm>1]=1
	lsm[lsm<1]=nan
	lsm2[lsm2<1]=1
	lsm2[lsm2>1]=nan
	print(np.mean(modisdata),np.nanmean(modisdata))
	print(np.mean(modeldata),np.nanmean(modeldata))
	print(np.mean(modeldata2),np.nanmean(modeldata2))
	print ('cellareamean')
	print(np.mean(modisdata),np.nansum(modisdata*cellarea)/np.sum(cellarea))
	print(np.mean(modeldata),np.nansum(modeldata*cellarea)/(np.sum(cellarea)))
	print(np.mean(modeldata2),np.nansum(modeldata2*cellarea)/(np.sum(cellarea)))
	print ('land')
	print(np.mean(modisdata),np.nansum(modisdata*cellarea*lsm)/np.nansum(cellarea*lsm))
	print(np.mean(modeldata),np.nansum(modeldata*cellarea*lsm)/(np.nansum(cellarea*lsm)))
	print(np.mean(modeldata2),np.nansum(modeldata2*cellarea*lsm)/(np.nansum(cellarea*lsm)))
	print ('ocean')
	print(np.mean(modisdata),np.nansum(modisdata*cellarea*lsm2)/np.nansum(cellarea*lsm2))
	print(np.mean(modeldata),np.nansum(modeldata*cellarea*lsm2)/(np.nansum(cellarea*lsm2)))
	print(np.mean(modeldata2),np.nansum(modeldata2*cellarea*lsm2)/(np.nansum(cellarea*lsm2)))

	f,ax=plt.subplots(ncols=1,figsize=(8,4))
	k=-1
	#print(np.shape(modisdata),modeldata.shape)
	lons, lats = np.meshgrid(modislon,modislat)
	#for exp in EXPS:
	k+=1
	bounds_load=[-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3]

	ax.set_title('diff')
	m=Basemap(projection='robin',lon_0=0,ax=ax)
	bounds = [-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	mycmap=plt.get_cmap('coolwarm',11) 

	#image=m.contourf(lons,lats,modeldata-modisdata,bounds_load,cmap=plt.cm.coolwarm,latlon=True)

	m.drawparallels(np.arange(-90.,90.,30.))
	m.drawmeridians(np.arange(-180.,180.,60.))
	m.drawcoastlines()
	cs = m.pcolormesh(lons,lats,((modeldata-modisdata).squeeze()),norm=norm,latlon=True,cmap=mycmap)
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7)
	c.set_label('AOD bias [TM5-MODIS]',fontsize=18)
	c.ax.tick_params(labelsize=10)
	print (np.mean(modisdata))
	print (np.mean(modeldata))
	print (np.mean(modeldata2))
	print (np.max(modisdata))
	print (np.max(modeldata))
	print (np.max(modeldata2))
	#print ('NEW-MODIS:',np.max(modeldata.mean(0)-modisdata.mean(0)))
	#print ('OLD-MODIS:',np.max(modeldata2.mean(0)-modisdata.mean(0)))
	#print ('NEW-OLD:',np.max(modeldata.mean(0)-modeldata2.mean(0)))
	

	fsoa,ax=plt.subplots(figsize=(10,7))
	m = Basemap(projection='robin',lon_0=0)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,120.,30.))
	m.drawmeridians(np.arange(0.,3060.,60.))
	mycmap=plt.get_cmap('Purples',11) 
	# define the bins and normalize
	bounds = [0.025,0.075,0.15,0.225,0.30,0.375,0.45,0.525,0.6,0.675,0.75]
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	plt.title('AOD collocated annual mean (TM5)',fontsize=18)
	m = Basemap(projection='robin',lon_0=0,ax=ax)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,120.,30.))
	m.drawmeridians(np.arange(0.,3060.,60.))
	#mycmap=plt.get_cmap('coolwarm',11) 
	# define the bins and normalize
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	#nlats = len(tm5lat)
	#nlons = len(tm5lon)
	#lons, lats = np.meshgrid(tm5lon, tm5lat)
	cs = m.pcolormesh(lons,lats,(modeldata.squeeze()),norm=norm,latlon=True,cmap=mycmap)
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7)
	c.set_label('AOD [TM5]',fontsize=18)
	c.ax.tick_params(labelsize=10)
	#c.set_label('AOD relative bias [(TM5-AERONET)/AERONET]')
	#fsoa.savefig(output_pdf_path+'MODIS/TM5-newsoa-ri_2010.pdf')
	#fsoa.savefig(output_png_path+'MODIS/TM5-newsoa-ri_2010.png',dpi=600)
	
	fmodis,ax=plt.subplots(figsize=(10,7))
	m = Basemap(projection='robin',lon_0=0)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,120.,30.))
	m.drawmeridians(np.arange(0.,3060.,60.))
	mycmap=plt.get_cmap('Purples',11) 
	# define the bins and normalize
	bounds = [0.025,0.075,0.15,0.225,0.30,0.375,0.45,0.525,0.6,0.675,0.75]
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	plt.title('AOD collocated annual mean (MODIS)',fontsize=18)
	m = Basemap(projection='robin',lon_0=0,ax=ax)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,120.,30.))
	m.drawmeridians(np.arange(0.,3060.,60.))
	#mycmap=plt.get_cmap('coolwarm',11) 
	# define the bins and normalize
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	cs = m.pcolormesh(lons,lats,(modisdata.squeeze()),norm=norm,latlon=True,cmap=mycmap)
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7)
	c.set_label('AOD [MODIS]',fontsize=18)
	c.ax.tick_params(labelsize=10)
	#c.set_label('AOD relative bias [(TM5-AERONET)/AERONET]')
	#fmodis.savefig(output_pdf_path+'MODIS/MODIS_2010.pdf')
	#fmodis.savefig(output_png_path+'MODIS/MODIS_2010.png',dpi=600)
def plot_modis_daily():
	EXPs=['newsoa-ri','oldsoa-final']#,'NOSOA']
	EXPs=['newsoa-ri','oldsoa-bhn']#,'NOSOA']
	#/col_MOD04_MYD04_QA2_L2_TM5_newsoa-ri_2010_1x1_monthlymean.nc 
	model='/Users/bergmant/Documents/tm5-soa/output/col_MOD04_MYD04_QA2_L2_TM5_'+EXPs[0]+'_2010_1x1_dailymean.nc'
	model2='/Users/bergmant/Documents/tm5-soa/output/col_MOD04_MYD04_QA2_L2_TM5_'+EXPs[1]+'_2010_1x1_dailymean.nc'
	#modis='/Users/bergmant/Documents/tm5-soa/output/processed/TM5-MODIS//Aggregated_MODIS_L2_2010_1x1_dailymean.nc'
	modis='/Users/bergmant/Documents/tm5-soa/output/processed/TM5-MODIS/MOD04_MYD04_L2_QA2_aggregated_2010_1x1_dailymean.nc'
	modisdata=np.squeeze(nc.Dataset(modis,'r').variables['AOD_550_Dark_Target_Deep_Blue_Combined'][:])
	modeldata=np.squeeze(nc.Dataset(model,'r').variables['od550aer'][:])
	#modeldata2=np.squeeze(nc.Dataset(model2,'r').variables['od550aer'][:])
	modislon=nc.Dataset(modis,'r').variables['longitude'][:]
	modislat=nc.Dataset(modis,'r').variables['latitude'][:]
	f,ax=plt.subplots(1)
	ax.plot(np.mean(modisdata,axis=(1,2)),'b')
	ax.plot(np.mean(modeldata,axis=(1,2)),'r')
	f,ax=plt.subplots(1)
	ax.plot(modisdata[:,90,180],'b')
	ax.plot(modeldata[:,90,180],'r')

def plot_diff_map(ax,coord,data,name):
	#ax.set_title('{0} season {1}'.format(season))
	#modis='/Users/bergmant/Documents/tm5-soa/output/TM5-MODIS//Aggregated_MODIS_L2_2010_1x1_dailymean.nc'
	#modislon=nc.Dataset(modis,'r').variables['longitude'][:]
	#modislat=nc.Dataset(modis,'r').variables['latitude'][:]
	lons, lats = np.meshgrid(coord[0],coord[1])
	m=Basemap(projection='robin',lon_0=0,ax=ax)
	bounds = [-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	mycmap=plt.get_cmap('coolwarm',11) 

	#image=m.contourf(lons,lats,modeldata-modisdata,bounds_load,cmap=plt.cm.coolwarm,latlon=True)

	m.drawparallels(np.arange(-90.,90.,30.))
	m.drawmeridians(np.arange(-180.,180.,60.))
	m.drawcoastlines()
	#print((modeldata-modisdata).squeeze().shape)
	#print ((modeldata[k,:,:]-modisdata[k,:,:]).mean())
	#cs = m.pcolormesh(lons,lats,((modeldata[k,:,:]-modisdata[k,:,:]).squeeze()),norm=norm,latlon=True,cmap=mycmap)
	#print (np.shape(data)),np.shape(lats),np.shape(lons)
	cs = m.pcolormesh(lons,lats,data,norm=norm,latlon=True,cmap=mycmap)
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.85,ax=ax)
	c.set_label('AOD bias [TM5-MODIS]',fontsize=14)
	c.ax.tick_params(labelsize=8)
	#print (np.mean(modisdata))
	#print (np.mean(modeldata))
	#print (np.mean(modeldata2))

def plot_seasonal_diff(data1,data2,title):
	seasons=[['DJF',[11,0,1]],['MAM',[2,3,4]],['JJA',[5,6,7]],['SON',[8,9,10]]]
	f,ax=plt.subplots(nrows=2,ncols=2,figsize=(16,10))
	k=-1
	for i in range(2):
		for j in range(2):
			k+=1
			data=(data1[seasons[k][1],:,:].mean(0)-data2[seasons[k][1],:,:].mean(0)).squeeze()
			#print (data.shape)
			bounds_load=[-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3]
			plot_diff_map(ax[i,j],[modislon,modislat],data.transpose(),'NEWSOA-MODIS')
			ax[i,j].set_title('{} season {}'.format(title,seasons[k][0]))
	return f,ax
def plot_modis_season():
	EXPs=['newsoa-ri','oldsoa-final']#,'NOSOA']
	EXPs=['newsoa-ri','oldsoa-bhn']#,'NOSOA']
	EXP_NAMEs=['NEWSOA','OLDSOA']#,'NOSOA']
	
	model='/Users/bergmant/Documents/tm5-soa/output/col_MOD04_MYD04_QA2_L2_TM5_'+EXPs[0]+'_2010_1x1_monthlymean.nc'  #Aggregated_lin_Col_TM5_'+EXPs[0]+'_MYD04_MOD04_L2_2010_1x1_monthlymean.nc'
	model2='/Users/bergmant/Documents/tm5-soa/output/col_MOD04_MYD04_QA2_L2_TM5_'+EXPs[1]+'_2010_1x1_monthlymean.nc'
	modis='/Users/bergmant/Documents/tm5-soa/output/processed/TM5-MODIS/MOD04_MYD04_L2_QA2_aggregated_2010_1x1_monthlymean.nc'
	modisdata=np.squeeze(nc.Dataset(modis,'r').variables['AOD_550_Dark_Target_Deep_Blue_Combined'][:])
	modeldata=np.squeeze(nc.Dataset(model,'r').variables['od550aer'][:])
	modeldata2=np.squeeze(nc.Dataset(model2,'r').variables['od550aer'][:])
	global modislon
	global modislat
	modislon=nc.Dataset(modis,'r').variables['longitude'][:]
	modislat=nc.Dataset(modis,'r').variables['latitude'][:]
	f,ax=plot_seasonal_diff(modeldata,modisdata,'NEWSOA-MODIS')
	f.savefig(output_pdf_path+'supplement/figS8_AOD-diff_NEW-MODIS_2010_allseasons.pdf')
	f.savefig(output_jpg_path+'supplement/figS8_AOD-diff_NEW-MODIS_2010_allseasons.jpg')
	f.savefig(output_png_path+'supplement/figS8_AOD-diff_NEW-MODIS_2010_allseasons.png',dpi=600)
	
	f,ax=plot_seasonal_diff(modeldata2,modisdata,'OLDSOA-MODIS')
			
	f.savefig(output_pdf_path+'supplement/figS9_AOD-diff_OLD-MODIS_2010_allseasons.pdf')
	f.savefig(output_jpg_path+'supplement/figS9_AOD-diff_OLD-MODIS_2010_allseasons.jpg')
	f.savefig(output_png_path+'supplement/figS9_AOD-diff_OLD-MODIS_2010_allseasons.png',dpi=600)


	# #f3,ax3=plt.subplots(nrows=2,ncols=2,figsize=(16,10))
	# k=-1
	# seasons=['DJF','MAM','JJA','SON']
	# #print(np.shape(modisdata),modeldata.shape)
	# lons, lats = np.meshgrid(modislon,modislat)
	# for i in range(2):
	# 	for j in range(2):
	# 		# f2,ax2=plt.subplots(nrows=1,ncols=1,figsize=(10,6))
	# 		k+=1
	# 		#print (np.mean(modeldata2[[-1,0,1],:,:],0).shape)
	# 		if k==0:
	# 			#data=((modeldata[-1,:,:]+modeldata[0,:,:]+modeldata[1,:,:])-(modeldata[-1,:,:]+modeldata[0,:,:]+modeldata[1,:,:]).mean()).squeeze()
	# 			data=(modeldata[[-1,0,1],:,:].mean(0)-modeldata2[[-1,0,1],:,:].mean(0)).squeeze()
	# 		elif k==1:
	# 			#data=((modeldata[-1,:,:]+modeldata[0,:,:]+modeldata[1,:,:])-(modeldata[-1,:,:]+modeldata[0,:,:]+modeldata[1,:,:]).mean()).squeeze()
	# 			data=(modeldata[[2,3,4],:,:].mean(0)-modeldata2[[2,3,4],:,:].mean(0)).squeeze()
	# 		elif k==2:
	# 			#data=((modeldata[-1,:,:]+modeldata[0,:,:]+modeldata[1,:,:])-(modeldata[-1,:,:]+modeldata[0,:,:]+modeldata[1,:,:]).mean()).squeeze()
	# 			data=(modeldata[[5,6,7],:,:].mean(0)-modeldata2[[5,6,7],:,:].mean(0)).squeeze()
	# 		elif k==3:
	# 			#data=((modeldata[-1,:,:]+modeldata[0,:,:]+modeldata[1,:,:])-(modeldata[-1,:,:]+modeldata[0,:,:]+modeldata[1,:,:]).mean()).squeeze()
	# 			data=(modeldata[[8,9,10],:,:].mean(0)-modeldata2[[8,9,10],:,:].mean(0)).squeeze()
	# 		#print (data.shape)
	# 		bounds_load=[-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3]
	# 		#print ('new,old',np.shape(modeldata),np.shape(modeldata2))

	# 		#ax3[i,j].set_title('AOD: NEWSOA-OLDSOA season {}'.format(seasons[k]),fontsize=18)
	# 		#plot_diff_map(ax3[i,j],[modislon,modislat],data.transpose(),'NEWSOA-OLDSOA')
	# 		# ax2.set_title('AOD: NEWSOA-OLDSOA season {}'.format(seasons[k]),fontsize=18)

	# 		#m=Basemap(projection='robin',lon_0=0,ax=ax3[i,j])

	# 		# m2=Basemap(projection='robin',lon_0=0,ax=ax2)
	# 		bounds = [-0.0375,-0.03,-0.0225,-0.015,-0.0075,-0.0025,0.0025,0.0075,0.015,0.0225,0.030,0.0375]
	# 		norm = mpl.colors.BoundaryNorm(bounds, 11)
	# 		mycmap=plt.get_cmap('coolwarm',11) 

	# 		# m2.drawparallels(np.arange(-90.,90.,30.))
	# 		# m2.drawmeridians(np.arange(-180.,180.,60.))
	# 		# m2.drawcoastlines()
	# 		# #print((modeldata-modisdata).squeeze().shape)
			
	# 		#print ((modeldata[k,:,:]-modisdata[k,:,:]).mean())
	# 		#print (np.shape(lons),np.shape(lats))

	# 		# cs2 = m2.pcolormesh(lons,lats,data.transpose(),norm=norm,latlon=True,cmap=mycmap)
	# 		# c2 = plt.colorbar(cs2,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.85,ax=ax2)
	# 		c2.set_label('AOD difference [NEWSOA-OLDSOA]',fontsize=14)
	# 		c2.ax.tick_params(labelsize=8)
	# 		print (np.mean(np.mean(modisdata,0)))
	# 		print (np.mean(np.mean(modeldata,0)))
	# 		print (np.mean(np.mean(modeldata2,0)))
	# 		print (np.max(modisdata))
	# 		print ('NEW-MODIS:',np.max(modeldata.mean(0)-modisdata.mean(0)))
	# 		print ('OLD-MODIS:',np.max(modeldata2.mean(0)-modisdata.mean(0)))
	# 		print ('NEW-OLD:',np.max(modeldata.mean(0)-modeldata2.mean(0)))
	
	# 		#f2.savefig(output_pdf_path+'MODIS/AOD-diff_OLD-NEW_2010_{}.pdf'.format(seasons[k]))
	# 		#f2.savefig(output_jpg_path+'MODIS/AOD-diff_OLD-NEW_2010_{}.jpg'.format(seasons[k]))
	# 		#f2.savefig(output_png_path+'MODIS/AOD-diff_OLD-NEW_2010_{}.png'.format(seasons[k]),dpi=600)

	f3,ax3=plot_seasonal_diff(modeldata2,modisdata,'NEWSOA-OLDSOA')
	f3.savefig(output_pdf_path+'/supplement/figS10_AOD-diff_OLD-NEW_2010_allseasons.pdf')
	f3.savefig(output_jpg_path+'/supplement/figS10_AOD-diff_OLD-NEW_2010_allseasons.jpg')
	f3.savefig(output_png_path+'/supplement/figS10_AOD-diff_OLD-NEW_2010_allseasons.png',dpi=600)
	
	# f11,ax11=plt.subplots(1,figsize=(10,6))
	# ax11.set_title('AOD: NEWSOA-OLDSOA',fontsize=18)
	# m=Basemap(projection='robin',lon_0=0,ax=ax11)
	# bounds = [-0.0375,-0.03,-0.0225,-0.015,-0.0075,-0.0025,0.0025,0.0075,0.015,0.0225,0.030,0.0375]
	# norm = mpl.colors.BoundaryNorm(bounds, 11)
	# mycmap=plt.get_cmap('coolwarm',11) 
	# m.drawparallels(np.arange(-90.,90.,30.))
	# m.drawmeridians(np.arange(-180.,180.,60.))
	# m.drawcoastlines()
	# data=(modeldata[:,:,:].mean(0)-modeldata2[:,:,:].mean(0)).squeeze()
	# #print (np.shape(modeldata),np.shape(lons),np.shape(lats))
	# cs = m.pcolormesh(lons,lats,data.transpose(),norm=norm,latlon=True,cmap=mycmap)
	# c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7,ax=ax11)
	# c.set_label('AOD difference [NEWSOA-OLDSOA]',fontsize=16)
	# c.ax.tick_params(labelsize=8)
	
	# f12,ax12=plt.subplots(1,figsize=(10,6))
	# ax12.set_title('AOD: MODIS',fontsize=18)
	# m=Basemap(projection='robin',lon_0=0,ax=ax12)
	# bounds = [0.025,0.075,0.15,0.225,0.30,0.375,0.45,0.525,0.6,0.675,0.75]
	# norm = mpl.colors.BoundaryNorm(bounds, 11)
	# mycmap=plt.get_cmap('Greens',11) 
	# m.drawparallels(np.arange(-90.,90.,30.))
	# m.drawmeridians(np.arange(-180.,180.,60.))
	# m.drawcoastlines()
	# data=(modisdata[:,:,:].mean(0)).squeeze()
	# cs = m.pcolormesh(lons,lats,data.transpose(),norm=norm,latlon=True,cmap=mycmap)
	# c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7,ax=ax12)
	# c.set_label('AOD MODIS',fontsize=16)
	# c.ax.tick_params(labelsize=8)
	
	# f13,ax13=plt.subplots(1,figsize=(10,6))
	# ax13.set_title('AOD: NEWSOA',fontsize=18)
	# m=Basemap(projection='robin',lon_0=0,ax=ax13)
	# bounds = [0.025,0.075,0.15,0.225,0.30,0.375,0.45,0.525,0.6,0.675,0.75]
	# norm = mpl.colors.BoundaryNorm(bounds, 11)
	# mycmap=plt.get_cmap('Greens',11) 
	# m.drawparallels(np.arange(-90.,90.,30.))
	# m.drawmeridians(np.arange(-180.,180.,60.))
	# m.drawcoastlines()
	# data=(modeldata[:,:,:].mean(0)).squeeze()
	# cs = m.pcolormesh(lons,lats,data.transpose(),norm=norm,latlon=True,cmap=mycmap)
	# c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7,ax=ax13)
	# c.set_label('AOD [NEWSOA]',fontsize=16)
	# c.ax.tick_params(labelsize=8)

	# f14,ax14=plt.subplots(1,figsize=(10,6))
	# ax14.set_title('AOD: OLDSOA',fontsize=18)
	# m=Basemap(projection='robin',lon_0=0,ax=ax14)
	# bounds = [0.025,0.075,0.15,0.225,0.30,0.375,0.45,0.525,0.6,0.675,0.75]
	# norm = mpl.colors.BoundaryNorm(bounds, 11)
	# mycmap=plt.get_cmap('Greens',11) 
	# m.drawparallels(np.arange(-90.,90.,30.))
	# m.drawmeridians(np.arange(-180.,180.,60.))
	# m.drawcoastlines()
	# data=(modeldata2[:,:,:].mean(0)).squeeze()
	# cs = m.pcolormesh(lons,lats,data.transpose(),norm=norm,latlon=True,cmap=mycmap)
	# c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7,ax=ax14)
	# c.set_label('AOD [OLDSOA]',fontsize=16)
	# c.ax.tick_params(labelsize=8)
	
	# f15,ax15=plt.subplots(1,figsize=(10,6))
	# ax15.set_title('AOD: NEWSOA-MODIS',fontsize=18)
	# m=Basemap(projection='robin',lon_0=0,ax=ax15)
	# bounds = [-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]
	# norm = mpl.colors.BoundaryNorm(bounds, 11)
	# mycmap=plt.get_cmap('coolwarm',11) 
	# m.drawparallels(np.arange(-90.,90.,30.))
	# m.drawmeridians(np.arange(-180.,180.,60.))
	# m.drawcoastlines()
	# data=(modeldata[:,:,:].mean(0)-modisdata[:,:,:].mean(0)).squeeze()
	# cs = m.pcolormesh(lons,lats,data.transpose(),norm=norm,latlon=True,cmap=mycmap)
	# c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7,ax=ax15)
	# c.set_label('AOD difference [NEWSOA-MODIS]',fontsize=16)
	# c.ax.tick_params(labelsize=8)
	
	# f16,ax16=plt.subplots(1,figsize=(10,6))
	# ax16.set_title('AOD: OLDSOA-MODIS',fontsize=18)
	# m=Basemap(projection='robin',lon_0=0,ax=ax16)
	# bounds = [-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]
	# norm = mpl.colors.BoundaryNorm(bounds, 11)
	# mycmap=plt.get_cmap('coolwarm',11) 
	# m.drawparallels(np.arange(-90.,90.,30.))
	# m.drawmeridians(np.arange(-180.,180.,60.))
	# m.drawcoastlines()
	# data=(modeldata2[:,:,:].mean(0)-modisdata[:,:,:].mean(0)).squeeze()
	# cs = m.pcolormesh(lons,lats,data.transpose(),norm=norm,latlon=True,cmap=mycmap)
	# c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7,ax=ax16)
	# c.set_label('AOD difference [OLDSOA-MODIS]',fontsize=16)
	# c.ax.tick_params(labelsize=8)

	#f11.savefig(output_pdf_path+'MODIS/AOD-diff_NEW-OLD_2010.pdf')
	#f11.savefig(output_jpg_path+'MODIS/AOD-diff_NEW-OLD_2010.jpg')
	#f11.savefig(output_png_path+'MODIS/AOD-diff_NEW-OLD_2010.png',dpi=600)
	#f12.savefig(output_pdf_path+'MODIS/AOD-MODIS_2010.pdf')
	#f12.savefig(output_jpg_path+'MODIS/AOD-MODIS_2010.jpg')
	#f12.savefig(output_png_path+'MODIS/AOD-MODIS_2010.png',dpi=600)
	#f13.savefig(output_pdf_path+'MODIS/AOD-NEWSOA_2010.pdf')
	#f13.savefig(output_jpg_path+'MODIS/AOD-NEWSOA_2010.jpg')
	#f13.savefig(output_png_path+'MODIS/AOD-NEWSOA_2010.png',dpi=600)
	#f14.savefig(output_pdf_path+'MODIS/AOD-OLDSOA_2010.pdf')
	#f14.savefig(output_jpg_path+'MODIS/AOD-OLDSOA_2010.jpg')
	#f14.savefig(output_png_path+'MODIS/AOD-OLDSOA_2010.png',dpi=600)
	#f15.savefig(output_pdf_path+'MODIS/AOD-diff_NEWSOA-MODIS_2010.pdf')
	#f15.savefig(output_jpg_path+'MODIS/AOD-diff_NEWSOA-MODIS_2010.jpg')
	#f15.savefig(output_png_path+'MODIS/AOD-diff_NEWSOA-MODIS_2010.png',dpi=600)
	
	#f16.savefig(output_pdf_path+'MODIS/AOD-diff_OLDSOA-MODIS_2010.pdf')
	#f16.savefig(output_jpg_path+'MODIS/AOD-diff_OLDSOA-MODIS_2010.jpg')
	#f16.savefig(output_png_path+'MODIS/AOD-diff_OLDSOA-MODIS_2010.png',dpi=600)
	
	cellarea_file='/Users/bergmant/Documents/tm5-soa/output/ec-ei-an0tr6-sfc-glb100x100-0000-oro.nc'
	cellarea=nc.Dataset(cellarea_file,'r').variables['cell_area'][:].transpose()

	nom=(modeldata[:,:,:].mean(0)-modisdata[:,:,:].mean(0)).sum()
	nom2=(modeldata2[:,:,:].mean(0)-modisdata[:,:,:].mean(0)).sum()
	denom=modisdata[:,:,:].mean(0).sum()
	bnom=((modeldata[:,:,:].mean(0)-modisdata[:,:,:].mean(0))*cellarea).sum()
	bnom2=((modeldata2[:,:,:].mean(0)-modisdata[:,:,:].mean(0))*cellarea).sum()
	bdenom=(modisdata[:,:,:].mean(0)*cellarea).sum()
	print ('NMB1:',nom/denom)
	print ('NMB2:',nom2/denom)
	print ('NMB1:',bnom/bdenom)
	print ('NMB2:',bnom2/bdenom)
if __name__ == "__main__":
	#process_modis()
	cellarea_file='/Users/bergmant/Documents/tm5-soa/output/ec-ei-an0tr6-sfc-glb100x100-0000-oro.nc'
	cellarea=nc.Dataset(cellarea_file,'r').variables['cell_area'][:]
	lsm_file='/Users/bergmant/Documents/tm5-soa/output/ec-ei-an0tr6-sfc-glb100x100-lsm.nc'
	lsm=nc.Dataset(lsm_file,'r').variables['lsm'][:]
	get_gridboxarea('TM53x2')
	#print(np.shape(cellarea))
	plot_modis_diff()
	
	#plot_modis_daily()
	plot_modis_season()
	plt.show()
