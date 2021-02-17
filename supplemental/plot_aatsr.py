from __future__ import print_function
import netCDF4  as nc
import numpy as np
from mpl_toolkits.basemap import Basemap
import sys
sys.path.append('/Users/bergmant/Documents/python')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lonlat import lonlat
from pylab import *
from aerocom_sites import aerocom_sites
import Colocate_MODIS as MODIS
import os
from settings import * #output_pdf_path,output_png_path,output_jpg_path
def plot_aatsr_diff(var='od550aer'):
	#output_pdf_path='/Users/bergmant/Documents/tm5-soa//figures/pdf/aatsr/'
	#output_png_path='/Users/bergmant/Documents/tm5-soa//figures/png/aatsr/'
	#output_jpg_path='/Users/bergmant/Documents/tm5-soa//figures/jpg/aatsr/'
	#EXPs=['soa-riccobono','oldsoa-final']#,'NOSOA']
	#EXPs=['newsoa-ri','oldsoa-bhn','nosoa']
	#EXP_NAMEs=['NEWSOA','OLDSOA']#,'NOSOA']
	model='/Users/bergmant/Documents/tm5-soa/output/processed/CCI/col_ESACCI_SU_TM5.2010.ym.nc'
	#model2='/Users/bergmant/Documents/tm5-soa/output/processed/output/Aggregated_lin_Col_TM5_'+EXPs[0]+'_MYD04_MOD04_L2_2010_1x1_yearmean.nc'
	AATSR='/Users/bergmant/Documents/tm5-soa/output/processed/CCI/ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated.2010.yearmean.nc'

	#model='/Users/bergmant/Documents/tm5-soa/output/processed/CCI/col_ESACCI_SU_TM5_'+exp+'.2010.mm.nc'
	model='/Volumes/Utrecht/CCI/cci_tm5_col/col_ESACCI_SU_TM5_'+EXPs[0]+'_2010.yearmean.nc'
	model2='/Volumes/Utrecht/CCI/cci_tm5_col/col_ESACCI_SU_TM5_'+EXPs[1]+'_2010.yearmean.nc'
	#model2='/Users/bergmant/Documents/tm5-soa/output/processed/output/Aggregated_lin_Col_TM5_'+EXPs[1]+'_MYD04_MOD04_L2_2010_1x1_monthlymean.nc'
	AATSR='/Volumes/Utrecht/CCI/aggregated_cci/ESACCI-L2P-AATSR-SU_agg.2010.monthlymean.nc'
	AATSR='/Users/bergmant/Documents/tm5-soa/output/processed/CCI/aggregated_cci/ESACCI-L2P-AATSR-SU_agg.2010.yearmean.nc'


	AATSRdata=np.squeeze(nc.Dataset(AATSR,'r').variables['AOD550'][:])
	modeldata=np.squeeze(nc.Dataset(model,'r').variables[var][:])
	modeldata2=np.squeeze(nc.Dataset(model2,'r').variables[var][:])
	AATSRlon=nc.Dataset(AATSR,'r').variables['longitude'][:]
	AATSRlat=nc.Dataset(AATSR,'r').variables['latitude'][:]
	cellarea_file='/Users/bergmant/Documents/tm5-soa//output/ec-ei-an0tr6-sfc-glb100x100-0000-oro.nc'
	cellarea=nc.Dataset(cellarea_file,'r').variables['cell_area'][:]
	lsm_file='/Users/bergmant/Documents/tm5-soa//output/ec-ei-an0tr6-sfc-glb100x100-lsm.nc'
	lsm=nc.Dataset(lsm_file,'r').variables['lsm'][:]
	lsm2=lsm.copy()
	lsm[lsm>1]=1
	lsm[lsm<1]=nan
	lsm2[lsm2<1]=1
	lsm2[lsm2>1]=nan
	print(np.mean(AATSRdata),np.nanmean(AATSRdata.transpose()))
	print(np.mean(modeldata),np.nanmean(modeldata.transpose()))
	print(np.mean(modeldata2),np.nanmean(modeldata2.transpose()))
	print('cellarea mean')
	print(np.mean(AATSRdata),np.nansum(AATSRdata.transpose()*cellarea)/np.sum(cellarea))
	print(np.mean(modeldata),np.nansum(modeldata.transpose()*cellarea)/(np.sum(cellarea)))
	print(np.mean(modeldata2),np.nansum(modeldata2.transpose()*cellarea)/(np.sum(cellarea)))
	print ('land')
	print(np.mean(AATSRdata),np.nansum(AATSRdata.transpose()*cellarea*lsm)/np.nansum(cellarea*lsm))
	print(np.mean(modeldata),np.nansum(modeldata.transpose()*cellarea*lsm)/(np.nansum(cellarea*lsm)))
	print(np.mean(modeldata2),np.nansum(modeldata2.transpose()*cellarea*lsm)/(np.nansum(cellarea*lsm)))
	print ('ocean')
	print(np.mean(AATSRdata),np.nansum(AATSRdata.transpose()*cellarea*lsm2)/np.nansum(cellarea*lsm2))
	print(np.mean(modeldata),np.nansum(modeldata.transpose()*cellarea*lsm2)/(np.nansum(cellarea*lsm2)))
	print(np.mean(modeldata2),np.nansum(modeldata2.transpose()*cellarea*lsm2)/(np.nansum(cellarea*lsm2)))
	
	f,ax=plt.subplots(ncols=1,figsize=(8,4))
	k=-1
	#print(np.shape(AATSRdata),modeldata.shape)
	lons, lats = np.meshgrid(AATSRlon,AATSRlat)
	print (lons)
	#for exp in EXPS:
	k+=1
	bounds_load=[-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3]

	ax.set_title(EXP_NAMEs[0]+'-AATSR')
	m=Basemap(projection='robin',lon_0=0,ax=ax)
	bounds = [-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	mycmap=plt.get_cmap('coolwarm',11) 

	#image=m.contourf(lons,lats,modeldata-AATSRdata,bounds_load,cmap=plt.cm.coolwarm,latlon=True)

	m.drawparallels(np.arange(-90.,90.,30.))
	m.drawmeridians(np.arange(-180.,180.,60.))
	m.drawcoastlines()
	print (np.shape(modeldata),np.shape(AATSRdata),np.shape(lons),np.shape(lats))
	cs = m.pcolormesh(lons,lats,((modeldata-AATSRdata).squeeze().transpose()),latlon=True,norm=norm,cmap=mycmap)
	#cs = m.pcontourf(lons,lats,((AATSRdata).squeeze()),latlon=True)#,norm=norm,cmap=mycmap)
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7)
	c.set_label('AOD bias [NEWSOA-AATSR]',fontsize=18)
	c.ax.tick_params(labelsize=10)
	print (np.mean(AATSRdata))
	print (np.mean(modeldata))
	#print (np.mean(modeldata2))
	print (np.max(AATSRdata))
	print (np.max(modeldata))
	#print (np.max(modeldata2))
	#print ('NEW-AATSR:',np.max(modeldata.mean(0)-AATSRdata.mean(0)))
	#print ('OLD-AATSR:',np.max(modeldata2.mean(0)-AATSRdata.mean(0)))
	#print ('NEW-OLD:',np.max(modeldata.mean(0)-modeldata2.mean(0)))
	f.savefig(output_pdf_path+'aatsr/AOD-diff_'+EXP_NAMEs[0]+'-AATSR_2010.pdf')
	f.savefig(output_png_path+'aatsr/AOD-diff_'+EXP_NAMEs[0]+'-AATSR_2010.png',dpi=600)

	f,ax=plt.subplots(ncols=1,figsize=(8,4))
	k=-1
	#print(np.shape(AATSRdata),modeldata.shape)
	lons, lats = np.meshgrid(AATSRlon,AATSRlat)
	print (lons)
	#for exp in EXPS:
	k+=1
	bounds_load=[-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3]

	ax.set_title(EXP_NAMEs[1]+'-AATSR')
	m=Basemap(projection='robin',lon_0=0,ax=ax)
	bounds = [-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	mycmap=plt.get_cmap('coolwarm',11) 

	#image=m.contourf(lons,lats,modeldata-AATSRdata,bounds_load,cmap=plt.cm.coolwarm,latlon=True)

	m.drawparallels(np.arange(-90.,90.,30.))
	m.drawmeridians(np.arange(-180.,180.,60.))
	m.drawcoastlines()
	print (np.shape(modeldata),np.shape(AATSRdata),np.shape(lons),np.shape(lats))
	cs = m.pcolormesh(lons,lats,((modeldata2-AATSRdata).squeeze().transpose()),latlon=True,norm=norm,cmap=mycmap)
	#cs = m.pcontourf(lons,lats,((AATSRdata).squeeze()),latlon=True)#,norm=norm,cmap=mycmap)
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7)
	c.set_label('AOD bias [OLDSOA-AATSR]',fontsize=18)
	c.ax.tick_params(labelsize=10)
	print (np.mean(AATSRdata))
	print (np.mean(modeldata))
	#print (np.mean(modeldata2))
	print (np.max(AATSRdata))
	print (np.max(modeldata))
	#print (np.max(modeldata2))
	#print ('NEW-AATSR:',np.max(modeldata.mean(0)-AATSRdata.mean(0)))
	#print ('OLD-AATSR:',np.max(modeldata2.mean(0)-AATSRdata.mean(0)))
	#print ('NEW-OLD:',np.max(modeldata.mean(0)-modeldata2.mean(0)))
	f.savefig(output_pdf_path+'aatsr/diff_'+EXP_NAMEs[1]+'-AATSR_2010.pdf')
	f.savefig(output_png_path+'aatsr/diff_'+EXP_NAMEs[1]+'-AATSR_2010.png',dpi=600)

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
	cs = m.pcolormesh(lons,lats,(modeldata.squeeze().transpose()),norm=norm,latlon=True,cmap=mycmap)
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7)
	c.set_label('AOD [TM5]',fontsize=18)
	c.ax.tick_params(labelsize=10)
	#c.set_label('AOD relative bias [(TM5-AERONET)/AERONET]')
	fsoa.savefig(output_pdf_path+'aatsr/TM5-NEWSOA_2010.pdf')
	fsoa.savefig(output_png_path+'aatsr/TM5-NEWSOA_2010.png',dpi=600)
	
	fAATSR,ax=plt.subplots(figsize=(10,7))
	m = Basemap(projection='robin',lon_0=0)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,120.,30.))
	m.drawmeridians(np.arange(0.,3060.,60.))
	mycmap=plt.get_cmap('Purples',11) 
	# define the bins and normalize
	bounds = [0.025,0.075,0.15,0.225,0.30,0.375,0.45,0.525,0.6,0.675,0.75]
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	plt.title('AOD collocated annual mean (AATSR)',fontsize=18)
	m = Basemap(projection='robin',lon_0=0,ax=ax)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,120.,30.))
	m.drawmeridians(np.arange(0.,3060.,60.))
	#mycmap=plt.get_cmap('coolwarm',11) 
	# define the bins and normalize
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	cs = m.pcolormesh(lons,lats,(AATSRdata.squeeze().transpose()),norm=norm,latlon=True,cmap=mycmap)
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7)
	c.set_label('AOD [AATSR]',fontsize=18)
	c.ax.tick_params(labelsize=10)
	#c.set_label('AOD relative bias [(TM5-AERONET)/AERONET]')
	fAATSR.savefig(output_pdf_path+'aatsr/AATSR_2010.pdf')
	fAATSR.savefig(output_png_path+'aatsr/AATSR_2010.png',dpi=600)
def plot_diff_map(ax,coord,data,name):
	#ax.set_title('{0} season {1}'.format(season))
	#AATSR='/Users/bergmant/Documents/tm5-soa//output/TM5-AATSR//Aggregated_AATSR_L2_2010_1x1_dailymean.nc'
	#AATSRlon=nc.Dataset(AATSR,'r').variables['longitude'][:]
	#AATSRlat=nc.Dataset(AATSR,'r').variables['latitude'][:]
	lons, lats = np.meshgrid(coord[0],coord[1])
	m=Basemap(projection='robin',lon_0=0,ax=ax)
	bounds = [-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	mycmap=plt.get_cmap('coolwarm',11) 

	#image=m.contourf(lons,lats,modeldata-AATSRdata,bounds_load,cmap=plt.cm.coolwarm,latlon=True)

	m.drawparallels(np.arange(-90.,90.,30.))
	m.drawmeridians(np.arange(-180.,180.,60.))
	m.drawcoastlines()
	#print((modeldata-AATSRdata).squeeze().shape)
	#print ((modeldata[k,:,:]-AATSRdata[k,:,:]).mean())
	#cs = m.pcolormesh(lons,lats,((modeldata[k,:,:]-AATSRdata[k,:,:]).squeeze()),norm=norm,latlon=True,cmap=mycmap)
	cs = m.pcolormesh(lons,lats,data.transpose(),norm=norm,latlon=True,cmap=mycmap)
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.85,ax=ax)
	c.set_label('AOD bias ['+name+']',fontsize=14)
	c.ax.tick_params(labelsize=8)
	#print (np.mean(AATSRdata))
	#print (np.mean(modeldata))
	#print (np.mean(modeldata2))
def plot_map(ax,coord,data,name):
	lons, lats = np.meshgrid(coord[0],coord[1])
	m=Basemap(projection='robin',lon_0=0,ax=ax)
	bounds = [0.025,0.075,0.15,0.225,0.30,0.375,0.45,0.525,0.6,0.675,0.75]
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	mycmap=plt.get_cmap('Purples',11) 

	#image=m.contourf(lons,lats,modeldata-AATSRdata,bounds_load,cmap=plt.cm.coolwarm,latlon=True)

	m.drawparallels(np.arange(-90.,90.,30.))
	m.drawmeridians(np.arange(-180.,180.,60.))
	m.drawcoastlines()
	#print((modeldata-AATSRdata).squeeze().shape)
	#print ((modeldata[k,:,:]-AATSRdata[k,:,:]).mean())
	#cs = m.pcolormesh(lons,lats,((modeldata[k,:,:]-AATSRdata[k,:,:]).squeeze()),norm=norm,latlon=True,cmap=mycmap)
	cs = m.pcolormesh(lons,lats,data.transpose(),norm=norm,latlon=True,cmap=mycmap)
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.85,ax=ax)
	c.set_label('AOD bias ['+name+']',fontsize=14)
	c.ax.tick_params(labelsize=8)
	#print (np.mean(AATSRdata))
	#print (np.mean(modeldata))
	#print (np.mean(modeldata2))
def plot_aatsr_season(iexp,var='od550aer',**kwargs):
	#output_pdf_path='/Users/bergmant/Documents/tm5-soa//figures/pdf/aatsr/'
	#output_png_path='/Users/bergmant/Documents/tm5-soa//figures/png/aatsr/'
	#output_jpg_path='/Users/bergmant/Documents/tm5-soa//figures/jpg/aatsr/'
	if 'annotate' in kwargs:
		l_annotate=True
	else:
		l_annotate=False
	figN=11+iexp

	#EXPs=['newsoa-ri','oldsoa-bhn']#,'NOSOA']
	#EXP_NAMEs=['NEWSOA','OLDSOA']#,'NOSOA']
	#model='/Users/bergmant/Documents/tm5-soa//CCI/col_ESACCI_SU_TM5_'+exp+'_2010.mm.nc'
	model='/Volumes/Utrecht/CCI/cci_tm5_col/col_ESACCI_SU_TM5_'+EXPs[iexp]+'_2010_'+var+'.monthlymean.nc'
	model2='/Volumes/Utrecht/CCI/cci_tm5_col/col_ESACCI_SU_TM5_'+EXPs[iexp]+'_2010_'+var+'.monthlymean.nc'
	#model2='/Users/bergmant/Documents/tm5-soa//output/Aggregated_lin_Col_TM5_'+EXPs[1]+'_MYD04_MOD04_L2_2010_1x1_monthlymean.nc'
	AATSR='/Volumes/Utrecht/CCI/aggregated_cci/ESACCI-L2P-AATSR-SU_agg.2010.monthlymean.nc'
	AATSR='/Users/bergmant/Documents/tm5-soa/output/processed/CCI/aggregated_cci/ESACCI-L2P-AATSR-SU_agg.2010.monthlymean.nc'
	#AATSR='/Users/bergmant/Documents/tm5-soa//CCI/ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated.2010.monthlymean.nc'
	#model='/Users/bergmant/Documents/tm5-soa//output/Aggregated_lin_Col_TM5_'+EXPs[0]+'_MYD04_MOD04_L2_2010_1x1_seasonalmean.nc'
	#model2='/Users/bergmant/Documents/tm5-soa//output/Aggregated_lin_Col_TM5_'+EXPs[1]+'_MYD04_MOD04_L2_2010_1x1_seasonalmean.nc'
	#AATSR='/Users/bergmant/Documents/tm5-soa//output/TM5-AATSR//Aggregated_AATSR_L2_2010_1x1_seasonalmean.nc'
	cell=np.squeeze(nc.Dataset('/Users/bergmant/Documents/tm5-soa//output/ec-ei-an0tr6-sfc-glb100x100-0000-oro.nc','r').variables['cell_area'][:])
	cellarea_file='/Users/bergmant/Documents/tm5-soa//output/ec-ei-an0tr6-sfc-glb100x100-0000-oro.nc'
	cellarea=nc.Dataset(cellarea_file,'r').variables['cell_area'][:]
	cellarea=cellarea.transpose()
	AATSRdata=np.squeeze(nc.Dataset(AATSR,'r').variables['AOD550'][:])
	print (model)
	modeldata=np.squeeze(nc.Dataset(model,'r').variables[var][:])
	#modeldata2=np.squeeze(nc.Dataset(model2,'r').variables['od550aer'][:])
	AATSRlon=nc.Dataset(AATSR,'r').variables['longitude'][:]
	AATSRlat=nc.Dataset(AATSR,'r').variables['latitude'][:]
	print (np.mean(np.mean(modeldata,0)*cellarea),np.mean(np.mean(AATSRdata,0)*cellarea))
	f,ax=plt.subplots(nrows=2,ncols=2,figsize=(16,10))
	faatsr,axaatsr=plt.subplots(nrows=2,ncols=2,figsize=(16,10))
	ftm5,axtm5=plt.subplots(nrows=2,ncols=2,figsize=(16,10))
	k=-1
	seasons=['DJF','MAM','JJA','SON']
	#print(np.shape(AATSRdata),modeldata.shape)
	lons, lats = np.meshgrid(AATSRlon,AATSRlat)
	#cellarea_file='/Users/bergmant/Documents/tm5-soa//output/ec-ei-an0tr6-sfc-glb100x100-0000-oro.nc'
	#cellarea=nc.Dataset(cellarea_file,'r').variables['cell_area'][:]
	lsm_file='/Users/bergmant/Documents/tm5-soa//output/ec-ei-an0tr6-sfc-glb100x100-lsm.nc'
	lsm=nc.Dataset(lsm_file,'r').variables['lsm'][:]
	lsm=lsm.transpose()
	lsm2=lsm.copy()
	lsm[lsm>1]=1
	lsm[lsm<1]=nan
	lsm2[lsm2<1]=1
	lsm2[lsm2>1]=nan
	#print ('land')
	#print(np.mean(AATSRdata),np.nansum(AATSRdata.transpose()*cellarea*lsm)/np.nansum(cellarea*lsm))
	#print(np.mean(modeldata),np.nansum(modeldata.transpose()*cellarea*lsm)/(np.nansum(cellarea*lsm)))
	#print ('ocean')
	#print(np.mean(AATSRdata),np.nansum(AATSRdata.transpose()*cellarea*lsm2)/np.nansum(cellarea*lsm2))
	#print(np.mean(modeldata),np.nansum(modeldata.transpose()*cellarea*lsm2)/(np.nansum(cellarea*lsm2)))
		
	indices=np.zeros((4,3),dtype=np.int8)
	indices[0,:]=[-1,0,1]
	indices[1,:]=[2,3,4]
	indices[2,:]=[5,6,7]
	indices[3,:]=[8,9,10]
	for i in range(2):
		for j in range(2):
			k+=1
			print (np.shape(modeldata))
			print (np.mean(modeldata[[-1,0,1],:,:],0).shape)
			print (indices)
			if k==0:
				#indices[0,:]
				#data=((modeldata[-1,:,:]+modeldata[0,:,:]+modeldata[1,:,:])-(AATSRdata[-1,:,:]+AATSRdata[0,:,:]+AATSRdata[1,:,:]).mean()).squeeze()
				data=(modeldata[indices[k,:],:,:].mean(0)-AATSRdata[indices[k,:],:,:].mean(0)).squeeze()
				aatsrdata=(AATSRdata[indices[k,:],:,:].mean(0)).squeeze()
				tm5data=(modeldata[indices[k,:],:,:].mean(0)).squeeze()
				print((modeldata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum(),(AATSRdata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum())
				tm5landaod=np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm)
				aatsrlandaod=np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm)
				tm5oceanaod=np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2)
				aatsroceanaod=np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2)


				print('land:',np.nansum(modeldata[indices[0,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm),np.nansum(AATSRdata[indices[0,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm))
				print('ocean:',np.nansum(modeldata[indices[0,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2),np.nansum(AATSRdata[indices[0,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2))
				if l_annotate:
					ax[i,j].annotate('TM5:   {:6.2f}'.format((modeldata[indices[0,:],:,:].mean(0)*cellarea).sum()/cellarea.sum()),xy=(-0.15,0.94),xycoords='axes fraction')
					ax[i,j].annotate('AATSR: {:6.2f}'.format((AATSRdata[indices[0,:],:,:].mean(0)*cellarea).sum()/cellarea.sum()),xy=(-0.15,1.00),xycoords='axes fraction')
					ax[i,j].annotate('TM5 l:   {:6.2f}'.format(tm5landaod),xy=(-0.15,0.85),xycoords='axes fraction')
					ax[i,j].annotate('AATSR l: {:6.2f}'.format(aatsrlandaod),xy=(-0.15,0.90),xycoords='axes fraction')
					ax[i,j].annotate('TM5 o:   {:6.2f}'.format(tm5oceanaod),xy=(-0.15,0.75),xycoords='axes fraction')
					ax[i,j].annotate('AATSR o: {:6.2f}'.format(aatsroceanaod),xy=(-0.15,0.80),xycoords='axes fraction')
			elif k==1:
				#data=((modeldata[-1,:,:]+modeldata[0,:,:]+modeldata[1,:,:])-(AATSRdata[-1,:,:]+AATSRdata[0,:,:]+AATSRdata[1,:,:]).mean()).squeeze()
				data=(modeldata[indices[k,:],:,:].mean(0)-AATSRdata[indices[k,:],:,:].mean(0)).squeeze()
				aatsrdata=(AATSRdata[indices[k,:],:,:].mean(0)).squeeze()
				tm5data=(modeldata[indices[k,:],:,:].mean(0)).squeeze()
				tm5landaod=np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm)
				aatsrlandaod=np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm)
				tm5oceanaod=np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2)
				aatsroceanaod=np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2)
				print((modeldata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum(),(AATSRdata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum())
				print('land:',np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm),  np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm))
				print('ocean:',np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2),np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2))
				if l_annotate:
					ax[i,j].annotate('TM5:   {:6.2f}'.format((modeldata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum()),xy=(0.00,0.94),xycoords='axes fraction')
					ax[i,j].annotate('AATSR: {:6.2f}'.format((AATSRdata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum()),xy=(0.00,1.00),xycoords='axes fraction')
					ax[i,j].annotate('TM5 l:   {:6.2f}'.format(tm5landaod),xy=(-0.15,0.85),xycoords='axes fraction')
					ax[i,j].annotate('AATSR l: {:6.2f}'.format(aatsrlandaod),xy=(-0.15,0.90),xycoords='axes fraction')
					ax[i,j].annotate('TM5 o:   {:6.2f}'.format(tm5oceanaod),xy=(-0.15,0.75),xycoords='axes fraction')
					ax[i,j].annotate('AATSR o: {:6.2f}'.format(aatsroceanaod),xy=(-0.15,0.80),xycoords='axes fraction')
			elif k==2:
				#data=((modeldata[-1,:,:]+modeldata[0,:,:]+modeldata[1,:,:])-(AATSRdata[-1,:,:]+AATSRdata[0,:,:]+AATSRdata[1,:,:]).mean()).squeeze()
				data=(modeldata[indices[k,:],:,:].mean(0)-AATSRdata[indices[k,:],:,:].mean(0)).squeeze()
				aatsrdata=(AATSRdata[indices[k,:],:,:].mean(0)).squeeze()
				tm5data=(modeldata[indices[k,:],:,:].mean(0)).squeeze()
				tm5landaod=np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm)
				aatsrlandaod=np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm)
				tm5oceanaod=np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2)
				aatsroceanaod=np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2)
				print((modeldata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum(),(AATSRdata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum())
				print('land:',np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm),  np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm))
				print('ocean:',np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2),np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2))
				if l_annotate:
					ax[i,j].annotate('TM5:   {:6.2f}'.format((modeldata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum()),xy=(0.00,0.94),xycoords='axes fraction')
					ax[i,j].annotate('AATSR: {:6.2f}'.format((AATSRdata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum()),xy=(0.00,1.00),xycoords='axes fraction')
					ax[i,j].annotate('TM5 l:   {:6.2f}'.format(tm5landaod),xy=(-0.15,0.85),xycoords='axes fraction')
					ax[i,j].annotate('AATSR l: {:6.2f}'.format(aatsrlandaod),xy=(-0.15,0.90),xycoords='axes fraction')
					ax[i,j].annotate('TM5 o:   {:6.2f}'.format(tm5oceanaod),xy=(-0.15,0.75),xycoords='axes fraction')
					ax[i,j].annotate('AATSR o: {:6.2f}'.format(aatsroceanaod),xy=(-0.15,0.80),xycoords='axes fraction')
			elif k==3:
				#data=((modeldata[-1,:,:]+modeldata[0,:,:]+modeldata[1,:,:])-(AATSRdata[-1,:,:]+AATSRdata[0,:,:]+AATSRdata[1,:,:]).mean()).squeeze()
				data=(modeldata[indices[k,:],:,:].mean(0)-AATSRdata[indices[k,:],:,:].mean(0)).squeeze()
				aatsrdata=(AATSRdata[indices[k,:],:,:].mean(0)).squeeze()
				tm5data=(modeldata[indices[k,:],:,:].mean(0)).squeeze()
				tm5landaod=np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm)
				aatsrlandaod=np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm)
				tm5oceanaod=np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2)
				aatsroceanaod=np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2)
				print((modeldata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum(),(AATSRdata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum())
				print('land:',np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm),  np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm)/np.nansum(cellarea*lsm))
				print('ocean:',np.nansum(modeldata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2),np.nansum(AATSRdata[indices[k,:],:,:].mean(0)*cellarea*lsm2)/np.nansum(cellarea*lsm2))
				if l_annotate:
					ax[i,j].annotate('TM5:   {:6.2f}'.format((modeldata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum()),xy=(0.00,0.94),xycoords='axes fraction')
					ax[i,j].annotate('AATSR: {:6.2f}'.format((AATSRdata[indices[k,:],:,:].mean(0)*cellarea).sum()/cellarea.sum()),xy=(0.00,1.00),xycoords='axes fraction')
					ax[i,j].annotate('TM5 l:   {:6.2f}'.format(tm5landaod),xy=(-0.15,0.85),xycoords='axes fraction')
					ax[i,j].annotate('AATSR l: {:6.2f}'.format(aatsrlandaod),xy=(-0.15,0.90),xycoords='axes fraction')
					ax[i,j].annotate('TM5 o:   {:6.2f}'.format(tm5oceanaod),xy=(-0.15,0.75),xycoords='axes fraction')
					ax[i,j].annotate('AATSR o: {:6.2f}'.format(aatsroceanaod),xy=(-0.15,0.80),xycoords='axes fraction')
			print (data.shape)
			bounds_load=[-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3]
			plot_diff_map(ax[i,j],[AATSRlon,AATSRlat],data,EXP_NAMEs[iexp]+'-AATSR')
			plot_map(axaatsr[i,j],[AATSRlon,AATSRlat],aatsrdata,'AATSR')
			plot_map(axtm5[i,j],[AATSRlon,AATSRlat],tm5data,EXP_NAMEs[iexp])
			ax[i,j].set_title(EXP_NAMEs[iexp]+'-AATSR season {}'.format(seasons[k]))
			
	f.savefig(output_pdf_path+'supplement/figS'+str(figN)+'_AOD-diff_'+EXP_NAMEs[iexp]+'-AATSR_2010_allseasons.pdf')
	f.savefig(output_jpg_path+'supplement/figS'+str(figN)+'_AOD-diff_'+EXP_NAMEs[iexp]+'-AATSR_2010_allseasons.jpg')
	f.savefig(output_png_path+'supplement/figS'+str(figN)+'_AOD-diff_'+EXP_NAMEs[iexp]+'-AATSR_2010_allseasons.png',dpi=600)
def do_plots():
	print ("plotting")
	#plot_aatsr_diff()
	plot_aatsr_season(0,'od550aer')
	plot_aatsr_season(1,'od550aer')
	#plot_aatsr_season(0,'od550soa')
	#plot_aatsr_season(1,'od550soa')
	plt.show()
if __name__ == "__main__":
	do_plots()
