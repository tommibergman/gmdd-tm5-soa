from __future__ import print_function
import netCDF4 as nc 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
from pylab import *
import os
import sys
from settings import *
import string
from settings import *

def read_aatsr(var='od550aer'):
	
	model=basepathprocessed+'CCI/col_ESACCI_SU_TM5.2010.ym.nc'
	#model2=basepathprocessed+'output/Aggregated_lin_Col_TM5_'+EXPs[0]+'_MYD04_MOD04_L2_2010_1x1_yearmean.nc'
	#AATSR=basepathprocessed+'CCI/ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated.2010.yearmean.nc'

	#model=basepathprocessed+'CCI/col_ESACCI_SU_TM5_'+exp+'.2010.mm.nc'
	model=raw_store+'CCI/cci_tm5_col/col_ESACCI_SU_TM5_'+EXPs[0]+'_2010.yearmean.nc'
	model2=raw_store+'CCI/cci_tm5_col/col_ESACCI_SU_TM5_'+EXPs[1]+'_2010.yearmean.nc'
	#model2=basepathprocessed+'output/Aggregated_lin_Col_TM5_'+EXPs[1]+'_MYD04_MOD04_L2_2010_1x1_monthlymean.nc'
	AATSR=raw_store+'CCI/aggregated_cci/ESACCI-L2P-AATSR-SU_agg.2010.monthlymean.nc'
	AATSR=basepathprocessed+'CCI/aggregated_cci/ESACCI-L2P-AATSR-SU_agg.2010.yearmean.nc'

	modeldata={}
	for expi in EXPS:
		model=raw_store+'CCI/cci_tm5_col/col_ESACCI_SU_TM5_'+expi+'_2010.yearmean.nc'
		modeldata[expi]=np.squeeze(nc.Dataset(model,'r').variables[var][:]).transpose()
		#modeldata2=np.squeeze(nc.Dataset(model2,'r').variables[var][:])

	AATSRdata=np.squeeze(nc.Dataset(AATSR,'r').variables['AOD550'][:]).transpose()
	AATSRlon=nc.Dataset(AATSR,'r').variables['longitude'][:]
	AATSRlat=nc.Dataset(AATSR,'r').variables['latitude'][:]
	cellarea_file=fixeddata+'/ec-ei-an0tr6-sfc-glb100x100-0000-oro.nc'
	cellarea=nc.Dataset(cellarea_file,'r').variables['cell_area'][:].transpose()
	
	return AATSRdata,modeldata,AATSRlon,AATSRlat
def plot_aatsr_diff(AATSRdata,modeldata,AATSRlon,AATSRlat):
	
	f,ax=plt.subplots(ncols=1,figsize=(8,4))
	k=-1
	#print(np.shape(AATSRdata),modeldata.shape)
	lons, lats = np.meshgrid(AATSRlon,AATSRlat)
	#print (lons)
	#for exp in EXPS:
	k+=1
	for i,expi in enumerate(EXPs):
		bounds_load=[-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3]

		m=Basemap(projection='robin',lon_0=0,ax=ax)
		bounds = [-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]
		norm = mpl.colors.BoundaryNorm(bounds, 11)
		mycmap=plt.get_cmap('coolwarm',11) 

		m.drawparallels(np.arange(-90.,90.,30.))
		m.drawmeridians(np.arange(-180.,180.,60.))
		m.drawcoastlines()
		#print (np.shape(modeldata),np.shape(AATSRdata),np.shape(lons),np.shape(lats))
		cs = m.pcolormesh(lons,lats,((modeldata[expi]-AATSRdata).squeeze()),latlon=True,norm=norm,cmap=mycmap)
		c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.9)
		c.set_label('AOD bias ['+EXP_NAMEs[i]+'-AATSR]',fontsize=24)
		c.ax.tick_params(labelsize=12)
		#f.savefig(output_pdf_path+'article/AOD-diff_'+EXP_NAMEs[i]+'-AATSR_2010.pdf')
		#f.savefig(output_png_path+'aatsr/AOD-diff_'+EXP_NAMEs[i]+'-AATSR_2010.png',dpi=600)

		
		fsoa,ax=plt.subplots(figsize=(10,7))
		m = Basemap(projection='robin',lon_0=0)
		m.drawcoastlines()
		m.drawparallels(np.arange(-90.,120.,30.))
		m.drawmeridians(np.arange(0.,3060.,60.))
		mycmap=plt.get_cmap('Purples',11) 
		# define the bins and normalize
		bounds = [0.025,0.075,0.15,0.225,0.30,0.375,0.45,0.525,0.6,0.675,0.75]
		norm = mpl.colors.BoundaryNorm(bounds, 11)
		#plt.title('AOD collocated annual mean (TM5)',fontsize=24)
		m = Basemap(projection='robin',lon_0=0,ax=ax)
		m.drawcoastlines()
		m.drawparallels(np.arange(-90.,120.,30.))
		m.drawmeridians(np.arange(0.,3060.,60.))
		#mycmap=plt.get_cmap('coolwarm',11) 
		# define the bins and normalize
		norm = mpl.colors.BoundaryNorm(bounds, 11)
		cs = m.pcolormesh(lons,lats,(modeldata[expi].squeeze()),norm=norm,latlon=True,cmap=mycmap)
		c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.9)
		c.set_label('AOD ['+EXP_NAMEs[i]+']',fontsize=24)
		c.ax.tick_params(labelsize=12)
		#c.set_label('AOD relative bias [(TM5-AERONET)/AERONET]')
		#fsoa.savefig(output_pdf_path+'aatsr/TM5-'+EXP_NAMEs[i]+'_2010.pdf')
		#fsoa.savefig(output_png_path+'aatsr/TM5-'+EXP_NAMEs[i]+'_2010.png',dpi=600)
	
	fAATSR,ax=plt.subplots(figsize=(10,7))
	m = Basemap(projection='robin',lon_0=0)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,120.,30.))
	m.drawmeridians(np.arange(0.,3060.,60.))
	mycmap=plt.get_cmap('Purples',11) 
	# define the bins and normalize
	bounds = [0.025,0.075,0.15,0.225,0.30,0.375,0.45,0.525,0.6,0.675,0.75]
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	#plt.title('AOD collocated annual mean (AATSR)',fontsize=24)
	m = Basemap(projection='robin',lon_0=0,ax=ax)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,120.,30.))
	m.drawmeridians(np.arange(0.,3060.,60.))
	#mycmap=plt.get_cmap('coolwarm',11) 
	# define the bins and normalize
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	cs = m.pcolormesh(lons,lats,(AATSRdata.squeeze()),norm=norm,latlon=True,cmap=mycmap)
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.9)
	c.set_label('AOD [AATSR]',fontsize=24)
	c.ax.tick_params(labelsize=12)
	#c.set_label('AOD relative bias [(TM5-AERONET)/AERONET]')
	#fAATSR.savefig(output_pdf_path+'aatsr/AATSR_2010.pdf')
	#fAATSR.savefig(output_png_path+'aatsr/AATSR_2010.png',dpi=600)
def read_modis(landsea_print=False):
	
	cellarea_file=fixeddata+'/ec-ei-an0tr6-sfc-glb100x100-0000-oro.nc'
	cellarea=nc.Dataset(cellarea_file,'r').variables['cell_area'][:]
	lsm_file=fixeddata+'/ec-ei-an0tr6-sfc-glb100x100-lsm.nc'
	lsm=nc.Dataset(lsm_file,'r').variables['lsm'][:]
	#modis=output+'/TM5-MODIS//Aggregated_MODIS_L2_2010_1x1_yearmean.nc'
	modis=basepathprocessed+'TM5-MODIS/MOD04_MYD04_L2_QA2_aggregated_2010_1x1_yearmean.nc'
	modisdata=np.squeeze(nc.Dataset(modis,'r').variables['AOD_550_Dark_Target_Deep_Blue_Combined'][:]).transpose()
	modislon=nc.Dataset(modis,'r').variables['longitude'][:]
	modislat=nc.Dataset(modis,'r').variables['latitude'][:]
	#print(np.shape(modisdata))
	lsm2=lsm.copy()
	lsm[lsm>1]=1
	lsm[lsm<1]=nan
	lsm2[lsm2<1]=1
	lsm2[lsm2>1]=nan
	modeldata={}
	for expi in EXPs:
		print (expi)
		model=basepathprocessed+'/col_MOD04_MYD04_QA2_L2_TM5_'+expi+'_2010_1x1_yearmean.nc'  #Aggregated_lin_Col_TM5_'+EXPs[0]+'_MYD04_MOD04_L2_2010_1x1_yearmean.nc'
		#print (model)
		#model2=output+'/col_MOD04_MYD04_QA2_L2_TM5_'+EXPs[1]+'_2010_1x1_yearmean.nc'
		modeldata[expi]=np.squeeze(nc.Dataset(model,'r').variables['od550aer'][:].transpose())
		#modeldata2=np.squeeze(nc.Dataset(model2,'r').variables['od550aer'][:].transpose())
		if landsea_print:
			print(np.mean(modisdata),np.nanmean(modisdata))
			print(np.mean(modeldata[expi]),np.nanmean(modeldata[expi]))
			print ('cellareamean')
			print(np.mean(modisdata),np.nansum(modisdata*cellarea)/np.sum(cellarea))
			print(np.mean(modeldata[expi]),np.nansum(modeldata[expi]*cellarea)/(np.sum(cellarea)))
			#print(np.mean(modeldata2),np.nansum(modeldata2*cellarea)/(np.sum(cellarea)))
			print ('land')
			print(np.mean(modisdata),np.nansum(modisdata*cellarea*lsm)/np.nansum(cellarea*lsm))
			print(np.mean(modeldata[expi]),np.nansum(modeldata[expi]*cellarea*lsm)/(np.nansum(cellarea*lsm)))
			#print(np.mean(modeldata2),np.nansum(modeldata2*cellarea*lsm)/(np.nansum(cellarea*lsm)))
			print ('ocean')
			print(np.mean(modisdata),np.nansum(modisdata*cellarea*lsm2)/np.nansum(cellarea*lsm2))
			print(np.mean(modeldata[expi]),np.nansum(modeldata[expi]*cellarea*lsm2)/(np.nansum(cellarea*lsm2)))
			#print(np.mean(modeldata2),np.nansum(modeldata2*cellarea*lsm2)/(np.nansum(cellarea*lsm2)))
	return modisdata, modeldata,modislon,modislat
def plot_modis_diff(modisdata,modeldata,modislon,modislat):
	k=-1
	lons, lats = np.meshgrid(modislon,modislat)
	k+=1
	bounds_load=[-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3]

	for i,expi in enumerate(EXPs):
		f,ax=plt.subplots(ncols=1,figsize=(8,4))
		#ax.set_title('diff')
		m=Basemap(projection='robin',lon_0=0,ax=ax)
		bounds = [-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]
		norm = mpl.colors.BoundaryNorm(bounds, 11)
		mycmap=plt.get_cmap('coolwarm',11) 
		m.drawparallels(np.arange(-90.,90.,30.))
		m.drawmeridians(np.arange(-180.,180.,60.))
		m.drawcoastlines()
		cs = m.pcolormesh(lons,lats,((modeldata[expi]-modisdata).squeeze()),norm=norm,latlon=True,cmap=mycmap)
		c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.9)
		c.set_label('AOD bias ['+EXP_NAMEs[i]+'-MODIS]',fontsize=24)
		c.ax.tick_params(labelsize=12)
		print (np.mean(modisdata))
		print (np.mean(modeldata[expi]))
		print (np.max(modisdata))
		print (np.max(modeldata[expi]))
		fsoa,ax=plt.subplots(figsize=(10,7))
		m = Basemap(projection='robin',lon_0=0)
		m.drawcoastlines()
		m.drawparallels(np.arange(-90.,120.,30.))
		m.drawmeridians(np.arange(0.,3060.,60.))
		mycmap=plt.get_cmap('Purples',11) 
		# define the bins and normalize
		bounds = [0.025,0.075,0.15,0.225,0.30,0.375,0.45,0.525,0.6,0.675,0.75]
		norm = mpl.colors.BoundaryNorm(bounds, 11)
		m = Basemap(projection='robin',lon_0=0,ax=ax)
		m.drawcoastlines()
		m.drawparallels(np.arange(-90.,120.,30.))
		m.drawmeridians(np.arange(0.,3060.,60.))
		# define the bins and normalize
		norm = mpl.colors.BoundaryNorm(bounds, 11)
		cs = m.pcolormesh(lons,lats,(modeldata[expi].squeeze()),norm=norm,latlon=True,cmap=mycmap)
		c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.9)
		c.set_label('AOD [TM5]',fontsize=24)
		c.ax.tick_params(labelsize=12)
		#fsoa.savefig(output_pdf_path+'MODIS/TM5-'+EXP_NAMEs[i]+'_2010.pdf')
		#fsoa.savefig(output_png_path+'MODIS/TM5-'+EXP_NAMEs[i]+'_2010.png',dpi=600)
		
	fmodis,ax=plt.subplots(figsize=(10,7))
	m = Basemap(projection='robin',lon_0=0)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,120.,30.))
	m.drawmeridians(np.arange(0.,3060.,60.))
	mycmap=plt.get_cmap('Purples',11) 
	# define the bins and normalize
	bounds = [0.025,0.075,0.15,0.225,0.30,0.375,0.45,0.525,0.6,0.675,0.75]
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	m = Basemap(projection='robin',lon_0=0,ax=ax)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,120.,30.))
	m.drawmeridians(np.arange(0.,3060.,60.))
	# define the bins and normalize
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	cs = m.pcolormesh(lons,lats,(modisdata.squeeze()),norm=norm,latlon=True,cmap=mycmap)
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.8)
	c.set_label('AOD [MODIS]',fontsize=24)
	c.ax.tick_params(labelsize=12)
	#fmodis.savefig(output_pdf_path+'MODIS/MODIS_2010.pdf')
	#fmodis.savefig(output_png_path+'MODIS/MODIS_2010.png',dpi=600)
def plot_both(data,bounds=[-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]):
	n=len(data)
	if n<1:
		exit()
	else:
		pass
	fcolumn,axit=plt.subplots(ncols=2,figsize=(20,6))
	name=''

	for i,idata in enumerate(data):
		lons, lats = np.meshgrid(data[i]['lon'],data[i]['lat'])
		mycmap=plt.get_cmap('Purples',11) 
		mycmap=plt.get_cmap('coolwarm',11) 
		# define the bins and normalize
		norm = mpl.colors.BoundaryNorm(bounds, 11)
		#plt.title('AOD collocated annual mean (MODIS)',fontsize=24)
		m = Basemap(projection='robin',lon_0=0,ax=axit[i])
		m.drawcoastlines()
		m.drawparallels(np.arange(-90.,120.,30.))
		m.drawmeridians(np.arange(0.,30.,60.))
		# define the bins and normalize
		norm = mpl.colors.BoundaryNorm(bounds, 11)
		cs = m.pcolormesh(lons,lats,((data[i]['modeldata']-data[i]['satdata']).squeeze()),norm=norm,latlon=True,cmap=mycmap)
		c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.02,shrink=1.05,ax=axit[i])
		c.ax.tick_params(labelsize=13)
		name=name+'-'+data[i]['name']
		c.set_label('AOD difference ['+data[i]['name']+']',fontsize=20)
		axit[i].annotate(string.ascii_lowercase[i]+')',xy=(0.05+float(i)*0.5,0.85),xycoords='figure fraction',fontsize=24)
	return fcolumn,axit
def main(all=False):
	modisdata,modismodeldata,modislon,modislat=read_modis()
	AATSRdata,aatsrmodeldata,aatsrlon,aatsrlat=read_aatsr()
	fs13,as13=plot_both([{"satdata":modismodeldata['oldsoa-bhn'],"modeldata":modismodeldata['newsoa-ri'],"lon":modislon,"lat":modislat,'name':'NEWSOA-OLDSOA'},{"satdata":aatsrmodeldata['oldsoa-bhn'],"modeldata":aatsrmodeldata['newsoa-ri'],"lon":aatsrlon,"lat":aatsrlat,'name':'NEWSOA-OLDSOA'}],[-0.0375,-0.03,-0.0225,-0.015,-0.0075,-0.0025,0.0025,0.0075,0.015,0.0225,0.030,0.0375])
	plt.tight_layout()
	fs13.savefig(output_png_path+'article/figS13_revised_bias-MODIS-AATSR-satellite_2010.png',dpi=600)
	fs13.savefig(output_pdf_path+'article/figS13_revised_bias-MODIS-AATSR-satellite_2010.pdf',dpi=600)
	if all:
		plot_aatsr_diff(AATSRdata,aatsrmodeldata,aatsrlon,aatsrlat)
		plot_modis_diff(modisdata,modismodeldata,modislon,modislat)
	f13,a13=plot_both([{"satdata":modisdata,"modeldata":modismodeldata['newsoa-ri'],"lon":modislon,"lat":modislat,'name':'MODIS'},{"satdata":AATSRdata,"modeldata":aatsrmodeldata['newsoa-ri'],"lon":aatsrlon,"lat":aatsrlat,'name':'AATSR'}])
	f13.savefig(output_png_path+'article/fig13_revised_bias-MODIS-AATSR-satellite_2010.png',dpi=600)
	f13.savefig(output_pdf_path+'article/fig13_revised_bias-MODIS-AATSR-satellite_2010.pdf',dpi=600)

	plt.show()
if __name__=='__main__':
	main()
