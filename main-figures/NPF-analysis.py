import numpy as np 
import matplotlib.pyplot as plt 
import netCDF4 as nc 
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import sys
#sys.path.append("/Users/bergmant/Documents/Project/ifs+tm5-validation/scripts")
from general_toolbox import lonlat,read_var,monthlengths,get_gb_xr
#from plot_m7 import read_var,modal_fraction,read_SD,plot_N_map,discretize_m7,discretize_mode,plot_mean_m7,plot_sd_pcolor,zonal
import xarray as xr
from mass_budget import mapit_log, mapit_boundary
from settings import *
EXPS=['newsoa-ri','oldsoa-bhn']
EXPnames=['NEWSOA','OLDSOA']
data={}
monthlengths=[31,28,31,30,31,30,31,31,30,31,30,31]
for exp in EXPS:
	data[exp]=xr.open_dataset('/Users/bergmant/Documents/tm5-soa/output/general_TM5_'+exp+'_2010.mm.nc')
gb=xr.open_dataset(fixeddata+'/griddef_62.nc')
gb=get_gb_xr()
#test=xr.open_dataset('/Volumes/Utrecht/newsoa-ri/general_TM5_newsoa-ri_2010.lev1.nc')

#((test.d_nuc*gb.area).sum(dim=['lat','lon'])/gb.area.sum()).plot()
#print test.d_nuc.where(test.d_nuc>0.00001).median(dim='time')
#plt.figure()
#a=test.d_nuc.median(dim='time').plot()
#plt.colorbar(a)


#plt.show()
#print test.d_nuc.median(dim=['lat','lon']).data
#print data
letters=[['a','b'],['c','d']]
f,ax=plt.subplots(nrows=2,ncols=2,figsize=(12,8))
for i,exp in enumerate(EXPS):
	#f,ax=plt.subplots(1)	
	#print i,data[i].d_nuc.mean(dim='time').sel(lev=1)
	print i,data[exp].d_nuc.mean(dim='time').sel(lev=1).mean().values

	mapit_boundary(data[exp].d_nuc.mean(dim='time').sel(lev=slice(1,10)).sum(dim='lev'),[0.0,1e-4,0.001,0.01,0.1,0.5,1,2.5,5,10,15],ax[0,i],False,cblabel='Particle formation [cm$^{-3}$ s$^{-1}$]')
	ax[0,i].set_title(EXPnames[i])
	ax[0,i].annotate(('%s)')%(letters[0][i]),xy=(0.0,1.0),xycoords='axes fraction',fontsize=24)
	#f.savefig(pngpath+'/production/OLDSOA_annual_production_SOA.png',dpi=400)
#for i in EXPS:
	#f,ax=plt.subplots(1)	
	#print data[i].N_NUS.mean(dim='time').sel(lev=1)
	#N=(data[exp].N_NUS.mean(dim='time').sel(lev=1)*1e-6+ data[exp].N_AIS.mean(dim='time').sel(lev=1)*1e-6 + data[exp].N_ACS.mean(dim='time').sel(lev=1)*1e-6 + data[exp].N_COS.mean(dim='time').sel(lev=1)*1e-6+
	#	data[exp].N_AII.mean(dim='time').sel(lev=1)*1e-6+data[exp].N_ACI.mean(dim='time').sel(lev=1)*1e-6 + data[exp].N_COI.mean(dim='time').sel(lev=1)*1e-6)
	N=(data[exp].N_NUS.mean(dim='time').sel(lev=1)*1e-6)
	#mapit_boundary(N,[10,50,100,250,500,750,1000,2500,5000,7500,10000,25000,50000],ax[1,i],False,cblabel='Nucleation mode \nnumber concentration [cm$^{-3}$]')
	mapit_boundary(N,[0.1,1,10,100,500,1000,5000,10000],ax[1,i],False,cblabel='Nucleation mode \nnumber concentration [cm$^{-3}$]')
	ax[1,i].annotate(('%s)')%(letters[1][i]),xy=(0.0,1.0),xycoords='axes fraction',fontsize=24)
	#ax.set_title('Annual production of OLDSOA in [g]')
	#f.savefig(pngpath+'/production/OLDSOA_annual_production_SOA.png',dpi=400)
plt.tight_layout()
f.savefig(output_png_path+'/article/fig7_npf-nnus-2x2.png',dpi=600)
f.savefig(output_pdf_path+'/article/fig7_npf-nnus-2x2.pdf',dpi=600)

# fzon,azon=plt.subplots(ncols=3,figsize=(18,6))

# data['newsoa-ri'].d_nuc.mean(dim=['time','lon']).plot(ax=azon[0],levels=[0,0.0001,0.001,0.01,0.1,0.5,1,2,3,4,5,10,100])
# data['oldsoa-bhn'].d_nuc.mean(dim=['time','lon']).plot(ax=azon[1],levels=[0,0.0001,0.001,0.01,0.1,0.5,1,2,3,4,5,10,100])
# ((data['newsoa-ri'].d_nuc.mean(dim=['time','lon'])-data['oldsoa-bhn'].d_nuc.mean(dim=['time','lon']))/(data['oldsoa-bhn'].d_nuc.mean(dim=['time','lon']))).plot(ax=azon[2],levels=[-5,-4,-3,-2,-1,-0.5,0,0.5,1,2,3,4,5,10,100,1000,10000,100000,1000000])

# print data['newsoa-ri'].d_nuc.isel(lev=1).max(dim=['lat','lon']).data
# print data['oldsoa-bhn'].d_nuc.isel(lev=1).max(dim=['lat','lon']).data
# print data['newsoa-ri'].d_nuc.isel(lev=1).min(dim=['lat','lon']).data
# print data['oldsoa-bhn'].d_nuc.isel(lev=1).min(dim=['lat','lon']).data

# f,ax=plt.subplots(1)
# Nn=(data['newsoa-ri'].N_NUS.mean(dim='time').sel(lev=1)*1e-6+ data['newsoa-ri'].N_AIS.mean(dim='time').sel(lev=1)*1e-6 + data['newsoa-ri'].N_ACS.mean(dim='time').sel(lev=1)*1e-6 + data['newsoa-ri'].N_COS.mean(dim='time').sel(lev=1)*1e-6+
# 		data['newsoa-ri'].N_AII.mean(dim='time').sel(lev=1)*1e-6+data['newsoa-ri'].N_ACI.mean(dim='time').sel(lev=1)*1e-6 + data['newsoa-ri'].N_COI.mean(dim='time').sel(lev=1)*1e-6)
# No=(data['oldsoa-bhn'].N_NUS.mean(dim='time').sel(lev=1)*1e-6+ data['oldsoa-bhn'].N_AIS.mean(dim='time').sel(lev=1)*1e-6 + data['oldsoa-bhn'].N_ACS.mean(dim='time').sel(lev=1)*1e-6 + data['oldsoa-bhn'].N_COS.mean(dim='time').sel(lev=1)*1e-6+
# 		data['oldsoa-bhn'].N_AII.mean(dim='time').sel(lev=1)*1e-6+data['oldsoa-bhn'].N_ACI.mean(dim='time').sel(lev=1)*1e-6 + data['oldsoa-bhn'].N_COI.mean(dim='time').sel(lev=1)*1e-6)
# mapit_boundary(Nn-No,[-7500,-5000,-2500,-1000,-500,-100,-10,10,100,500,1000,2500,5000,7500],ax,True,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title(' N(TOT) diff')
	
# f,ax=plt.subplots(1)
# Nn=(data['newsoa-ri'].N_NUS.mean(dim='time').sel(lev=1)*1e-6+ data['newsoa-ri'].N_AIS.mean(dim='time').sel(lev=1)*1e-6 + data['newsoa-ri'].N_ACS.mean(dim='time').sel(lev=1)*1e-6 + data['newsoa-ri'].N_COS.mean(dim='time').sel(lev=1)*1e-6)
# No=(data['oldsoa-bhn'].N_NUS.mean(dim='time').sel(lev=1)*1e-6+ data['oldsoa-bhn'].N_AIS.mean(dim='time').sel(lev=1)*1e-6 + data['oldsoa-bhn'].N_ACS.mean(dim='time').sel(lev=1)*1e-6 + data['oldsoa-bhn'].N_COS.mean(dim='time').sel(lev=1)*1e-6)	
# mapit_boundary(Nn-No,[-7500,-5000,-2500,-1000,-500,-100,-10,10,100,500,1000,2500,5000,7500],ax,True,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('Soluble N diff')
# f,ax=plt.subplots(1)
# Nn=(data['newsoa-ri'].N_AIS.mean(dim='time').sel(lev=1)*1e-6 )
# No=( data['oldsoa-bhn'].N_AIS.mean(dim='time').sel(lev=1)*1e-6 )	
# mapit_boundary(Nn-No,[-7500,-5000,-2500,-1000,-500,-100,-10,10,100,500,1000,2500,5000,7500],ax,True,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('Soluble N(AIS) diff')
# f,ax=plt.subplots(1)
# Nn=(data['newsoa-ri'].N_AIS.mean(dim='time').sel(lev=1)*1e-6 +data['newsoa-ri'].N_AII.mean(dim='time').sel(lev=1)*1e-6 )
# No=(data['oldsoa-bhn'].N_AIS.mean(dim='time').sel(lev=1)*1e-6+data['oldsoa-bhn'].N_AII.mean(dim='time').sel(lev=1)*1e-6 )	
# mapit_boundary(Nn-No,[-7500,-5000,-2500,-1000,-500,-100,-10,10,100,500,1000,2500,5000,7500],ax,True,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('Soluble N(AIS+AII) diff')
# f,ax=plt.subplots(1)
# Nn=(data['newsoa-ri'].N_AII.mean(dim='time').sel(lev=1)*1e-6 )
# No=(data['oldsoa-bhn'].N_AII.mean(dim='time').sel(lev=1)*1e-6 )	
# mapit_boundary(Nn-No,[-1000,-500,-400,-300,-200,-100,-10,10,100,200,300,400,500,1000],ax,True,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('Soluble N(AII) diff')
# f,ax=plt.subplots(1)
# Nn=( data['newsoa-ri'].N_ACS.mean(dim='time').sel(lev=1)*1e-6 )
# No=( data['oldsoa-bhn'].N_ACS.mean(dim='time').sel(lev=1)*1e-6)	
# mapit_boundary(Nn-No,[-7500,-5000,-2500,-1000,-500,-100,-10,10,100,500,1000,2500,5000,7500],ax,True,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('Soluble N(ACS) diff')
# f,ax=plt.subplots(1)
# Nn=( data['newsoa-ri'].N_ACI.mean(dim='time').sel(lev=1)*1e-6 )
# No=( data['oldsoa-bhn'].N_ACI.mean(dim='time').sel(lev=1)*1e-6)	
# mapit_boundary(Nn-No,[-7500,-5000,-2500,-1000,-500,-100,-10,10,100,500,1000,2500,5000,7500],ax,True,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('Soluble N(ACI) diff')
# f,ax=plt.subplots(1)
# Nn=( data['newsoa-ri'].N_ACS.mean(dim='time').sel(lev=1)*1e-6 +data['newsoa-ri'].N_ACI.mean(dim='time').sel(lev=1)*1e-6 )
# No=( data['oldsoa-bhn'].N_ACS.mean(dim='time').sel(lev=1)*1e-6+data['oldsoa-bhn'].N_ACI.mean(dim='time').sel(lev=1)*1e-6)	
# mapit_boundary(Nn-No,[-7500,-5000,-2500,-1000,-500,-100,-10,10,100,500,1000,2500,5000,7500],ax,True,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('Soluble N(ACS+ACI) diff')
# f,ax=plt.subplots(1)
# Nn=(  data['newsoa-ri'].N_COS.mean(dim='time').sel(lev=1)*1e-6)
# No=(  data['oldsoa-bhn'].N_COS.mean(dim='time').sel(lev=1)*1e-6)	
# mapit_boundary(Nn-No,[-7500,-5000,-2500,-1000,-500,-100,-10,10,100,500,1000,2500,5000,7500],ax,True,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('Soluble N(COS) diff')
# f,ax=plt.subplots(1)
# Nn=(  data['newsoa-ri'].N_COI.mean(dim='time').sel(lev=1)*1e-6)
# No=(  data['oldsoa-bhn'].N_COI.mean(dim='time').sel(lev=1)*1e-6)	
# mapit_boundary(Nn-No,[-7500,-5000,-2500,-1000,-500,-100,-10,10,100,500,1000,2500,5000,7500],ax,True,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('Soluble N(COI) diff')
# f,ax=plt.subplots(1)
# Nn=(data['newsoa-ri'].N_NUS.mean(dim='time').sel(lev=1)*1e-6)
# No=(data['oldsoa-bhn'].N_NUS.mean(dim='time').sel(lev=1)*1e-6)	
# mapit_boundary(Nn-No,[-7500,-5000,-2500,-1000,-500,-100,-10,10,100,500,1000,2500,5000,7500],ax,True,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('Soluble N(NUS) diff')
# f,ax=plt.subplots(1)
# Nn=(data['newsoa-ri'].N_NUS.mean(dim='time').sel(lev=1)*1e-6)
# No=(data['oldsoa-bhn'].N_NUS.mean(dim='time').sel(lev=1)*1e-6)	
# mapit_boundary(No,[0,5,10,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,7500],ax,False,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('Soluble N(NUS) OLDSOA')
# f,ax=plt.subplots(1)
# Nn=(data['newsoa-ri'].N_NUS.mean(dim='time').sel(lev=1)*1e-6)
# No=(data['oldsoa-bhn'].N_NUS.mean(dim='time').sel(lev=1)*1e-6)	
# mapit_boundary(Nn,[0,5,10,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000],ax,False,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('Soluble N(NUS) NEWSOA')
# f,ax=plt.subplots(1)
# Nn=(data['newsoa-ri'].emioa.mean(dim='time'))
# No=(data['oldsoa-bhn'].emioa.mean(dim='time'))	
# mapit_boundary(Nn-No,[-7500,-5000,-2500,-1000,-500,-100,-10,10,100,500,1000,2500,5000,7500],ax,True,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('emi diff')

# f,ax=plt.subplots(1)
# Nn=(data['newsoa-ri'].gph3D.mean(dim='time').sel(lev=10))
# No=(data['oldsoa-bhn'].gph3D.mean(dim='time').sel(lev=10))	
# mapit_boundary(Nn-No,[-100,-80,-60,-40,-10,10,40,60,80,100],ax,True,cblabel='Number concentration [cm$^{-3}$]')
# ax.set_title('emi diff')

plt.show()
