import numpy as np 
import matplotlib.pyplot as plt 
import netCDF4 as nc 
from matplotlib.colors import BoundaryNorm
#from matplotlib.colors import DivergingNorm #3.X? with python 3
from matplotlib.ticker import MaxNLocator,LinearLocator
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import sys
#sys.path.append("/Users/bergmant/Documents/Project/ifs+tm5-validation/scripts")
#sys.path.append("/Users/bergmant/Documents/python/aeronet/")
from colocate_aeronet import do_colocate
from general_toolbox import read_var#,read_SD
from lonlat import lonlat
#from plot_m7 import read_var,modal_fraction,read_SD,plot_N_map,discretize_m7,discretize_mode,plot_mean_m7,plot_sd_pcolor,zonal
#import ngl
from settings import *
def read_soa(infile):
	comp='SOA'
	modes=['NUS','AIS','ACS','COS','AII']
	data={}
	for i in modes:
		var='M_'+comp+i
		vardata=read_var(infile,var)
		print vardata[0]
		data[i]=vardata[0]
	outdata=np.zeros_like(data['NUS'][:])
	print outdata
	for i in modes:
		outdata+=data[i]
	return outdata
def vertical_sampling(levelheight,data,gb):
	ndims=data.shape
	outdata=np.array()
	for i in range(5):
		n_i=0
		for j in range(34):
			outdata[i]+=np.sum(data[j,:,:]*gb)/np.sum(gb)
			n_i+=1
		outdata[i]=outdata[i]/n_i
	return outdata

def zonal(data):
	pass

hyam = np.array([3.287814, 171.6739985, 817.2142485, 2153.9015505, 4216.4748535, 
    6889.527832, 9949.709961, 13104.40625, 16025.6206055, 18367.185547, 
    19833.893555, 20333.9638675, 20134.3564455, 19552.1708985, 18687.7871095, 
    17590.826172, 16314.338379, 14915.0317385, 13452.9536135, 11988.190918, 
    10568.022461, 9219.092285, 7953.046875, 6770.654541, 5431.807861, 
    4011.101074, 2825.2906495, 1877.659729, 1160.5548705, 654.315796, 
    327.661621, 140.4025535, 48.790634, 10.706806])
hybm = np.array([0.997102, 0.983835, 0.9543115, 0.9053375, 0.8354515, 0.7468045, 
    0.6437705, 0.5321665, 0.4186095, 0.3108785, 0.2178225, 0.144535, 
    0.096878, 0.0681255, 0.0458505, 0.029208, 0.0173485, 0.009415, 0.0045435, 
    0.0018825, 0.0006395, 0.000167, 2.75e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])

gbfile='/Users/bergmant/Documents/python/tm5/griddef_62.nc'
ncgb=nc.Dataset(gbfile,'r')
gb=ncgb.variables['area']

fnew=output+'/general_TM5_newsoa-ri_2010.mm.nc'
fold=output+'/general_TM5_oldsoa-bhn_2010.mm.nc'

# p=nc.Dataset(fnew).variables['pressure'][:]
# T=nc.Dataset(fnew).variables['temp'][:]
# gphnew=nc.Dataset(fnew).variables['gph3D'][:]
# gphold=nc.Dataset(fold).variables['gph3D'][:]
# lon=nc.Dataset(fnew).variables['lon'][:]
# lat=nc.Dataset(fnew).variables['lat'][:]

# psnew=nc.Dataset(fnew).variables['ps'][:]
# psold=nc.Dataset(fold).variables['ps'][:]

# Tann=T.mean(axis=0)
# pann=p.mean(axis=0)
# print T.shape,p.shape,
# h=287.05/9.80666*(Tann[:,40,40]+Tann[:,40,40])/2*np.log(pann[0,40,40]/pann[:,40,40])
# h2=287.05/9.80666*Tann[:,:,:]*np.log(1e5/pann[:,:,:])
# # plt.figure()
# # plt.plot(gphnew.mean(axis=(0,2,3)),h2.mean(axis=(1,2)))
# # plt.figure()
# # plt.plot(gphnew.mean(axis=(0,2,3)),gphnew.mean(axis=(0,2,3)))
# # plt.show()
# Rnew,Nnew=read_SD(fnew)
# Rold,Nold=read_SD(fold)
# f,a=plt.subplots(10,figsize=(4,10))
# for i in range(0,30,3):
# 	print i
# 	ins,sol,lnr=discretize_m7(Rnew[:,:,i,40,40],Nnew[:,:,i,40,40])
# 	plot_mean_m7(a[(i-0)/3],ins,sol,lnr,'r','new',True,[0,3e3])
# 	ins,sol,lnr=discretize_m7(Rold[:,:,i,40,40],Nold[:,:,i,40,40])
# 	plot_mean_m7(a[(i-0)/3],ins,sol,lnr,'b','old',True,[0,3e3])
# 	a[(i-0)/3].set_yscale('linear')
# 	a[(i-0)/3].set_title(str(h[i]))
# f,a = plt.subplots(1,figsize=(8,8))

# a.plot(Nnew[:,:,:,40,40].mean(axis=1).sum(axis=0),np.linspace(0,34,34),'r')
# a.plot(Nold[:,:,:,40,40].mean(axis=1).sum(axis=0),np.linspace(0,34,34),'b')
# a.plot(Nnew[0,:,:,40,40].mean(axis=0),np.linspace(0,34,34),'--r')
# a.plot(Nold[0,:,:,40,40].mean(axis=0),np.linspace(0,34,34),'--b')
# a.plot(Nnew[1,:,:,40,40].mean(axis=0),np.linspace(0,34,34),'.-r')
# a.plot(Nold[1,:,:,40,40].mean(axis=0),np.linspace(0,34,34),'.-b')
# #a.plot(Nnew[2,:,:,40,40].mean(axis=0),np.linspace(0,34,34),'s-r')
# #a.plot(Nold[2,:,:,40,40].mean(axis=0),np.linspace(0,34,34),'s-b')

# f,a = plt.subplots(1,figsize=(8,8))

# a.plot(Nnew[:,:,0:15,:,:].mean(axis=(1,3,4)).sum(axis=0),gphnew[:,0:15,:,:].mean(axis=(0,2,3)),'r')
# a.plot(Nold[:,:,0:15,:,:].mean(axis=(1,3,4)).sum(axis=0),gphold[:,0:15,:,:].mean(axis=(0,2,3)),'b')
# a.plot(Nnew[1:7,:,0:15,:,:].mean(axis=(1,3,4)).sum(axis=0),gphnew[:,0:15,:,:].mean(axis=(0,2,3)),'m')
# a.plot(Nold[1:7,:,0:15,:,:].mean(axis=(1,3,4)).sum(axis=0),gphold[:,0:15,:,:].mean(axis=(0,2,3)),'g')
# a.set_xlim([1e8,1e10])
# a.set_xscale('log')
# print np.min(gphnew[:,0,:,:].mean(axis=(0)))
# print np.max(gphnew[:,0,:,:].mean(axis=(0)))
# print gphnew[:,0:15,:,:].mean(axis=(0,2,3))
# print h2.shape
# print h2[0:15,:,:].mean(axis=(1,2))
SOAnew=read_soa(fnew)
SOAold=read_soa(fold)

f,a = plt.subplots(ncols=2,figsize=(8,8))

print SOAnew.shape
# f,a = plt.subplots(ncols=2,figsize=(8,8))
# a[0].plot(SOAnew[:,0:15,:,:].mean(axis=(0,2,3))*1e9,gphnew[:,0:15,:,:].mean(axis=(0,2,3)),'r')
# a[0].plot(SOAold[:,0:15,:,:].mean(axis=(0,2,3))*1e9,gphold[:,0:15,:,:].mean(axis=(0,2,3)),'b')
# a[0].set_ylim([0,7000])
# a[0].set_xscale('log')
# ratio=(SOAnew[:,0:15,:,:].mean(axis=(0,2,3))-SOAold[:,0:15,:,:].mean(axis=(0,2,3)))/SOAold[:,0:15,:,:].mean(axis=(0,2,3))
# a[1].plot(ratio,gphnew[:,0:15,:,:].mean(axis=(0,2,3)),'r')
# a[1].plot([0,0],[0,10000],'--k')
# a[1].set_ylim([0,7000])
# a[1].set_xlim([-0.75,0.75])
# f,a = plt.subplots(ncols=2,figsize=(8,8))
# a[0].plot(SOAnew[:,0:15,:,:].mean(axis=(0,2,3))*1e9,h2[0:15,:,:].mean(axis=(1,2)),'r')
# a[0].plot(SOAold[:,0:15,:,:].mean(axis=(0,2,3))*1e9,h2[0:15,:,:].mean(axis=(1,2)),'b')
# a[0].set_ylim([0,7000])
# a[0].set_xscale('log')
# ratio=(SOAnew[:,0:15,:,:].mean(axis=(0,2,3))-SOAold[:,0:15,:,:].mean(axis=(0,2,3)))/SOAold[:,0:15,:,:].mean(axis=(0,2,3))
# a[1].plot(ratio,h2[0:15,:,:].mean(axis=(1,2)),'r')
# a[1].plot([0,0],[0,10000],'--k')
# a[1].set_ylim([0,7000])
# a[1].set_xlim([-0.75,0.75])

f,a = plt.subplots(ncols=3,figsize=(16,8))
print SOAnew.shape,(SOAnew[:,:,:,:].mean(axis=0)*gb[:,:]).sum(axis=2).shape
print SOAnew[:,:,:,:].mean(axis=0).mean(axis=2)[10,45]
print ((SOAnew[:,:,:,:].mean(axis=0)*gb[:,:]).sum(axis=2)/(gb[:,:]).sum(axis=1))[10,45]
data1=((SOAnew[:,:,:,:].mean(axis=0)*gb[:,:]).sum(axis=2)/(gb[:,:]).sum(axis=1))
data2=((SOAold[:,:,:,:].mean(axis=0)*gb[:,:]).sum(axis=2)/(gb[:,:]).sum(axis=1))
zmin=min(data1.min(),data2.min())
zmax=max(data1.max(),data2.max())
print zmin
print zmax
lon,lat=lonlat('TM53x2')
levels = MaxNLocator(nbins=15).tick_values(zmin*1e9,zmax*1e9)
cmap = plt.get_cmap('Greens')
print data1.shape
c1=a[0].contourf(lat,np.linspace(1,35,34),data1*1e9,levels=levels,cmap=cmap)
cb1= f.colorbar(c1,ax=a[0], orientation='horizontal')
c2=a[1].contourf(lat,np.linspace(1,35,34),data2*1e9,levels=levels,cmap=cmap)
cb2= f.colorbar(c2,ax=a[1], orientation='horizontal')

change=((data1-data2)/data2)*100
levels = MaxNLocator(nbins=25).tick_values(-change.max(),change.max())
cmap = plt.get_cmap('RdBu_r')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#norm = DivergingNorm(vmin=change.min(),vcenter=0.,vmax=change.max())
c3=a[2].contourf(lat,np.linspace(1,35,34),change,norm=norm,levels=levels,cmap=cmap)
cb3= f.colorbar(c3,ax=a[2], orientation='horizontal')



f,a = plt.subplots(ncols=2,figsize=(8,8))

data=(SOAnew[:,:,:,:].sum(axis=0)*gb[:,:]).sum(axis=(1,2))/(gb[:,:]).sum()
a[0].plot(data*1e9,np.linspace(1,35,34),'r')
#a[0].plot(SOAnew[:,:,:,:].mean(axis=(0,2,3))*1e9,np.linspace(1,35,34),'r')
data=(SOAold[:,:,:,:].mean(axis=0)*gb[:,:]).sum(axis=(1,2))/(gb[:,:]).sum()
a[0].plot(data*1e9,np.linspace(1,35,34),'b')
#a[0].plot(SOAold[:,:,:,:].mean(axis=(0,2,3))*1e9,np.linspace(1,35,34),'b')
a[0].set_ylabel('Model level')
a[0].set_xlabel('SOA [ug m$^{-3}$]')
a[0].set_title('Concentration of SOA')
a[0].set_ylim([0,34])
#a[0].set_xscale('log')
a[0].set_xlim([1e-2,1e0])
ratio=(SOAnew[:,:,:,:].mean(axis=(0,2,3))-SOAold[:,:,:,:].mean(axis=(0,2,3)))/SOAold[:,:,:,:].mean(axis=(0,2,3))
a[1].plot(ratio,np.linspace(1,35,34),'r')
a[1].plot([0,0],[0,35],'--k')
a[1].set_ylim([0,34])
a[1].set_xlim([-0.75,0.75])
a[1].set_ylabel('Model level')
a[1].set_xlabel('ratio (NEWSOA-OLDSOA)/OLDSOA')
a[1].set_title('Ratio of concentrations')
f.savefig(output_png_path+'/SOA_concentrations_vertical.png',dpi=600)
f.savefig(output_pdf_path+'/SOA_concentrations_vertical.pdf')
f.savefig(output_jpg_path+'/SOA_concentrations_vertical.jpg',dpi=600)
print type(gb)
ggb=gb[:,:]
n4gb=ggb[np.newaxis,np.newaxis,:,:]
print n4gb.shape
print ((SOAnew[:,:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0))*1e9
print ((SOAnew[:,:2,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0))*1e9
print ((SOAnew[:,2:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0))*1e9
print ((SOAnew[:,:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0))/((SOAnew[:,:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()
print ((SOAnew[:,1:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()/((SOAnew[:,:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()
print ((SOAnew[:,2:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()/((SOAnew[:,:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()
print ((SOAnew[:,:2,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()/((SOAnew[:,:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()
print ((SOAnew[:,1,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()/((SOAnew[:,:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()

print ((SOAold[:,:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0))/((SOAold[:,:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()
print ((SOAold[:,1:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()/((SOAold[:,:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()
print ((SOAold[:,2:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()/((SOAold[:,:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()
print ((SOAold[:,:2,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()/((SOAold[:,:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()
print ((SOAold[:,1,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()/((SOAold[:,:,:,:]*n4gb).sum(axis=(2,3))/n4gb.sum()).sum(axis=(0)).sum()
#fas
print (SOAnew[:,:,:,:]).sum(axis=(2,3)).mean(axis=(0))*1e9
f,a = plt.subplots(ncols=3,figsize=(18,8))
mycmap=plt.get_cmap('Greens')
levels=[0,0.01,0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.4]
norm = BoundaryNorm(boundaries=levels, ncolors=256)
data=(SOAnew[:,:,:,:].mean(axis=(0,3)))*1e9
print norm
cs1=a[0].contourf(lat,np.linspace(1,35,34),data,levels=levels,cmap=mycmap,norm=norm)
a[0].set_ylabel('Model level')
a[0].set_xlabel('Latitude')
a[0].set_title('NEWSOA')
a[0].set_ylim([1,34])
a[0].annotate('a)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=14)
f.colorbar(cs1,ax=a[0],label='SOA [ug m$^{-3}$]', orientation='horizontal')

data=(SOAold[:,:,:,:].mean(axis=(0,3)))*1e9
cs2=a[1].contourf(lat,np.linspace(1,35,34),data,levels=levels,cmap=mycmap,norm=norm)
a[1].set_ylabel('Model level')
#a[1].set_xlabel('SOA [ug m$^{-3}$]')
a[1].set_xlabel('Latitude')
f.colorbar(cs2,ax=a[1],label='SOA [ug m$^{-3}$]', orientation='horizontal')
a[1].set_title('OLDSOA')
a[1].set_ylim([1,34])
a[1].annotate('b)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=14)
mycmap=plt.get_cmap('RdBu_r')
ratio=(SOAnew[:,:,:,:].mean(axis=(0,3))-SOAold[:,:,:,:].mean(axis=(0,3)))/SOAold[:,:,:,:].mean(axis=(0,3))*100
cs3=a[2].contourf(lat,np.linspace(1,35,34),ratio,cmap=mycmap,levels=[-125,-100,-75,-50,-25,-5,5,25,50,75,100,125])
a[2].set_ylim([1,34])
a[2].set_ylabel('Model level')
a[2].set_xlabel('Latitude')
cbar=f.colorbar(cs3, orientation='horizontal',label='Change in SOA concentration [%]',ticks=[-125,-100,-75,-50,-25,-5,5,25,50,75,100,125])

#cbar.ax.set_yticklabels()
#a[2].clabel('ratio (NEWSOA-OLDSOA)/OLDSOA')
a[2].set_title('Relative difference')
a[2].annotate('c)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=14)
f.savefig(output_png_path+'/article/fig6_SOA_concentrations_zonal.png',dpi=600)
f.savefig(output_pdf_path+'/article/fig6_SOA_concentrations_zonal.pdf')
f.savefig(output_jpg_path+'/article/fig6_SOA_concentrations_zonal.jpg',dpi=600)
plt.show()
