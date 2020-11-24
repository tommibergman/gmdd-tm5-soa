#from lonlat import lonlat
import netCDF4 as nc 
import xarray as xr
import os
def get_gridboxarea(grid, path=''):
	if grid=='TM53x2':
		gridareafile=path+'griddef_62.nc'
		if os.path.isfile(gridareafile):
			ncgb=nc.Dataset(gridareafile,'r')
			gridarea=ncgb.variables['area']
			return gridarea
		else:
			print 'gridarea file: '+gridareafile+' not found!'
			return False
	else:
		print 'grid name unkown'
		return False
def get_gb_xr():
	gbfile='griddef_62.nc'
	dsgb=xr.open_dataset(gbfile)
#	gb=ncgb.variables['area']
	return dsgb
def lonlat(grid):
    #returns gridpoint coordinates T63 grid
    import numpy as np

    if grid=="T63":
        lon = np.array([0, 1.875, 3.75, 5.625, 7.5, 9.375, 11.25, 13.125, 15, 16.875, 18.75, 
                        20.625, 22.5, 24.375, 26.25, 28.125, 30, 31.875, 33.75, 35.625, 37.5, 
                        39.375, 41.25, 43.125, 45, 46.875, 48.75, 50.625, 52.5, 54.375, 56.25, 
                        58.125, 60, 61.875, 63.75, 65.625, 67.5, 69.375, 71.25, 73.125, 75, 
                        76.875, 78.75, 80.625, 82.5, 84.375, 86.25, 88.125, 90, 91.875, 93.75, 
                        95.625, 97.5, 99.375, 101.25, 103.125, 105, 106.875, 108.75, 110.625, 
                        112.5, 114.375, 116.25, 118.125, 120, 121.875, 123.75, 125.625, 127.5, 
                        129.375, 131.25, 133.125, 135, 136.875, 138.75, 140.625, 142.5, 144.375, 
                        146.25, 148.125, 150, 151.875, 153.75, 155.625, 157.5, 159.375, 161.25, 
                        163.125, 165, 166.875, 168.75, 170.625, 172.5, 174.375, 176.25, 178.125, 
                        180, 181.875, 183.75, 185.625, 187.5, 189.375, 191.25, 193.125, 195, 
                        196.875, 198.75, 200.625, 202.5, 204.375, 206.25, 208.125, 210, 211.875, 
                        213.75, 215.625, 217.5, 219.375, 221.25, 223.125, 225, 226.875, 228.75, 
                        230.625, 232.5, 234.375, 236.25, 238.125, 240, 241.875, 243.75, 245.625, 
                        247.5, 249.375, 251.25, 253.125, 255, 256.875, 258.75, 260.625, 262.5, 
                        264.375, 266.25, 268.125, 270, 271.875, 273.75, 275.625, 277.5, 279.375, 
                        281.25, 283.125, 285, 286.875, 288.75, 290.625, 292.5, 294.375, 296.25, 
                        298.125, 300, 301.875, 303.75, 305.625, 307.5, 309.375, 311.25, 313.125, 
                        315, 316.875, 318.75, 320.625, 322.5, 324.375, 326.25, 328.125, 330, 
                        331.875, 333.75, 335.625, 337.5, 339.375, 341.25, 343.125, 345, 346.875, 
                        348.75, 350.625, 352.5, 354.375, 356.25, 358.125])
        lat = np.array([88.5721685140073, 86.7225309546681, 84.8619702920424, 
                        82.9989416428375, 81.1349768376774, 79.2705590348597, 77.4058880820788, 
                        75.541061452879, 73.6761323132091, 71.8111321142745, 69.9460806469834, 
                        68.0809909856513, 66.2158721139987, 64.3507304088721, 62.4855705220364, 
                        60.6203959268265, 58.7552092693799, 56.8900126013571, 55.0248075383117, 
                        53.1595953700197, 51.2943771389511, 49.429153697123, 47.5639257479787, 
                        45.6986938777018, 43.8334585789513, 41.9682202690754, 40.1029793042494, 
                        38.2377359905648, 36.3724905928122, 34.507243341501, 32.6419944385177, 
                        30.7767440617232, 28.9114923687178, 27.0462394999448, 25.1809855812706, 
                        23.3157307261409, 21.4504750373982, 19.5852186088223, 17.7199615264474, 
                        15.8547038696949, 13.9894457123567, 12.1241871234558, 10.2589281680064, 
                        8.39366890769239, 6.52840940147998, 4.66314970617789, 2.79788987695673, 
                        0.932629967838004, -0.932629967838004, -2.79788987695673, 
                        -4.66314970617789, -6.52840940147998, -8.39366890769239, 
                        -10.2589281680064, -12.1241871234558, -13.9894457123567, 
                        -15.8547038696949, -17.7199615264474, -19.5852186088223, 
                        -21.4504750373982, -23.3157307261409, -25.1809855812706, 
                        -27.0462394999448, -28.9114923687178, -30.7767440617232, 
                        -32.6419944385177, -34.507243341501, -36.3724905928122, 
                        -38.2377359905648, -40.1029793042494, -41.9682202690754, 
                        -43.8334585789513, -45.6986938777018, -47.5639257479787, 
                        -49.429153697123, -51.2943771389511, -53.1595953700197, 
                        -55.0248075383117, -56.8900126013571, -58.7552092693799, 
                        -60.6203959268265, -62.4855705220364, -64.3507304088721, 
                        -66.2158721139987, -68.0809909856513, -69.9460806469834, 
                        -71.8111321142745, -73.6761323132091, -75.541061452879, 
                        -77.4058880820788, -79.2705590348597, -81.1349768376774, 
                        -82.9989416428375, -84.8619702920424, -86.7225309546681, -88.5721685140073 ])
    elif (grid=='TM53x2'):
        lon=np.array([-178.5, -175.5, -172.5, -169.5, -166.5, -163.5, -160.5, -157.5,
            -154.5, -151.5, -148.5, -145.5, -142.5, -139.5, -136.5, -133.5, -130.5,
            -127.5, -124.5, -121.5, -118.5, -115.5, -112.5, -109.5, -106.5, -103.5, 
            -100.5, -97.5, -94.5, -91.5, -88.5, -85.5, -82.5, -79.5, -76.5, -73.5, 
            -70.5, -67.5, -64.5, -61.5, -58.5, -55.5, -52.5, -49.5, -46.5, -43.5,
            -40.5, -37.5, -34.5, -31.5, -28.5, -25.5, -22.5, -19.5, -16.5, -13.5,
             -10.5, -7.5, -4.5, -1.5, 1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5,
             25.5, 28.5, 31.5, 34.5, 37.5, 40.5, 43.5, 46.5, 49.5, 52.5, 55.5, 58.5, 
             61.5, 64.5, 67.5, 70.5, 73.5, 76.5, 79.5, 82.5, 85.5, 88.5, 91.5, 94.5,
              97.5, 100.5, 103.5, 106.5, 109.5, 112.5, 115.5, 118.5, 121.5, 124.5,
              127.5, 130.5, 133.5, 136.5, 139.5, 142.5, 145.5, 148.5, 151.5, 154.5,
              157.5, 160.5, 163.5, 166.5, 169.5, 172.5, 175.5, 178.5])
        lat = np.array([-89, -87, -85, -83, -81, -79, -77, -75, -73, -71, -69, -67, -65,   
            -63, -61, -59, -57, -55, -53, -51, -49, -47, -45, -43, -41, -39, -37,
            -35, -33, -31, -29, -27, -25, -23, -21, -19, -17, -15, -13, -11, -9, -7,
             -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
             33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67,
             69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89])
    else:
        print "lonlat.py: unkonwn grid"
        return None
    
    return lon,lat

def write_netcdf_file(datain, varnames, filename, lat=None, lon=None, timein=None,
                      var_longnames='NetCDF variable',
                      var_units='',
                      time_name='time',
                      time_units='days since 2010-01-01 00:00:00',
                      time_longname='time',
                      time_standardname='time',
                      time_calender='gregorian',
                      lon_name='lon',
                      lon_longname='longitude',
                      lon_standardname='longitude',
                      lon_units='degrees_east',
                      lon_axis='X',
                      lat_name='lat',
                      lat_longname='latitude',
                      lat_standardname='latitude',
                      lat_units='degrees_north',
                      lat_axis='Y',
                      netcdf_format='NETCDF4',
                      data_description='Data saved from Python'):
    """
    Write a 1D. 2D or 3D data into a netCDF file

    Data should be formatted so that it is either ntime for 1D data, nlat x xlon
    for 2D data, or ntime x nlat x nlon for 3D data. Other formats will not be
    saved correctly or at all.

    """
    
    # Remove old file
    if os.path.isfile(filename):
        os.remove(filename)    
    
    root_grp = Dataset(filename, 'w', format=netcdf_format)
    

    # Find the length of the time axis if any
    if timein is None:
        timedim = None
    else:
        timedim = len(timein)
        root_grp.createDimension(time_name, timedim)
        times = root_grp.createVariable(time_name, 'f8', (time_name,))

    nvars = len(datain)

    root_grp.description = data_description

    # dimensions
    
    
    if lat is not None:    
        root_grp.createDimension(lat_name, len(lat))
        latitudes = root_grp.createVariable(lat_name, 'f4', (lat_name,))
    
    if lon is not None:
        root_grp.createDimension(lon_name, len(lon))
        longitudes = root_grp.createVariable(lon_name, 'f4', (lon_name,))
   
    # variables
    vars = [] #Initialize empty list to hold netCDF variables
    
    print (nvars,np.shape(nvars))
    for i in range(nvars):
        print (i,varnames[i])
        if lat is None:
            vars.append(root_grp.createVariable(varnames[i], 'f4',time_name))
        else:
            vars.append(root_grp.createVariable(varnames[i], 'f4',
                                       (time_name, lat_name, lon_name,)))
        
    # Assign values to netCDF variables
    if lat is not None: latitudes[:] = lat
    if lon is not None: longitudes[:] = lon
    print (type(timein[0]),timein)
    if (isinstance(timein[0], datetime.datetime)):
      test=nc.date2num(timein,'days since 2010-01-01 00:00:00',calendar='standard')
      print (test)
      newtime=[]
      for i in test:
        newtime.append(i)
      if timein is not None: times[:] = newtime #timein
    else:
      if timein is not None: times[:] = timein
      
        
    
    for i in range(nvars):
        print (datain[i])
        if np.ndim(datain[i])==1:                        
            time_in_len=len(datain[i])            
            vars[i][:]=np.NaN
            vars[i][0:time_in_len] = datain[i]
            print (datain[i])
        elif np.ndim(datain[i])==3:
            time_in_len=datain[i].shape[0]
            vars[i][:,:,:]=np.NaN
            vars[i][0:time_in_len,:,:] = datain[i]
        elif np.ndim(datain[i])==4:
            time_in_len=datain[i].shape[0]
            vars[i][:,:,:,:]=np.NaN
            vars[i][0:time_in_len,1,:,:] = datain[i]
        else:
            print('Unsupported dimension for the variable. Exiting.')
            sys.exit()

    # Define attributes
    if timein is not None:
        times.units = time_units
        times.long_name = time_longname
        times.standard_name = time_standardname
        times.calender = time_calender

    if lon is not None:    
        longitudes.long_name = lon_longname
        longitudes.standard_name = lon_standardname
        longitudes.units = lon_units

    if lat is not None:    
        latitudes.long_name = lat_longname
        latitudes.standard_name = lat_standardname
        latitudes.units = lat_units


    for i in range(nvars):
        if type(var_longnames) is list:
          vars[i].long_name = var_longnames[i]
        else:
          vars[i].long_name = var_longnames

        if type(var_units) is list:                  
            vars[i].units = var_units[i]
        else:
            vars[i].units = var_units

    

    root_grp.history = 'Created ' + time.ctime(time.time())
    root_grp.source = ''
    root_grp.contact='tommi.bergman@fmi.fi'

    root_grp.close()
