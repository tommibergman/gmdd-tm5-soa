# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:47:43 2016

@author: bergmant

"""

import glob
import subprocess
import os
def collocate_MODIS(TM5_input='TM5-AP3-data/full/TM5_daily_subset_new/',TM5_data='aerocom3_TM5_AP3-ctrl2016_global_2010_hourly.od550aer.',EXP='TM5/'):
	#input_TM5_data = 'od550aer:TM5_daily_subset_new/aerocom3_TM5_AP3-ctrl2016_global_2010_hourly.od550aer.'
	Basepath='/Users/bergmant/Documents/Project/ifs+tm5-validation/'
	TM5_datapath=Basepath+TM5_input
	#TM5_datapath='/Users/bergmant/Documents/Project/ifs+tm5-validation/'
	output_datapath='/Users/bergmant/Documents/Project/ifs+tm5-validation/'

	if os.path.exists(TM5_datapath):#+TM5_input):
		input_TM5_location=TM5_datapath#+TM5_input

	else:
		print ' TM5 input dir not found: ',TM5_datapath
		brer
	MODIS_datapath='/Volumes/clouds/'
	input_TM5_data = 'od550aer:'+input_TM5_location+TM5_data
	#input_TM5_data = 'abs440aer:'+input_TM5_location+TM5_data
	input_MODIS_terra_location = 'Masked_data_MOD04_ctrl2016/'
	input_MODIS_aqua_location = 'Masked_data_MYD04_ctrl2016/'
	#input_MODIS_location = ['Masked_data_MOD04_ctrl2016/','Masked_data_MYD04_ctrl2016/']
	output_data_location = output_datapath+'Masked_collocated_TM5_ctrl2016_MODIS_test/'+EXP+'/'
	if not os.path.exists(output_data_location):
		os.makedirs(output_data_location)
	variable = 'variable=AOD_550_Dark_Target_Deep_Blue_Combined'

	for day in range(31,366):
		for hours in range(24):
			for minutes in range(0,60,5):
				print "Now procesing: day "+str(day).zfill(3)+" time "+str(hours).zfill(2)+str(minutes).zfill(2) #prints dayDDDtimeHHMM
				print "Terra platform"				
				MODIS = MODIS_datapath+input_MODIS_terra_location+"Masked_MOD04_L2_2010_"+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc' #filelocation and filename
				TM5=input_TM5_data+'%d.nc '%day
				#TM5=input_TM5_data+'nc '
				out_data_terra=output_data_location+'lin_col_Masked_TM5_MOD04_L2_2010_'+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc'
				if len(glob.glob(MODIS)) == 1 and not os.path.isfile(out_data_terra): #checks if the file in MODIS exists
					print('cis col '+TM5+MODIS+':'+variable+',collocator=lin -o '+out_data_terra) #output_data_location+'lin_col_Masked_TM5_MOD04_L2_2010_'+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2))+'.nc'
					#subprocess.call('cis col '+TM5+MODIS+':'+variable+',collocator=lin -o '+output_data_location+'lin_col_Masked_TM5_MOD04_L2_2010_'+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc', shell=True)
					subprocess.call('cis col '+TM5+MODIS+':'+variable+',collocator=lin -o '+out_data_terra, shell=True)
					#above code runs CIS collocation command in terminal and writes the file in output_data_location
				else:
					print "done already or no modis data"
				print "Aqua platform"				
				MODIS = MODIS_datapath+input_MODIS_aqua_location+"Masked_MYD04_L2_2010_"+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc' #filelocation and filename
				TM5=input_TM5_data+'%d.nc '%day
				#TM5=input_TM5_data+'nc '
				out_data_aqua=output_data_location+'lin_col_Masked_TM5_MYD04_L2_2010_'+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc'
				if len(glob.glob(MODIS)) == 1 and not os.path.isfile(out_data_aqua): #checks if the file in MODIS exists
					print('cis col '+TM5+MODIS+':'+variable+',collocator=lin -o '+out_data_aqua) #output_data_location+'lin_col_Masked_TM5_MYD04_L2_2010_'+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2))+'.nc'
					#subprocess.call('cis col '+TM5+MODIS+':'+variable+',collocator=lin -o '+output_data_location+'lin_col_Masked_TM5_MYD04_L2_2010_'+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc', shell=True)
					subprocess.call('cis col '+TM5+MODIS+':'+variable+',collocator=lin -o '+out_data_aqua, shell=True)
					#above code runs CIS collocation command in terminal and writes the file in output_data_location
				else:
					print "done already or no modis data"
def Mask_QA_MODIS():
	input_MODIS_location = '/Volumes/SAMSUNG/MODIS/6/MYD04_L2/2010/'
	output_masked_location = 'Masked_data_MYD04_ctrl2016/'

	QA_Flag = 'AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag'
	AOD = 'AOD_550_Dark_Target_Deep_Blue_Combined'
	Level=1
	for day in range(314,366):
		for hours in range(24):
			for minutes in range(0,60,5):
				print "day"+str(day).zfill(3)+"time"+str(hours).zfill(2)+str(minutes).zfill(2) #prints dayDDDtimeHHMM
				
				MODIS_Filename = "MYD04_L2.A2010"+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+"*"
				MODIS = input_MODIS_location+str(day).zfill(3)+'/'+MODIS_Filename
				Masked_data_filename = 'MYD04_L2_2010_'+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc'
				
				if len(glob.glob(MODIS)) == 1: #checks if the file in MODIS exists
					subprocess.call("cis eval "+QA_Flag+"=QA,"+AOD+"=AOD:"+MODIS+" 'numpy.ma.masked_where(QA <= "+str(Level)+", AOD)' 1 -o "+AOD+":"+output_masked_location+"Masked_"+Masked_data_filename, shell=True)
					#Masks AOD variable where QA_Flag smaller or equal to 1

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:51:50 2016

@author: terpstra
"""


def subset_TM5(datagroup = 'od550aer:newdata/aerocom3_TM5_AP3-ctrl2016_global_2010_hourly.od550aer.nc',dir_out='TM5_daily_subset_new/', file_out1 = 'aerocom3_TM5_AP3-ctrl2016_global_2010_hourly.od550aer.'):
	if not os.path.exists(dir_out):
		os.makedirs(dir_out)
	file_out=dir_out+file_out1

	daycount=0
	print 'processing January'
	for day in range(1,31): #January
		print day,file_out
		raw_input()
		month=1
		daycount+=1
		if daycount<9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day==9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day<31:
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True)
		else: 
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month+1, 1, file_out, day), shell=True) 

	print '\nprocessing February'
	daycount = 0
	for day in range(32,60): #February
		print day,
		month = 2
		daycount+=1
		if daycount<9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day==9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day<59:
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True)
		else: 
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month+1, 1, file_out, day), shell=True) 

	print '\nprocessing March'
	daycount = 0
	for day in range(60,91): #March
		print day,
		month = 3
		daycount+=1
		if daycount<9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day==9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day<90:
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True)
		else: 
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month+1, 1, file_out, day), shell=True) 

	print '\nprocessing April'
	daycount = 0
	for day in range(91,121): #April
		print day,
		month = 4
		daycount+=1
		if daycount<9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day==9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day<120:
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True)
		else: 
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month+1, 1, file_out, day), shell=True) 

	print '\nprocessing May'
	daycount = 0
	for day in range(121,152): #May
		print day,
		month = 5
		daycount+=1
		if daycount<9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day==9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day<151:
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True)
		else: 
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month+1, 1, file_out, day), shell=True) 

	print '\nprocessing June'
	daycount = 0
	for day in range(152,182): #June
		print day,
		month = 6
		daycount+=1
		if daycount<9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day==9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day<181:
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True)
		else: 
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month+1, 1, file_out, day), shell=True) 

	print '\nprocessing July'
	daycount = 0
	for day in range(182,213): #July
		print day,
		month = 7
		daycount+=1
		if daycount<9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day==9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day<212:
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True)
		else: 
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month+1, 1, file_out, day), shell=True) 

	print '\nprocessing August'
	daycount = 0
	for day in range(213,244): #August
		print day,
		month = 8
		daycount+=1
		if daycount<9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day==9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day<243:
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True)
		else: 
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month+1, 1, file_out, day), shell=True) 

	print '\nprocessing September'
	daycount = 0
	for day in range(244,274): #September
		print day,
		month = 9
		daycount+=1
		if daycount<9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day==9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day<273:
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True)
		else: 
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month+1, 1, file_out, day), shell=True) 

	print '\nprocessing October'
	daycount = 0
	for day in range(274,305): #October
		print day,
		month = 10
		daycount+=1
		if daycount<9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day==9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day<304:
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True)
		else: 
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month+1, 1, file_out, day), shell=True) 

	print '\nprocessing November'
	daycount = 0
	for day in range(305,335): #November
		print day,
		month = 11
		daycount+=1
		if daycount<9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day==9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day<334:
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True)
		else: 
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month+1, 1, file_out, day), shell=True) 

	print '\nprocessing December'
	daycount = 0
	for day in range(335,365): #december
		print day,
		month = 12
		daycount+=1
		if daycount<9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day==9:
			subprocess.call('cis subset %s t=[2010-0%d-0%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True) 
		elif day<365:
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month, daycount+1, file_out, day), shell=True)
		else: 
			subprocess.call('cis subset %s t=[2010-0%d-%dT00:00:00,2010-0%d-0%dT01:00:00] -o %s%d.nc' % (datagroup, month, daycount, month+1, 1, file_out, day), shell=True) 

	subprocess.call('cis %s t=[2010-12-31T00:00:00,2010-12-31T23:59:59] -o %s365.nc', shell=True) 
	#Dataset does not include 2011-01-01T00:00:00 so last statement is to avoid complications

def bias(MODISpath='Masked_data_',MODELpath= 'Masked_Collocated_data_TM5/',OUTPUT='Bias_masked/'):
	#MODIS_location = 
	#coldata_location 
	#output_bias_location = '/run/media/terpstra/SAMSUNG/

	MODIS_AOD = 'AOD_550_Dark_Target_Deep_Blue_Combined'
	MODEL_AOD = 'od550aer'
	for day in range(1,366):
		for hours in range(24):
			for minutes in range(0,60,5):
				print "day "+str(day).zfill(3)+" time "+str(hours).zfill(2)+str(minutes).zfill(2) #prints dayDDDtimeHHMM
				
				coldataaqua = MODELpath+"lin_col_Masked_TM5_MYD04_L2_2010_"+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc' #filelocation and filename
				coldataterra = MODELpath+"lin_col_Masked_TM5_MOD04_L2_2010_"+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc' #filelocation and filename
				MODISaqua = MODISpath+"MYD04_ctrl2016/Masked_MYD04_L2_2010_"+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+".nc"
				MODISterra = MODISpath+"MOD04_ctrl2016/Masked_MOD04_L2_2010_"+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+".nc"
				Biasaqua = OUTPUT+"Bias_TM5_MYD04_L2_2010_"+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc'
				Biasterra = OUTPUT+"Bias_TM5_MOD04_L2_2010_"+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc'
				print coldataterra
				print MODISterra
				print Biasterra
				
				if len(glob.glob(MODISterra)) == 1 and os.path.isfile(coldataterra) and  not os.path.isfile(Biasterra) : #checks if the file in MODIS exists
					print('cis eval '+MODEL_AOD+'=a:'+coldataterra+' '+MODIS_AOD+'=b:'+MODISterra+" a-b 1 -o "+Biasterra)
					subprocess.call('cis eval '+MODEL_AOD+'=a:'+coldataterra+' '+MODIS_AOD+'=b:'+MODISterra+' a-b 1 -o bias:'+Biasterra, shell=True)
					#calculate bias by subtracting the masked MODIS L2 data from the collocated modeldata
				if len(glob.glob(MODISaqua)) == 1 and os.path.isfile(coldataaqua) and not os.path.isfile(Biasaqua): #checks if the file in MODIS exists
					print('cis eval '+MODEL_AOD+'=a:'+coldataaqua+' '+MODIS_AOD+'=b:'+MODISaqua+" a-b 1 -o "+Biasaqua)
					subprocess.call('cis eval '+MODEL_AOD+'=a:'+coldataaqua+' '+MODIS_AOD+'=b:'+MODISaqua+' a-b 1 -o bias:'+Biasaqua, shell=True)
					#calculate bias by subtracting the masked MODIS L2 data from the collocated modeldata

def aggregate_bias(bias_location = 'Bias_MYD04/',output_aggregated_bias_location = 'Aggregated_Bias/'):
	for day in range(1,366):
		Biasaqua = bias_location+"Bias_TM5_MYD04_L2_2010_"+str(day).zfill(3)+'*'
		Biasterra = bias_location+"Bias_TM5_MOD04_L2_2010_"+str(day).zfill(3)+'*'
		Aggregated_Bias = output_aggregated_bias_location+"Aggregated_Bias_TM5_MOD04_L2_2010_1x1_dailymean_"+str(day).zfill(3)+'.nc'
		if not os.path.isfile(Aggregated_Bias):
			print('cis aggregate bias:'+Biasterra+":kernel=mean x=[-180,180,1],y=[-90,90,1],t -o "+Aggregated_Bias)
			subprocess.call('cis aggregate bias:'+Biasterra+":kernel=mean x=[-180,180,1],y=[-90,90,1],t -o "+Aggregated_Bias, shell=True)
		Aggregated_Bias = output_aggregated_bias_location+"Aggregated_Bias_TM5_MYD04_L2_2010_1x1_dailymean_"+str(day).zfill(3)+'.nc'
		if not os.path.isfile(Aggregated_Bias):
			print('cis aggregate bias:'+Biasaqua+":kernel=mean x=[-180,180,1],y=[-90,90,1],t -o "+Aggregated_Bias)
			subprocess.call('cis aggregate bias:'+Biasaqua+":kernel=mean x=[-180,180,1],y=[-90,90,1],t -o "+Aggregated_Bias, shell=True)

if __name__ == "__main__":
	import sys
	from timeit import default_timer as timer
	if sys.version_info[0] > 2:
		print("Cistools will not work with Python 3")
		sys.exit(1)
	start=timer()
	#subset_TM5('od550aer:SOA-runs/SOA/aerocom3_TM5_SOA_global_2010.od550aer.nc','SOA-runs/SOA/TM5-subset/','aerocom3_TM5_SOA_global_2010.od550aer.')
	end=timer()
	print "time used for subsetting",end-start
	#collocate_MODIS()
	
	"""
	Here we collocate the MODIS data with TM5 data	
	"""
	start=timer()
	collocate_MODIS('SOA-runs/SOA/TM5-subset/','aerocom3_TM5_SOA_global_2010.od550aer.','SOA2')
	end=timer()
	print "time used for collocation",end-start
	raw_input()
	""" Calculate the bias between TM5 and MODIS"""
	start=timer()
	bias('/Volumes/clouds/Masked_data_','Masked_collocated_TM5_ctrl2016_MODIS_test/SOA2/','BIAS/')
	""" Aggregate and average the bias data"""
	end=timer()
	print "time used for bias calculation (eval a-b)",end-start
	
	start=timer()
	aggregate_bias('BIAS/','Aggregated_Bias')
	end=timer()
	print "time used for aggregation (aggregate)",end-start
	
