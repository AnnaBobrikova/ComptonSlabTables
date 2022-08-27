###############################
#This program can be used to calculate and save polarized pulse profiles from accreting millisecond pulsars
###############################

from polpulse import compf, reading_table

mass = 1.4 # 1.4
rad = 12.0 # 12.0
incl = 60.0 #60.0 #40.0 # 40.0
theta = 20.0 #20.0 #60.0 #-120.0#60.0 # 60.0
rho = 1.0 #1.0 # 10.0
pol = 0.0 #0.1171 #0.0 #0.1171 #0.0 #0.1171 #0.0 #0.1171 # 0.1171 #USE NOW ONLY 0.0 or 0.1171

reading_table()

loop = True
#if len(sys.argv) > 1:
if(loop):
	#print("Choosing param values for i,theta,rho from CL arguments:")
	#incl = float(sys.argv[1])
	#theta = float(sys.argv[2])
	#rho = float(sys.argv[3])
	#antpd = bool(sys.argv[4])
	#spath = str(sys.argv[5])

	print("Running in loop mode! Make sure not overwriting old results.")        
	incls = [60.0]#[50.0,60.0,70.0]
	thetas = [20.0]#, 120.0]#120.0]#[80.0, 100.0, 120.0,140.0,160.0]#[20.0]#[10.0,20.0,30.0]
	rhos = [1.0]#[30.0]#[1.0]
	antipds = [True]#[True,False]
	for ir in range(len(rhos)):
		for i in range(len(incls)):
			for it in range(len(thetas)):
				for ai in range(len(antipds)):
					rho = rhos[ir]
					incl = incls[i]
					theta = thetas[it]
					antpd = antipds[ai]
					spath = "pulses/pulse_compt"+str(int(antpd)+1)+"_r"+str(int(rho))+"t"+str(int(theta))+"i"+str(int(incl))
					#spath = "pulses/test/pulse_test_thom"+str(int(antpd)+1)+"_r"+str(int(rho))+"t"+str(int(theta))+"i"+str(int(incl))+"p"+str(int(10000*pol))
					#if(incl!=60.0 and theta!=20.0):
					#	continue
					Flux = compf(mass,rad,incl,theta,rho,pol,spherical=False,antipodal=antpd,spath=spath,savePulse=True)
else:
	Flux = compf(mass,rad,incl,theta,rho,pol,spherical=False,antipodal=False,spath="pulses/test/pulse_testX",savePulse=True) 

#import subprocess
#subprocess.call("./pulse_rename.sh",shell=True)