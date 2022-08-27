from interpolation import interp, reading_table
import numpy as np

NEnergy = 150
NZenith = 9

Task_Te = np.array([97.8])#, 122.0, 190.0]) #value Te (keV) = Task_Te*511keV/1000, 97.8 ~ 50 keV
Task_Tbb = np.array([0.002])#, 0.00158, 0.0028]) #value Tbb (kev) = Task_Tbb*511 keV, 0.002 ~ 1keV
Task_tau = np.array([1.0])#, 1.95, 3.25]) 

reading_table()

for i in Task_Te:
    for ii in Task_tau:
        for iii in Task_Tbb:
            interp(i, iii, ii)
            
#this code is generally prepared to go over a set of points, as we're gonna need to run the pulse profiling many times in our task. 
#surely all the cycles can be removed, and there is generally no need to have a separate driver code :)
#and it doesn't save anything anywhere, basically it just helps me check that everything is working smoothly