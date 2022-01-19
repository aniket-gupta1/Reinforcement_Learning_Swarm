import os
import sys
import dronekit
import getpass
from time import sleep

N = int(sys.argv[1])
launch_string_mopso="gnome-terminal"
launch_string="gnome-terminal"
for i in range(N):
	os.chdir("/home/"+getpass.getuser()+"/IAF/Integrated_CodeBase/")
	launch_string += " --tab -- python3 modified_server7.py " + str(i)
	sleep(0.1)

os.system(launch_string)

sleep(3)
print("sleep_over")
for i in range(N):
	os.chdir("/home/"+getpass.getuser()+"/IAF/Integrated_CodeBase/")
	launch_string_mopso += " --tab -- python3 Iteration6_Integrated_Swarm.py " + str(i)
	sleep(0.1)
print("doing_this")
os.system(launch_string_mopso)
print("done")