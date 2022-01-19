import os
import sys
import threading
from time import sleep

N = int(sys.argv[1])
k=0
string = "mavproxy.py"

def Launch_SITL_instances(i):
	os.chdir("/home/aniket/ardupilot/ArduCopter/")
	os.system("sim_vehicle.py -I"+str(i)+" --out=127.0.0.1:"+ str(14552 + i*10) +" --sysid "+str(i+1)+" -L L"+str(i+1))


for i in range(N):
	string += " --master=127.0.0.1:"+str(14551 + i*10)
	Launch_SITL_instances_thread = threading.Thread(target = Launch_SITL_instances, args = (i,))
	Launch_SITL_instances_thread.start()
	print("Done ")
	sleep(1)

os.system("gnome-terminal -e '"+string+" --map'")
