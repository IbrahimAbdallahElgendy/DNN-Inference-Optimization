#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import threading
import time
import os
import string
import sys
import multiprocessing
import signal
import psutil
class ControlThread(threading.Thread):
 
 
  def __init__(self, sleep_interval):
    threading.Thread.__init__(self)
    self.runflag = True 
    self.sleep_interval = sleep_interval
 
  def run(self):
    while self.runflag:
      # os.popen('usleep ' + sys.argv[5])
      time.sleep(self.sleep_interval)
      #time.sleep(float(sys.argv[5]))
 
  def stop(self):
    self.runflag = False

threadList=[]
 
print('Start Thread Number:' + sys.argv[2] + '\tSleep Time(ms):'+ sys.argv[5])
 
def quit(signum, frame):
  print('You choose to stop me.')
  sys.exit(0)

signal.signal(signal.SIGINT, quit)
signal.signal(signal.SIGTERM, quit)

core_count = multiprocessing.cpu_count()

for i in range(0,int(sys.argv[3])):
  thread = ControlThread(float(sys.argv[5]) / 1000.0)
  threadList.append(thread)
  thread.start()
 
while True:
  # output = 100 - float(os.popen('sar 1 1 | grep ^Average | awk \'{print $8}\'').read())
  # output *= core_count
  cpu_usage = psutil.cpu_percent(1)
  print('CPU Usage:' + str(cpu_usage) + '\tCurrent Thread Number:' + str(len(threadList)))
 
  if cpu_usage < int(sys.argv[1]):
   for i in range(0,int(sys.argv[4])):
    thread = ControlThread(float(sys.argv[5]) / 1000.0)
    thread.start()
    threadList.append(thread)
   print("+++++")
  if cpu_usage > int(sys.argv[2]):
   for i in range(0,int(sys.argv[4])):  
    thread = threadList.pop()
    thread.stop()
   print("-----")

