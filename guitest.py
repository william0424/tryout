# Import required modules
import pyautogui
import time
from time import gmtime, strftime

import math
import numpy as np

# FAILSAFE to FALSE feature is enabled by default
# so that you can easily stop execution of
# your pyautogui program by manually moving the
# mouse to the upper left corner of the screen.
# Once the mouse is in this location,
# pyautogui will throw an exception and exit.
pyautogui.FAILSAFE = False

# We want to run this code for infinite
# time till we stop it so we use infinite loop now
COUNTER = 0
while True:
    
    if not COUNTER%100:
        print("Local: ",strftime("%a, %d %b %Y %I:%M:%S %p %Z"))
    
    x,y = pyautogui.position()
    time.sleep(5)
    x1,y1 = pyautogui.position()
    
    t = x**2 + y**2
    t1 = x1**2 + y1**2
    
    if t== t1:
    
    

        # This for loop is used to move the mouse
        # pointer to 500 pixels in this case(5*100)
        x,y = pyautogui.position()
        angel = 0
        N = 100
        C=10
        L=1
        
        for i in range(0, N):
            
            x,y = pyautogui.position()
            angel += 2*C*math.pi/N
            
            x-= int(np.cos(angel)*L)
            y-= int(np.sin(angel)*L)
            
            pyautogui.moveTo(x, y)
        # This for loop is used to press keyboard keys,
        # in this case the harmless key shift key is
        # used. You can change it according to your
        # requirement. This works with all keys.
        for i in range(0, 3):
            pyautogui.press('shift')

        # time.sleep(t) is used to give a break of
        # specified time t seconds so that its not
        # too frequent

    
    time.sleep(40)
    COUNTER += 1
