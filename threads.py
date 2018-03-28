import thread
import time

def f1():
    count = 0
    while count < 3:
        time.sleep(1)
        print "hello1"
        count += 1

def f2():
    count = 0
    while count < 3:
        time.sleep(1)
        print "hello2"
        count += 1

try:
    thread.start_new_thread( f1, () )
    thread.start_new_thread( f2, () )
except:
    print "Error: unable to start thread"

while 1:
    pass
