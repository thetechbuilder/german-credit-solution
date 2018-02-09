import time
from sys import stdout

class Timer(object):
    def __enter__(self):
        self.__start = time.time()
        self.__busy = True
    
    def __exit__(self, type, value, traceback):
        # Error handling here
        self.__finish = time.time()
        self.__busy = False
        
    def duration_in_seconds(self):
        return (time.time() if self.__busy else self.__finish) - self.__start

class Progress(Timer):
    def update(self, progress):
        stdout.write("\r[{0:=3}%]".format(int(progress)))
        stdout.flush()

class Enum(object):
    def __init__(self, **enums):
        for x in enums:
            setattr(self, x, enums[x])
