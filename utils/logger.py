import sys, os

class Logger(object):
    def __init__(self, logpath, logterminal):

        self.terminal = sys.stdout
        self.log = open(f'{logpath}', 'a')
        self.logterminal = logterminal
    
    def write(self, message):
        if self.logterminal: self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass 