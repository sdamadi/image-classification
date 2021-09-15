import os
import datetime
import time

class Scenario(object):

    def __init__(self, args):

        self.args = args
        now = datetime.datetime.now()
        self.time  = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
                self.curr_scen_name = \
        f'{self.time}_'\
        f'data_{self.args.dataname}_arch_{self.args.arch}_init_{self.args.init_policy}_'\
        f'GPUs_{self.args.world_size}_minib_{self.args.batch_size}_'\
        f'opt_{self.args.optimizer}_lr_{self.args.lr}_lrpolicy_{self.args.lr_policy}_'\
        f'ep_{self.args.epochs}'
    
