import os
import datetime
import time

class Scenario(object):

    def __init__(self, args):

        self.args = args
        now = datetime.datetime.now()
        self.time  = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
        
        # if not self.args.prepruned_model:
        self.curr_scen_name = \
        f'{self.time}_'\
        f'data_{self.args.dataname}_arch_{self.args.arch}_init_{self.args.init_policy}_'\
        f'GPUs_{self.args.world_size}_minib_{self.args.batch_size}_'\
        f'opt_{self.args.optimizer}_lr_{self.args.lr}_lrpolicy_{self.args.lr_policy}_'\
        f'ep_{self.args.epochs}'
        
        # else:
        #     self.curr_scen_name = \
        #     f'{self.time}_'\
        #     f'data_{self.args.dataname}_arch_{self.args.arch}_init_{self.args.init_policy}_'\
        #     f'GPUs_{self.args.world_size}_minib_{self.args.batch_size}_'\
        #     f'opt_{self.args.optimizer}_lr_{self.args.lr}_lrpolicy_{self.args.lr_policy}_'\
        #     f'ep_{self.args.epochs}_strategy_{self.args.pruning_strategy}_'\
        #     f'initstg_{self.args.initial_stage}_stg_{self.args.stages}'

    
