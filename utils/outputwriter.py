import numpy as np
import os
import torch

class Outputwriter(object):

    def __init__(self, model, traval, writer, args, curr_scen_name):
        self.model = model
        self.traval = traval
        self.writer = writer
        self.args = args
        self.curr_scen_name = curr_scen_name
        self.scen_name, self.scen_time = self.time_remover(self.curr_scen_name)
        
        self.best_loss_val, self.best_top1, self.best_top5  = -np.log(np.exp(1/1000)), 0, 0
        
        self.GradNorm, self.GradDot, self.delta2 = [] , [], []
        self.Best_Stage, self.Best_Val_Loss, self.Best_Val_Acc, self.Best_Percent = ([] for i in range(4) )
        self.Prune_Percent, self.Global_Threshold = [] , []

   
    def net_params_init(self):
        root = 'history/variables'
        torch.save(self.model.state_dict(), f'./history/saved/{self.args.dataname}/{self.args.arch}/'\
                        f'saved_initialization/{self.scen_time}/'\
                        f'initialization_{self.scen_name}.pth.tar')
        # saving each layer as a variable
        for name, weight in self.model.named_parameters():

            self.writer.add_histogram('Initializations/Layer:'+ name, weight, 1)
            new_name = name.replace('.','_').replace('module_', '')
                      
   
    def net_params(self):

        root = 'history/variables'

        torch.save(self.model.state_dict(), f'./history/saved/{self.args.dataname}/{self.args.arch}/'\
                f'saved_end_of_training/{self.scen_time}/'\
                f'_{self.scen_name}.pth.tar')
        

    def close(self):
        root = 'history/variables'

        path1 = f'./{root}/{self.args.dataname}/{self.args.arch}/main'
        self.folder_builder(path = path1, folder_name = self.scen_time)
        path = f'{path1}/{self.scen_time}/'

  
    
    def time_remover(self, curr_scen_name):
        scen_name = "_".join(curr_scen_name.split('_')[6:])
        scen_time = "_".join(curr_scen_name.split('_')[:6])

        return scen_name, scen_time

    def folder_builder(self, path, folder_name):
        if not os.path.exists(f'{path}/{folder_name}'):
            os.makedirs(f'{path}/{folder_name}')


