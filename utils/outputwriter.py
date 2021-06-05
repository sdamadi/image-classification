import numpy as np
import os
import torch

class Outputwriter(object):

    def __init__(self, model, poda, traval, writer, args, curr_scen_name):
        self.model = model
        self.poda = poda
        self.traval = traval
        self.writer = writer
        self.args = args
        self.curr_scen_name = curr_scen_name
        self.scen_name, self.scen_time = self.time_remover(self.curr_scen_name)

        if self.args.percent == 0 and not self.args.prepruned_model:
            self.mode = 'trainandtest'
        elif self.args.prepruned_model:
            self.mode = 'trainedsparse'
        elif self.args.percent != 0 and not self.args.prepruned_model:
            self.mode = 'pruned'
        
        self.golden_stage = 0
        self.best_loss_val, self.best_top1, self.best_top5  = -np.log(np.exp(1/1000)), 0, 0
        self.best_pruned = self.poda.pruned
        
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
                      

    def threshold(self, stage): 
        if self.args.local_rank == 0:
            if self.args.pruning_strategy in {'asni', 'lottery'}: 
                self.writer.add_scalar('Pruning/Global Threshold', self.poda.global_threshold, stage)
                self.Global_Threshold.append(self.poda.global_threshold)
                self.writer.add_scalar('Pruning/Percentage', self.poda.pruned, stage)
                self.Prune_Percent.append(self.poda.pruned)
    
    def net_params(self, stage):

        last = self.args.save_stages if stage != (self.args.initial_stage+self.args.stages-1) else True
        if self.args.local_rank == 0 and (last or self.args.save_stages ):
            
            root = 'history/variables'
  
            if not self.args.prepruned_model: 
                torch.save(self.model.state_dict(), f'./history/saved/{self.args.dataname}/{self.args.arch}/'\
                        f'saved_at_each_stage_of_pruning/{self.scen_time}/'\
                        f'pruned_at_stg_{stage}_eta_{100-self.poda.pruned:.1f}'
                        f'_{self.scen_name}.pth.tar')
            elif self.args.prepruned_model:
                torch.save(self.model.state_dict(), f'./history/saved/{self.args.dataname}/{self.args.arch}/'\
                        f'saved_retrained_prepruned_net_at_each_stage/{self.scen_time}/'\
                        f'trainedsparse_at_stg_{stage}_eta_{100-self.poda.pruned:.1f}'
                        f'_{self.scen_name}.pth.tar')
        

    def close(self):
        root = 'history/variables'

        if self.args.percent !=0:  

            path1 = f'./{root}/{self.args.dataname}/{self.args.arch}/{self.mode}/main'
            self.folder_builder(path = path1, folder_name = self.scen_time)
            path = f'{path1}/{self.scen_time}/'

            file_name = f'Prune_Percent_{self.scen_name}.npy'
            np.save(path + file_name, self.Prune_Percent)  

            file_name = f'Global_Threshold_{self.scen_name}.npy'
            np.save(path + file_name, self.Global_Threshold)
    
    def time_remover(self, curr_scen_name):
        scen_name = "_".join(curr_scen_name.split('_')[6:])
        scen_time = "_".join(curr_scen_name.split('_')[:6])

        return scen_name, scen_time

    def folder_builder(self, path, folder_name):
        if not os.path.exists(f'{path}/{folder_name}'):
            os.makedirs(f'{path}/{folder_name}')


