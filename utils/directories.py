import os

class Directories(object):

    def __init__(self, args, curr_scen_name):

        self.args = args
        
        self.curr_scen_name = curr_scen_name

        self.general_roots = [self.args.dataname, self.args.arch]

        self.datanames = ['mnist', 'cifar10', 'fashionmnist', 'cifar100', 'imagenet']
        self.archs = ['fc', 'conv2', 'conv4', 'conv6', 'conv2a', 'conv4a', 'conv6a',
                        'vgg11', 'vgg13', 'vgg16', 'vgg19',
                        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                        'mobilenet_v2']

        self.level_1 = ['pruned', 'trainandtest', 'trainedsparse']
        self.level_2 = ['main'] #, 'parameters'
        self.level_3 = ['initialization', 'final']
        self.level_4 = ['layers', 'network']

        # extract the time and name of the scenario from `curr_scen_name`
        self.scen_name, self.scen_time = self.time_remover(self.curr_scen_name)
        # creates folders wherein logs are saved
        self.logfolders()
        # create the path of log file
        self.log_path()
        # creates folder to save pruned models
        self.saved_pruned_model_folders()
        # creates folder to save numpy variables 
        self.numpyfolders()
 
    #creates the name of the text files that logs terminal output
    def log_path(self):
        root = 'history/logs'
        if self.args.prepruned_model:
            self.logpath = f'./{root}/{self.args.dataname}/{self.args.arch}/'\
                            f'trainedsparse/{self.curr_scen_name}'.replace('pth.tar','txt')
        elif not self.args.prepruned_model:
            if self.args.percent == 0 and self.args.pruning_strategy in {'lottery'}:
                self.logpath = f'./{root}/{self.args.dataname}/{self.args.arch}/'\
                               f'trainandtest/{self.curr_scen_name}.txt'
            else:
                self.logpath = f'./{root}/{self.args.dataname}/{self.args.arch}/'\
                            f'pruned/{self.curr_scen_name}.txt'

    #creates folders under which each scenario's output is logged
    def logfolders(self):
        root = 'history/logs'
        for d in self.datanames:
            if self.args.dataname == d:
                if not os.path.exists(f'./{root}/{d}'):
                    os.makedirs(f'./{root}/{d}') 
                for ar in self.archs:
                    if self.args.arch == ar:
                        if not os.path.exists(f'./{root}/{d}/{ar}'): 
                            os.makedirs(f'./{root}/{d}/{ar}') 
                        for j in self.level_1:
                            if not os.path.exists(f'./{root}/{d}/{ar}/{j}'):
                                os.makedirs(f'./{root}/{d}/{ar}/{j}')
    
    #creates folders that saves pruned models at different stage of pruning
    def saved_pruned_model_folders(self):
        root = 'history/saved'
        for d in self.datanames:
            if self.args.dataname == d:
                if not os.path.exists(f'./{root}/{d}'):
                    os.makedirs(f'./{root}/{d}') 
                for ar in self.archs:
                    if self.args.arch == ar:
                        if not os.path.exists(f'./{root}/{d}/{ar}'):
                            os.makedirs(f'./{root}/{d}/{ar}')
                        path = f'./{root}/{d}/{ar}'
                        
                        file_name = 'saved_initialization'
                        self.folder_builder(path, file_name)
                        # for i in range(100): print(path, file_name)
                        self.folder_builder(f'{path}/{file_name}', self.scen_time)
                    
                        if not self.args.prepruned_model: #self.args.percent != 0 and 
                            if not os.path.exists(f'./{root}/{d}/{ar}/saved_at_each_stage_of_pruning'): 
                                os.makedirs(f'./{root}/{d}/{ar}/saved_at_each_stage_of_pruning')
                            path = f'./{root}/{d}/{ar}/saved_at_each_stage_of_pruning'
                            self.folder_builder(path, self.scen_time)

                        elif self.args.prepruned_model:
                            #training a sparse model
                            file_name = 'saved_retrained_prepruned_net_at_each_stage'
                            self.folder_builder(path, file_name)
                            self.folder_builder(f'{path}/{file_name}', self.scen_time) 
    
    #creates folders saving numpy variables including loss, parameters, etc. 
    def numpyfolders(self):
        root = 'history/variables'
        for d in self.datanames:
            if self.args.dataname == d:
                if not os.path.exists(f'./{root}/{d}'): 
                    os.makedirs(f'./{root}/{d}') 
                for ar in self.archs:
                    if self.args.arch == ar:
                        if not os.path.exists(f'./{root}/{d}/{ar}'):
                            os.makedirs(f'./{root}/{d}/{ar}') 
                        for j in self.level_1:
                            if not os.path.exists(f'./{root}/{d}/{ar}/{j}'): 
                                os.makedirs(f'./{root}/{d}/{ar}/{j}') 
                            for k in self.level_2:
                                if not os.path.exists(f'./{root}/{d}/{ar}/{j}/{k}'):
                                    os.makedirs(f'./{root}/{d}/{ar}/{j}/{k}')
                               
    
    def time_remover(self, curr_scen_name):
        scen_name = "_".join(curr_scen_name.split('_')[6:])
        scen_time = "_".join(curr_scen_name.split('_')[:6])
        return scen_name, scen_time

    def folder_builder(self, path, folder_name):
        if not os.path.exists(f'{path}/{folder_name}'):
            os.makedirs(f'{path}/{folder_name}')




       