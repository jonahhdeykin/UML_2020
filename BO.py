import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
from IPython.utils import io

import pyro
import pyro.contrib.gp as gp
import pandas as pd



import numpy as np
import torch
import shutil
import os
from evaluate_clustering import make_cluster_dict, evaluate_clustering
import subprocess
import math

class Transform_log_scale():
    def __init__(self, to_lb, to_ub):
        assert((to_lb>0) and (to_ub>0))
        assert(to_lb < to_ub)
        self.to_lb = to_lb
        self.to_ub = to_ub
        self.to_lb_log = math.log(self.to_lb)
        self.to_ub_log = math.log(self.to_ub)
        self.b0 = self.to_lb_log
        self.b1 = self.to_ub_log - self.to_lb_log
        
    def forward(self, value):
        assert(np.logical_and(value>=0, value<=1))
        return math.exp(self.b0+self.b1*value)
        
    def backward(self, value):
        assert(np.logical_and(value>=self.to_lb, value<=self.to_ub))
        return((math.log(value)-self.b0)/self.b1)


class BO():
    
    def __init__(self):
        pass
        
        
    def transform(self, num_range, num):
        return num_range[0] + num*(num_range[1]-num_range[0])
    
    
    def make_settings(self, hyp_names, hyp_ranges, hyp_vals, defaults_dict, temp_path, log=False):
        try:
            shutil.rmtree(temp_path)
        except:
            pass

        os.mkdir(temp_path)

        for name in range(0, len(hyp_names)):
            if log:
                T = Transform_log_scale(hyp_ranges[name][0], hyp_ranges[name][1])

                defaults_dict[hyp_names[name]] = T.forward(hyp_vals[name])

            else:
                defaults_dict[hyp_names[name]] = self.transform(hyp_ranges[name], hyp_vals[name])
        
        with open('{}/settings.txt'.format(temp_path), 'w') as f:
            
            f.write('DEPTH {}\n'.format(int(defaults_dict['DEPTH'])))
            
            temp = 'ETA'
            for _ in range(0, int(defaults_dict['DEPTH'])):
                temp = '{} {}'.format(temp, defaults_dict['ETA'])

            temp = '{}\n'.format(temp)
            f.write(temp)

            temp = 'GAM'
            for _ in range(0, int(defaults_dict['DEPTH']) - 1):
                temp = '{} {}'.format(temp, defaults_dict['GAM'])

            temp = '{}\n'.format(temp)
            f.write(temp)
                        
            f.write('GEM_MEAN {}\n'.format(defaults_dict['GEM_MEAN']))

            f.write('GEM_SCALE {}\n'.format(int(defaults_dict['GEM_SCALE'])))

            f.write('SCALING_SHAPE {}\n'.format(defaults_dict['SCALING_SHAPE']))

            f.write('SCALING_SCALE {}\n'.format(defaults_dict['SCALING_SCALE']))

            f.write('SAMPLE_ETA {}\n'.format(int(defaults_dict['SAMPLE_ETA'])))

            f.write('SAMPLE_GEM {}'.format(int(defaults_dict['SAMPLE_GEM'])))

            
    def run_hlda(self, hype_names, hype_ranges, hype_vals, defaults_dict, temp_path, dat_path, order_path, log=False):

        self.make_settings(hype_names, hype_ranges, hype_vals, defaults_dict, temp_path, log = log)
        result = subprocess.run(['hlda-master/main', 'gibbs', dat_path, '{}/settings.txt'.format(temp_path), temp_path], stdout=subprocess.PIPE)
        #process = subprocess.Popen('hlda-master/main gibbs {} {}/settings.txt {}'.format(dat_path, temp_path, temp_path),  stdout=subprocess.PIPE)
        #process.wait()

        scores, a, max_depth = make_cluster_dict('{}/run000/mode.assign'.format(temp_path), order_path)

        score = evaluate_clustering(scores, max_depth)
        
        return torch.tensor((1 - (score[0]/score[1])))
        
    def update_posterior(self, new_x, gpmodel, hyp_names, hyp_ranges, defaults_dict, temp_path, dat_path, order_path, log = False):
        
                   
            
            
            
            
        y = self.run_hlda(hyp_names, hyp_ranges, new_x, defaults_dict, temp_path, dat_path, order_path, log=log)
        # evaluate f at new point.
        # For the wired "is double, not float" error
        y = y.float()
        X = torch.cat([gpmodel.X, new_x.view(-1, new_x.shape[0])]) # incorporate new evaluation
        y = torch.cat([gpmodel.y, y.view(1)])
        gpmodel.set_data(X, y)
        # optimize the GP hyperparameters using Adam with lr=0.001
        optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
        gp.util.train(gpmodel, optimizer)   
        
        return y[-1]
        
    def lower_confidence_bound(self, x, gpmodel, kappa=2):
        assert(len(x.shape) == 1)
        # Object of x: [] -> [[]]
        x = x.view(-1, x.shape[0])
        mu, variance = gpmodel(x, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        return mu - (kappa * sigma)
    
    def find_a_candidate(self, gpmodel, x_init, lower_bound=0, upper_bound=1):
        assert(len(x_init.shape) == 1)
        # transform x to an unconstrained domain
        constraint = constraints.interval(lower_bound, upper_bound)
        # ????? What is this step ?????
        unconstrained_x_init = transform_to(constraint).inv(x_init)
        # Object of unconstrained_x_init: [] -> [[]]
        unconstrained_x_init = unconstrained_x_init.view(-1,x_init.shape[0])
        unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
        minimizer = optim.LBFGS([unconstrained_x])
    
        def closure():
            minimizer.zero_grad()
            x = transform_to(constraint)(unconstrained_x)
            # Object of x: [[]] -> []
            x = x[0]
            y = self.lower_confidence_bound(x, gpmodel)
            autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
            return y
    
        minimizer.step(closure)
        # after finding a candidate in the unconstrained domain,
        # convert it back to original domain.
        # Object of unconstrained_x: [[]] -> []
        unconstrained_x = unconstrained_x[0]
        x = transform_to(constraint)(unconstrained_x)
    
        return x.detach()
    
    def next_x(self, gpmodel, lower_bound=0, upper_bound=1, num_candidates=5, explicit_init=None):
        candidates = []
        values = []
        if explicit_init is None:
            x_init = gpmodel.X[-1:]
            x_init = x_init[0]
        else:
            x_init = explicit_init
        # Object of x_init: [[]] -> []
        
        for i in range(num_candidates):
            x = self.find_a_candidate(gpmodel, x_init, lower_bound, upper_bound)
            y = self.lower_confidence_bound(x, gpmodel)
            candidates.append(x)
            values.append(y)
            # Object x_init: []
            x_init = x.new_empty(x.shape[0]).uniform_(lower_bound, upper_bound)
    
        argmin = torch.min(torch.cat(values), dim=0)[1].item()
        return candidates[argmin]
    
    def run_BO(self, hyp_names, hyp_ranges, defaults_dict, num_random, num_BO, temp_path, dat_path, order_path, out_path, seed=None, log=False):
        
        if seed is not None:
            np.random.seed(seed)
            pyro.set_rng_seed(seed)
       
        X = torch.tensor(np.random.rand(num_random, len(hyp_names)))  
        y = []
    
        for i in range(num_random):
            
            
                        
           new_y = torch.tensor(self.run_hlda(hyp_names, hyp_ranges, X[i], defaults_dict, temp_path, dat_path, order_path, log=log))
           y.append(new_y)
           print('x_min: {}'.format(X[i]))
           print('new_y: {}'.format(new_y))
           print('RANDOM ROUND {}: seed {} '.format(i, seed))
        
    
        y = torch.tensor(y)
        
        # !!! Need to convert X,y from tensor double to tensor float !!!
        # otherwise get error "expected device cpu and dtype Double but got device cpu and dtype Float"
        # https://discuss.pytorch.org/t/gpytorch-runtimeerror-expected-backend-cpu-and-dtype-double-but-got-backend-cpu-and-dtype-float/44309
        X = X.float()
        y = y.float()
        
        
    
        gpmodel = gp.models.GPRegression(X, 
                                             y, 
                                             gp.kernels.Matern52(
                                                 input_dim=len(hyp_names),
                                                 lengthscale = torch.tensor([0.01])
                                             ),
                                             noise=torch.tensor(.005), 
                                             jitter=1.0e-4)
        optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.01)
        # Documentation: http://docs.pyro.ai/en/0.3.0-release/contrib.gp.html#module-pyro.contrib.gp.util
        with io.capture_output() as captured:
            print(captured)
            
        gp.util.train(gpmodel, optimizer)
            
        pyro.enable_validation(True)  # can help with debugging
        
        
        for i in range(num_BO):
            # def next_x(lower_bound=0, upper_bound=1, num_candidates=5):
            
            xmin = self.next_x(gpmodel, 0, 1, 5)
    
                
            new_y = self.update_posterior(xmin, gpmodel, hyp_names, hyp_ranges, defaults_dict, temp_path, dat_path, order_path, log = log)
            
           
            print('EVALUATION ROUND {}'.format(i))
        ##### Iteration Statistics #####
        X_final = gpmodel.X.numpy()
    
        for i in range(num_random + num_BO):
            for arg in range(len(hyp_names)):
                if not log:
                    X_final[i][arg] =  self.transform(hyp_ranges[arg], float(X_final[i][arg]))
                else:
                    T = Transform_log_scale(hyp_ranges[arg][0], hyp_ranges[arg][1])
                    X_final[i][arg] = T.forward(float(X_final[i][arg]))
            
        
        dim = len(hyp_names)
        col_X_names = list(map(lambda x: 'axis' + str(x), list(np.arange(dim) + 1)))
        
        arr_iter = np.concatenate(
            (
                np.repeat(0, X.shape[0]), 
                np.arange(gpmodel.X.numpy().shape[0] - X.shape[0]) + 1
            )
        )
        df_iter = pd.DataFrame(arr_iter, columns=['n_iter'])
        df_X = pd.DataFrame(X_final, columns=hyp_names)
        df_y = pd.DataFrame(gpmodel.y.numpy(), columns=['exploitability']) 
        df_bo = pd.concat([df_iter, df_X, df_y], axis=1)
        df_bo.to_csv(out_path, index = False)

if __name__ == '__main__':
    def_dict = {'DEPTH': 8, 'ETA': 5e-4, 'GAM': 1.0, 'GEM_MEAN': 0.5, 'GEM_SCALE': 100, 'SCALING_SHAPE': 1.0, 'SCALING_SCALE': 0.5, 'SAMPLE_ETA': 1, 'SAMPLE_GEM': 1}
    hyp_names = ['DEPTH', 'ETA', 'GAM', 'GEM_MEAN', 'GEM_SCALE']
    hyp_ranges = [(7, 11), (1e-5, 100), (0.1, 1.0), (0.1, 1.0), (10, 1000)]
    n_random = 10
    n_bo = 20
    temp_path = 'temp'
    dat_path = 'n_cuts_dats/ORB_50_hlda_data.dat'
    order_path = 'n_cuts_vecs/order.pkl'
    out_path = 'BO_log.csv'
    B_OP = BO()
    B_OP.run_BO(hyp_names, hyp_ranges, def_dict, n_random, n_bo, temp_path, dat_path, order_path, out_path, log=True)






