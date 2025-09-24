import numpy as np
import torch
from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer
import torch.nn as nn
import torch.nn.init as init
from abom_net import ABOM_net
import time

class ABOM(Basic_Optimizer):
    """
    # Introduction
    Adaptive Meta Black-box Evolutionary Optimization Model (ABOM).
    
    # Args:
    - config (object): Configuration object containing agent and training parameters.
    # Attributes:
    
    # arch param
    - num_layers (int): Number of layers in the ABOM model.
    - d_model (int): Dims of q/k/v.
    - dim_mutation(int): Dims of mutation.
    - dropout_c (float): Dropout rate of crossover.
    - dropout_m (float): Dropout rate of mutation.
    - input_problem_dim (int): Dims of problem.

    # evolutionary param
    - NP (int): Size of the population.

    # Adaptive update param
    - optimizer: AdamW
    - criterion: MSELoss
    - lr (float): Learning rate.
    - wd (float): Weight decay.
    """
    def __init__(self, config):
        """
        # Introduction
        Initializes the optimizer with the given configuration, setting up neural network components, device, and various hyperparameters for the optimization process.
        # Args:
        - config (object): Config object containing optimizer settings.
            - Attributes needed for the ABOM_Optimizer are the following:
                - device (str or torch.device): Device on which computations will be performed.Default is 'cpu'.
                - maxFEs (int): Maximum number of function evaluations allowed.
                - log_interval (int): Interval for logging progress.Default is 100.
                - n_logpoint (int): Number of log points for logging.Default is 50.
        # Built-in Attribute:
        - self.device (torch.device): Device on which computations will be performed.
        - self.max_fes (int): Maximum number of function evaluations allowed.
        - self.cost (any): Variable for recording cost or loss. Default is None.
        - self.log_index (any): Index for logging. Default is None.
        - self.log_interval (int): Interval for logging progress.
        - self.fes (int or None): Current function evaluation count. Default is None.
        # Returns:
        - None
        # Raises:
        - None
        """
        
        super().__init__(config)
        self.__config = config
        self.device = self.__config.device
        self.max_fes = self.__config.maxFEs

        # for record
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval
        self.fes = None
        self.evolution_info = None

        # arch param
        self.num_layers = 1
        self.dim_mutation = None 
        self.dropout_c = 0.95
        self.dropout_m = 0.95
        self.d_model = None 

        # evolutionary param
        self.NP=16

        # adaptive update param
        self.abom = None
        self.lr = 1e-3
        self.wd = 1e-5

    
    def __str__(self):
        """
        # Introduction
        Returns a string representation of the ABOM_Optimizer object.
        # Returns:
        - str: The string "ABOM_Optimizer", representing the class name.
        """
        
        return "ABOM"

    def sort(self, population, c_cost):
        """
        # Introduction
        Sorts the population and corresponding cost values in ascending order based on the cost.
        # Built-in Attribute:
        - population (numpy): The current population of solutions.
        - c_cost (numpy): The cost values associated with each member of the population.
        # Returns:
        None. Updates `population` and `c_cost` in-place to reflect the sorted order.
        """
        
        index = np.argsort(c_cost)
        population_sort = population[index]
        c_cost_sort = c_cost[index]

        return population_sort, c_cost_sort
    
    def init(self, problem):
        """
        # Introduction
        Initializes the population with random values within the problem's bounds.
        # Built-in Attribute:
        - problem (object): The problem instance for which the optimizer is being used.
        # Returns:
        - None. Updates `self.evolution_info['parents']` in-place.
        """
        
        # init population
        population=self.rng.uniform(low=problem.lb,high=problem.ub,size=(self.NP, problem.dim))
        population = np.clip(population, problem.lb, problem.ub)

        if problem.optimum is None:
            costs=problem.eval(population)
        else:
            costs=problem.eval(population)-problem.optimum
        # 
        population_sort, costs_sort = self.sort(population, costs)
        # 
        self.evolution_info = {'parents': population_sort,
                'parents_cost':costs_sort,
                'generation_counter': 0, 
                'xbest':population_sort[0],
                'gbest':costs_sort[0]}
        self.fes = self.NP
        self.cost = [costs_sort[0]]
        self.log_index = 1
        if self.__config.full_meta_data:
            self.meta_X.append(self.evolution_info['parents'].copy())
            self.meta_Cost.append(self.evolution_info['parents_cost'].copy())
        # init model
        if self.d_model==None:
            self.d_model = problem.dim

        if self.dim_mutation == None:
            self.dim_mutation = 1<<((problem.dim).bit_length()-1)

        if self.device == 'cpu':
            rng = self.rng_cpu
        else:
            rng = self.rng_gpu
        self.abom = ABOM_net(self.num_layers, self.d_model, self.dim_mutation, self.dropout_c, self.dropout_m, problem.dim,self.device,rng).to(self.device)
        for name, param in self.abom.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.constant_(param, 0)

    def update(self, problem):

        
        population = self.evolution_info['parents']
        
        population = torch.from_numpy(population).unsqueeze(0)
        cost = torch.from_numpy(self.evolution_info['parents_cost']).unsqueeze(0).unsqueeze(-1)
        # 
        pop_f_nor = torch.argsort(torch.argsort(cost, dim=1), dim=1).double()

        # 
        output = self.abom(population.to(self.device), pop_f_nor.to(self.device), self.evolution_info['generation_counter'])

        criterion = nn.SmoothL1Loss(beta=1.0, reduction='sum')
        optimizer = torch.optim.AdamW(self.abom.parameters(), lr=self.lr, weight_decay=self.wd)

        with torch.no_grad():  
            offspring = output.clone() 

            if not isinstance(problem.ub, (int, float)):
                ub = torch.from_numpy(problem.ub).to(offspring.device).view(1, 1, -1)
                lb = torch.from_numpy(problem.lb).to(offspring.device).view(1, 1, -1)
                offspring = torch.clamp(offspring, lb, ub)
            else:
                offspring = torch.clamp(offspring, problem.lb, problem.ub)
            offspring_e = offspring

            offspring_e = offspring_e.squeeze(0).to('cpu').numpy()

            if problem.optimum is None:
                cost_off=problem.eval(offspring_e)
            else:
                cost_off=problem.eval(offspring_e)-problem.optimum
            self.fes += offspring_e.shape[0]
            # 
            compop_x = np.concatenate((self.evolution_info['parents'], offspring_e))
            compop_f = np.concatenate((self.evolution_info['parents_cost'], cost_off))
            sorted_indices = np.argsort(compop_f)
            best_pop = compop_x[sorted_indices][:compop_x.shape[0]//2]
            best_fitness = compop_f[sorted_indices][:compop_f.shape[0]//2]

        optimizer.zero_grad()  # 
        best_pop_torch = torch.from_numpy(best_pop).unsqueeze(0)
        best_pop_torch = best_pop_torch[:, 0, :].unsqueeze(1).repeat(1, output.size(1), 1)

        loss = criterion(output, best_pop_torch.to(self.device))
        loss.backward()
        optimizer.step()

        print(loss.item())

        # update evolution information
        self.evolution_info['generation_counter'] += 1
        self.evolution_info['parents'] = best_pop
        self.evolution_info['parents_cost'] = best_fitness
        
        if best_fitness[0] <= self.evolution_info['gbest']:
            self.evolution_info['gbest'] = best_fitness[0]
            self.evolution_info['xbest'] = best_pop[0]

    
    def run_episode(self, problem):
        """
        # Introduction
        Executes a single episode of ABOM on the given optimization problem, tracking the best solution found and optionally collecting meta-data about the optimization process.
        # Args:
        - problem (object): An object representing the optimization problem to solve. Must have attributes `lb` (lower bounds), `ub` (upper bounds), `dim` (dimension), `eval` (evaluation function), and optionally `optimum` (known optimum value).
        # Returns:
        - dict: A dictionary containing:
            - 'cost' (list of float): The best fitness value found at each logging interval.
            - 'fes' (int): The total number of function evaluations performed.
            - 'metadata' (dict, optional): If `self.full_meta_data` is True, includes:
                - 'X' (list of np.ndarray): The population positions at each logging interval.
                - 'Cost' (list of float): The fitness values of the population at each logging interval.
        # Raises:
        - AttributeError: If required attributes are missing from the `problem` object.
        - Exception: For errors during the optimization process, such as invalid configuration or evaluation failures.
        """

        self.seed()
        # init population
        self.init(problem)
        
        is_end = False
        # 
        while not is_end:
            # 
            self.update(problem)

            # 
            print('gen',self.evolution_info['generation_counter'])
            print('gbest',self.evolution_info['gbest'])
            if self.__config.full_meta_data:
                self.meta_X.append(self.evolution_info['parents'].copy())
                self.meta_Cost.append(self.evolution_info['parents_cost'].copy())

            if self.fes >= self.log_index * self.log_interval:
                self.log_index += 1
                self.cost.append(self.evolution_info['gbest'])
            is_end = self.fes >= self.max_fes
            if is_end:
                if len(self.cost) >= self.__config.n_logpoint + 1:
                    self.cost[-1] = self.evolution_info['gbest']
                else:
                    while len(self.cost) < self.__config.n_logpoint + 1:
                        self.cost.append(self.evolution_info['gbest'])

        results = {'cost': self.cost, 'fes': self.fes}
        if self.__config.full_meta_data:
            results['metadata'] = {'X':self.meta_X, 'Cost':self.meta_Cost}
        print(results)
        return results
    

# for test
from metaevobox import Config, Tester, get_baseline
from metaevobox.environment.problem.utils import construct_problem_set
from metaevobox.environment.problem import Basic_Problem


class MyProblem(Basic_Problem):
# torch version : class MyProblem(Basic_Problem_Torch)
    def __init__(self,dim=10):
        # Init parameters
        self.opt = None
        self.optimum = None
        self.dim = dim
        self.ub = 5
        self.lb = -5

    def get_optimial(self):
        return self.opt

    def eval(self, x):
        return np.sum(x ** 2, axis=1)
    
if __name__ == '__main__':
    # specify your configuration
    config = {
        'device':'cuda',
    }
    config = Config(config)
    problem = MyProblem()
    abom = ABOM(config)
    res = abom.run_episode(problem)
