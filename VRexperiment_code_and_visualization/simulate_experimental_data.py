# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:26:39 2018

@author: nesti
"""


import numpy as np
import pickle
from skopt import Optimizer
from skopt.plots import plot_evaluations, plot_objective

from skopt.benchmarks import hart6 as hart6_
def hart6(x):
    return hart6_(x[:6])

noise_level = 0.0
def objective(x, noise_level=noise_level):
#    return np.sin(x[0])**2 + 1*np.cos(x[1])**2 - .3*x[2]**2 + noise_level * np.random.randn();      # optimum should be around [pi/2, pi, 0]

    return -x[0]**2 - x[0] + (-0.5*x[1]**2  +  2.5*x[1]) + (- .7*x[2]**2) + noise_level * np.random.randn();  

try:
    with open('my_optimizer_old.pkl', 'rb') as f:
        opt = pickle.load(f)
except:
#    opt = Optimizer([(-2.0, 2.0)], "GP", acq_func = 'LCB', acq_func_kwargs = )
    #### high k means explore, low kappa means exploit

#     bounds = [(0., 1.),] * 8
     opt = Optimizer(base_estimator = 'GP', acq_func = 'LCB', acq_optimizer = 'auto',
                 dimensions         = [(-5.0, 5.0),   # range for param 1 (eg trajectory final height?)
                                       (-5.0, 5.0),   # range for param 2 (eg trajectory final pitch?)
                                       (-5.0, 5.0)],  # range for param 4 (eg vibrational state duration?)
                 acq_func_kwargs    = {'kappa': 10}, # we should prefer explore (high kappa). howver, with higher dim it will naturally tend to diversify, so kappa could be decreased
                 n_initial_points   = 10
                 )


"""

kappa [float, default=1.96]: Controls how much of the variance in the predicted values should be taken into account. Used when the acquisition is "LCB" (lower confidence bound). 
    If set high, then we are favouring exploration over exploitation and vice versa.

xi [float, default=0.01]: Controls how much improvement one wants over the previous best values. Used when the acquisition is either "EI" or "PI".
    to use this, i think i need a way to scale xi by the variance of the signal, which howver will depend on the participant and will have to be adapted online...

"""

# run the experiment
for i in range(50):
    next_x = opt.ask()
    f_val = -1 * objective(next_x)   # hart6(next_x) ## branin(next_x)
    result = opt.tell(next_x, f_val)
    #print('iteration:', i, next_x, f_val)
    with open('my_optimizer.pkl', 'wb') as f:
        pickle.dump(opt, f)
        

# visualize evaluations and partial dependence plots ( Friedman (2001) (doi:10.1214/aos/1013203451 section 8.2))
# good explanation: https://www.kaggle.com/dansbecker/partial-dependence-plots
#visulize how the value of xj influences the average predicted values y (marginalize over of all other variables).
#        marginalize: find the probability distribution of a random variable REGARDLESS of the value taken 
#        by all other random variables. concretly, summ joint values of the variable we are interested in
#        and the one we want to marginalize over all possible values of the variable we want to marginalize
plot_evaluations(result, bins=10)
plot_objective(result)
print(result.x)


#save_fig('kappa_def')

#gp_minimize(a=1)






