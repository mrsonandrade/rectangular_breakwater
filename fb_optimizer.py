#!/usr/bin/env python
# coding: utf-8

__author__      = "Emerson Martins de Andrade"
__version__   = "1.0.0"
__date__   = "2024"

# tested in Python 3.12.2

import numpy as np
from pymoo.optimize import minimize
from pymoode.algorithms import DE
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.termination import Termination
import pickle

def load_model(filename):
    with open(filename+'.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model_SL = load_model("ml_models/length_Gradient_Boosted_Tree")
model_SB = load_model("ml_models/base_Gradient_Boosted_Tree")

import warnings
warnings.filterwarnings("ignore")

class ObjectiveFunction(ElementwiseProblem):
        
    def __init__(self, lower_limit, upper_limit, dimensions, model_SL, model_SB):
        self.model_SL, self.model_SB = model_SL, model_SB
        dimensions = dimensions
        xl = lower_limit
        xu = upper_limit
        super().__init__(n_var=dimensions, n_obj=1, n_ieq_constr=0, xl=xl, xu=xu)

    def objective_function(self, x):

        L =      x[0]
        B =      x[1]
        T =      x[2]

        # Normalize the data using Min-Max scaling
        B_ = (B - 100.) / (500. - 100.)
        L_ = (L - 5.) / (25. - 5.)
        T_ = (T - 5.) / (25. - 5.)

        zeta = 10**-5
        for Lambda in [25., 75., 150., 300., 600.]:
            for Kt in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                Lambda_ = (Lambda - 25.) / (600. - 25.)
            
                X_now = np.array([[Kt,B_,L_,T_,Lambda_]])

                SL = self.model_SL.predict(X_now)
                SB = self.model_SB.predict(X_now)

                # Denormalize
                SL_min_val, SL_max_val = (0.1, 2000.)
                SB_min_val, SB_max_val = (19.3, 500.)

                SL_result = SL[0] * (SL_max_val - SL_min_val) + SL_min_val
                SB_result = SB[0] * (SB_max_val - SB_min_val) + SB_min_val

                if SL_result>=100. and SB_result>=100.:
                    zeta += np.exp(-Kt+0.5)**9

        result = (L*B*T) / zeta

        return result
            
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.objective_function(x)

class FunctionCallTermination(Termination):

    def __init__(self, ref, tol, n_max_evals=float("inf")) -> None:
        super().__init__()

        self.tol = tol
        self.ref = ref
        self.n_max_evals = n_max_evals

    def _update(self, algorithm):

        # the object from the current iteration
        current = self._data(algorithm)

        if self.n_max_evals is None:
            return 0.0

        elif (current-self.ref) <= self.tol:
            return 1.0

        else:
            return algorithm.evaluator.n_eval / self.n_max_evals

    def _data(self, algorithm):
        opt = algorithm.opt
        f = opt.get("f")

        if len(f) > 0:
            return f.min()
        else:
            return np.inf

class Optimization():

    def __init__(self, model_SL, model_SB, iterations, population_size, F, CR, seed, max_evals):
        
        self.iterations = iterations
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.seed = seed
        self.max_evals = max_evals
        self.dimensions = 3
        self.variant = "DE/rand/1/bin"
        #                                L,      B,      T
        self.lower_limit = np.array([    5.,    100.,   5.], dtype=float)
        self.upper_limit = np.array([   25.,    500.,  25.], dtype=float)
        self.pos = None

    def optimize(self):

        fx = DE(pop_size=self.population_size,
                variant=self.variant,
                CR=self.CR,
                F=self.F,
                gamma=1e-4,
                de_repair="to-bounds")

        termination = FunctionCallTermination(ref = 0.0,
                                              tol = 1e-8,
                                              n_max_evals = self.max_evals)

        result = minimize(ObjectiveFunction(self.lower_limit,
                                            self.upper_limit,
                                            self.dimensions,
                                            model_SL,
                                            model_SL),
                          fx,
                          termination,
                          seed=self.seed,
                          save_history=True,
                          verbose=False)

        # UNCOMMENT BELOW TO SAVE THE FITNESS/EVOLUTION HISTORY
        best_f1 = np.array([e.opt[0].F for e in result.history])
        avg_pop_f1 = np.array([e.pop.get("F").mean() for e in result.history])
        best_f2 = np.array([aa[0] for aa in best_f1])
        all_best = best_f2
        all_gen = np.array([aa for aa in range(len(result.history))])
        data = np.asarray([all_gen, all_best]).T
        np.savetxt("history_seed_%d_pop_%d_F_%d_CR_%d.csv" % (self.seed, self.population_size, int(self.F*10), int(self.CR*10)), data, delimiter=",")
        pos = result.X
        self.pos = pos
        final_result = []
        for res_i in self.pos:
            final_result.append(res_i)

        return final_result

for seed in range(1, 31):

    b_list = []
    population_size = 10
    F = 0.8
    CR = 0.7
    max_evals = 1000
    iterations = int(max_evals/population_size)
    b = Optimization(model_SL, model_SB, iterations, population_size, F, CR, seed, max_evals)
    result = b.optimize()

    if np.any(np.isnan(result))==False:
        b_list.append(result)
    else:
        b_list.append(b_list[-1])

    b_list=np.asarray(b_list)
    np.savetxt("seed_%d_pop_%d_F_%d_CR_%d.csv" % (seed, population_size, int(F*10), int(CR*10)), b_list, delimiter=",")

                    

