import gurobipy as gb
import pandas as pd
import numpy as np

from benders_stochastic_subproblem import Benders_Subproblem

####
# Benders' decomposition, stochastic version
# Generators' production are set day ahead.
# Subproblems find costs associated with that setting
# depending on which demand scenario occurs.
####


# Class which can have attributes set
class expando(object):
    pass


class Benders_Master:
    def __init__(self, max_iters=25, verbose=True, numscenarios=100, demand_avg=200.0, demand_std=20.0, epsilon=0.001, delta=0.001):
        '''
            Class which solves the benders decomposed version of the dispatch problem.

            Parameters
            ----------
            max_iters: int, default 25
                    Maximum number of Benders iterations to run.
            verbose: boolean, default True
                    Print information on upper and lower bounds for each iteration
            numscenarios: int, default 100
                    Number of scenarios to use for subproblems
            demand_avg: float, default 200.0
                    Average demand, used as day-ahead bid.
            demand_std: float, default 20.0
                    Standard deviation for demand in scenario generation.
            epsilon: float, default 0.001
                    Relative threshold for benders iterations.
                    Iterations will stop if ub - lb > |epsilon * lb|
            delta: float, default 0.001
                    Absolute threshold for benders iterations.
                    Iterations will stop if ub < lb + delta
        '''
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self.params = expando()

        self.params.max_iters = max_iters
        self.params.verbose = verbose
        self.params.numscenarios = numscenarios
        self.params.demand_avg = demand_avg
        self.params.demand_std = demand_std

        self._init_benders_params(epsilon=epsilon, delta=delta)
        self._load_data()
        self._build_model()

    def optimize(self, force_submodel_rebuild=False):
        # initial solution
        self.model.optimize()

        # Only build submodels if they don't exist or a rebuild is forced.
        if not hasattr(self, 'submodels') or force_submodel_rebuild:
            self.submodels = {s: Benders_Subproblem(self, scenario=s) for s in self.data.scenarios}
        # Update fixed variables for submodels and rebuild.
        [sm.update_fixed_vars(self) for sm in self.submodels.itervalues()]
        [sm.optimize() for sm in self.submodels.itervalues()]

        # Update bounds based on submodel rebuild
        self._update_bounds()
        self._save_vars()
        # Build cuts until we reach absolute and relative tolerance,
        # or max_iters cuts have been generated.
        while (
            (self.data.ub > self.data.lb + self.data.delta or
             self.data.ub - self.data.lb > abs(self.data.epsilon * self.data.lb)) and
                len(self.data.cutlist) < self.params.max_iters):
            # Generate new cut.
            if self.params.verbose:
                print('********')
                print('* Benders\' step {0}:'.format(len(self.data.upper_bounds)))
                print('* Upper bound: {0}'.format(self.data.ub))
                print('* Lower bound: {0}'.format(self.data.lb))
                print('********')
            self._do_benders_step()
        pass

    def _do_benders_step(self):
            self._add_cut()
            self._start_from_previous()
            self.model.optimize()
            [sm.update_fixed_vars(self) for sm in self.submodels.itervalues()]
            [sm.optimize() for sm in self.submodels.itervalues()]
            self._update_bounds()
            self._save_vars()

    def _init_benders_params(self, epsilon=0.001, delta=0.001):
        self.data.cutlist = []
        self.data.upper_bounds = []
        self.data.lower_bounds = []
        self.data.mipgap = []
        self.data.solvetime = []
        self.data.alphas = []
        self.data.lambdas = {}
        self.data.epsilon = epsilon
        self.data.delta = delta
        self.data.ub = gb.GRB.INFINITY
        self.data.lb = -gb.GRB.INFINITY

    ###
    # Data Loading
    ###
    def _load_data(self):
        self._load_generator_data()
        self._load_demand_data()

    def _load_generator_data(self):
        self.data.geninfo = pd.read_csv('benders_stochastic_gens.csv', index_col='gen', skipinitialspace=True)
        self.data.generators = self.data.geninfo.index

    def _load_demand_data(self):
        self.data.VOLL = 1000
        self.data.demand_da = self.params.demand_avg
        self.data.scenarios = ['s'+str(i) for i in xrange(self.params.numscenarios)]
        self.data.demand_rt = pd.Series(
            data=np.random.normal(self.params.demand_avg, self.params.demand_std, size=self.params.numscenarios),
            index=self.data.scenarios)
        self.data.scenarioprobs = {s: 1.0/self.params.numscenarios for s in self.data.scenarios}
        # Dump load
        self.data.dumploadprice = 10
        self.data.dumploadmax = self.data.demand_da

    ###
    # Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model
        gens = self.data.generators
        geninfo = self.data.geninfo

        self.variables.gprod_da = {}
        for g in gens:
            self.variables.gprod_da[g] = m.addVar(lb=0, ub=geninfo.maxprod[g])

        self.variables.load_da = m.addVar(lb=0, ub=self.data.demand_da)

        # Benders' proxy variable
        self.variables.alpha = m.addVar(lb=-self.data.demand_da*self.data.VOLL, ub=gb.GRB.INFINITY)

        m.update()

    def _build_objective(self):
        m = self.model
        gens = self.data.generators
        geninfo = self.data.geninfo

        self.objective = m.setObjective(
            gb.quicksum(geninfo.price[g] * self.variables.gprod_da[g] for g in gens) -
            self.data.VOLL*self.variables.load_da +
            self.variables.alpha)

    def _build_constraints(self):
        m = self.model
        gens = self.data.generators
        geninfo = self.data.geninfo

        self.constraints.powerbalance_da = m.addConstr(
            gb.quicksum(self.variables.gprod_da[g] for g in gens),
            gb.GRB.EQUAL,
            self.variables.load_da)

        self.constraints.cuts = {}

    def _add_cut(self):
        gens = self.data.generators
        geninfo = self.data.geninfo

        cut = len(self.data.cutlist)
        self.data.cutlist.append(cut)

        # Get sensitivities from subproblem
        sens_gen = {
            g: sum(self.data.scenarioprobs[s] * self.submodels[s].constraints.fixed_da[g].pi for s in self.data.scenarios)
            for g in gens}
        self.data.lambdas[cut] = sens_gen
        sens_load = sum(self.data.scenarioprobs[s] * self.submodels[s].constraints.fixed_load_da.pi for s in self.data.scenarios)
        # Get subproblem objectives)
        z_sub = sum(self.data.scenarioprobs[s] * self.submodels[s].model.ObjVal for s in self.data.scenarios)
        # Generate cut
        self.constraints.cuts[cut] = self.model.addConstr(
            self.variables.alpha,
            gb.GRB.GREATER_EQUAL,
            z_sub +
            gb.quicksum(sens_gen[g] * self.variables.gprod_da[g] for g in gens) -
            sum(sens_gen[g] * self.variables.gprod_da[g].x for g in gens) +
            sens_load * (self.variables.load_da - self.variables.load_da.x)
        )

    def _clear_cuts(self):
        self.data.cutlist = []
        self.data.lambdas = {}
        self.model.update()
        for con in self.constraints.cuts.values():
            self.model.remove(con)
        self.constraints.cuts = {}
        self.data.ub = gb.GRB.INFINITY
        self.data.lb = -gb.GRB.INFINITY
        self.data.upper_bounds = []
        self.data.lower_bounds = []

    ###
    # Update upper and lower bounds for Benders' iterations
    ###
    def _update_bounds(self):
        z_sub = sum(self.data.scenarioprobs[s] * self.submodels[s].model.ObjVal for s in self.data.scenarios)
        z_master = self.model.ObjVal
        # The best upper bound is the best incumbent with
        # alpha replaced by the sub problems' actual cost
        self.data.ub = z_master - self.variables.alpha.x + z_sub
        # The best lower bound is the current bestbound,
        # This will equal z_master at optimality
        try:
            self.data.lb = self.model.ObjBound
        except gb.GurobiError:
            self.data.lb = self.model.ObjVal
        self.data.upper_bounds.append(self.data.ub)
        self.data.lower_bounds.append(self.data.lb)
        self.data.mipgap.append(self.model.params.IntFeasTol)
        self.data.solvetime.append(self.model.Runtime)

    def _save_vars(self):
        # self.data.xs.append(self.variables.x.x)
        # self.data.ys.append(self.submodel.variables.y.x)
        self.data.alphas.append(self.variables.alpha.x)

    def _start_from_previous(self):
        '''
            Used to warm-start MIP problems.
        '''
        pass
