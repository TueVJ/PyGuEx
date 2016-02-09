# Import Gurobi Library
import gurobipy as gb

####
#   Benders decomposition via Gurobi + Python
#   Example 3.1 from Conejo et al.'s book on optimization techniques
####

##
# To Run:
# m = Benders_Master()
# m.optimize()
##


# Class which can have attributes set.
class expando(object):
    pass


# Master problem
class Benders_Master:
    def __init__(self, benders_gap=0.001, max_iters=10):
        self.max_iters = max_iters
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._load_data(benders_gap=benders_gap)
        self._build_model()

    def optimize(self, simple_results=False):
        # Initial solution
        self.model.optimize()
        # Build subproblem from solution
        self.submodel = Benders_Subproblem(self)
        self.submodel.update_fixed_vars(self)
        self.submodel.optimize()
        self._add_cut()
        self._update_bounds()
        self._save_vars()
        while self.data.ub > self.data.lb + self.data.benders_gap and len(self.data.cutlist) < self.max_iters:
            self.model.optimize()
            self.submodel.update_fixed_vars(self)
            self.submodel.optimize()
            self._add_cut()
            self._update_bounds()
            self._save_vars()
        pass

    ###
    #   Loading functions
    ###

    def _load_data(self, benders_gap=0.001):
        self.data.cutlist = []
        self.data.upper_bounds = []
        self.data.lower_bounds = []
        self.data.lambdas = {}
        self.data.benders_gap = benders_gap
        self.data.ub = gb.GRB.INFINITY
        self.data.lb = -gb.GRB.INFINITY
        self.data.xs = []
        self.data.ys = []
        self.data.alphas = []

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model

        self.variables.x = m.addVar(lb=0.0, ub=16.0, name='x')
        self.variables.alpha = m.addVar(lb=-25.0, ub=gb.GRB.INFINITY, name='alpha')
        m.update()

    def _build_objective(self):
        self.model.setObjective(
            -self.variables.x/4 + self.variables.alpha,
            gb.GRB.MINIMIZE)

    def _build_constraints(self):
        self.constraints.cuts = {}
        pass

    ###
    # Cut adding
    ###
    def _add_cut(self):
        x = self.variables.x
        cut = len(self.data.cutlist)
        self.data.cutlist.append(cut)
        # Get sensitivity from subproblem
        sens = self.submodel.constraints.fix_x.pi
        z_sub = self.submodel.model.ObjVal
        # Generate cut
        self.constraints.cuts[cut] = self.model.addConstr(
            self.variables.alpha,
            gb.GRB.GREATER_EQUAL,
            z_sub + sens * (x - x.x))

    ###
    # Update upper and lower bounds
    ###
    def _update_bounds(self):
        z_sub = self.submodel.model.ObjVal
        z_master = self.model.ObjVal
        self.data.ub = z_master - self.variables.alpha.x + z_sub
        # The best lower bound is the current bestbound,
        # This will equal z_master at optimality
        self.data.lb = self.model.ObjBound
        self.data.upper_bounds.append(self.data.ub)
        self.data.lower_bounds.append(self.data.lb)

    def _save_vars(self):
        self.data.xs.append(self.variables.x.x)
        self.data.ys.append(self.submodel.variables.y.x)
        self.data.alphas.append(self.variables.alpha.x)


# Subproblem
class Benders_Subproblem:
    def __init__(self, MP):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._build_model()
        self.data.MP = MP
        self.update_fixed_vars()

    def optimize(self):
        self.model.optimize()

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model

        # Power flow on line l
        self.variables.y = m.addVar(lb=0.0, ub=gb.GRB.INFINITY, name='y')
        self.variables.x_free = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, name='x_free')

        m.update()

    def _build_objective(self):
        m = self.model

        self.model.setObjective(
            - self.variables.y,
            gb.GRB.MINIMIZE)

    def _build_constraints(self):
        m = self.model
        y = self.variables.y
        x = self.variables.x_free

        self.constraints.c1 = m.addConstr(y - x, gb.GRB.LESS_EQUAL, 5.)
        self.constraints.c2 = m.addConstr(y - x/2., gb.GRB.LESS_EQUAL, 15./2.)
        self.constraints.c3 = m.addConstr(y + x/2., gb.GRB.LESS_EQUAL, 35./2.)
        self.constraints.c4 = m.addConstr(-y + x, gb.GRB.LESS_EQUAL, 10.)
        self.constraints.fix_x = m.addConstr(x, gb.GRB.EQUAL, 0.)

    def update_fixed_vars(self, MP=None):
        if MP is None:
            MP = self.data.MP
        self.constraints.fix_x.rhs = MP.variables.x.x
