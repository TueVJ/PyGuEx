# Import Gurobi Library
import gurobipy as gb

####
#   Benders decomposition via Gurobi + Python
####

# Class which can have attributes set.
class expando(object):
    pass


# Optimization class
class Benders_Subproblem:
    def __init__(self, MP, scenario=0):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        
        self.MP = MP
        self.data.scenario = scenario

        self._build_model()
        self.update_fixed_vars(MP)

    def optimize(self):
        self.model.optimize()

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self.model.setParam('OutputFlag', False)
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model

        dumploadmax = self.MP.data.dumploadmax
        gens = self.MP.data.generators
        geninfo = self.MP.data.geninfo
        s = self.data.scenario
        demandmax = self.MP.data.demand_rt[s]

        # Production of generator g, up and downregulation
        # Up and down regulation are limited by the generators' capability.
        self.variables.gprod_da = {}
        self.variables.gprod_rt = {}
        self.variables.gprod_rt_up = {}
        self.variables.gprod_rt_down = {}
        for g in gens:
            self.variables.gprod_da[g] = m.addVar(lb=0.0, ub=geninfo.maxprod[g])
            self.variables.gprod_rt[g] = m.addVar(lb=0.0, ub=geninfo.maxprod[g])
            self.variables.gprod_rt_up[g] = m.addVar(lb=0.0, ub=geninfo.maxprod[g] * geninfo.upflex[g])
            self.variables.gprod_rt_down[g] = m.addVar(lb=0.0, ub=geninfo.maxprod[g] * geninfo.downflex[g])

        self.variables.loadserved = m.addVar(lb=0.0, ub=demandmax)
        self.variables.loadserved_DA = m.addVar(lb=0.0, ub=gb.GRB.INFINITY)
        self.variables.dumpload = m.addVar(lb=0.0, ub=dumploadmax)

        m.update()

    def _build_objective(self):
        m = self.model
        gens = self.MP.data.generators
        geninfo = self.MP.data.geninfo
        VOLL = self.MP.data.VOLL
        dumploadprice = self.MP.data.dumploadprice

        m.setObjective(
            gb.quicksum((geninfo.price[g] + geninfo.uppremium[g])*self.variables.gprod_rt_up[g] for g in gens) +
            gb.quicksum((- geninfo.price[g] + geninfo.downpremium[g])*self.variables.gprod_rt_down[g] for g in gens) -
            VOLL*(self.variables.loadserved-self.variables.loadserved_DA) +
            dumploadprice * self.variables.dumpload
        )

    def _build_constraints(self):
        m = self.model
        gens = self.MP.data.generators

        self.constraints.powerbalance_rt = m.addConstr(
            gb.quicksum(self.variables.gprod_rt[g] for g in gens),
            gb.GRB.EQUAL,
            self.variables.loadserved + self.variables.dumpload)

        self.constraints.coupling_da = {}
        self.constraints.fixed_da = {}
        for g in gens:
            self.constraints.coupling_da[g] = m.addConstr(
                self.variables.gprod_rt[g],
                gb.GRB.EQUAL,
                self.variables.gprod_da[g] + self.variables.gprod_rt_up[g] - self.variables.gprod_rt_down[g])
            self.constraints.fixed_da[g] = m.addConstr(
                self.variables.gprod_da[g],
                gb.GRB.EQUAL,
                0.0)
        self.constraints.fixed_load_da = m.addConstr(
            self.variables.loadserved_DA,
            gb.GRB.EQUAL,
            0.0)

    def update_fixed_vars(self, MP):
        for g in self.MP.data.generators:
                self.constraints.fixed_da[g].rhs = MP.variables.gprod_da[g].x
        self.constraints.fixed_load_da.rhs = MP.variables.load_da.x
