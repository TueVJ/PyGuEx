import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from benders_stochastic_master import Benders_Master

sns.set_style('ticks')

m = Benders_Master()
m.model.Params.OutputFlag = False
m.optimize()

rtdf = pd.DataFrame({g: {m.data.demand_rt[s]: m.submodels[s].variables.gprod_rt[g].x for s in m.data.scenarios} for g in m.data.generators})
updf = pd.DataFrame({g: {m.data.demand_rt[s]: m.submodels[s].variables.gprod_rt_up[g].x for s in m.data.scenarios} for g in m.data.generators})
downdf = pd.DataFrame({g: {m.data.demand_rt[s]: m.submodels[s].variables.gprod_rt_down[g].x for s in m.data.scenarios} for g in m.data.generators})

dacost = m.model.ObjVal - m.variables.alpha.x
rscostseries = pd.Series({m.data.demand_rt[s]: sum(
    m.data.geninfo.price[g]*m.submodels[s].variables.gprod_rt[g].x +
    m.data.geninfo.uppremium[g]*m.submodels[s].variables.gprod_rt_up[g].x +
    m.data.geninfo.downpremium[g]*m.submodels[s].variables.gprod_rt_down[g].x
    for g in m.data.generators) for s in m.data.scenarios})


plt.ion()

plt.figure(figsize=(12, 8))
ax = plt.subplot(221)
rtdf.plot(ax=ax, marker='.')
plt.xlabel('Realised real-time demand [MW]')
plt.ylabel('Generator setting [MW]')

ax = plt.subplot(222)
rscostseries.plot(ax=ax, marker='.')
plt.xlabel('Realised real-time demand [MW]')
plt.ylabel('Final cost [$]')

ax = plt.subplot(223)
updf.plot(ax=ax, marker='.')
plt.xlabel('Realised real-time demand [MW]')
plt.ylabel('Generator upregulation [MW]')

ax = plt.subplot(224)
downdf.plot(ax=ax, marker='.')
plt.xlabel('Realised real-time demand [MW]')
plt.ylabel('Generator downregulation [MW]')
plt.tight_layout()

m.model.Params.OutputFlag = False
m.params.verbose = False
demands = np.linspace(160, 240, 81)
costs = []
for demand in demands:
    m.variables.load_da.ub = demand
    m.variables.load_da.lb = demand
    m._clear_cuts()
    m.optimize()
    costs.append(-m.model.ObjVal)
m.variables.load_da.ub = 200
m.variables.load_da.lb = 0

plt.figure()
plt.plot(demands, costs)
plt.ylabel('Social Welfare [$]')
plt.xlabel('Demand cleared in DA market [MW]')
