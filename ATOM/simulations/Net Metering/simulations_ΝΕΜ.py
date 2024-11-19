"""
    Copyright (C) 2022 Technoeconomics of Energy Systems laboratory - University of Piraeus Research Center (TEESlab-UPRC)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# Import packages
###################
import os
import pickle
import datetime
import numpy as np
import pandas as pd
from pyDOE import lhs
from tqdm import trange
import matplotlib.pyplot as plt
from errorbar_plot import plot_errorbars
from models import Parameters, NetMeteringModel, PVOption
import datetime
from dateutil.relativedelta import relativedelta

# Supporting Functions
#######################

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def create_parameters(sample):
    params = Parameters()
    params.max_budget_mu = 0               #not used in the decision-making process of the agent in this version of ATOM
    params.max_budget_sigma = 0
    params.belief_mean_mu   = sample[0] 
    params.belief_mean_sigma = sample[1]
    params.belief_var_mu = sample[2]
    params.belief_var_sigma = sample[3]
    params.w1_mu = sample[4]
    params.w1_sigma = sample[5]
    params.w2_mu = sample[6]
    params.w2_sigma = sample[7]
    params.propensity_mu = sample[8]
    params.propensity_sigma = sample[9]
    params.inertia = 0.04
    params.discount_rate = 0.04
    return params

def rescale(column, params, index):
    interval = params.iloc[index]
    return column * (interval['Max'] - interval['Min']) + interval['Min']

################################
#                              #
#         Main Program         #
#                              #
################################
#
# Parameters - Inputs
##########################################################################################################

num_samples      = 25
start_year       = 2023
end_year         = 2030
initial_demand   = 4162
init_pvprice     = 1.333
pv_price_change  = False
pv_dfact         = - 1.28 # Annual percentage Increase/Decrease in PV initial investment cost (+ for increase, - for decrease)
demand_change    = False
demand_dfact     = 1.19  # Annual percentage Mean Household Electricity Demand Increase/Decrease (+ for increase, - for decrease)
ret_price_change = False
init_ret_price1  = 0.205
init_ret_price2  = 0.05087
ret_price_change_rate = 1.31 # Annual percentage Increase/Decrease in Γ1 Residential Tariff


params_name = 'NEM'
filename = 'data_{}/params_{}.csv'.format(str(params_name), str(params_name))
params = pd.read_csv(filename)
design = lhs(10, samples=num_samples, criterion='maximin')

for i in range(10):
    design[:, i] = rescale(design[:, i], params, i)
design_df = pd.DataFrame.from_dict(design)
design_df.to_csv('outputs_{}/design_{}.csv'.format(str(params_name), str(params_name)))


##########################################################################################################

# Simulation
#############

end_date = datetime.date(end_year, 1, 1)

options = [PVOption('1.8 kWp', 6), PVOption('2.4kWp', 8),PVOption('3 kWp', 10),PVOption('3.6 kWp', 12), PVOption('4.8 kWp', 16), PVOption('6 kWp', 20) ]

results = {}

for i in trange(num_samples):
    sim = NetMeteringModel(1000, 5, create_parameters(design[i, :]), options, start_year, end_year,
                                    init_pvprice, pv_price_change,pv_dfact,
                                    initial_demand,demand_change,demand_dfact,    
                                                   ret_price_change,
                                                   init_ret_price1, 
                                                   init_ret_price2, 
                                                   ret_price_change_rate) 
    
    
    sim.run(end_date)
    data = sim.datacollector.get_model_vars_dataframe()
    results[i] = data['Capacity (kW)']
    

# Results
##########
capacity = pd.DataFrame.from_dict(results)

# Scaling-up in MW
###################
capacity = capacity/1000
capacity = 86.2635 * capacity #upscaler based on calibration results


start = datetime.date(2023, 1, 1)
x_labels = pd.date_range(start, start+relativedelta(months=len(capacity)), freq='M')
capacity_save = capacity
capacity_save.index = x_labels.map(lambda s: s.date())
labels = {"x": "Month of simulation", "y": "PV capacity (MW)",
          "title": " PV capacities"}
capacity_save.to_csv('outputs_{}/design_outputs_capacity_{}.csv'.format(str(params_name),str(params_name)))

###################################
#                                 #
#            Plotting             #
#                                 #
###################################

string = 'New PV capacity addition expected from the {} scheme'.format(str(params_name))

# Processing the data - Statistics
num = num_samples
cap = capacity
cap_mu = [0 for k in range(len(cap))]
cap_var1 = np.zeros((len(cap)))
cap_var2 = np.zeros((len(cap)))
cap_var = np.zeros((2,len(cap)))


for i in range(len(cap)):
    buff = [0 for k in range(num)]
    k = 0
    for j in range(num):
        buff[k] = cap[j][i]
        k += 1
    cap_mu[i] = np.median(buff)
    cap_var[0][i] = cap_mu[i] - np.min(buff)
    cap_var[1][i] = np.max(buff) - cap_mu[i]
    cap_var1[i] = cap_mu[i] - np.min(buff)
    cap_var2[i] = np.max(buff) - cap_mu[i]

# Plotting the results
#######################
t = [i for i in range(len(cap))]
plt.figure(2,figsize = cm2inch(30, 20))
plt.errorbar(t, cap_mu, cap_var,  mfc='red', ms=20, mew=4, alpha = 0.3, animated = 0)
plt.plot(t,cap_mu, color='#CC4F1B')
plt.fill_between(t, cap_mu-cap_var1, cap_mu+cap_var2, alpha=0.3, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.suptitle(string, fontsize=12)
plt.xlabel('Month of simulation ({}-{})'.format(str(start_year),str(end_year)), fontsize=10)
plt.ylabel('PV capacity additions (MW)', fontsize=10)
plt.savefig("outputs_{}/Fig.{}a.png".format(str(params_name),str(params_name)), bbox_inches='tight')
plt.show()

# Extra Plots
##############

capacity.plot(legend=False)
plt.suptitle(string , fontsize=12)
plt.xlabel('Month of simulation (2023-2030)', fontsize=10)
plt.ylabel('PV capacity additions (MW)', fontsize=10)
plt.savefig("outputs_{}/Fig.{}b.png".format(str(params_name),str(params_name)), bbox_inches='tight')
plt.show()

