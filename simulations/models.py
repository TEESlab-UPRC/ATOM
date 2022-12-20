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

import datetime
import functools
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from dateutil.relativedelta import relativedelta
from agents import FutureAgent_NetMetering, indicator
from networkx.generators.random_graphs import watts_strogatz_graph


# Supporting Functions
#######################

def track_capacity(model):
    capacities = [agent.capacity for agent in model.schedule.agents if agent.has_invested]
    return sum(capacities)

def track_adopters(model):
    return model.num_invested / model.num_agents

def start_year(model):
    return model.start_year


class Parameters(object):
    def __init__(self, init=None):
        if init is None:
            init = np.zeros(14)

        self.max_budget_mu = init[0]
        self.max_budget_sigma = init[1]

        self.belief_mean_mu = init[2]
        self.belief_mean_sigma = init[3]
        self.belief_var_mu = init[4]
        self.belief_var_sigma = init[5]

        self.w1_mu = init[6]
        self.w1_sigma = init[7]
        self.w2_mu = init[8]
        self.w2_sigma = init[9]

        self.propensity_mu = init[10]
        self.propensity_sigma = init[11]

        self.inertia = init[12]
        self.discount_rate = init[13]



##--------FutureModel-------------------------------------------------------------------------

def find_period(s):
    if s.month in [4, 5, 10, 11]:
        return 1
    elif s.month in [6, 7, 8, 9]:
        return 2
    else:
        return 3

class PVOption(object):

    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity




##---------------------------FutureModel- Net-metering --------------------------------------------

###################################################################################################

class NetMeteringModel(Model):

    def __init__(self, N, k, parameters, options,  start_year, end_year,
                                    init_pvprice, pv_price_change,pv_dfact,
                                    initial_demand,demand_change,demand_dfact,    
                                                   ret_price_change,
                                                   init_ret_price1, 
                                                   init_ret_price2, 
                                                   ret_price_change_rate):
        self.num_agents = N
        self.options = options
        self.num_invested = 0
        self.schedule = SimultaneousActivation(self)
        self.now = datetime.date(start_year, 1, 1)
        self.start_year = start_year
        self.end_year = end_year
        self.init_pvprice = init_pvprice
        self.pv_price_change = pv_price_change
        self.pv_dfact = pv_dfact
        self.initial_demand = initial_demand
        self.demand_change = demand_change
        self.demand_dfact =  demand_dfact   
        self.ret_price_change = ret_price_change
        self.init_ret_price1 = init_ret_price1 
        self.init_ret_price2 = init_ret_price2
        self.ret_price_change_rate = ret_price_change_rate
        
        # Create network
        G = watts_strogatz_graph(N, k, 0.2)

        # Create parameters
        max_budget = np.random.normal(parameters.max_budget_mu,
                                      scale=parameters.max_budget_sigma, size=N)

        belief_mean = np.random.normal(parameters.belief_mean_mu,
                                       scale=parameters.belief_mean_sigma, size=N)

        belief_var = np.random.normal(parameters.belief_var_mu,
                                      scale=parameters.belief_var_sigma, size=N)

        w1 = np.random.normal(parameters.w1_mu, scale=parameters.w1_sigma, size=N)
        w2 = np.random.normal(parameters.w2_mu, scale=parameters.w2_sigma, size=N)

        propensity = np.random.normal(parameters.propensity_mu,
                                      scale=parameters.propensity_sigma, size=N)

        self.inertia = parameters.inertia
        self.discount_rate = parameters.discount_rate

        # Create agents
        for i in range(N):
            a = FutureAgent_NetMetering( i, self, max_budget[i],
                             belief_mean[i], belief_var[i],
                             w1[i], w2[i], propensity[i]
                           )
            G.add_node(i, agent=a)
            self.schedule.add(a)

        for i in range(N):
            G.nodes[i]['agent'].add_contacts([G.nodes[j]['agent'] for j in G.neighbors(i)])

        self.datacollector = DataCollector(
            model_reporters={'Capacity (kW)': track_capacity,
                             'Adopters (%)': track_adopters})

    def calculate_revenue(self, option, initial_demand, demand_change,
                                        demand_dfact, init_ret_price1, init_ret_price2,
                                        ret_price_change, ret_price_change_rate, start_year, end_year):

        try:
            self.average_profile
        except AttributeError:
            self.average_profile = pd.read_csv('data_NEM/mean_profiles.csv')
           
            

        try:
            self.production
        except AttributeError:
            start = pd.Timestamp(datetime.date(2016, 1, 1))
            end = start + pd.Timedelta(days=366)
            index = pd.date_range(start=start, end=end, freq='H')

            dc_production = pd.read_csv('data_NEM/dc_profile.csv')
            dc_production['index'] = index[:-1]
            dc_production = dc_production.set_index('index')
            self.production = dc_production * 0.00096

        return self.get_benefit(option,initial_demand, demand_change,
                                        demand_dfact, init_ret_price1, init_ret_price2,
                                        ret_price_change, ret_price_change_rate, start_year, end_year )

    @functools.lru_cache(maxsize=None)
    def get_benefit(self, option, initial_demand, demand_change,
                                        demand_dfact, init_ret_price1, init_ret_price2,
                                        ret_price_change, ret_price_change_rate, start_year, end_year):
        production = option.capacity * self.production
        if demand_change:
            year_cnt = self.now.year - start_year 
        else:
            year_cnt = 0
        total_demand = ((1 + (demand_dfact/100)) ** year_cnt) * initial_demand
        net_energy = min(production.sum().values, total_demand)
        energy_from_grid = 0
        g = production.groupby([lambda s: find_period(s), lambda s: s.hour])

        for i, j in [(x,y) for x, y in itertools.product([1,2,3], range(24))]:
                energy_from_grid += production.loc[g.groups[(i, j)]]['p_mp'].map(lambda x:
                        indicator(self.average_profile.iloc[j]['Period ' + str(i)]*total_demand/3807.27 - x)).sum()
                        
        dur = end_year - start_year
        if ret_price_change:
            year_cnt = self.now.year - start_year
        else:
            year_cnt = 0
        result = ((1 + (ret_price_change_rate/100))** year_cnt) * init_ret_price1 * net_energy + ((1 + (ret_price_change_rate/100))** year_cnt) * init_ret_price2 * (total_demand-energy_from_grid)
        result = result/option.capacity
        return result

    def get_capex(self, option, init_pvprice, pv_price_change, pv_dfact, start_year):

        if pv_price_change:
            end = self.now.year
            year_cnt = end - start_year
        else:
            year_cnt = 0
        init_PV_cost = ((1 + (pv_dfact/100)) ** year_cnt) * init_pvprice * 1000 # 1000 is for Wp to kWp
        pv_cost = (option.capacity * 0.3) * init_PV_cost
        return pv_cost

    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()

    def run(self, end_date, verbose=False):
        if verbose:
            with tqdm(total=156) as pbar:
                while self.now <= end_date:
                    self.step()
                    self.now += relativedelta(months=1)
                    pbar.update(1)
        else:
            while self.now <= end_date:
                self.step()
                self.now += relativedelta(months=1)

###################################################################################################
###################################################################################################
