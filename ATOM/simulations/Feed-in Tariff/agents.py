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
import logging
import numpy as np
import numpy_financial as npf
from mesa import Agent
from typing import List
from functools import wraps

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

handler = logging.FileHandler('agents.log', mode='w')
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


# Supporting Functions
#######################

def memoize(func):
    cache = {}
    @wraps(func)
    def wrap(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrap

@memoize
def get_discount(rate, n):
    return npf.npv(rate, np.ones(n))

def indicator_3(x):
    if x > 3.:
        return x
    else:
        return 3
    
def agent_update_vars():
    belief_var_low = 0.95
    belief_var_up  = 1
    belief_mean_weight = 0.5
    revenue_low = 0.2
    revenue_up = 0.5
    return belief_var_low, belief_var_up, belief_mean_weight, revenue_low, revenue_up


##----------FutureAgent-------------------------------------------------------------------------------------------------------------##

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def indicator(x):
    if x > 0:
        return x
    else:
        return 0

######################################################################################################################################
##-----------------------------------------FutureAgent_Feed in Tarrif---------------------------------------------------------------##
######################################################################################################################################


class FutureAgent_FiT(Agent):

    def __init__(self, unique_id, model,
                 max_budget, belief_mean, belief_var,
                 w1, w2, propensity
                ):

        super().__init__(unique_id, model)

        self.max_budget = indicator(max_budget)

        self.belief_mean = abs(belief_mean) 
        self.belief_var = abs(belief_var)

        self.w1 = abs(w1)
        self.w2 = abs(w2)
        self.propensity = abs(propensity)

        self.has_invested = False
        self.has_sent = False
        self.revenue = 0


    def __repr__(self):
        return 'Agent: ID{}'.format(self.unique_id)

    __str__ = __repr__

    def add_contacts(self, agents):
        self.contacts = agents

    def send_experience(self):
        for agent in self.contacts:
            if not agent.has_invested:
                agent.inform(self.revenue, self.my_option)
                logger.debug('%s sent revenue %d for %s to %s',
                             self, self.revenue, self.my_option.name, agent)

    def inform(self, revenue, option):
        if revenue is None:
            revenue = 0.0
        belief_mean, belief_var = self.expectations[option.name]
        logger.debug('%s had initial beliefs for option %s mean:%d and var:%d',
                         self, option.name, belief_mean, belief_var)
        
        a, b, c, d, e = agent_update_vars()
        self.expectations[option.name][1] = indicator_3(np.random.uniform(a, b)*belief_var)
        self.expectations[option.name][0] = c*belief_mean + revenue*np.random.uniform(d,e)
        logger.debug('%s updated its beliefs for option %s to mean:%d and var:%d',
                               self, option.name, self.expectations[option.name][0],
                               self.expectations[option.name][1])

    def estimate_payback_period(self, option):
        try:
            self.expectations
        except AttributeError:
            self.expectations = {}
            for _option in self.model.options:
                self.expectations[_option.name] = [self.belief_mean,
                                                   self.belief_var]

        belief_mean, belief_var = self.expectations[option.name]
        for i in range(1, 25):
            capex = self.model.get_capex(option, self.model.init_pvprice, self.model.pv_price_change,
                                                           self.model.pv_dfact, self.model.start_year)
            z = (belief_mean *option.capacity * get_discount(self.model.discount_rate, i) -
                       capex) / (i * belief_var)

            if z >= 1.28:
                logger.debug('%s estimated payback period for %s as %d', self, option.name, i)
                return i

        logger.debug('%s estimated payback period for %s as >20 years', self, option.name)
        return np.inf

    def update_resistance(self, option):
        adopters = self.model.num_invested / self.model.num_agents
        res = self.w1*self.estimate_payback_period(option) + self.w2*(1 - adopters)

        if np.isinf(res):
            logger.debug('%s has inf resistance', self)
        else:
            logger.debug('%s has resistance %d and threshold %d', self, res, self.propensity)

        return res

    def make_decision(self):
        quadrant = []
        resistances = []

        for option in self.model.options:
            res = self.update_resistance(option)
            if res <= self.propensity: #and self.model.get_capex(option) <= self.max_budget:
                quadrant.append(option)
                resistances.append(res)

        if quadrant:
            if np.random.binomial(1, self.model.inertia):
                choice = np.random.multinomial(1, softmax(resistances))
                self.my_option = quadrant.pop(np.where(choice==True)[0][0])
                logger.debug('%s invested in %s', self, self.my_option.name)
                return self.my_option

    def update_revenue(self):
        self.revenue = self.model.calculate_revenue(self.my_option, self.model.initial_FIT, self.model.FIT_evolution,
                                                           self.model.FIT_dfact, self.model.start_year, self.model.end_year)

    def step(self):
        if self.has_invested and not self.has_sent:
            self.update_revenue()
            self.send_experience()
            self.has_sent = True

    def advance(self):
        if not self.has_invested:
            decision = self.make_decision()
            if decision:
                self.has_invested = True
                self.model.num_invested += 1 
                self.capacity = decision.capacity * 0.3 # kW
                
######################################################################################################################################
######################################################################################################################################
