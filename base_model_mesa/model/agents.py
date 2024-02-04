# Importing necessary libraries
import random
import numpy as np
from mesa import Agent
from shapely.geometry import Point
from shapely import contains_xy
import networkx as nx
from economic_risk import calculate_economic_risk

# Import functions from functions.py
from functions import generate_random_location_within_map_domain, get_flood_depth, calculate_basic_flood_damage, \
    floodplain_multipolygon, calculate_economic_flood_damage


# Define the Households agent class
class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "Household"

        # Set initial state of adaptation measures
        self.is_adapted_utilities = False  # Initial adaptation status set to False
        self.is_adapted_barrier = False
        self.is_adapted_drainage = False


        # Savings are assumed to be stable over time.
        # This helps simplify the model.
        self.savings = 2*np.random.lognormal(mean=self.model.settings["savings_mean"][self.model.settings_ind],
                                             sigma=self.model.settings["savings_std"][self.model.settings_ind])
        # assume 2 bank accounts per household
        self.house_size = 0.042 * self.savings * \
                          np.random.normal(loc=self.model.settings["house_size_loc"][self.model.settings_ind],
                                           scale=self.model.settings["house_size_scale"][self.model.settings_ind])
        self.money_spent = 0
        self.economic_damage = 0
        self.friend_radius = self.model.settings["friends_influence_radius"][self.model.settings_ind]
        self.adaption_chance = self.model.settings["agent_step_adaption_chance"][self.model.settings_ind]
        
        # Set actual efficacy and costs of measures
        self.actual_values = dict(utilities_efficacy=self.model.settings["actual_efficacy_utilities"]
                                                [self.model.settings_ind],
                                  barrier_efficacy=self.model.settings["actual_efficacy_barrier"]
                                                [self.model.settings_ind],
                                  drainage_efficacy=self.model.settings["actual_efficacy_drainage"]
                                                [self.model.settings_ind],
                                  utilities_cost=self.model.settings["actual_cost_utilities"]
                                                [self.model.settings_ind],
                                  barrier_cost= self.model.settings["actual_cost_barrier"]
                                                [self.model.settings_ind] * 4 * np.sqrt(self.house_size),
                                  # Barrier cost based on square house, x sandbags (5 ish?) per m around perrimeter,
                                  # y (roughly 3) dollar per sandbag
                                  drainage_cost=self.model.settings["actual_cost_drainage"]
                                                [self.model.settings_ind]*10.764*self.house_size,
                                  # Based on 2-5 dollar per square foot of surface area, multiplied by 10.764 to
                                  # convert to dollar per m2
                                  probability=1/(4*self.model.flood_event_occurrence)
                                  ) # damage is calculated after defining self.flood_damage_estimated

        # Probability of flood is set by the input file, as it is a lever

        # Set convincable level and informedness level for interaction with Government Agent.

        self_lower = self.model.settings["agent_self_lower"][self.model.settings_ind]
        self_upper = self.model.settings["agent_self_upper"][self.model.settings_ind]
        self.friend_influence_weight_self = \
            self.model.settings["friends_influence_weight_self"][self.model.settings_ind]
        self.convincable_level = random.uniform(self_lower, self_upper)
        self.informedness = random.uniform(self_lower, self_upper)
        # Set self-efficacy perception.
        self.self_efficacy_perception = random.uniform(self_lower, self_upper)
        self.informedness_old = self.informedness

        # getting flood map values
        # Get a random location on the map
        loc_x, loc_y = generate_random_location_within_map_domain()
        self.location = Point(loc_x, loc_y)

        # Check whether the location is within floodplain
        self.in_floodplain = False
        if contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y):
            self.in_floodplain = True

        # Get the estimated flood depth at those coordinates. the estimated flood depth is calculated based on the
        # flood map (i.e., past data) so this is not the actual flood depth. Flood depth can be negative if the
        # location is at a high elevation
        self.flood_depth_estimated = get_flood_depth(corresponding_map=model.flood_map, location=self.location,
                                                     band=model.band_flood_img)
        # handle negative values of flood depth
        if self.flood_depth_estimated < 0:
            self.flood_depth_estimated = 0
        
        # calculate the estimated flood damage given the estimated flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated)
        # update actual estimated flood damage
        self.actual_values['economic_damage'] = self.flood_damage_estimated * self.house_size * \
                                                self.model.economic_damage_m2 #788 euro/m2 to dollars

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since
        # there is no flood yet and will update its value when there is a shock
        # (i.e., actual flood). Shock happens at some point during the simulation
        self.flood_depth_actual = 0

        # calculate the actual flood damage given the actual flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_actual = calculate_basic_flood_damage(flood_depth=self.flood_depth_actual)

        # Perceptions that can be influenced go into a dictionary
        self.perceptions = dict(utilities_efficacy=0, barrier_efficacy=0, drainage_efficacy=0,
                                utilities_cost=0, barrier_cost=0, drainage_cost=0,
                                probability=0, economic_damage=0)
        # Set initial values for these perceptions
        spread_lower = self.model.settings["agent_perception_spread_lower"][self.model.settings_ind]
        spread_upper = self.model.settings["agent_perception_spread_upper"][self.model.settings_ind]
        for x in self.perceptions:
            self.perceptions[x] = random.uniform(1 - spread_lower, 1 + spread_upper) * self.actual_values[x]
    
    # Function to count friends who can be influential.
    def count_friends(self, radius):
        """Count the number of neighbors within a given radius (number of edges away). This is social
         relation and not spatial"""
        friends = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=radius)
        return len(friends)

    def set_economic_risk(self, risk_aversion):

        self.investment_keys = ["nothing", "utilities", "barrier", "drainage"]
        self.economic_risk, self.economic_optimum = \
            calculate_economic_risk(
                                    calculate_economic_flood_damage,
                                    [0, self.actual_values["utilities_cost"], self.actual_values["barrier_cost"],
                                     self.actual_values["drainage_cost"]],
                                    [1, self.actual_values["utilities_efficacy"],
                                     self.actual_values["barrier_efficacy"],
                                     self.actual_values["drainage_efficacy"]], risk_aversion,
                                    flood_event_height=self.flood_depth_estimated,
                                    flood_event_occurrence=self.model.flood_event_occurrence,
                                    damage_function_kws = {"housesize": self.house_size,
                                                           "damage_m2": self.model.settings["economic_damage_m2"][self.model.settings_ind] 
                                                           }
                                    )

    # Function to find friends and adjust opinions
    def influence_of_friends(self, radius):

        # identify friends within given radius
        friends_nodes = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=radius)
        friends = self.model.grid.get_cell_list_contents(friends_nodes)

        # make list of distances from friends
        # NB: distances not used yet!
        friends_distances = np.array(
            [nx.shortest_path_length(self.model.G, source=self.pos, target=friend.pos) for friend in friends])
        friends_weights = friends_distances / sum(friends_distances)

        # get friends' perceptions
        friends_perceptions = self.perceptions.copy()
        for x in self.perceptions:
            # get friends' perceptions
            friends_perceptions[x] = np.array([friend.perceptions[x] for friend in friends])
            if x == "drainage_cost" or x == "barrier_cost":
                friends_perceptions[x] = np.array([friend.perceptions[x]/friend.house_size for friend in friends])
            # update own perception (DeGroot, distance weight)
            if len(friends_weights) > 0:
                if x == "drainage_cost" or x == "barrier_cost":
                    self.perceptions[x] = self.friend_influence_weight_self * self.perceptions[x] + \
                                          (1 - self.friend_influence_weight_self) * np.average(friends_perceptions[x],
                                                            weights=friends_weights) * self.house_size
                elif x != "economic_damage":
                    self.perceptions[x] = self.friend_influence_weight_self * self.perceptions[x] + \
                                          (1 - self.friend_influence_weight_self) *\
                                          np.average(friends_perceptions[x], weights=friends_weights)



    # Function to update perceptions based on informedness level
    #This updates every step
    def influence_of_informedness(self):
        # Probability & damage perceptions
        # TODO: self.probability_perception = self.informedness * ????
        informedness_difference = self.informedness - self.informedness_old
        for x in self.perceptions:
            self.perceptions[x] = informedness_difference * self.actual_values[x] + \
                                  (1-informedness_difference) * self.perceptions[x]

        self.informedness_old = self.informedness

    # Motivation for utilities measure, following Protection Motivation Theory
    def motivation(self, name):
        threat_appraisal = self.perceptions['probability'] * self.perceptions['economic_damage']/self.savings
        coping_appraisal = self.self_efficacy_perception * self.perceptions[name + '_efficacy'] / \
            (self.perceptions[name + '_cost']/self.savings)
        return min(threat_appraisal * coping_appraisal, 1)

    # Function to move utilities to higher storeys to lower flood damage (arbitrary 10% lowering of flood damage)
    def adap_measure(self,name):
        if self.savings > self.actual_values[f"{name}_cost"]:
            self.savings -= self.actual_values[f"{name}_cost"]
            self.money_spent += self.actual_values[f"{name}_cost"]
            self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated) * \
                                          (1 - self.actual_values[f"{name}_efficacy"])
            self.flood_damage_actual = calculate_basic_flood_damage(flood_depth=self.flood_depth_actual) * \
                                       (1 - self.actual_values[f"{name}_efficacy"])
            if name == "utilities":
                self.is_adapted_utilities = True
            elif name == "barrier":
                self.is_adapted_barrier = True
            elif name == "drainage":
                self.is_adapted_drainage = True

    def step(self):

        self.influence_of_friends(self.friend_radius)
        self.influence_of_informedness()

        if self.motivation('utilities') > self.model.motivation_threshold and \
                self.adaption_chance > random.uniform(0,1) and not self.is_adapted_utilities:

            self.adap_measure("utilities")
        if self.motivation('barrier') > self.model.motivation_threshold and \
                self.adaption_chance > random.uniform(0,1) and not self.is_adapted_barrier:

            self.adap_measure("barrier")
        if self.motivation('drainage') > self.model.motivation_threshold and \
                self.adaption_chance > random.uniform(0,1) and not self.is_adapted_drainage:

            self.adap_measure("drainage")

        if self.model.gov_activate and (self.model.gov_information_campaign or self.model.gov_subsidies):

            # Government tries to notify everyone in the area that is most prone to flooding to take measures.
            if self.model.gov_information_campaign:
                lower = self.model.settings["gov_information_lower"][self.model.settings_ind]
                upper = self.model.settings["gov_information_upper"][self.model.settings_ind]
                if (self.flood_depth_estimated > self.model.government.danger_level
                        and self.model.government.conviction_strength > random.random()):
                    val = random.uniform(lower, upper)  #Spread of information effectiveness
                    self.informedness += self.convincable_level * self.model.government.conviction_strength * val

                    if self.informedness > 1:
                        self.informedness = 1
                    if self.informedness < 0:
                        self.informedness = 0

        if self.model.gov_subsidies:
            chance = self.model.settings["gov_subsidy_chance"][self.model.settings_ind]
            percent_subsidy = self.model.settings["gov_percent_subsidy"][self.model.settings_ind]
            informedness_threshold = self.model.settings["subsidy_informedness_threshold"][self.model.settings_ind]
            if self.informedness >= informedness_threshold and chance > random.random():
                name = self.investment_keys[self.economic_optimum]
                if self.economic_optimum != 0:
                    if self.motivation(name) > self.model.motivation_threshold / 2:
                        if self.model.government.budget >= percent_subsidy * self.actual_values[f"{name}_cost"]\
                                and self.savings >= (1-percent_subsidy) * self.actual_values[f"{name}_cost"]:

                            self.model.government.budget -= percent_subsidy * self.actual_values[f"{name}_cost"]
                            self.savings += percent_subsidy * self.actual_values[f"{name}_cost"]
                            self.adap_measure(f"{name}")


# Define the Government agent class
class Government(Agent):
    """
    A government agent can perform a number of actions. One of them is subsidising
    """

    def __init__(self, unique_id, model, danger_level=0.2, conviction_strength=0.2,
                 budget=1e6,
                 risk_aversion=0.5
                 ):
        super().__init__(unique_id, model)
        self.type = "Government"
        self.danger_level = danger_level
        self.conviction_strength = conviction_strength
        self.budget = budget
        self.base_budget = budget
        self.risk_aversion = risk_aversion


    def step(self):
        pass



# More agent classes can be added here, e.g. for insurance agents.
