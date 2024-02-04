# Importing necessary libraries
import networkx as nx
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import geopandas as gpd
import rasterio as rs
import matplotlib.pyplot as plt
import random
import numpy as np

# Import the agent class(es) from agents.py
from agents import Households, Government


# Import functions from functions.py
from functions import get_flood_map_data, calculate_basic_flood_damage
from functions import map_domain_gdf, floodplain_gdf


# Define the AdaptationModel class
class AdaptationModel(Model):
    """
    The main model running the simulation. It sets up the network of household agents,
    simulates their behavior, and collects data. The network type can be adjusted based on study requirements.
    """

    def __init__(self, 
                 seed = None,
                 settings = None,
                 settings_ind = 0
                 # number_of_households = 25, # number of household agents
                 # # Simplified argument for choosing flood map. Can currently be "harvey", "100yr", or "500yr".
                 # flood_map_choice='harvey',
                 # # ### network related parameters ###
                 # # The social network structure that is used.
                 # # Can currently be "erdos_renyi", "barabasi_albert", "watts_strogatz", or "no_network"
                 # network = 'watts_strogatz',
                 # # likeliness of edge being created between two nodes
                 # probability_of_network_connection = 0.4,
                 # # number of edges for BA network
                 # number_of_edges = 3,
                 # # number of nearest neighbours for WS social network
                 # number_of_nearest_neighbours = 5,
                 #
                 # gov_subsidies = True,
                 # gov_information_campaign = True,
                 # gov_risk_aversion = 0.2,
                 # motivation_threshold = 0.3,
                 # flood_timestep = 5
                 ):
        
        super().__init__(seed = seed)
        self.settings = settings
        self.settings_ind = settings_ind
        # add flood map choice to class
        flood_map_choice = settings["flood_map_choice"][settings_ind]
        self.flood_timestep = settings["flood_timestep"][settings_ind]
        # defining the variables and setting the values
        self.number_of_households = settings["number_of_households"][settings_ind]  # Total number of household agents
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        # network
        self.network = settings["network"][settings_ind]  # Type of network to be created
        self.probability_of_network_connection = settings["probability_of_network_connection"][settings_ind]
        self.number_of_edges = settings["number_of_edges"][settings_ind]
        self.number_of_nearest_neighbours = settings["number_of_nearest_neighbours"][settings_ind]
        self.economic_damage_m2 = settings["economic_damage_m2"][settings_ind]
        # generating the graph according to the network used and the network parameters specified
        self.G = self.initialize_network()
        # create grid out of network graph
        self.grid = NetworkGrid(self.G)

        # Initialize maps
        self.initialize_maps(flood_map_choice)

        # Set flood risk occurence
        self.flood_event_occurrence = settings["flood_occurrence"][settings_ind]
            #how often does this occur in YEARS
            #Found a source that says that the rainfall equivalent has a 1-18 % chance of happening the further in time we go
            # https://www.pnas.org/doi/full/10.1073/pnas.1716222114


        self.motivation_threshold = settings["motivation_threshold"][settings_ind]
        # set schedule for agents
        self.schedule = RandomActivation(self)  # Schedule for activating agents

        # create households through initiating a household on each node of the network graph
        for i, node in enumerate(self.G.nodes()):
            household = Households(unique_id=i, model=self)
            household.set_economic_risk(settings["gov_risk_aversion"][settings_ind])
            self.schedule.add(household)
            self.grid.place_agent(agent=household, node_id=node)

        self.households = self.schedule.agents.copy()

        # Add a government agent
        self.government = Government(unique_id=-1,model=self, risk_aversion=settings["gov_risk_aversion"][settings_ind])
        self.schedule.add(self.government)
        self.gov_subsidies = settings["gov_subsidies"][settings_ind]
        self.gov_information_campaign = settings["gov_information_campaign"][settings_ind]
        self.gov_activate = False
        self.gov_interval = settings["gov_info_interval"][settings_ind]

        # Add economic risk

        # Data collection setup to collect data
        model_metrics = {
                        "total_adapted_households": self.total_adapted_households,
                        "total_adapted_utilities": self.total_adapted_utilities,
                        "total_adapted_barrier": self.total_adapted_barrier,
                        "total_adapted_drainage": self.total_adapted_drainage,
                        "total_costs": self.total_costs,
                        "total_house_size": self.total_house_size,
                        "total_savings": self.total_savings
                        # ... other reporters ...
                        }
        
        agent_metrics = {
                        "FloodDepthEstimated": "flood_depth_estimated",
                        "FloodDamageEstimated": "flood_damage_estimated",
                        "FloodDepthActual": "flood_depth_actual",
                        "FloodDamageActual": "flood_damage_actual",
                        #"IsAdapted": "is_adapted",
                        "FriendsCount": lambda a: a.count_friends(radius=1) if a.type == "Household" else None,
                        "location": "location",
                        "budget": "budget",
                        "savings": "savings",
                        "economic_risk": "economic_risk",
                        "is_adapted_utilities": "is_adapted_utilities",
                        "is_adapted_barrier": "is_adapted_barrier",
                        "is_adapted_drainage": "is_adapted_drainage",
                        "informedness": "informedness",
                        "utilities_efficacy": lambda a: a.perceptions["utilities_efficacy"] if a.type == "Household" else None,
                        "barrier_efficacy": lambda a: a.perceptions["barrier_efficacy"] if a.type == "Household" else None,
                        "drainage_efficacy": lambda a: a.perceptions["drainage_efficacy"] if a.type == "Household" else None,
                        "p_utilities_cost": lambda a: a.perceptions["utilities_cost"] if a.type == "Household" else None,
                        "p_barrier_cost": lambda a: a.perceptions["barrier_cost"] if a.type == "Household" else None,
                        "p_drainage_cost": lambda a: a.perceptions["drainage_cost"] if a.type == "Household" else None,
                        "actual_utilities_cost": lambda a: a.actual_values["utilities_cost"] if a.type == "Household" else None,
                        "actual_barrier_cost": lambda a: a.actual_values["barrier_cost"] if a.type == "Household" else None,
                        "actual_drainage_cost": lambda a: a.actual_values["drainage_cost"] if a.type == "Household" else None,
                        "motivation_utilities": lambda a: a.motivation("utilities") if a.type == "Household" else None,
                        "motivation_barrier": lambda a: a.motivation("barrier") if a.type == "Household" else None,
                        "motivation_drainage": lambda a: a.motivation("drainage") if a.type == "Household" else None,
                        "p_probability": lambda a: a.perceptions["probability"] if a.type == "Household" else None,
                        "p_economic_damage":  lambda a: a.perceptions["economic_damage"] if a.type == "Household" else None,
                        "actual_economic_damage": lambda a: a.actual_values["economic_damage"] if a.type == "Household" else None,
                        "house_size": "house_size",
                        "savings": "savings"
                        }
        #set up the data collector 
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)

    def initialize_network(self):
        """
        Initialize and return the social network graph based on the provided network type using pattern matching.
        """
        if self.network == 'erdos_renyi':
            return nx.erdos_renyi_graph(n=self.number_of_households,
                                        p=self.number_of_nearest_neighbours / self.number_of_households,
                                        seed=self.seed)
        elif self.network == 'barabasi_albert':
            return nx.barabasi_albert_graph(n=self.number_of_households,
                                            m=self.number_of_edges,
                                            seed=self.seed)
        elif self.network == 'watts_strogatz':
            return nx.watts_strogatz_graph(n=self.number_of_households,
                                        k=self.number_of_nearest_neighbours,
                                        p=self.probability_of_network_connection,
                                        seed=self.seed)
        elif self.network == 'no_network':
            G = nx.Graph()
            G.add_nodes_from(range(self.number_of_households))
            return G
        else:
            raise ValueError(f"Unknown network type: '{self.network}'. "
                            f"Currently implemented network types are: "
                            f"'erdos_renyi', 'barabasi_albert', 'watts_strogatz', and 'no_network'")


    def initialize_maps(self, flood_map_choice):
        """
        Initialize and set up the flood map related data based on the provided flood map choice.
        """
        # Define paths to flood maps
        flood_map_paths = {
            'harvey': r'../input_data/floodmaps/Harvey_depth_meters.tif',
            '100yr': r'../input_data/floodmaps/100yr_storm_depth_meters.tif',
            '500yr': r'../input_data/floodmaps/500yr_storm_depth_meters.tif'  # Example path for 500yr flood map
        }

        # Throw a ValueError if the flood map choice is not in the dictionary
        if flood_map_choice not in flood_map_paths.keys():
            raise ValueError(f"Unknown flood map choice: '{flood_map_choice}'. "
                             f"Currently implemented choices are: {list(flood_map_paths.keys())}")

        # Choose the appropriate flood map based on the input choice
        flood_map_path = flood_map_paths[flood_map_choice]

        # Loading and setting up the flood map
        self.flood_map = rs.open(flood_map_path)
        self.band_flood_img, self.bound_left, self.bound_right, self.bound_top, self.bound_bottom = get_flood_map_data(
            self.flood_map)

    def total_adapted_households(self):
        """Return the total number of households that have adapted."""
        #BE CAREFUL THAT YOU MAY HAVE DIFFERENT AGENT TYPES SO YOU NEED TO FIRST CHECK IF THE AGENT IS ACTUALLY A HOUSEHOLD AGENT USING "ISINSTANCE"
        #Fixed by using the self.households list
        adapted_count = sum(
            [
                1 for agent in self.households
                if (
                    agent.is_adapted_drainage
                    or agent.is_adapted_barrier
                    or agent.is_adapted_utilities
                )
            ]
        )
        return adapted_count

    def total_adapted_utilities(self):
        """Return the total number of households that have adapted utilities."""
        return sum([1 for agent in self.households if agent.is_adapted_utilities])

    def total_adapted_barrier(self):
        """Return the total number of households that have adapted barriers."""
        return sum([1 for agent in self.households if agent.is_adapted_barrier])

    def total_adapted_drainage(self):
        """Return the total number of households that have adapted drainage."""
        return sum([1 for agent in self.households if agent.is_adapted_drainage])

    def total_costs(self):
        return sum([(agent.economic_damage + agent.money_spent) for agent in self.households])

    def total_savings(self):
        return sum([agent.savings for agent in self.households])
    def total_house_size(self):
        return sum([agent.house_size for agent in self.households])
    def plot_model_domain_with_agents(self):
        fig, ax = plt.subplots()
        # Plot the model domain
        map_domain_gdf.plot(ax=ax, color='lightgrey')
        # Plot the floodplain
        floodplain_gdf.plot(ax=ax, color='lightblue', edgecolor='k', alpha=0.5)

        # Collect agent locations and statuses

        for agent in self.households:
            if agent.is_adapted_barrier == True:
                color = "blue"
            elif agent.is_adapted_utilities == True:
                color = "green"
            elif agent.is_adapted_drainage == True:
                color = "yellow"
            else:
                color = "red"

            ax.scatter(agent.location.x, agent.location.y, color=color, s=10, label=color.capitalize() if not ax.collections else "")
            ax.annotate(str(agent.unique_id), (agent.location.x, agent.location.y), textcoords="offset points", xytext=(0,1), ha='center', fontsize=9)
        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Red: not adapted, Blue, green, yellow: adapted")

        # Customize plot with titles and labels
        plt.title(f'Model Domain with Agents at Step {self.schedule.steps}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def step(self):
        """
        introducing a shock: 
        at time step 5, there will be a global flooding.
        This will result in actual flood depth. Here, we assume it is a random number
        between 0.5 and 1.2 of the estimated flood depth. In your model, you can replace this
        with a more sound procedure (e.g., you can devide the floop map into zones and 
        assume local flooding instead of global flooding). The actual flood depth can be 
        estimated differently
        """
        if self.schedule.steps == self.flood_timestep:
            lower = self.settings["flood_depth_actual_lower"][self.settings_ind]
            upper = self.settings["flood_depth_actual_upper"][self.settings_ind]
            update_factor = self.settings["flood_perceptions_update"][self.settings_ind]
            for agent in self.schedule.agents:
                if isinstance(agent, Households):
                    # Calculate the actual flood depth as a random number between 0.5 and 1.2 times the estimated flood depth
                    agent.flood_depth_actual = random.uniform(lower, upper) * agent.flood_depth_estimated
                    #Set estimated as equal to actual
                    agent.flood_depth_estimated = agent.flood_depth_actual
                    # calculate the actual flood damage given the actual flood depth

                    agent.flood_damage_actual = calculate_basic_flood_damage(agent.flood_depth_actual) * \
                                                (1 - agent.actual_values[
                                                    "utilities_efficacy"] * agent.is_adapted_utilities) * \
                                                (1 - agent.actual_values["barrier_efficacy"] * agent.is_adapted_barrier) \
                                                * \
                                                (1 - agent.actual_values["drainage_efficacy"] * agent.is_adapted_drainage)
                    agent.actual_values["economic_damage"] = agent.flood_damage_actual * \
                                                             agent.house_size * self.economic_damage_m2
                    agent.economic_damage += agent.actual_values["economic_damage"]
                    # Calculate the money of the agent based on the flood damage (ervanuit gaande dat ELine iets met money heeft gedaan)
                    #TODO: uitvogelen hoe het omtoveren van absolute damage naar monetary damage
                    #agent.money = agent.money - agent.flood_damage_actual * 5000 #misschien nog iets met huisgrootte
                    agent.perceptions['economic_damage'] = (agent.perceptions['economic_damage'] * update_factor +
                                                           (1 - update_factor) * agent.actual_values["economic_damage"])
                    #Hier actual probability ipv flood damage actual
                    agent.perceptions['probability'] = (agent.perceptions['probability'] * update_factor
                                                        + 1/(4*self.flood_event_occurrence) * (1 - update_factor))

        
        # Collect data and advance the model by one step
        self.datacollector.collect(self)

        if self.schedule.steps % self.gov_interval == self.gov_interval - 1:
            self.gov_activate = True
        elif self.schedule.steps % self.gov_interval == 0:
            self.gov_activate = False
        self.schedule.step()
