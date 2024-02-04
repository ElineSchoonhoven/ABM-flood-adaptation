import numpy as np

def create_flood_probability_distr(occurrence, flood_height, parameter=0.8,
                                   max_height=10, steps=100, plot=False):
    """Helper function that creates an exponential probability distribution based on the input datapoint.

    It first scales the input flood heights so that the distribution function behaves nicely.
    Then, the probabilities are calculated using the passed (or default) parameter, and returned.

    Parameters
    -----------
    occurrence: int
        How often the flood event takes place in years.

    flood_height: float
        The height of the flood event

    parameter: float, optional
        The parameter of the exponential distribution function.

        This parameter alters the steepness of the distribution.
        See https://en.wikipedia.org/wiki/Exponential_distribution for more information.
        0.8 is used as a default.

    max_height: float, optional
        The maximum flood height considered, in meters.
        This, together with steps, is used generate the different flood heights for which the function is calculated.
        10 m is used as a default.

    steps: int, optional
        The number of intervals of flood heigths (between 0 and max_height) to return the distribution for.
        100 is chosen as a default.

    plot: bool, optional
        If true, the resultant probability distribution is plotted.

    Returns
    ----------
    distr: numpy array
        The resultant distribution in a 1D ordered numpy array of lenght steps.
        """

    # We scale the flood_height to work with an exponential probability distribution
    distr_flood_height = np.log(parameter * occurrence) / parameter
    flood_distr_scale = flood_height / distr_flood_height
    distr_flood_heights = np.linspace(0, max_height * flood_distr_scale, steps)
    distr = parameter * np.exp(-parameter * distr_flood_heights)
    if plot:
        plt.plot(distr_flood_heights * flood_distr_scale, distr)

    return distr


def calculate_economic_risk(damage_function, investments_costs, investments_damage_reduction_factors, risk_aversion,
                            flood_heights=None,
                            probabilities=None,
                            flood_event_occurrence=None,
                            flood_event_height=None,
                            max_flood_height=10,
                            flood_height_steps=15,
                            discount_rate=1.05,
                            exp_distr_parameter=0.8,
                            damage_function_kws = None):
    """Function that calculates the economic risk of different investment options, given flooding
    and agent risk aversion information.

    This calculation is based on chapter 4 of the paper by Bas Jonkman et al., (2002) titled
    'An overview of quantitative risk measures and their application for calculation of flood risk'.
    The user has the freedom to use this function with either a single flooding event and its occurence,
    or with a set of flooding events and their probabilities.
    TODO: type the parameter docstrings
    Parameters
    ----------
    damage_function: function
        Explanation
    investment_costs: array

    investment_damage_reduction_factors: array

    risk_aversion: float

    flood_heights: array

    probabilities: array

    flood_event_occurrence: int

    flood_event_height: float

    max_flood_height=: float

    flood_height_steps: int

    discount_rate=1.05: float

    exp_distr_parameter: float

    Returns
    ----------
    total_cost_distr: array
        Total expected cost for all the investments passed.

    optimum_ind: int
        Index of the economic optimum in the total_cost_dist array.

    Raises
    ---------
    ValueError
        If the input provided does not match one of the two input options.

    """

    # Check if the right combination of conditional arguments are passed for our two use cases.
    if flood_heights is None:
        flood_heights = np.linspace(0, max_flood_height, flood_height_steps)
        if flood_event_occurrence is None or flood_event_height is None:
            raise ValueError("When using a single flooding event, "
                             "flood_event_height and flood_event_occurence have to be specified.")
    elif probabilities is None:
        raise ValueError("When a predetermined set of flood heights is used, "
                         "their probabilities also have to be passed.\n"
                         "This is only recommended when using real world data for both.")

    if probabilities is None:
        probabilities = create_flood_probability_distr(flood_event_occurrence, flood_event_height,
                                                       parameter=exp_distr_parameter, steps=flood_height_steps)

    damages = damage_function(flood_heights, damage_function_kws)
    base_expected_economic_damages = probabilities * damages / discount_rate

    total_cost = investments_costs \
                 + base_expected_economic_damages[:, np.newaxis] * investments_damage_reduction_factors

    total_cost_distr = np.mean(total_cost, axis=0) + risk_aversion * np.std(total_cost, axis=0)

    # Finding the economic optimum (lowest total cost)
    optimum_ind = np.argmin(total_cost_distr)

    return total_cost_distr, optimum_ind


if __name__ == "__main__":
    from functions import calculate_basic_flood_damage
    import matplotlib.pyplot as plt

    # Testing the code
    par = 0.6
    occurence = 100
    flood_height = 6

    investments = {"cost": np.array([0.2, 0.1, 0.05]),
                   "damage_reduction": np.array([0.7, 0.8, 0.9])
                   }
    cost, opt_ind = calculate_economic_risk(calculate_basic_flood_damage, investments["cost"],
                                            investments["damage_reduction"], 0.8,
                                            flood_event_height=3, flood_event_occurrence=100)

    print(f"outcomes: {cost} cost, {opt_ind} optimum ind")
