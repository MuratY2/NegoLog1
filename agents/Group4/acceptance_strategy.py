# agents/Group4/acceptance_strategy.py
import math
from nenv import Bid

def acceptance_strategy(
    preference,
    last_bid: Bid,
    t: float,
    my_reservation: float
) -> bool:
    """
    A time-based acceptance threshold that starts near 1.0 and goes down
    to ~0.6, never going below the reservation value.

    :param preference: Self preference object (i.e., self.preference)
    :param last_bid: The opponent's last offer
    :param t: Time in [0, 1]
    :param my_reservation: The agent's reservation value
    :return: True if the agent accepts, False otherwise
    """
    if not last_bid:
        return False

    # threshold = 0.6 + 0.4 * e^(-2.0 * t)
    threshold = 0.6 + 0.4 * (math.e ** (-2.0 * t))

    # Make sure we do not drop below the reservation value
    if threshold < my_reservation:
        threshold = my_reservation

    incoming_utility = preference.get_utility(last_bid)
    return (incoming_utility >= threshold)
