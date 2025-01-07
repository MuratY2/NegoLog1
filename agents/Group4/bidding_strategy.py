# agents/Group4/bidding_strategy.py
import math
import random
from nenv import Bid

def bidding_strategy(
    preference,
    opponent_model,
    t: float,
    my_reservation: float
) -> Bid:
    """
    A time-based target from ~1.0 down to ~0.6, then pick a candidate that
    maximizes the (estimated) opponent utility.

    :param preference: Self preference object
    :param opponent_model: Opponent model object (which has .preference)
    :param t: Time in [0, 1]
    :param my_reservation: The agent's reservation value
    :return: A single bid
    """
    target = 0.6 + 0.4 * (math.e ** (-2.0 * t))
    if target < my_reservation:
        target = my_reservation

    # ±0.05 around target
    candidate_bids = preference.get_bids_at(target, 0.05, 0.05)
    if not candidate_bids:
        # fallback to near reservation
        candidate_bids = preference.get_bids_at(my_reservation, 0.0, 0.1)
        if not candidate_bids:
            # ultimate fallback: random
            all_bids = preference.all_bids
            return random.choice(all_bids) if all_bids else None

    estimated_pref = opponent_model.preference
    best_bid = None
    best_opp_utility = float("-inf")

    for b in candidate_bids:
        # get_utility(b) is now valid on estimated_pref
        opp_u = estimated_pref.get_utility(b)
        if opp_u > best_opp_utility:
            best_opp_utility = opp_u
            best_bid = b

    return best_bid
