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
    A time-based threshold from ~1.0 down to ~0.6 for the agent's own utility,
    then choose a bid that balances self-utility and opponent-utility.

    We'll define a combined score:
        score = alpha * my_utility + (1 - alpha) * opp_utility

    We'll keep alpha=0.7 by default, so we weigh our utility 70%
    and the opponent's 30%.

    :param preference: Self preference object
    :param opponent_model: Opponent model object (which has .preference)
    :param t: Time in [0, 1]
    :param my_reservation: The agent's reservation value
    :return: A single bid
    """
    alpha = 0.7

    # Time-based target for self utility
    target = 0.6 + 0.4 * (math.e ** (-2.0 * t))
    if target < my_reservation:
        target = my_reservation

    # ±0.05 around the target
    candidate_bids = preference.get_bids_at(target, 0.05, 0.05)
    if not candidate_bids:
        # fallback to near reservation
        candidate_bids = preference.get_bids_at(my_reservation, 0.0, 0.1)
        if not candidate_bids:
            # ultimate fallback: random from entire space
            all_bids = preference.all_bids
            return random.choice(all_bids) if all_bids else None

    best_bid = None
    best_score = float("-inf")

    for b in candidate_bids:
        my_util = preference.get_utility(b)
        opp_util = opponent_model.preference.get_utility(b)
        combined_score = alpha * my_util + (1 - alpha) * opp_util

        if combined_score > best_score:
            best_score = combined_score
            best_bid = b

    return best_bid
