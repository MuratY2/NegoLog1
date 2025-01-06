# agents/Group4/opponent_model.py
import nenv
from nenv import Bid

def create_opponent_model(preference):
    """
    Create the base model (ClassicFrequencyOpponentModel from nenv).
    """
    return nenv.OpponentModel.ClassicFrequencyOpponentModel(preference)

def update_opponent_model(opponent_model, bid: Bid, t: float):
    """
    Extend ClassicFrequencyOpponentModel with mild 'decay'
    to emphasize more recent bids.

    :param opponent_model: the instance of ClassicFrequencyOpponentModel
    :param bid: the Bid from the opponent
    :param t: time in [0, 1]
    """
    # 1) Standard frequency counting
    opponent_model.update(bid, t)

    # 2) Time-decay example: multiply frequencies by 0.99..0.90 depending on t
    decay_factor = 0.99 - 0.05 * t  # ~0.99 at t=0, ~0.95 at t=1
    if decay_factor < 0.90:
        decay_factor = 0.90

    try:
        for issue in opponent_model.issue_frequency:
            opponent_model.issue_frequency[issue] *= decay_factor

        for issue in opponent_model.value_frequency:
            for val in opponent_model.value_frequency[issue]:
                opponent_model.value_frequency[issue][val] *= decay_factor
    except AttributeError:
        # The model might not have the same fields, or at initialization time
        pass
