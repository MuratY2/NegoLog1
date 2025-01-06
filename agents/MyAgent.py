from typing import Union

import nenv
from nenv import Action, Bid


class MyAgent(nenv.AbstractAgent):
    @property
    def name(self) -> str:
        return "My Agent"

    def initiate(self, opponent_name: Union[None, str]):
        self.opponent_model = nenv.OpponentModel.ClassicFrequencyOpponentModel(self.preference)

    def receive_offer(self, bid: Bid, t: float):
        self.opponent_model.update(bid, t)

    def act(self, t: float) -> Action:
        # Target Utility [1.0, 0.5]
        reservation_value = self.preference.reservation_value

        target_utility = 1.0 - t * 0.5

        if target_utility < reservation_value:
            target_utility = reservation_value

        # AC_Next
        if self.can_accept() and self.last_received_bids[-1].utility >= target_utility:
            return self.accept_action

        estimated_preference = self.opponent_model.preference

        # Target Utility = 0.8, 0.8 - 0.03 -> 0.8 + 0.05
        candidate_bids = self.preference.get_bids_at(target_utility, 0.03, 0.05)

        my_bid = None
        utility = -1

        for bid in candidate_bids:
            estimated_opp_utility = estimated_preference.get_utility(bid)

            if estimated_opp_utility > utility:
                utility = estimated_opp_utility
                my_bid = bid

        return nenv.Offer(my_bid)
