from typing import Union
import nenv
from nenv import Action, Bid
from .acceptance_strategy import acceptance_strategy
from .bidding_strategy import bidding_strategy
from .opponent_model import OpponentModel

class Group4(nenv.AbstractAgent):
    """
    This agent demonstrates:
      - A custom acceptance strategy
      - A custom bidding strategy
      - An frequency-based opponent model with time decay
    """

    @property
    def name(self) -> str:
        return "Group4 Agent"

    # ---------------------------------------------------------
    # 1) INITIATION
    # ---------------------------------------------------------
    def initiate(self, opponent_name: Union[None, str]):
        """
        Called at the start of each negotiation session.
        We'll create or reset anything here.
        """
        # Use the OpponentModel with time decay:
        self.opponent_model = OpponentModel(self.preference)

        # Optional: store the opponent's name
        self.opponent_name = opponent_name

        # For acceptance strategy / bidding strategy parameters
        self.my_reservation = self.preference.reservation_value

    # ---------------------------------------------------------
    # 2) OPPONENT MODEL UPDATE
    # ---------------------------------------------------------
    def update_opponent_model(self, bid: Bid, t: float):
        self.opponent_model.update(bid, t)

    # ---------------------------------------------------------
    # 3) ACCEPTANCE STRATEGY
    # ---------------------------------------------------------
    def acceptance_strategy(self, last_bid: Bid, t: float) -> bool:
        return acceptance_strategy(
            self.preference,
            last_bid,
            t,
            self.my_reservation
        )

    # ---------------------------------------------------------
    # 4) BIDDING STRATEGY
    # ---------------------------------------------------------
    def bidding_strategy(self, t: float) -> Bid:
        return bidding_strategy(
            self.preference,
            self.opponent_model,
            t,
            self.my_reservation
        )

    # ---------------------------------------------------------
    # 5) receive_offer (hook)
    # ---------------------------------------------------------
    def receive_offer(self, bid: Bid, t: float):
        self.update_opponent_model(bid, t)

    # ---------------------------------------------------------
    # 6) act (hook)
    # ---------------------------------------------------------
    def act(self, t: float) -> Action:
        last_bid = self.last_received_bids[-1] if self.last_received_bids else None

        # If we accept the last offer
        if last_bid and self.acceptance_strategy(last_bid, t):
            return self.accept_action

        # Otherwise propose a new bid
        proposed_bid = self.bidding_strategy(t)
        if proposed_bid is None:
            return self.accept_action

        return nenv.Offer(proposed_bid)
