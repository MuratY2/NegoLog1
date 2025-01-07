# agents/Group4/opponent_model.py

from typing import List, Dict
from nenv.OpponentModel.AbstractOpponentModel import AbstractOpponentModel
from nenv.Preference import Preference
from nenv.Bid import Bid, Issue

class OpponentModel(AbstractOpponentModel):
    """
    A Frequency-Based Opponent Model with Time Decay:

    """

    def __init__(self, reference: Preference, decay_factor: float = 0.95):
        
        super().__init__(reference)

        self.alpha = 0.1
        self.decay_factor = decay_factor
        self.opponent_bids: List[Bid] = []

        self.issue_counts: Dict[Issue, float] = {}
        self.value_counts: Dict[Issue, Dict[str, float]] = {}

        for issue in reference.issues:
            self.issue_counts[issue] = self._pref[issue]

            self.value_counts[issue] = {}
            for value_str in issue.values:
                self.value_counts[issue][value_str] = self._pref[issue, value_str]

        self._pref.normalize()

    @property
    def name(self) -> str:
        return "Time Decay Frequency Opponent Model"

    def update(self, bid: Bid, t: float):
        """
        Called when we receive a new opponent bid.

        """
        if not bid:
            return

        self.opponent_bids.append(bid)

        for issue, value_str in bid:
            if value_str not in self.value_counts[issue]:
                self.value_counts[issue][value_str] = 0.0
            self.value_counts[issue][value_str] += 1.0

            if len(self.opponent_bids) >= 2:
                previous_bid = self.opponent_bids[-2]
                if previous_bid[issue] == value_str:
                    self.issue_counts[issue] += self.alpha * (1 - t) * self.decay_factor

        self.apply_time_decay(t)
        self.update_weights()

    def apply_time_decay(self, t: float):
        """
        Applies time-based decay to stored frequencies
        """
        decay = self.decay_factor ** t
        if decay < 0.8:  
            decay = 0.8

        for issue in self.issue_counts:
            self.issue_counts[issue] *= decay

        for issue in self.value_counts:
            for value_str in self.value_counts[issue]:
                self.value_counts[issue][value_str] *= decay

    def update_weights(self):
        """
        Normalize the data into the parent's preference object .
        """
        sum_issues = sum(self.issue_counts.values()) if self.issue_counts else 1.0
        if sum_issues == 0:
            sum_issues = 1.0

        for issue in self._pref.issues:
            self._pref[issue] = self.issue_counts[issue] / sum_issues

            max_val_count = max(self.value_counts[issue].values()) if self.value_counts[issue] else 1.0
            if max_val_count == 0:
                max_val_count = 1.0

            for val_str in issue.values:
                freq = self.value_counts[issue].get(val_str, 0.0)
                self._pref[issue, val_str] = freq / max_val_count

        self._pref.normalize()

    def predict_preference(self, bid: Bid) -> float:
        """
        Return a predicted utility for the given bid
        """
        return self._pref.get_utility(bid)


    @property
    def preference(self):
        """
        The bidding strategy references `opponent_model.preference`

        """
        return self

    def get_utility(self, bid: Bid) -> float:
        """So that `opponent_model.preference.get_utility(bid)` works."""
        return self.predict_preference(bid)
