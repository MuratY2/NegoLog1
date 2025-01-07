# agents/Group4/opponent_model.py

from typing import List, Dict
from nenv.OpponentModel.AbstractOpponentModel import AbstractOpponentModel
from nenv.Preference import Preference
from nenv.Bid import Bid, Issue

class OpponentModel(AbstractOpponentModel):
    """
    A Frequency-Based Opponent Model with Time Decay:
    - Tracks frequency of issues/values in opponent bids.
    - Applies time decay to emphasize recent bids.
    - Dynamically updates issue and value weights.
    """

    def __init__(self, reference: Preference, decay_factor: float = 0.95):
        
        super().__init__(reference)

        self.alpha = 0.1
        self.decay_factor = decay_factor
        self.opponent_bids: List[Bid] = []

        # Dictionary for tracking how many times each issue changes or repeats
        # issue_counts[issue] -> float
        self.issue_counts: Dict[Issue, float] = {}
        # Dictionary for tracking how frequently each value is chosen
        # value_counts[issue][value_str] -> float
        self.value_counts: Dict[Issue, Dict[str, float]] = {}

        # Initialize counts from reference preference
        # self._pref is the parent's internal preference copy
        for issue in reference.issues:
            # Start with whatever weight the reference might have assigned
            self.issue_counts[issue] = self._pref[issue]

            self.value_counts[issue] = {}
            for value_str in issue.values:
                # Use the reference's initial weight for (issue, value)
                self.value_counts[issue][value_str] = self._pref[issue, value_str]

        # Normalize once at the beginning
        self._pref.normalize()

    @property
    def name(self) -> str:
        return "Time Decay Frequency Opponent Model"

    def update(self, bid: Bid, t: float):
        """
        Called when we receive a new opponent bid.
        We'll track repeated values, apply partial time decay,
        and then re-normalize preference weights.
        """
        if not bid:
            return

        self.opponent_bids.append(bid)

        # For every (issue, value) in the new bid:
        for issue, value_str in bid:
            # Bump the frequency for the chosen value
            if value_str not in self.value_counts[issue]:
                self.value_counts[issue][value_str] = 0.0
            self.value_counts[issue][value_str] += 1.0

            # If the last round had the same value for this issue, increment issue importance
            if len(self.opponent_bids) >= 2:
                previous_bid = self.opponent_bids[-2]
                if previous_bid[issue] == value_str:
                    # Weighted by alpha*(1 - t)*decay_factor
                    self.issue_counts[issue] += self.alpha * (1 - t) * self.decay_factor

        # Apply time decay, then update self._pref
        self.apply_time_decay(t)
        self.update_weights()

    def apply_time_decay(self, t: float):
        """
        Applies time-based decay to stored frequencies, emphasizing recent bids.
        e.g., if decay_factor=0.95, we might do (0.95^t) or something like that.
        """
        # We'll exponentiate the base decay_factor by t to get a partial effect
        decay = self.decay_factor ** t
        if decay < 0.8:  # You can choose a cutoff
            decay = 0.8

        # Decay the issue counts
        for issue in self.issue_counts:
            self.issue_counts[issue] *= decay

        # Decay the value counts
        for issue in self.value_counts:
            for value_str in self.value_counts[issue]:
                self.value_counts[issue][value_str] *= decay

    def update_weights(self):
        """
        Normalize the data into the parent's preference object (self._pref).
        """
        # 1) Summation for issues
        sum_issues = sum(self.issue_counts.values()) if self.issue_counts else 1.0
        if sum_issues == 0:
            sum_issues = 1.0

        # 2) Write back to self._pref for issues
        for issue in self._pref.issues:
            self._pref[issue] = self.issue_counts[issue] / sum_issues

            # For values, find the maximum frequency for that issue
            max_val_count = max(self.value_counts[issue].values()) if self.value_counts[issue] else 1.0
            if max_val_count == 0:
                max_val_count = 1.0

            # Normalize each value
            for val_str in issue.values:
                freq = self.value_counts[issue].get(val_str, 0.0)
                self._pref[issue, val_str] = freq / max_val_count

        # 3) Finally, let NegoLog finalize normalizing
        self._pref.normalize()

    def predict_preference(self, bid: Bid) -> float:
        """
        Return a predicted utility for the given bid,
        using the underlying self._pref. This is the parent's built-in
        linear additive approach, now updated by our frequency model.
        """
        return self._pref.get_utility(bid)

    # -----------------------------------------------------------------
    # Add these so your bidding_strategy code can do:
    #    opp_util = opponent_model.preference.get_utility(bid)
    # -----------------------------------------------------------------
    @property
    def preference(self):
        """
        The bidding strategy references `opponent_model.preference`
        as if it's a "fake preference" with `.get_utility(bid)`.
        We can just return `self`, but define get_utility(...) below.
        """
        return self

    def get_utility(self, bid: Bid) -> float:
        """So that `opponent_model.preference.get_utility(bid)` works."""
        return self.predict_preference(bid)
