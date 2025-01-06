from typing import List, Dict
from nenv.OpponentModel.AbstractOpponentModel import AbstractOpponentModel
from nenv.Preference import Preference
from nenv.Bid import Bid, Issue

class OpponentModel(AbstractOpponentModel):
    """
    Time andFrequency-Based Opponent Model:
    - Tracks the frequency of issues and values in opponent bids.
    - Applies a time decay factor to prioritize recent bids.
    - Dynamically ranks the opponent's preferred values and trends.
    """

    issue_counts: Dict[Issue, float]                #: Tracks the number of changes for each issue.
    value_counts: Dict[Issue, Dict[str, float]]     #: Tracks the frequency of each value under each issue.
    alpha: float                                   #: Learning rate for issue importance.
    decay_factor: float                            #: Time decay factor for weighting updates.
    opponent_bids: List[Bid]                       #: List of received bids.

    def __init__(self, reference: Preference, decay_factor: float = 0.95):
        super().__init__(reference)

        self.alpha = 0.1
        self.decay_factor = decay_factor
        self.opponent_bids = []
        self.issue_counts = {}
        self.value_counts = {}

        # Initialize counts based on reference preference
        for issue in reference.issues:
            self.issue_counts[issue] = self._pref[issue]
            self.value_counts[issue] = {}
            for value in issue.values:
                self.value_counts[issue][value] = self._pref[issue, value]

        self._pref.normalize()

    @property
    def name(self) -> str:
        return "Opponent Model"

    def update(self, bid: Bid, t: float):
        """
        Updates the model with the opponent's latest bid.

        :param bid: The opponent's bid.
        :param t: Time in [0, 1] indicating the progress of the negotiation.
        """
        self.opponent_bids.append(bid)

        # Update value and issue frequencies
        for issue, value in bid:
            self.value_counts[issue][value] += 1

            # Increment issue importance for repeated values with time decay
            if len(self.opponent_bids) >= 2 and self.opponent_bids[-2][issue] == value:
                self.issue_counts[issue] += self.alpha * (1 - t) * self.decay_factor

        self.apply_time_decay(t)
        self.update_weights()

    def apply_time_decay(self, t: float):
        """
        Applies time decay to issue and value counts to prioritize recent bids.

        :param t: Time in [0, 1].
        """
        decay = self.decay_factor ** t

        for issue in self.issue_counts:
            self.issue_counts[issue] *= decay

        for issue in self.value_counts:
            for value in self.value_counts[issue]:
                self.value_counts[issue][value] *= decay

    def update_weights(self):
        """
        Updates the preference weights based on normalized issue and value counts.
        """
        sum_issues = sum(self.issue_counts.values())

        for issue in self._pref.issues:
            # Normalize issue weights
            self._pref[issue] = self.issue_counts[issue] / sum_issues if sum_issues > 0 else 0

            max_value = max(self.value_counts[issue].values(), default=1)

            for value in issue.values:
                # Normalize value weights
                self._pref[issue, value] = self.value_counts[issue][value] / max_value if max_value > 0 else 0

        self._pref.normalize()

    def predict_preference(self, bid: Bid) -> float:
        """
        Predicts the utility of a given bid based on the opponent's preferences.

        :param bid: The bid to evaluate.
        :return: Predicted utility of the bid.
        """
        return self._pref.get_utility(bid)

    def get_preferred_values(self) -> Dict[Issue, str]:
        """
        Returns the most preferred value for each issue based on observed frequencies.

        :return: A dictionary of issues and their most preferred values.
        """
        preferred_values = {}
        for issue, values in self.value_counts.items():
            preferred_values[issue] = max(values, key=values.get, default=None)
        return preferred_values

    def analyze_trends(self) -> Dict[Issue, str]:
        """
        Analyzes trends in opponent preferences based on changes in bids.

        :return: A dictionary of issues and trends (e.g., increasing, decreasing, stable).
        """
        trends = {}
        for issue in self.issue_counts:
            if len(self.opponent_bids) < 2:
                trends[issue] = "stable"
            else:
                last_value = self.opponent_bids[-1][issue]
                second_last_value = self.opponent_bids[-2][issue]
                if last_value == second_last_value:
                    trends[issue] = "stable"
                else:
                    trends[issue] = "changing"
        return trends