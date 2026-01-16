"""Time Series Segmentation Evaluation Metrics.

This module provides comprehensive evaluation metrics for time series segmentation
tasks, specifically focused on boundary detection problems. It implements standard
information retrieval metrics (precision, recall, F1-score) adapted for temporal
boundary detection with configurable tolerance windows.

The evaluator handles the matching of predicted boundaries to ground truth boundaries
within specified tolerance ranges, ensuring fair evaluation of segmentation algorithms
that may have slight temporal offsets.

Author: Hugo Demule
Date: January 2026
"""


class TS_Evaluator:
    """Time Series Segmentation Evaluator for boundary detection tasks.

    This class provides comprehensive evaluation metrics for time series segmentation
    algorithms, particularly those focused on detecting temporal boundaries. It implements
    tolerance-based matching between predicted and ground truth boundaries, computing
    standard classification metrics adapted for temporal data.

    The evaluator uses a configurable tolerance window to determine if predicted
    boundaries match ground truth boundaries, handling the inherent uncertainty
    in exact temporal boundary placement.

    Attributes:
        tolerance_seconds (float): Tolerance window in seconds for boundary matching.

    Example:
        >>> evaluator = TS_Evaluator(tolerance_seconds=0.5)
        >>> y_true = [10.0, 20.0, 30.0]  # Ground truth boundaries
        >>> y_pred = [9.8, 20.2, 35.0]  # Predicted boundaries
        >>> metrics = evaluator.evaluate(y_true, y_pred)
        >>> print(f"F1 Score: {metrics['f1_score']:.3f}")

    Note:
        All boundary timestamps should be provided in seconds as floating-point numbers.
        The tolerance window is applied symmetrically around each ground truth boundary.
    """

    def __init__(self, tolerance_seconds=15.0):
        """Initialize the TS_Evaluator with a specified tolerance window.

        Args:
            tolerance_seconds (float): Tolerance window in seconds for boundary
                matching. A predicted boundary is considered a match if it falls
                within ±tolerance_seconds of a ground truth boundary.
                Defaults to 15 seconds.
        """
        self.tolerance_seconds = tolerance_seconds

    def true_positives(self, y_true, y_pred):
        """Count true positives using tolerance-based boundary matching.

        Each ground truth boundary can be matched by at most one prediction
        within the tolerance window. If multiple predictions fall within the
        tolerance of a single ground truth boundary, only the closest one
        is considered a match.

        Args:
            y_true (list): Ground truth boundary timestamps in seconds.
            y_pred (list): Predicted boundary timestamps in seconds.

        Returns:
            int: Number of true positive detections.

        Note:
            This implements a greedy matching strategy where each ground truth
            boundary is matched with its closest prediction within tolerance.
        """
        matched_pred = set()
        num_tp = 0

        for true_time in y_true:
            # Find all unmatched predictions within tolerance
            candidates = [
                i
                for i, pred_time in enumerate(y_pred)
                if i not in matched_pred
                and abs(pred_time - true_time) <= self.tolerance_seconds
            ]
            if candidates:
                # Match the closest prediction (optional, but standard)
                closest_idx = min(candidates, key=lambda i: abs(y_pred[i] - true_time))
                matched_pred.add(closest_idx)
                num_tp += 1

        return num_tp

    def false_positives(self, y_true, y_pred):
        """Count false positives: predicted boundaries with no matching ground truth.

        A predicted boundary is considered a false positive if it does not fall
        within the tolerance window of any ground truth boundary.

        Args:
            y_true (list): Ground truth boundary timestamps in seconds.
            y_pred (list): Predicted boundary timestamps in seconds.

        Returns:
            int: Number of false positive detections.
        """
        num_fp = 0

        for pred_time in y_pred:
            # Check if this prediction is NOT within tolerance of any reference boundary
            is_match = False
            for true_time in y_true:
                if abs(pred_time - true_time) <= self.tolerance_seconds:
                    is_match = True
                    break

            if not is_match:
                num_fp += 1

        return num_fp

    def false_negatives(self, y_true, y_pred):
        """Count false negatives: ground truth boundaries with no matching prediction.

        A ground truth boundary is considered a false negative if no predicted
        boundary falls within its tolerance window.

        Args:
            y_true (list): Ground truth boundary timestamps in seconds.
            y_pred (list): Predicted boundary timestamps in seconds.

        Returns:
            int: Number of false negative detections (missed boundaries).
        """
        num_fn = 0

        for true_time in y_true:
            # Check if this reference boundary has NO prediction within tolerance
            is_detected = False
            for pred_time in y_pred:
                if abs(pred_time - true_time) <= self.tolerance_seconds:
                    is_detected = True
                    break

            if not is_detected:
                num_fn += 1

        return num_fn

    def precision(self, y_true, y_pred):
        """Calculate precision: TP / (TP + FP).

        Precision measures the fraction of predicted boundaries that are correct.
        A high precision indicates few false positive detections.

        Args:
            y_true (list): Ground truth boundary timestamps in seconds.
            y_pred (list): Predicted boundary timestamps in seconds.

        Returns:
            float: Precision score between 0.0 and 1.0. Returns 0.0 if no
                   predictions are made (TP + FP = 0).
        """
        num_tp = self.true_positives(y_true, y_pred)
        num_fp = self.false_positives(y_true, y_pred)

        if num_tp + num_fp == 0:
            return 0.0

        return num_tp / (num_tp + num_fp)

    def recall(self, y_true, y_pred):
        """Calculate recall: TP / (TP + FN).

        Recall measures the fraction of ground truth boundaries that are correctly
        detected. A high recall indicates few missed boundaries.

        Args:
            y_true (list): Ground truth boundary timestamps in seconds.
            y_pred (list): Predicted boundary timestamps in seconds.

        Returns:
            float: Recall score between 0.0 and 1.0. Returns 0.0 if no
                   ground truth boundaries exist (TP + FN = 0).
        """
        num_tp = self.true_positives(y_true, y_pred)
        num_fn = self.false_negatives(y_true, y_pred)

        if num_tp + num_fn == 0:
            return 0.0

        return num_tp / (num_tp + num_fn)

    def f1_score(self, y_true, y_pred):
        """Calculate F1 score: 2 * (precision * recall) / (precision + recall).

        F1 score is the harmonic mean of precision and recall, providing a single
        metric that balances both measures. It's particularly useful when you need
        to consider both false positives and false negatives equally.

        Args:
            y_true (list): Ground truth boundary timestamps in seconds.
            y_pred (list): Predicted boundary timestamps in seconds.

        Returns:
            float: F1 score between 0.0 and 1.0. Returns 0.0 if both precision
                   and recall are 0.0.
        """
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def weighted_f1_score(self, y_true, y_pred, beta=0.5):
        """Calculate weighted F1 score (F-beta score).

        The F-beta score allows weighting the relative importance of precision
        versus recall. This is useful when one type of error (false positives
        or false negatives) is more costly than the other.

        Formula: (1 + β²) * (precision * recall) / (β² * precision + recall)

        Args:
            y_true (list): Ground truth boundary timestamps in seconds.
            y_pred (list): Predicted boundary timestamps in seconds.
            beta (float): Weighting factor for recall relative to precision.
                - beta < 1.0: Precision weighted more heavily
                - beta = 1.0: Equal weighting (standard F1 score)
                - beta > 1.0: Recall weighted more heavily
                Defaults to 0.5 (precision-focused).

        Returns:
            float: Weighted F1 score between 0.0 and 1.0.

        Example:
            >>> evaluator = TS_Evaluator()
            >>> # Precision-focused evaluation
            >>> f_half = evaluator.weighted_f1_score(y_true, y_pred, beta=0.5)
            >>> # Recall-focused evaluation
            >>> f2 = evaluator.weighted_f1_score(y_true, y_pred, beta=2.0)
        """
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)

        beta_squared = beta**2

        if (beta_squared * precision + recall) == 0:
            return 0.0

        return (
            (1 + beta_squared)
            * precision
            * recall
            / (beta_squared * precision + recall)
        )

    def evaluate(self, y_true, y_pred):
        """Perform comprehensive evaluation returning all metrics.

        Computes all evaluation metrics in a single call, providing a complete
        assessment of boundary detection performance.

        Args:
            y_true (list): Ground truth boundary timestamps in seconds.
            y_pred (list): Predicted boundary timestamps in seconds.

        Returns:
            dict: Dictionary containing all evaluation metrics:
                - 'precision': Precision score (0.0 to 1.0)
                - 'recall': Recall score (0.0 to 1.0)
                - 'f1_score': F1 score (0.0 to 1.0)
                - 'true_positives': Number of true positives (int)
                - 'false_positives': Number of false positives (int)
                - 'false_negatives': Number of false negatives (int)
                - 'tolerance_seconds': Tolerance window used (float)

        Example:
            >>> evaluator = TS_Evaluator(tolerance_seconds=0.5)
            >>> metrics = evaluator.evaluate([10.0, 20.0], [9.8, 20.2, 30.0])
            >>> print(f"Precision: {metrics['precision']:.3f}")
            >>> print(f"Recall: {metrics['recall']:.3f}")
        """
        num_tp = self.true_positives(y_true, y_pred)
        num_fp = self.false_positives(y_true, y_pred)
        num_fn = self.false_negatives(y_true, y_pred)

        precision, recall, f1 = (
            self.precision(y_true, y_pred),
            self.recall(y_true, y_pred),
            self.f1_score(y_true, y_pred),
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": num_tp,
            "false_positives": num_fp,
            "false_negatives": num_fn,
            "tolerance_seconds": self.tolerance_seconds,
        }

    def plot_evaluation(self, y_true, y_pred, save_path=None, figsize=(12, 4)):
        """Create a visualization of boundary detection evaluation results.

        Generates a comprehensive plot showing ground truth boundaries, tolerance
        bands, and predicted boundaries classified as true positives or false
        positives. The plot includes performance metrics in the title.

        Args:
            y_true (list): Ground truth boundary timestamps in seconds.
            y_pred (list): Predicted boundary timestamps in seconds.
            save_path (str, optional): Path to save the plot. If None, displays
                the plot instead of saving. Defaults to None.
            figsize (tuple): Figure size as (width, height) in inches.
                Defaults to (12, 4).

        Note:
            Requires matplotlib to be installed. The plot shows:
            - Blue shaded regions: Tolerance bands around ground truth boundaries
            - Black lines: Ground truth boundaries
            - Green lines: True positive predictions
            - Red lines: False positive predictions
            - Title includes precision, recall, F1 score, and false negative count

        Example:
            >>> evaluator = TS_Evaluator(tolerance_seconds=0.5)
            >>> evaluator.plot_evaluation(y_true, y_pred, save_path="eval.png")
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        # Plot tolerance bands around true boundaries
        for true_time in y_true:
            ax.axvspan(
                true_time - self.tolerance_seconds,
                true_time + self.tolerance_seconds,
                alpha=0.2,
                color="blue",
                label="Tolerance Band" if true_time == y_true[0] else "",
            )

        # Plot true boundaries as dark vertical lines
        ax.eventplot(
            y_true,
            lineoffsets=2,
            colors="black",
            linewidths=3,
            linelengths=0.3,
            label="True Boundaries",
        )

        # Classify predictions as TP or FP and plot accordingly
        tp_predictions = []
        fp_predictions = []

        for pred_time in y_pred:
            is_tp = False
            # Check if this prediction is within tolerance of any true boundary
            for true_time in y_true:
                if abs(pred_time - true_time) <= self.tolerance_seconds:
                    is_tp = True
                    break

            if is_tp:
                tp_predictions.append(pred_time)
            else:
                fp_predictions.append(pred_time)

        # Plot True Positives as green lines
        if tp_predictions:
            ax.eventplot(
                tp_predictions,
                lineoffsets=1,
                colors="green",
                linewidths=2,
                linelengths=0.4,
                label=f"True Positives ({len(tp_predictions)})",
            )

        # Plot False Positives as red lines
        if fp_predictions:
            ax.eventplot(
                fp_predictions,
                lineoffsets=0.5,
                colors="red",
                linewidths=2,
                linelengths=0.4,
                label=f"False Positives ({len(fp_predictions)})",
            )

        # Calculate and display metrics
        metrics = self.evaluate(y_true, y_pred)
        num_fn = metrics["false_negatives"]

        # Set plot properties
        ax.set_ylim(0, 2.5)
        ax.set_yticks([0.5, 1, 2], ["FP", "TP", "True"])
        ax.set_xlabel("Time (s)")
        ax.set_title(
            f"Boundary Detection Evaluation (Tolerance: ±{self.tolerance_seconds}s)\n"
            f"P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, "
            f"F1: {metrics['f1_score']:.3f}, FN: {num_fn}"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
