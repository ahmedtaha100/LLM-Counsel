from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationBucket:
    confidence_range: tuple[float, float]
    total: int
    correct: int
    accuracy: float
    avg_confidence: float


@dataclass
class CalibrationResult:
    buckets: list[CalibrationBucket]
    ece: float
    mce: float
    overconfident: bool


class CalibrationAnalyzer:
    def __init__(self, num_buckets: int = 10):
        self._num_buckets = num_buckets

    def analyze(
        self,
        confidences: list[float],
        correctness: list[bool],
    ) -> CalibrationResult:
        if not confidences or len(confidences) != len(correctness):
            return CalibrationResult(
                buckets=[],
                ece=0.0,
                mce=0.0,
                overconfident=False,
            )

        conf_arr = np.array(confidences)
        corr_arr = np.array(correctness)

        boundaries = np.linspace(0, 1, self._num_buckets + 1)
        buckets = []
        ece_sum = 0.0
        mce = 0.0

        for i in range(self._num_buckets):
            low, high = boundaries[i], boundaries[i + 1]
            mask = (conf_arr >= low) & (conf_arr < high)

            if high == 1.0:
                mask = mask | (conf_arr == 1.0)

            bucket_conf = conf_arr[mask]
            bucket_corr = corr_arr[mask]

            if len(bucket_conf) == 0:
                continue

            avg_conf = float(np.mean(bucket_conf))
            acc = float(np.mean(bucket_corr))
            gap = abs(avg_conf - acc)

            ece_sum += gap * len(bucket_conf)
            mce = max(mce, gap)

            buckets.append(
                CalibrationBucket(
                    confidence_range=(low, high),
                    total=len(bucket_conf),
                    correct=int(np.sum(bucket_corr)),
                    accuracy=acc,
                    avg_confidence=avg_conf,
                )
            )

        ece = ece_sum / len(confidences) if confidences else 0.0
        avg_conf_total = float(np.mean(conf_arr))
        avg_acc_total = float(np.mean(corr_arr))
        overconfident = avg_conf_total > avg_acc_total + 0.05

        return CalibrationResult(
            buckets=buckets,
            ece=ece,
            mce=mce,
            overconfident=overconfident,
        )
