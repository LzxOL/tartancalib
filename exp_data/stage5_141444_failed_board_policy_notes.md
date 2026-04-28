# Stage5 Failed-Board Policy Comparison Notes

## Scope

This note compares three policies on the 141444 full dataset:

- A / off-rescue: keep outer observations even when internal regeneration fails.
- strict failed-board drop: Kalibr-style strict board-observation acceptance, no rescue.
- rescue gate8: wide-FOV pose rescue enabled with outer RMSE acceptance gate 8 px.

The 140151 full baseline and strict result are included as a sanity check.

## Main Takeaway

Strict failed-board drop reaches the best holdout overall RMSE among the three 141444 policies without using rescue.
It drops all 11 failed holdout board observations and improves holdout outer-only RMSE from 4.21086 to 3.8359.
Holdout internal RMSE remains unchanged from A because strict removes boards that had no valid internal regeneration.

Rescue gate8 is not favored by this result. It accepts two previously failed boards, but holdout overall RMSE worsens
from 7.04054 to 7.30215 and holdout internal RMSE worsens from 7.3607 to 7.64423.

On 140151, strict is essentially neutral relative to the official frozen baseline and does not hurt the result.

## Interpretation

The strict policy is a conservative Kalibr-style experiment branch, not yet a new baseline. Current evidence suggests:

- It is better than rescue for the 141444 edge-board failure case.
- It is safe on the 140151 full baseline dataset.
- It should be preferred over wide-FOV rescue unless a future dataset shows rescue provides a clear benefit.

Recommended next step: keep strict as a candidate paper method and run any future full datasets with both official baseline
and strict branch before deciding whether to freeze a new baseline.
