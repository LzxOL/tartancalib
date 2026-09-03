# Left Checkerboard Parameter Comparison

This record separates the two left-camera checkerboard sequences that were
previously mixed in one parameter table.

## Source groups

`checkerboard_3_25_left` is the 106-view
`checkboard_3_25_left-clear` sequence. The Ours row is valid for that sequence,
but there are no external canonical parameters calibrated from the same views.

`stereo_4_2_3_left` is the 84-view
`stereo_4_2-3_images/left` sequence. Ours, BabelCalib, Kalibr, and TartanCalib
are comparable within this group. The Ours DS run uses all 84 MAT views and is
stored at:

`result_may/stage5_mat_stereo-4-2-3-all_left_20260715_same_source_audit/ds`

## Rule

Do not compare the `checkerboard_3_25_left` Ours row against the
`stereo_4_2_3_left` external rows as a same-sequence calibration comparison.
That mixed table is diagnostic only and explains the apparent `cu/cv` shift.

## Protocol

- Model: DS
- Image size: 4512 x 4512
- Checkerboard: 11 x 8 inner corners
- Square size: 0.03 m
- Calibration: all MAT views, no train/test split
- Independent holdout: none; duplicate observations are diagnostic only
