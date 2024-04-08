README for 100- scripts

100_vdr_threshold.ipynb : exploratory analysis of VDR properties, to determine best liquid water threshold
101_vdr_ref_mask.ipynb : one-off processing of new VDR reference mask
102_vdr_ref_mask.py : script for processing of new VDR reference mask for full dataset

- Current version of new reference mask saves to CHCH_v3_reference directory. Changes from previous version:
    - Adds classification as MPC (i.e. SLCC except SLWC)

- Changes in CHCH_v2_reference:
    - Classifies as WLCC, SLWC, SLCC or IC
    - Follows reviewer suggestions for liquid and mixed phase thresholds
    - Adds undetermined and unknown categories

103_training_dataset.ipynb : one-off processing of peak selection for preparation of training dataset
104_training_dataset.py : script for processing of peak dataset and saving to CSV

- Current version saves to CHCH_v3_reference/peaks_training_df.csv. Records metadata, peak properties and reference mask target. 

105_training_dataset_analysis.ipynb : exploratory analysis of peak dataset and properties
106_train_model.ipynb : XGBoost model training