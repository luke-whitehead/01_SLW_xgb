README for 100- scripts

100_vdr_threshold.ipynb : exploratory analysis of VDR properties, to determine best liquid water threshold

101_vdr_ref_mask.ipynb : one-off processing of new VDR reference mask
102_vdr_ref_mask.py : script for processing of new VDR reference mask for full dataset

- Current version of new reference mask saves to CHCH_v3_reference directory. Changes from previous version:
    - Adds MPC category (i.e. SLCC except SLWC)

- Changes in CHCH_v2_reference:
    - Classifies as WLCC, SLWC, SLCC or IC
    - Follows reviewer suggestions for liquid and mixed phase thresholds
    - Adds undetermined and unknown categories

103_training_dataset.ipynb : one-off processing of peak selection for preparation of training dataset
104_training_dataset.py : script for processing of peak dataset and saving to CSV

- Current version saves to CHCH_v3_reference/peaks_training_df.csv. Records metadata, peak properties and reference mask target. 

105_training_dataset_analysis.ipynb : exploratory analysis of peak dataset and properties

106_train_model.ipynb : XGBoost model training

- Current version saves to CHCH_v3_reference/models/. Model versions correspond to different scoring methods from hyperparameter grid searches.

107_apply_model.ipynb : one-off processing of applying model to create cloud mask
108_apply_model.py : script for processing new cloud masks from applying model

- Current version saves to CHCH_v4_model_processed/{model_name} for a given model. Also can process AG model. Changes from previous version (v3):
    - Wider mask around a peak (+- 2*width)
    - Added 'Other' type for AG model
    - Added 'Any peaks' mask (called classified_peaks_mask)
    - Only processing days for which AMPS temperature data is available

109_model_figures.ipynb : processing of reference mask and model mask time x altitude figures, including AG

110_model_stats.ipynb : calculations of precision/recall statistics for model mask. Also creates and plots altitude/frequency figure.

111_model_shap.ipynb : analysis of model performance, with confusion matrices and SHAP statistics

112_reduced_feature_models.ipynb : analysis to produce Section 3.6

113_mpl2nc_processing.ipynb : see 114. Replacing 000_MPL_processing.
114_mpl2nc_processing.py : preprocessing script. Takes mpl2nc, AMPS and ALCF and produces layer averaged VDR, for classification with script 102.