# Prawn Size Measurement Project

This project focuses on automated measurement of the freshwater prawn *Macrobrachium rosenbergii* from underwater images.

## Goal
- Measure **carapace length** (carapace-start → eyes).
- Measure **total body length** (rostrum → tail).

## Approach
- Underwater images are processed with **YOLO-based keypoint models**.
- Predictions are validated against ground-truth Excel annotations (ImageJ + manual labels).
- Pixel distances are converted to millimeters using **refraction-aware camera calibration**.

## Repository Structure

```

prawn-size-project/
├─ batch_eval_dual_set_...py # main evaluation scripts
├─ gamma_batch_for2025_test.py # batch script for 2025 test images
├─ ImageJ_measurements_2025.xlsx # ImageJ-based ground truth
├─ ImageJ_measurements_2025_CLEAN.xlsx # cleaned version
├─ models/ # trained model weights
├─ outputs/ # evaluation results, tables, visualizations
├─ results_summary.py # summary generator
├─ test_images_2024/ # pond images (2024, processed)
├─ test_images_2024_orginalSize/ # pond images (2024, original size)
├─ test_images_2025/ # pond images (2025, processed)
├─ test_images_2025_gamma/ # pond images (2025, gamma-corrected)
├─ TEST_2025_Body.xlsx # GT body measurements
├─ TEST_2025_Carapace.xlsx # GT carapace measurements
├─ TEST_2025_Body_WITH_OBB_v2.xlsx # GT with OBB (body)
├─ TEST_2025_Carapace_WITH_OBB_v2.xlsx # GT with OBB (carapace)
├─ updated_filtered_data_with_lengths_body-all.xlsx
├─ updated_filtered_data_with_lengths_carapace-all.xlsx
└─ docs/

   └─ Prawn_Size_Project_Overview.pdf             # detailed project explanation

```


## Output
- Per-image and per-pond summaries of prediction error (MAE, MPE).
- Comparison of multiple models trained across 2024 and 2025 datasets.


**📄 Full details and methodology: [Prawn Size Project Overview (PDF)](delete/docs/Prawn_Size_Project_Overview.pdf)
*

