# fg2020_abaw
This repository is Robolab@UBD's submission for FG-2020 Competition: ABAW Track 2 Expression Challenge and contains the following:

* data - contains the processed dataset with all the images in its respective folder according to its expressions (e.g 0 - neutral)
* features - contains numpy files of dataset for model training and testing purposes.
* model - contains trained models checkpoints and weights.
* results - contains all 222 predictions submission generated after running the baseline model.
* src - contains codes for baseline model, dataset extraction and assembling.

## Process workflow
This shows the overall workflow on how to use our scripts to generate prediction results.

* `affwild2_extract.py` is run in the root folder of aff-wild2 dataset to generate training/validation labeled images in a folder named `aff-wild2/labelled_image`. Additionally, a test images folder will be generated (i.e `aff-wild2/labelled_test_image`) which will contain all test images labeled with dummy annotations.
* The two generated folders (i.e `aff-wild2/labelled_image`, `aff-wild2/labelled_test_image`) should be placed in the `baseline_model/data` folder in two seperate folders (i.e `aff-wild2`, `aff-wild2_test` respectively).
* `assembler.py` is run to generate numpy files which are required for both training and testing. The numpy files for training and validation can be placed in the folder of `baseline_model/features` while numpy files for testing should be placed in a folder (i.e `baseline_model/features/abaw2020_affwild_test_set`)
* `baseline_model.py` is used to train and test a model using aff-wild2 dataset. This will also generate prediction results file for all 222 tests.

## Acknowledgment
The model used here is exactly based on [mayurmadnani](https://github.com/mayurmadnani/fer/blob/master/FER_CNN.ipynb)'s work. However, we have plans on extending his work to include a Landmark-net architecture to process facial landmark features to improve expression recognition accuracy.
