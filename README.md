# FG-2020 Competition: Affective Behavior Analysis in-the-wild (ABAW)
This repository is Robolab@UBD's submission for [FG-2020 Competition: ABAW Track 2 Expression Challenge](https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/). The model was ranked 8th out of 12 with a total score of 0.342 (Accuracy: 63%, F1-Score: 20%) in our first and only attempt. 

If you have used this repository in any of your scientific work, please consider citing my work using this [BibTeX Citation](#bibtex-citation).

The repository contains the following:

* data - contains the processed dataset with all the images in its respective folder according to its expressions (e.g 0 - neutral)
* features - contains NumPy files of the dataset for model training and testing purposes.
* model - contains trained model checkpoints and weights.
* results - contains all 223 predictions submission generated after running the baseline model.
* src - contains codes for the baseline model, dataset extraction, and assembling.

## Process workflow
This shows the overall workflow on how to use our scripts to generate prediction results.

* `affwild2_extract.py` is run in the root folder of the aff-wild2 dataset to generate training/validation labeled images in a folder named `aff-wild2/labelled_image`. Additionally, a test images folder will be generated (i.e `aff-wild2/labelled_test_image`) which will contain all test images labeled with dummy annotations.
* The contents of the two generated folders (i.e `aff-wild2/labelled_image`, `aff-wild2_test/labelled_test_image`) should be placed in the `/data` folder in two separate folders (i.e `aff-wild2`, `aff-wild2_test` respectively).
* `assembler.py` is run to generate NumPy files which are required for both training and testing. The NumPy files for training and validation can be placed in the folder of `/features` while NumPy files for testing should be placed in a folder (i.e `/features/abaw2020_affwild_test_set`)
* `baseline_model.py` is used to train and test a model using the aff-wild2 dataset. This will also generate a prediction results file for all 223 tests.
* `fix_missing_predictions.py` is run after prediction results are available, this is to add undetected face frames into the prediction files.

## Hardware
* Operating System: Windows 10
* CPU: Intel i7-9700
* GPU: Nvidia RTX2070 8GB
* RAM: 64GB DDR4

## Installing packages
Create a virtual environment with Python 3.7.6 64-bit and install the following packages with `pip install`: <br />
* opencv-python
* keras
* tensorflow
* matplotlib
* cmake
* dlib
* tqdm
* scikit-learn
* imutils
* iterative_stratification
* tensorflow-gpu

## BibTeX Citation
If you have used this repository in any of your scientific work, please consider citing my work:
```
@article{anas2020deep,
  title={Deep convolutional neural network based facial expression recognition in the wild},
  author={Anas, Hafiq and Rehman, Bacha and Ong, Wee Hong},
  journal={arXiv preprint arXiv:2010.01301},
  year={2020}
}
```

## Acknowledgment
The model used here is exactly based on [mayurmadnani](https://github.com/mayurmadnani/fer/blob/master/FER_CNN.ipynb)'s work. We would like to thank him for making his repository public and available for everyone to use.
