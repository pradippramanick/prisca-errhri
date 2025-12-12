# prisca-errhri

This repository contains the code submission to the ERR@HRI 2024 Challenge at ICMI 2024 from the PRISCA team.

## Publication
Please read our paper for more details about our approach here https://dl.acm.org/doi/abs/10.1145/3678957.3688387


# Data

To get the data, please ask the challenge organizers. More info at https://arxiv.org/abs/2407.06094

## Preprocessing
There are two steps to reproduce the preprocessing for training and validation. 

1. First, run preprocess.py, setting the data path in the main function. This will produce a train-val.csv file inside the data folder. Since we use the same preprocessing script as shared in the official GitHub repo, you can directly go to step 2, if you already have the train-val.csv (unnormalized).
2. Run create_splits.py 3 times. Please set the label_column variable to the label
column names, e.g., UserAwkwardness. This will generate 6 files with .npz extension
(compressed, serialized numpy arrays), two for each label type, one for training and validation.

## Training

Please follow these steps to reproduce the training and validation steps.
1. Cd into /models
2. Run train_[task name].py three times for each task (UA,RM, IR). This will create the model weights. Please move the learned weights to the /weights folder.

3. For validation please run evaluate_[task name].py, after setting the paths in the main function. This script may throw an error when importing metrics.err_hri_calc. Please, either remove this import or import the shared get_metrics function as per your codebase.

## Testing

We generated the predictions by running the preprocess_test_and_serialize.py which preprocesses the test data, produces a single test.csv (without labels), and
dumps testing X_sequences as test.npz. This will create a [task name]_preds.csv for each label type.

Therefore, the three CSV files that correspond to the predictions for the three tasks are
- RM_preds, UA_preds, and IR_preds.

## Citation

If you use this code for your research, please cite our paper:

```@inproceedings{pramanick2024prisca,
  title={PRISCA at ERR@ HRI 2024: Multimodal Representation Learning for Detecting Interaction Ruptures in HRI},
  author={Pramanick, Pradip and Rossi, Silvia},
  booktitle={Proceedings of the 26th International Conference on Multimodal Interaction},
  pages={666--670},
  year={2024}
}
```
