# distracted-driving-behaviors

This project focuses on detecting distracted driving behaviors using machine learning models trained on the State Farm Distracted Driver Detection dataset. The models classify images into 10 classes representing various distracted and non-distracted driving activities.

# Project Directory Structure

- **Custom CNN**
  - `custom_cnn_best_model.pth`: Saved weights for the Custom CNN model.
  - `custom_cnn_submission.csv`: Kaggle submission file for the Custom CNN model.
  - `custom_cnn.ipynb`: Notebook for training and evaluating the Custom CNN.

- **Non-Pre-Trained-VGG**
  - `custom_vgg_best_model.pth`: Saved weights for the non-pretrained VGG16 model.
  - `custom_vgg_submission.csv`: Kaggle submission file for the non-pretrained VGG16 model.
  - `non_pretrained_vgg.ipynb`: Notebook for training and evaluating the non-pretrained VGG16.

- **Pretrained_VGG**
  - `best_vgg_model.pth`: Saved weights for the pretrained VGG16 model.
  - `vgg_submission.csv`: Kaggle submission file for the pretrained VGG16 model.
  - `pretrained_vgg.ipynb`: Notebook for training and evaluating the pretrained VGG16.

- **Resnet**
  - `best_resnet_model.pth`: Saved weights for the ResNet18 model.
  - `resnet_submission.csv`: Kaggle submission file for the ResNet18 model.
  - `resnet.ipynb`: Notebook for training and evaluating ResNet18.

- **Ensemble**
  - `best_performers_weighted_ensemble_submission.csv`: Kaggle submission file for the best performers weighted ensemble.
  - `best_performers_weighted_ensemble.ipynb`: Notebook for the best performers weighted ensemble method.
  - `hybrid_ensemble_submission.csv`: Kaggle submission file for the hybrid ensemble.
  - `hybrid_ensemble.ipynb`: Notebook for the hybrid ensemble method.
  - `weighted_ensemble_submission.csv`: Kaggle submission file for the weighted ensemble.
  - `weighted_ensemble.ipynb`: Notebook for the weighted ensemble method.

- **imgs**
  - `train/`: Folder containing training images.
  - `test/`: Folder containing test images for submission predictions.

- **state-farm-distracted-driver-detection**
  - `driver_imgs_list.csv`: CSV file mapping drivers to their respective images.
  - `sample_submission.csv`: Example submission format provided by Kaggle.

- **README.md**: Project overview and description.

## Models Overview
We trained four different models using PyTorch:

- ResNet18 (Transfer Learning): A pretrained ResNet18 model fine-tuned for this classification task.
- Pretrained VGG16 (Transfer Learning): A pretrained VGG16 model fine-tuned for this classification task.
- Custom CNN: A custom-built convolutional neural network designed from scratch.
- Non-Pretrained VGG16: A VGG16 model trained from scratch without any pretraining.

## Model Performance
The performance of each model was evaluated using Kaggle's multiclass logarithmic loss metric:

| Model                          | Multiclass Log Loss |
|--------------------------------|---------------------|
| **ResNet18**                   | **0.28200**        |
| **Pretrained VGG16**           | **0.41719**        |
| **Custom CNN**                 | **1.47378**        |
| **Non-Pretrained VGG16**       | **2.30182**        |


## Ensemble Methods
Three ensemble methods were implemented to combine the strengths of individual models:

- Hybrid Ensemble: Combined predictions using a hybrid of majority voting and confidence scores.
- All Model Weighted Ensemble: Weighted average of predictions from all four models. Weights were inversely proportional to their individual log loss scores.
- Best Performers Weighted Ensemble: Weighted average using only the two best-performing models (ResNet18 and Pretrained VGG16).

## Ensemble Performance

| Ensemble Method                     | Multiclass Log Loss |
|-------------------------------------|---------------------|
| **Hybrid Ensemble**                 | **0.57478**        |
| **All Model Weighted Ensemble**     | **0.28683**        |
| **Best Performers Weighted Ensemble** | **0.21775**        |


## Submission Files
The following submission files were generated:

- custom_cnn_submission.csv: Predictions from the Custom CNN.
- custom_vgg_submission.csv: Predictions from the non-pretrained VGG16.
- vgg_submission.csv: Predictions from the pretrained VGG16.
- resnet_submission.csv: Predictions from the ResNet18.
- hybrid_ensemble_submission.csv: Predictions from the hybrid ensemble method.
- weighted_ensemble_submission.csv: Predictions from the all-model weighted ensemble.
- best_performers_weighted_ensemble_submission.csv: Predictions from the best performers weighted ensemble.


## Key Findings
- Transfer learning significantly outperformed custom models trained from scratch.
- The Best Performers Weighted Ensemble yielded the best overall performance, leveraging the strengths of ResNet18 and Pretrained VGG16.
- Non-pretrained VGG16 performed poorly, emphasizing the value of pretraining on large-scale datasets.

# Setting Up the Project

Follow these steps to set up and run the project in a virtual environment.

## 1. Navigate to the Project Directory

Move into the root directory of the project where the `requirements.txt` file is located.

bash:
cd /path/to/your/project

## 2. Create a Virtual environment

python -m venv venv

## 3. Activate the Virtual Environment

Windows:
venv\Scripts\activate

Mac/ Linux:
source venv/bin/activate

## 4. Install required dependancies

pip install -r requirements.txt

## 5. Verify the installation

pip list

# 6. Download the Dataset from the Kaggle Competition

Download the dataset from: https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data 

Place the dataset in the root of the directory

# 7. Change the paths to the test and training sets

Each file contains multiple paths to the training and test dataset as well as paths to the trained models and they must be updated with your unique paths. These are generally found near the top of every section of code. This was setup so indivual parts could be run seperately without the need to retrain the entire model.

# 8. Run the code 

You can now run any of the .ipynb notebooks or .py scripts in the project.

Ensure Jupyter notebooks are installed:

pip install jupyter

GPU Setup: If you plan to use GPU acceleration with CUDA, ensure that your PyTorch version is compatible with your CUDA version. You can check compatibility here.
Python Version: This project was developed using Python 3.8+. Ensure your Python version is compatible to avoid issues.

## How to Reproduce
- Train individual models by navigating to their respective folders and running the training notebooks (*.ipynb).
- Generate ensemble submissions using the notebooks in the Ensemble/ folder.
- Follow Kaggle's submission format when creating new submission files.

## Conclusion
This project demonstrates the power of transfer learning and ensemble methods in addressing challenging image classification tasks. Future improvements could involve fine-tuning hyperparameters and exploring additional ensemble techniques.