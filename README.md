
# Agricultural Pests Classification

## Project Overview

The **Agricultural Pests Classification** project aims to develop an efficient deep learning model using the ResNet-50 V2 architecture to classify various agricultural pests. By accurately identifying pests from images, this model aids farmers and agricultural professionals in implementing timely and appropriate pest management strategies, ultimately enhancing crop yields and minimizing losses.

## Dataset

The datasets used in this project are sourced from Kaggle and comprise images of different agricultural pests and their respective labels. The datasets are divided into two parts:

1. **Agricultural Pests Dataset**: This dataset includes a collection of labeled images of various agricultural pests. The images are organized into a training set and a testing set.
   - **Training Set**: Contains images and labels used to train the model.
   - **Testing Set**: Contains images used to evaluate the model's performance after training.

   You can access the dataset here:
   - [Agricultural Pests Dataset](https://www.kaggle.com/datasets/abhijithpadma/agril-pests)

2. **Agricultural Pests Image Dataset**: This dataset contains a larger collection of pest images, contributing to the model's robustness and accuracy.
   - [Agricultural Pests Image Dataset](https://www.kaggle.com/datasets/sakshamjain/agril-pests-image-dataset)

3. **ResNet-50 V2 Pre-trained Model**: This is a pre-trained deep learning model that serves as the backbone for the pest classification task.
   - [ResNet-50 V2 Agricultural Pests Classification Model](https://www.kaggle.com/datasets/sakshamjain/resnet50-v2-agricultural-pests-classification)

## Model

The core of this project is based on the **ResNet-50 V2** architecture, which is known for its ability to effectively handle image classification tasks. ResNet employs residual learning, allowing for deeper networks to be trained without the risk of vanishing gradients. This feature enhances the model's performance by enabling it to learn more complex features.

### Implementation Details

- **Model Loading**: The ResNet-50 V2 model is loaded and fine-tuned using the training dataset.
- **Prediction**: The trained model is then utilized to predict the class labels of both the training and testing datasets.
- **Evaluation**: The model's performance is evaluated using metrics such as classification reports, which provide insights into precision, recall, and F1-score.

## Installation

To run this project, ensure that you have the following dependencies installed:

```bash
pip install lobe[all]
pip install pandas
pip install scikit-learn
```

## Usage

Clone this repository and run the Jupyter Notebook provided to execute the classification process. Make sure to have access to the Kaggle datasets mentioned above.

## Contributing

Contributions are welcome! If you would like to improve this project, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Links

- GitHub Repository: [PESTS-CLASSIFICATION](https://github.com/shiwanshra1/PESTS-CLASSIFICATION)

