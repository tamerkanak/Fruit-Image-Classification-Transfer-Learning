# Fruit Image Classification with Transfer Learning

This repository contains code for classifying fruit images using transfer learning with pre-trained models. The project utilizes the [Fruits 360 dataset](https://www.kaggle.com/moltean/fruits) and explores different pre-trained models for image classification.

## Overview

This project focuses on classifying different types of fruits using the Fruits 360 dataset and transfer learning techniques. We experiment with the following pre-trained models:

- **MobileNetV2**
- **EfficientNetB0**
- **DenseNet121**

The project includes:

- **Data Preparation:** Loading and preprocessing images using `ImageDataGenerator`.
- **Transfer Learning:** Utilizing pre-trained models from TensorFlow Keras applications.
- **Model Training:** Training the models with a custom top layer added.
- **Model Evaluation:** Calculating classification reports, confusion matrices, and loss/accuracy graphs.
- **Visualization:** Generating GradCAM visualizations to understand model focus areas.

## Project Structure

├── fruit_classifier_tl.py # Main Python script
├── MobileNetV2_fruit_classifier.h5 # Trained MobileNetV2 model
├── EfficientNetB0_fruit_classifier.h5 # Trained EfficientNetB0 model
├── DenseNet121_fruit_classifier.h5 # Trained DenseNet121 model
├── MobileNetV2_classification_report.txt # Classification report for MobileNetV2
├── EfficientNetB0_classification_report.txt # Classification report for EfficientNetB0
├── DenseNet121_classification_report.txt # Classification report for DenseNet121
├── README.md # This README file

- `fruit_classifier_tl.py`: The main Python script containing all the code for data preparation, model training, evaluation, and visualization.
- `*_fruit_classifier.h5`: Saved trained models.
- `*_classification_report.txt`: Classification reports for each model.

## How to Run

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd Fruit-Image-Classification-Transfer-Learning
    ```

2.  **Install required libraries:**

    ```bash
    pip install numpy matplotlib tensorflow scikit-learn seaborn tf-keras-vis
    ```
    or you can use:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set your data paths:**

    Modify the `DATASET_PATH` variable in `fruit_classifier_tl.py` to point to your local directory where the Fruits-360 dataset is located.

    ```python
    DATASET_PATH = r"C:\\Users\\tamer\\Desktop\\Deep Learning\\fruits-360_dataset_original-size\\fruits-360-original-size"
    ```

4.  **Run the script:**

    ```bash
    python fruit_classifier_tl.py
    ```

## Key Findings

- This project shows how to perform transfer learning using pre-trained models effectively.
- Performance comparison of MobileNetV2, EfficientNetB0, and DenseNet121 models on the Fruits 360 dataset.
- The effect of fine-tuning the last layers of the pre-trained models are explored.
- Visualization of GradCAM heatmaps highlight areas of interest for the models when predicting fruit images.
- Custom stopper is added to the model to stop the training when the model reaches 100% accuracy and validation accuracy.

## Results

The results of the experiments include:
- Trained models saved as `.h5` files
- Classification reports saved as `.txt` files
- Confusion matrices
- Loss and accuracy graphs during training
- GradCAM visualizations

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Feel free to fork this repository and contribute by submitting a pull request.

## Contact

For any questions or feedback, feel free to contact:
Tamer Kanak
tamerkanak75@gmail.com
