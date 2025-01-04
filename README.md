# COVID-19 Image Classifier

![Project Banner](https://www.itnonline.com/sites/default/files/Chest.jpeg)

## 🚀 Introduction

This project aims to leverage AI and machine learning to assist the medical community in identifying COVID-19 and other respiratory conditions through radiography images. Utilizing a Kaggle dataset and a ResNet18 model, the classifier achieves an impressive **94.96% accuracy**, making it a reliable tool for diagnostic assistance.

The COVID-19 pandemic has emphasized the importance of rapid and accurate diagnostic tools. This classifier is designed to provide an additional layer of support to radiologists and healthcare professionals.

---

## 📊 Dataset

The model is trained on the **COVID-19 Radiography Dataset**, sourced from Kaggle. The dataset includes:

- **COVID**
- **Lung Opacity**
- **Normal**
- **Viral Pneumonia**

The dataset is preprocessed to normalize pixel values for consistent model input.

Dataset Link: [COVID-19 Radiography Dataset on Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

---

## 🛠️ Features

1. **High Accuracy:** Achieves 94.96% accuracy on the test set.
2. **Multi-Class Prediction:** Supports classification of four distinct classes.
3. **Pretrained Model:** Built upon ResNet18, a robust and reliable convolutional neural network.
4. **Medical Support:** Assists healthcare professionals in making quick diagnostic decisions.

---

## 📂 Project Structure

```plaintext
.
├── calculatevalues.py  # Calculates dataset mean and std for normalization.
├── creating_model.py   # Contains model training and evaluation scripts.
├── test_model.py       # Tests the model with sample images.
├── README.md           # This file.
```

---

## 🧠 Model Overview

The classifier is built on **ResNet18**, a convolutional neural network pretrained on ImageNet. The model is fine-tuned to handle the unique features of the COVID-19 Radiography Dataset.

### Training Details:
- **Loss Function:** Weighted CrossEntropy Loss (to handle class imbalance)
- **Optimizer:** Adam
- **Learning Rate Scheduler:** StepLR
- **Training Epochs:** 10
- **Augmentations:** Random Resizing and Cropping

---

## 📈 Performance Metrics

The model achieves the following performance on the test set:

- **Accuracy:** 94.96%
- **Loss:** Reduced consistently across epochs

---

## 💻 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/abdullahlali/covid19-image-classifier.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model (optional):
   ```bash
   python creating_model.py
   ```
4. Test the model:
   ```bash
   python test_model.py
   ```

Ensure you have the required dataset in the specified directory.

---

## 📂 Sample Predictions

```plaintext
Input Image: pneumonia.png
Predicted Class: Viral Pneumonia

Input Image: normal.png
Predicted Class: Normal
```

---

## 🌟 Future Enhancements

1. Deploy the model as a web application for easy accessibility.
2. Explore other deep learning architectures for improved performance.
3. Expand the dataset to include additional respiratory conditions.
4. Integrate explainability methods to provide insights into predictions.

---

## 🙏 Acknowledgements

- Kaggle for the dataset.
- The PyTorch community for the ResNet18 implementation.

---

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 💌 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or new features.

---

## 🏥 Impact on Healthcare

This project is driven by a desire to assist the medical community in combating COVID-19 and similar respiratory illnesses. With a focus on accuracy and accessibility, this AI tool aims to provide timely diagnostic support, saving lives and reducing the burden on healthcare systems.

---

## 📬 Contact

For any questions or collaborations, feel free to reach out:

- **Email:** abdullahliaqat.dev@gmail.com
- **GitHub Issues:** [Submit an Issue](https://github.com/abdullahlali/issues)

---

Thank you for checking out this project! Together, let's make a difference in healthcare with AI.

