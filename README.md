<!-- Badges -->
<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange.svg?style=flat&logo=Jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Version-1.0.0-blue" alt="Version">
  <img src="https://img.shields.io/github/stars/engcarlo/TransferLearning-Datasets?style=social" alt="License">
  <img src="https://img.shields.io/github/forks/engcarlo/TransferLearning-Datasets?style=social" alt="Stars">
</p>

**Author:** [engcarlo](https://www.github.com/engcarlo)

---
# 🔗 Summary

- [🔗 Summary](#-summary)
- [📘 Project: Transfer Learning for Image Classification with TensorFlow and Keras](#-project-transfer-learning-for-image-classification-with-tensorflow-and-keras)
  - [1. Project Introduction](#1-project-introduction)
  - [2. Objective](#2-objective)
    - [Techniques and Architectures Used](#techniques-and-architectures-used)
    - [The Dataset](#the-dataset)
  - [3. Repository Structure](#3-repository-structure)
    - [Cloning the Repository](#cloning-the-repository)
    - [Prerequisites](#prerequisites)
    - [Installing Dependencies](#installing-dependencies)
  - [4. Usage Examples](#4-usage-examples)
    - [Running the Jupyter Notebook](#running-the-jupyter-notebook)
    - [Exploring the Notebook](#exploring-the-notebook)
  - [5. Repository File/Directory Structure](#5-repository-filedirectory-structure)
    - [Project Files](#project-files)
  - [6. Project Roadmap](#6-project-roadmap)
  - [7. Contribution Guidelines](#7-contribution-guidelines)
  - [8. License](#8-license)


---


# 📘 Project: Transfer Learning for Image Classification with TensorFlow and Keras

*This project features an interactive Python notebook that demonstrates how to apply Transfer Learning to solve image classification problems efficiently.*


## 1. Project Introduction

This project presents an **interactive Python notebook** using **Tensorflow** and **Keras** to apply **Transfer Learning** to image classification problems. The main objective is to demonstrate how to leverage pre-trained models on large datasets (e.g., ImageNet) and adapt them for specific tasks, saving time and computational resources.

The notebook uses **Convolutional Neural Networks (CNNs)**. Specifically, it uses the VGG16 architecture, which is a pre-trained CNN for transfer learning, and also builds a neural network from scratch using convolutional (Conv2D) and pooling (MaxPooling2D) layers, which are the fundamental components of modern CNNs.

The core technique involves adapting a CNN pre-trained on a large-scale dataset (like ImageNet) for our new classification task." (Slightly more formal and descriptive).


---

## 2. Objective

The primary goal is to showcase how to leverage powerful, pre-trained models (like VGG16, trained on the ImageNet dataset) and fine-tune them for a custom task. This approach dramatically reduces training time and the need for vast amounts of data, making deep learning more accessible.



### Techniques and Architectures Used

Along the notebook it will be shown step-by-step the flow of that task following:

- __Convolutional Neural Networks (CNNs):__ We explore the fundamental building blocks of modern computer vision.
- __Training from Scratch:__ A custom CNN is built and trained on our dataset to establish a baseline performance metric.
- __Transfer Learning (Feature Extraction):__ The core of this project. We use the __VGG16__ architecture, freezing its convolutional base and training only a new, custom classifier head.

### The Dataset

The models are trained and evaluated on the classic __"Cats vs. Dogs"__ dataset, a perfect use case for demonstrating the power of adapting a general-purpose model to a specific binary classification task.


---

## 3. Repository Structure

To get started with `TransferLearning-Datasets`, follow these simple steps.

### Cloning the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/engcarlo/TransferLearning-Datasets.git
cd TransferLearning-Datasets
```

### Prerequisites

Ensure you have Python 3.11+ and `pip` installed on your system. It's highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Installing Dependencies

Navigate into the cloned directory and install the necessary Python packages. While no `requirements.txt` is provided, common libraries for machine learning and Jupyter Notebooks are usually needed.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---
## 4. Usage Examples

Once installed, you can launch the Jupyter Notebook and explore the transfer learning examples.

The notebook generate automatically:

- Plot Loss and Accuracy of model;

- Classification examples applying trained model to predict images of one category.

By the end of the notebook, you will see a significant performance improvement in the transfer learning model compared to the model trained from scratch, highlighting why this technique is a cornerstone of modern computer vision tasks.

By comparing the two approaches, this notebook clearly illustrates the power of transfer learning. You will observe a significant improvement in classification accuracy (~80% or higher) with the VGG16-based model compared to the CNN trained from scratch (~50%), highlighting why this technique is a cornerstone of modern computer vision.


### Running the Jupyter Notebook

To open the main transfer learning notebook, run the following command from the project root directory:

```bash
jupyter notebook M2_transfer_learning.ipynb
```

This will open the notebook in your web browser, where you can execute cells and interact with the code and datasets.

The execution flow inside of notebook:

1. Import libraries;

2. Load and Pre-processing Data;

3. Setting the model from absolutelly Zero and from Pretrained;

4. Dataset Training and Validation;

5. Evaluating the results.


### Exploring the Notebook

The `M2_transfer_learning.ipynb` notebook data organization:

- __Number of Classes:__ The dataset is correctly identified as having __2__ classes.

- __Data Split:__ The data is partitioned into three distinct sets to ensure robust model evaluation:

  - __Training Set:__ Comprises 80% of the total data, used for training the neural network.
  - __Validation Set:__ Comprises 10% of the data, used to monitor the model's performance during training and prevent overfitting.
  - __Test Set:__ The final 10% of the data, reserved for an unbiased evaluation of the trained model's accuracy.

- __Data Shape:__

  - __Images (`x_train`):__ The input images are processed into a 4D tensor with the shape `(number_of_images, 224, 224, 3)`, where `224x224` is the image resolution required by VGG16 and `3` represents the RGB color channels.
  - __Labels (`y_train`):__ The corresponding labels are one-hot encoded into a 2D tensor with the shape `(number_of_images, num_classes)`, which will be `(number_of_images, 2)` for this dataset.

---

## 5. Repository File/Directory Structure

The present jupyter notebook assume the following layout (ex.: after cloning '"TransferLearning-Datasets"'):

```sh
TransferLearning-Datasets/ # Main repository
├─Dataset/                 # Dataset directory
│   ├─Cats/
│   │  ├─ 1.jpg
│   │  └─ ...
│   └─Dogs/
│      ├─ 1.jpg
│      └─ ...
├─models/                   # Saved trained models and checkpoint
└─outputs/                  # Results, metrics and graphs
```

###  Project Files

```sh
M2_transfer_learning.ipynb   # Notebook with Transfer Learning flow analysis
DataAug-App.ipynb            # Notebook to apply data augmentation
README.md                    # Project documentation
```


In Dataset directory, each subdir represents one class/category. The directories' names (Cats, Dogs, etc...) are used to build the categories automatically.
 
---
## 6. Project Roadmap

Future implementation for `TransferLearning-Datasets` includes continuous improvement and expansion:

*   ➕ **More Datasets:** Expand the `Dataset/` directory with additional curated datasets for others categories and subcategories like species/race/type (e.g., animals species, objects, fruits, plants, minerals).
*   📌 **Aditional Implementation:** Expand the functionalities by implementing functions to provide data augmentation, and report generation.
*   🚀 **User Interface:** Create UI using the framework of Streamlit.
*   📈 **Advanced Notebooks:** Introduce more complex transfer learning scenarios, including domain adaptation and few-shot learning.
*   📊 **Performance Benchmarks:** Include benchmarks for common transfer learning tasks to compare different approaches.
*   📚 **Documentation & Tutorials:** Enhance documentation and add dedicated tutorials for beginners.

---

##  7. Contribution Guidelines

Feel free to contribute for `TransferLearning-Datasets` repository! To ensure a smooth collaboration, please follow these guidelines:

1. __Fork the Repository__
   - Start by forking the repository to your own GitHub account. This creates a personal copy where you can work without affecting the original project.
2. __Clone Your Fork__
   - Clone your forked repository to your local machine:
```bash
git clone https://github.com/{{YOUR-USERNAME}}/TransferLearning-Datasets.git
cd TransferLearning-Datasets
```
3. __Create a New Branch__
   - Create a dedicated branch for your changes. Use a descriptive name that reflects the feature or fix you're working on.
```bash
# Example for a new feature
git checkout -b feature/add-resnet-model

# Example for a bug fix
git checkout -b fix/data-loading-error
```
4. __Make Your Changes__
   - Now, you can modify the code. As you work, please keep the following guidelines in mind:
     - __Code Style:__ Adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code. For Jupyter Notebooks, ensure cells are clean, well-commented, and that you __clear all outputs__ before committing.
     - __Testing:__ Run all cells in any modified notebooks to confirm that your changes haven't introduced errors and that everything executes correctly from top to bottom.
5. __Commit Your Work__
   - Once you've made your changes, commit them with a clear and concise message describing what you've done. We follow the Conventional Commits standard.
```bash
# Example for a new feature
git commit -m "feat: Add ResNet50 as a new transfer learning option"

# Example for a bug fix
git commit -m "fix: Correct path issue in data loader"
```
6. __Push to Your Fork__
   - Push your committed changes from your local branch to your forked repository on GitHub. This makes your work available online so you can create a pull request.
```bash
git push origin feature/add-resnet-model
```
7. __Open a Pull Request (PR)__ 
   - Finally, go to the original `TransferLearning-Datasets` repository on GitHub and open a pull request from your branch to the `main` branch.
   - Provide a detailed title and description for your PR, explaining the changes and why they are valuable.
   - If your changes are related to an existing issue, be sure to link it in the description.
   - Ensure all existing code runs without errors.
   - If adding new features or datasets, include relevant documentation or examples.

🤝 Thank you for your contribution!

---

## 8. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE)📜 file for more details.

