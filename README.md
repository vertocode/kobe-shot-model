# Kobe Shot Model

Github: https://github.com/vertocode/kobe-shot-model

## Jump To

- [Setup](/docs/SETUP.md)
- [Introduction](#introduction)
- [Project Flow Diagram](#project-flow)
- [How Do Streamlit, MLFlow, PyCaret, and Scikit-Learn Support the Pipeline?](#how-do-streamlit-mlflow-pycaret-and-scikit-learn-support-the-pipeline)

## Introduction

The **Kobe Shot Model** project is part of my **postgraduate** course in **Machine Learning Engineering** at **INFNET University**. The goal of this project is to predict whether Kobe Bryant successfully made a basket based on various features of his shot attempts throughout his NBA career. This machine learning pipeline leverages historical game data, including shot distance, game period, and player position, to create models that predict the success or failure of each shot.

In this project, we will explore two machine learning approaches:
- **Regression** to predict the likelihood of a successful shot.
- **Classification** to determine whether the shot was successful or missed.

The project follows the **TDSP (Team Data Science Process)** framework, ensuring a structured approach to data collection, preparation, model training, evaluation, and deployment. The goal is to deliver a robust model that can predict shot outcomes and be easily deployed for future use.

This README will guide you through the steps and processes involved in the project, from data ingestion to model deployment and monitoring.

## Project Flow

Below is the diagram that outlines the steps and processes involved in the **Kobe Shot Model** project, from data ingestion to model deployment and monitoring.

![Project Flow](docs/images/project_flow.png)

## How Do Streamlit, MLFlow, PyCaret, and Scikit-Learn Support the Pipeline?

**Q: How do these tools assist in building the machine learning pipeline?**

**A:** Each tool plays a crucial role in different stages of the pipeline:

- **MLflow**:
  - **Experiment Tracking**: Logs model parameters, metrics, and artifacts to track experiments.
  - **Model Registry**: Stores different versions of models, facilitating easy comparison and deployment.
  - **Monitoring**: Helps in evaluating model performance over time.

- **PyCaret**:
  - **Automated Training**: Simplifies the process of training and comparing multiple models with minimal code.
  - **Model Evaluation**: Provides built-in tools to assess model performance using various metrics.

- **Scikit-Learn**:
  - **Feature Engineering & Preprocessing**: Offers robust tools for feature selection, scaling, and transformation.
  - **Model Training**: Implements various machine learning algorithms, including logistic regression and decision trees.

- **Streamlit**:
  - **Interactive Dashboard**: Enables visualization of predictions and model insights in a user-friendly web app.
  - **Deployment**: Facilitates easy deployment of models with an intuitive UI for end users.

By integrating these tools, the pipeline ensures efficient model training, evaluation, deployment, and monitoring.

## Dataset Dimensions

As part of the data preprocessing for the **Kobe Shot Model**, the dataset was filtered to include only relevant columns, and rows with missing data were removed. The resulting dataset has the following dimensions:

- **Filtered dataset dimensions**: (20285, 7)

This result can be verified in the notebook **dimension_initial_dataset.ipynb** located in the **notebooks** folder. The notebook demonstrates how the filtering process was applied, resulting in a cleaned dataset ready for model training.

You can inspect the details of the dataset and the preprocessing steps by reviewing the notebook.