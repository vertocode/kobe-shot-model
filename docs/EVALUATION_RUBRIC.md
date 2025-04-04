# Final Project Evaluation Rubric

Each criterion should be marked as:

- ✅ Demonstrated the rubric item  
- ❌ Did not demonstrate the rubric item

---

## 1. Develop a data collection system using public APIs

| Criterion                                                                                  | Status                                                                                                                                                                                                                                                    |
|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Did the student correctly categorize the collected data?                                   | ✅ - The result can be found in the following file: [data_feature.py](../src/kedro_ml/nodes/data_feature.py)                                                                                                                                               |
| Did the student properly integrate data fetching into their solution?                      | ✅ - The result can be found in the following file: [data_acquisition.py](../src/kedro_ml/nodes/data_feature.py)                                                                                                                                           |
| Did the student deploy the model in production (as an API or embedded solution)?           | ✅ - We can observe the result in the [README.md ](../README.md) file using streamlit. And also we can observe that in the the [main.py](../streamlit/main.py) file                                                                                        |
| Did the student indicate whether the model is compatible with a new dataset?               | ✅ - Two separate pipelines were created — one for development and another for production — as shown [here](../src/kedro_ml/pipelines/__init__.py). The model was successfully applied in the production pipeline, confirming compatibility with new data. |
---

## 2. Create a data streaming solution using pipelines

| Criterion                                                                                                    | Status                                                                                                                                                                                                                                                           |
|--------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Did the student create a Git repository following the Microsoft TDSP framework structure?                   | ✅ - As discussed in class with instructor Felipe, we were allowed to use the Kedro structure, which is similar to TDSP. The project was created following this structure, as shown in [this repository](https://github.com/vertocode/kobe-shot-model/tree/main). |
| Did the student create a diagram showing all steps required for building the models?                        | ✅ - The diagram is included in the [README.md](../README.md), and the image can also be viewed directly [here](../docs/images/project_flow.png).                                                                                                                 |
| Did the student train a regression model using PyCaret and MLflow?                                          | ✅ - The training of the regression model can be found in [models.py](../src/kedro_ml/nodes/models.py), and the results are being logged using MLflow, as configured in [catalog.yml](../conf/base/catalog.yml).                                                  |
| Did the student calculate the Log Loss for the regression model and log it in MLflow?                       | ✅ - The Log Loss calculation and its logging in MLflow can be found in [best_model.py](../src/kedro_ml/nodes/best_model.py).                                                                                                                                     |
| Did the student train a decision tree model using PyCaret and MLflow?                                       | ✅ - The training of the regression model can be found in [models.py](../src/kedro_ml/nodes/models.py), and the results are being logged using MLflow, as configured in [catalog.yml](../conf/base/catalog.yml).                                                  |
| Did the student calculate Log Loss and F1 Score for the decision tree model and log them in MLflow?         | ✅ - The Log Loss calculation and its logging in MLflow can be found in [best_model.py](../src/kedro_ml/nodes/best_model.py).                                                                                                                                     |

---

## 3. Prepare a pre-trained model for a streaming data solution

| Criterion                                                                                                            | Status                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Did the student define the project objective and describe each created artifact in detail?                          | ✅ - We can observe the result in the [README.md ](../README.md) file                                       |
| Did the student cover all artifacts presented in the proposed diagram?                                               | ✅ - We can observe the result in the [README.md ](../README.md) file                                       |
| Did the student use MLflow to log the "Data Preparation" run with relevant metrics and parameters?                  | ✅ - We can observe the result in the [README.md ](../README.md) file                                       |
| Did the student remove missing data from the dataset?                                                                | ✅ - We can observe the result in the [README.md ](../README.md) file                                       |
| Did the student select the indicated columns to train the model?                                                    | ✅ - We can observe the result in the [data_preparation.py](../src/kedro_ml/nodes/data_preparation.py) file |
| Did the student indicate the dimensions of the preprocessed dataset?                                                | ✅ - We can observe the result in the [README.md ](../README.md) file                                       |
| Did the student create files for each processing phase and store them in the appropriate folders?                   | ✅ - We can observe the result in the [README.md ](../README.md) file                                       |
| Did the student split the data into two sets: one for training and one for testing?                                 | ✅ - We can observe the result in the [data_splitting.py](../src/kedro_ml/nodes/data_splitting.py) file   |
| Did the student create a pipeline called "Training" in MLflow?                                                      | ✅ - We can observe the result in the [README.md ](../README.md) file                                                                                                                    |

---

## 4. Establish a method to update the model in production

| Criterion                                                                                                                          | Status |
|-------------------------------------------------------------------------------------------------------------------------------------|--------|
| Did the student identify differences between the development and production datasets?                                              |  ✅ - We can observe the result in the [README.md ](../README.md) file          |
| Did the student describe how to monitor model health with and without the availability of the target variable?                    |        |
| Did the student implement a monitoring dashboard for the operation using Streamlit?                                                |  ✅ - We can observe the result in the [README.md ](../README.md) file          |
| Did the student describe reactive and predictive retraining strategies for the production model?                                  |        |
