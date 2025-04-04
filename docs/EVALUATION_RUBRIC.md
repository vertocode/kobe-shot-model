# Final Project Evaluation Rubric

Each criterion should be marked as:

- ✅ Demonstrated the rubric item  
- ❌ Did not demonstrate the rubric item

---

## 1. Develop a data collection system using public APIs

| Criterion                                                                                  | Status                                                                                               |
|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Did the student correctly categorize the collected data?                                   | ✅ - The result can be observed in this file [data_feature.py](../src/kedro-ml/nodes/data_feature.py) |
| Did the student properly integrate data fetching into their solution?                      |                                                                                                      |
| Did the student deploy the model in production (as an API or embedded solution)?           |                                                                                                      |
| Did the student indicate whether the model is compatible with a new dataset?               |                                                                                                      |

---

## 2. Create a data streaming solution using pipelines

| Criterion                                                                                                    | Status |
|--------------------------------------------------------------------------------------------------------------|--------|
| Did the student create a Git repository following the Microsoft TDSP framework structure?                   |        |
| Did the student create a diagram showing all steps required for building the models?                        |        |
| Did the student train a regression model using PyCaret and MLflow?                                          |        |
| Did the student calculate the Log Loss for the regression model and log it in MLflow?                       |        |
| Did the student train a decision tree model using PyCaret and MLflow?                                       |        |
| Did the student calculate Log Loss and F1 Score for the decision tree model and log them in MLflow?         |        |

---

## 3. Prepare a pre-trained model for a streaming data solution

| Criterion                                                                                                            | Status |
|----------------------------------------------------------------------------------------------------------------------|--------|
| Did the student define the project objective and describe each created artifact in detail?                          |        |
| Did the student cover all artifacts presented in the proposed diagram?                                               |        |
| Did the student use MLflow to log the "Data Preparation" run with relevant metrics and parameters?                  |        |
| Did the student remove missing data from the dataset?                                                                |        |
| Did the student select the indicated columns to train the model?                                                    |        |
| Did the student indicate the dimensions of the preprocessed dataset?                                                |        |
| Did the student create files for each processing phase and store them in the appropriate folders?                   |        |
| Did the student split the data into two sets: one for training and one for testing?                                 |        |
| Did the student create a pipeline called "Training" in MLflow?                                                      |        |

---

## 4. Establish a method to update the model in production

| Criterion                                                                                                                          | Status |
|-------------------------------------------------------------------------------------------------------------------------------------|--------|
| Did the student identify differences between the development and production datasets?                                              |        |
| Did the student describe how to monitor model health with and without the availability of the target variable?                    |        |
| Did the student implement a monitoring dashboard for the operation using Streamlit?                                                |        |
| Did the student describe reactive and predictive retraining strategies for the production model?                                  |        |
