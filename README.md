# Rainfall Prediction using Azure Automated Machine Learning
This repository contains an automated machine learning (AutoML) project that predicts rainfall in Australia based on weather data. The project uses Azure Machine Learning service to run an AutoML experiment and find the best model for the prediction task.

## Dataset
The dataset used for this project is the Rain in Australia dataset from Kaggle. It contains daily weather observations from various locations in Australia, such as temperature, humidity, wind speed, sunshine, etc. The target variable is RainTomorrow, which indicates whether it rained the next day or not.

The dataset has 145,460 rows and 23 columns. It has some missing values and categorical features that need to be handled before feeding it to the AutoML experiment.

## AutoML Experiment
The AutoML experiment is configured with the following settings:

- Task: Classification
- Primary metric: Accuracy
- Validation: 5-fold cross-validation
- Max concurrent iterations: 5
- Max cores per iteration: -1 (use all available cores)
- Featurization: Auto (automatically handle missing values and categorical features)
The experiment runs on a local compute target with 4 cores and 16 GB of RAM. It tries various algorithms and hyperparameters to find the best model for the prediction task.

## Results
The best model found by the AutoML experiment is a Voting Ensemble model with an accuracy of 0.8588 on the validation set. The Voting Ensemble model combines the predictions of several base models, such as LightGBM, XGBoost, Logistic Regression, etc.

## Deployment
The best model can be deployed as a web service using Azure ML Studio. The web service can receive input data as JSON and return the predicted labels as JSON. The web service also provides a Swagger UI for testing and documentation.

To deploy the best model as a web service, follow these steps:

1. Register the best model in Azure ML Studio.
1. Create an inference config file that specifies the scoring script and the environment for the web service.
1. Create an Azure Container Instance (ACI) or an Azure Kubernetes Service (AKS) as the compute target for the web service.
1. Deploy the model to the compute target using Azure ML SDK or Azure ML Studio.
1. Test the web service using Postman or curl.
For more details on how to deploy models with Azure Machine Learning, see [" Deploy models with Azure Machine Learning ".](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where)

## Scripts
This repository contains two scripts that perform different tasks related to the project. Here is a brief description of each script:

* **conda_env_v_1_0_0.yml** : YAML file which contains the configuration pre-requisites for this experiment
* **script_run_notebook.ipynb**: A Jupyter notebook that contains the code to load the data, configure the AutoML settings, call **script.py** , run the experiment, and evaluate the results.
* **script.py** : contains the script to define the logging functions , featurization steps, 
* **scoring_file_v_2_0_0.py**: A scoring script that defines how to load and use the best model for inference
* **model.pkl** : the serialized version of the model

