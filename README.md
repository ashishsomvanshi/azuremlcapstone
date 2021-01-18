*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.
## Note :
     The Deletion Section Shows an Error simply because the deployed Endpoint Service was deleted priror to deletion of compute target. Also the Notebooks are direct download of the code on Azure ML Studio. The same can be observed in the Full Video Recording Provided in the Screencast Section. 

## ML Task to determine death event baaed on various risk factors from different body parameters
*TODO:* Write a short introduction to your project.
    The Experiment is to predict the death event baaed on various risk factors from diffrent body parameters from the chossen datset using Azure Machine Learning Studio and select the models generated via Hyperdive Confugration and via Azure AutoML.
## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.
    The requireed azure ml workspace and experiment was created and azure ml studio was used to run Azure ML Notebooks which inturn ran on coumte vms and calulations were done via the jupyter notebook on a 4 node cluster of cloud based VMs of size STANDARD_D2_V2. To avoid duplication same compute resourses and datset created were used for both hyperdrive method and automl method.

## Dataset
## Overview
*TODO*: Explain about the data you are using and where you got it from.
    The Dataset website: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
    The Dataset URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv
## Source:
    The current version of the dataset was elaborated by Davide Chicco (Krembil Research Institute, Toronto, Canada) and donated to the University of California Irvine Machine Learning Repository under the same Attribution 4.0 International (CC BY 4.0) copyright in January 2020. Davide Chicco can be reached at <davidechicco '@' davidechicco.it>

## Data Set Information:
    A detailed description of the dataset can be found in the Dataset section of the following paper:
    Davide Chicco, Giuseppe Jurman: "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". BMC Medical Informatics and Decision Making 20, 16 (2020).

## Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
    The task is a two class classification task to predict the DEATH_EVENT column as 0 or 1.
    Various risk factors from diffrent body parameters are recored to predict the death event in the DEATH_EVENT column.

## Sample of features in the dataset: 
                "age": 45,
                "anaemia": 0,
                "creatinine_phosphokinase": 2413,
                "diabetes": 0,
                "ejection_fraction": 38,
                "high_blood_pressure": 0,
                "platelets": 140000,
                "serum_creatinine": 1.4,
                "serum_sodium": 140,
                "sex": 1,
                "smoking": 1,
                "time": 280,
                "DEATH_EVENT": 0

## Attribute Information:
    - age: age of the patient (years)
    - anaemia: decrease of red blood cells or hemoglobin (boolean)
    - high blood pressure: if the patient has hypertension (boolean)
    - creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
    - diabetes: if the patient has diabetes (boolean)
    - ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
    - platelets: platelets in the blood (kiloplatelets/mL)
    - sex: woman or man (binary)
    - serum creatinine: level of serum creatinine in the blood (mg/dL)
    - serum sodium: level of serum sodium in the blood (mEq/L)
    - smoking: if the patient smokes or not (boolean)
    - time: follow-up period (days)
    - [target] death event: if the patient deceased during the follow-up period (boolean)

## Access
*TODO*: Explain how you are accessing the data in your workspace.
    The Data was accesed via the Dataset URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv and loded into Azure ML Studio via python code in juypter notebook and required tabular Azure ML Dataset was created and consumed as reuired. To avoid duplication same Azure ML DAtaset was used for both hyperdrive method and automl method.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
## Classification technique used:
    Multiple Alogritims 

## Best Run Selection: 
    Selected out of multiple runs with diffrent algorititms with there auto generated hyperparameters. 

## Confugration Selected: 
    `
    automl_settings = {
        "experiment_timeout_hours" : 0.3,
        "enable_early_stopping" : True, (Enable early termination if the score is not improving in the short term) 
        "iteration_timeout_minutes": 5,
        "max_concurrent_iterations": 4,
        "max_cores_per_iteration": -1,
        "primary_metric": 'accuracy',
        "featurization": 'auto',
        "verbosity": logging.INFO,
    }
    automl_config = AutoMLConfig(
        experiment_timeout_minutes=30,
        debug_log = 'automl_errors.log',
        compute_target=compute_target,
        task="classification",
        training_data= train,
        label_column_name="DEATH_EVENT",    (Collumn to be classified)
        enable_onnx_compatible_models=True, (to emable saving output model in ONNX format)
        n_cross_validations= 3,
        **automl_settings)
    `
## Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
    The best performing model a model with acuuracy of aproximately .0.8926726726726727 with AUC_weight =  0.9208363636363636 using VotingEnsemble algoritim.
    
    Screenshot 1: RunDetails widget that shows the progress of the training runs of the different experiment
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl%20and%20Active%20Endpoint/AutoML%20Best%20Run%20Details%20Notebook.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

     Screenshot 2: RunDetails Wizard Completed
<img src=" https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl%20and%20Active%20Endpoint/AutoML%20RunDetailWizard%20Completed%20.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

     Screenshot 3: Remote Run Completed
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl%20and%20Active%20Endpoint/Automl%20Run%20Completion%20Details%201.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

     Screenshot 4: Best Run Details Completed (Visual 1)
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl%20and%20Active%20Endpoint/Automl%20Best%20Run%20Shown.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

     Screenshot 5: Best Run Details Completed 
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl%20and%20Active%20Endpoint/AutoML%20Best%20Run%20Details%20Notebook.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

     Screenshot 6: Best Run Metrics
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl%20and%20Active%20Endpoint/Automl%20Best%20Run.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

     Screenshot 7: Best Run Metrics (Visual 1)
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl%20and%20Active%20Endpoint/AutoML%20Best%20Run%2041%20.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

     Screenshot 8: Best Run Metrics (Visual 2)
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl%20and%20Active%20Endpoint/AutoML%20Best%20Run%2041%20%20Metrics.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

     Screenshot 9: Best Run Metrics (Visual 3)
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl%20and%20Active%20Endpoint/AutoML%20Best%20Run%2041%20%20Metrics%202.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

## Improving Results
    In future we can improve the best model by choosing different primary metrics and different classification methods and calculating and comparing the values of mean_squared_error, to study how our predictions have deviated from actual values and, we study mean absolute percent error (MAPE) in detail. We can also study the impact of increasing number of clusters used to study to get faster results. All these could help us in reducing error in our model and help us to study the model much faster. We can also add more data to the model, or we can add more columns. Also, we can make new columns with existing ones with feature engineering and by applying our domain knowledge and obtain better results. Also, we can provide a more user-friendly user interface wile consuming the api’s and the swager documentation. Lot of steps run on command-line can be ran directly from Jyupiter notebook or could be automated in a single script. We do have room to select more hyperparameters to tune in addtion to the two selected and tuned.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
    First the data was loaded into dataset and proper compute infra was created to run with suitable hyperdrive configuration. Random Sampling was choosen as it supports early termination of low-performance runs. In random sampling, hyperparameter values are randomly selected from the defined search space.
    
    Random Sampling was choosen as it supports early termination of low-performance runs. In random sampling, hyperparameter values are randomly selected from the defined search space with two hyperparameters '--C' (Reqularization Strength) and '--max_iter' (Maximum iterations to converge).
    
    enable_early_stopping = true, enables early termination if the score is not improving in the short term. BanditPolicy is an "aggressive" early stopping policy with the meaning that cuts more runs.

## Classification technique used: 
    Logistic Regression (Collumn to be classifed : DEATH_EVENT)

## Best Run Selection: 
    Selected out of multiple runs with same algorititm with diffrent hyperparameters.

## Confrugration Selected:
    Primary_metric= 'Accuracy'
    '
    early_termination_policy = BanditPolicy(slack_factor = 0.1, evaluation_interval = 1, delay_evaluation=5)
    (allowable slack = 0.1, frequency for applying the policy = 1, First policy evaluation doene after 5 intervals)
    ps = RandomParameterSampling(
        {
            "--C" :        choice(0.001,0.01,0.1, 0.5, 1,1.5,10,20,50,100,200,500,1000),
            "--max_iter" : choice(25,50,75,100,200,300)
        }
    )
    src = ScriptRunConfig(source_directory=script_folder,
                         script='train.py',
                         compute_target=compute_target,
                         environment=sklearn_env)
    hyperdrive_run_config =  HyperDriveConfig(
        hyperparameter_sampling = ps,
        primary_metric_name = 'Accuracy',
        primary_metric_goal = PrimaryMetricGoal.MAXIMIZE, 
        max_total_runs =  100,
        max_concurrent_runs = 4,
        policy = early_termination_policy,
        run_config = src
    )
    `
## Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
    With the above selected confugration the Hyperdrive Pipeline shows best results with --c = 500 and --max_iter = 50 giving accuracy of  0.8133333333333334(approximate).
    Screenshots: https://github.com/ashishsomvanshi/azuremlcapstone/tree/master/images/Hyperdrive
## Improving Results
    In future we can improve the best model by choosing different primary metrics and different classification methods and calculating and comparing the values of mean_squared_error, to study how our predictions have deviated from actual values and, we study mean absolute percent error (MAPE) in detail. We can also study the impact of increasing number of clusters used to study to get faster results. All these could help us in reducing error in our model and help us to study the model much faster. We can also add more data to the model, or we can add more columns. Also, we can make new columns with existing ones with feature engineering and by applying our domain knowledge and obtain better results. Also, we can provide a more user-friendly user interface wile consuming the api’s and the swager documentation. Lot of steps run on command-line can be ran directly from Jyupiter notebook or could be automated in a single script. We do have room to select more hyperparameters to tune in addtion to the two selected and tuned.
    
## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
    The best performing model was a model with acuuracy of aproximately .0.8926726726726727 with AUC_weight =  0.9208363636363636 using VotingEnsemble algoritim obtained via AutoML Method. The respective model was registered and deployed as rest endpoint via a scoring uri and the deployment was verified by sending two sets of data picked from the original dataset and sent as JSON data to the service and the service predicted the output correctly.
    
    Screenshot 1: Active Endpoint
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl%20and%20Active%20Endpoint/Active%20Endpoint.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

     Screenshot 2: Active Endpoint for Depoloyed Model Tested
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl%20and%20Active%20Endpoint/AutoML%20Depyment%20Test%20Result.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
 - A working model
 - Demo of the deployed  model
 - Demo of a sample request sent to the endpoint and its response
 
     Full Recording : https://1drv.ms/u/s!AqbnW4s20s0s1gynpKjiB2YCDMGI?e=efm01A
     
     Short Recording (5 Minitues) : https://1drv.ms/v/s!AqbnW4s20s0s1grRH4rLy6A0vbe7

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
   
    Convert Your Model to ONNX Format
    a) Hyperdrvive Method: The code to comvert and save the .joblib model to .onnx model was excecuted from the jupyter notebook.
    b) AutoML Method: The automl config was sent a setting to enable the onnx output via the jupyter code and respective ..onnx model was saved in .onnx format.
    
    Screenshot 1: Code excecution of .joblib model converstion to .onnx model for HyperDerive Method
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Standouts/Hyperdrive%20ONNX%20Convertion.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

     Screenshot 2: Code excecution of genration of .onnx model for AutoML Method
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl%20and%20Active%20Endpoint/AutoML%20Best%20model%20and%20Onnx%20Model.png"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />
    
    Enable logging in your deployed web app
    Respective sevrive logs were colleced and application insights was enabled as true via jupyter notebook code.
    
     Screenshot 3: Application Insights Were Enlabled
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Standouts/Application%20Insight%20True.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />
