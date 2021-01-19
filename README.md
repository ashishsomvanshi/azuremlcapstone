## ML Task to determine death event baaed on various risk factors from different body parameters
    The Experiment is to predict the death event baaed on various risk factors from diffrent body parameters from the chossen datset using Azure Machine Learning Studio and select the models generated via Hyperdive Confugration and via Azure AutoML.
## Project Set Up and Installation
    The requireed azure ml workspace and experiment was created and azure ml studio was used to run Azure ML Notebooks which inturn ran on coumte vms and calulations were done via the jupyter notebook on a 4 node cluster of cloud based VMs of size STANDARD_D2_V2. To avoid duplication same compute resourses and datset created were used for both hyperdrive method and automl method.

## Dataset
## Overview
    The Dataset website: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
    The Dataset URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv
    
## Source:
    The current version of the dataset was elaborated by Davide Chicco (Krembil Research Institute, Toronto, Canada) and donated to the University of California Irvine Machine Learning Repository under the same Attribution 4.0 International (CC BY 4.0) copyright in January 2020. Davide Chicco can be reached at <davidechicco '@' davidechicco.it>

## Data Set Information:
    A detailed description of the dataset can be found in the Dataset section of the following paper:
    Davide Chicco, Giuseppe Jurman: "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". BMC Medical Informatics and Decision Making 20, 16 (2020).

## Task
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
    The Data was accesed via the Dataset URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv and loded into Azure ML Studio via python code in juypter notebook and required tabular Azure ML Dataset was created and consumed as reuired. To avoid duplication same Azure ML DAtaset was used for both hyperdrive method and automl method.
    Size of Training Dataset : 75% of Sample Dataset
    Size of Test Datset: 25% of Sample Dataset

## Automated ML
## Classification technique used:
    Multiple Alogritims 

## Best Run Selection: 
    Selected out of multiple runs with diffrent algorititms with there auto generated hyperparameters. 

## Confugration Selected: 
       
     automl_settings = {
         "experiment_timeout_hours": 0.5, (Exit Criteria: experiment should continue to run for maximum of .05 hours)
         "enable_early_stopping": True, (Enable early termination if the score is not improving in the short term) 
         "iteration_timeout_minutes": 5, (An iteration should continue to run for maximum of 5 minutes)
         "max_concurrent_iterations": 4, (Maximum 4 iterations should run concurrently)
         "max_cores_per_iteration": -1, (Use all the possible cores per iteration per child-run)
         "primary_metric": 'accuracy', (Primary Metric is selected as Accuracy)
         "featurization": 'auto', (During the preprocessing, data guardrails and featurization steps are performed automatically)
         "verbosity": logging.INFO, (Verbosity (log level) of log is selected as logging.INFO)
     }
     automl_config = AutoMLConfig(
         experiment_timeout_minutes=30, (Exit Criteria: Experiment should continue to run for maximum of 30 Minutes)
         debug_log = 'automl_errors.log', (Selection of file for debug logs)
         compute_target=compute_target, (Compute Target or the Cluster to run the task on is selected as compute_target)
         task="classification", (ML Task is selected as Classification)
         training_data= train, (Training Dataset is selected as train)
         label_column_name="DEATH_EVENT", (Column to be classified)
         enable_onnx_compatible_models=True, (To enable saving output model in ONNX format)
         n_cross_validations= 3, (Since the data set is smaller than 20,000 rows, cross validation approach is preferred for validation with 3 folds)
         **automl_settings)

## Results
    The best performing model a model with acuuracy of aproximately .0.87934 with AUC_weight =  0.92554 using VotingEnsemble algoritim.

     Screenshot 1: RunDetails widget that shows the progress of the training runs of the different experiment
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl/screenshot1.jpg"
     alt="RunDetails widget that shows the progress of the training runs of the different experiment"
     style="float: left; margin-right: 10px;" />
     
     Screenshot 1A: RunDetails widget that shows the progress of the training runs of the different experiment
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl/screenshot1A.jpg"
     alt="RunDetails widget that shows the progress of the training runs of the different experiment (A)"
     style="float: left; margin-right: 10px;" />
     
     Screenshot 2: RunDetails Wizard Completed
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl/screenshot2.jpg"
     alt="RunDetails Wizard Completed"
     style="float: left; margin-right: 10px;" />

     Screenshot 3: Remote Run Completed
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl/screenshot3.jpg"
     alt="Remote Run Completed"
     style="float: left; margin-right: 10px;" />

     Screenshot 4: Best Run Details Completed (Visual 1)
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl/screenshot4.jpg"
     alt="Best Run Details Completed (Visual 1)"
     style="float: left; margin-right: 10px;" />

     Screenshot 5: Best Run Details Completed 
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl/screenshot5.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

     Screenshot 6: Best Run Metrics
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl/screenshot6.jpg"
     alt="Best Run Metrics"
     style="float: left; margin-right: 10px;" />

     Screenshot 7: Best Run Metrics (Visual 1)
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl/screenshot7.jpg"
     alt="Best Run Metrics (Visual 1)"
     style="float: left; margin-right: 10px;" />

     Screenshot 8: Best Run Metrics (Visual 2)
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl/screenshot8.jpg"
     alt="Best Run Metrics (Visual 2)"
     style="float: left; margin-right: 10px;" />

     Screenshot 9: Best Run Metrics (Visual 3)
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Automl/screenshot9.jpg"
     alt="Best Run Metrics (Visual 3)"
     style="float: left; margin-right: 10px;" />

## Improving Results
    In future we can improve the best model by choosing different primary metrics and different classification methods and calculating and comparing the values of mean_squared_error, to study how our predictions have deviated from actual values and, we study mean absolute percent error (MAPE) in detail. We can also study the impact of increasing number of clusters used to study to get faster results. All these could help us in reducing error in our model and help us to study the model much faster. We can also add more data to the model, or we can add more columns. Also, we can make new columns with existing ones with feature engineering and by applying our domain knowledge and obtain better results. Also, we can provide a more user-friendly user interface wile consuming the api’s and the swager documentation. Lot of steps run on command-line can be ran directly from Jyupiter notebook or could be automated in a single script. We do have room to select more hyperparameters to tune in addtion to the two selected and tuned.

## Hyperparameter Tuning
    First the data was loaded into dataset and proper compute infra was created to run with suitable hyperdrive configuration. Random Sampling was choosen as it supports early termination of low-performance runs. In random sampling, hyperparameter values are randomly selected from the defined search space.
    
    Random Sampling was choosen as it supports early termination of low-performance runs. In random sampling, hyperparameter values are randomly selected from the defined search space with two hyperparameters '--C' (Reqularization Strength) and '--max_iter' (Maximum iterations to converge).
    
    enable_early_stopping = true, enables early termination if the score is not improving in the short term. BanditPolicy is an "aggressive" early stopping policy with the meaning that cuts more runs.

## Classification technique used: 
    Logistic Regression (Collumn to be classifed : DEATH_EVENT)

## Best Run Selection: 
    Selected out of multiple runs with same algorititm with diffrent hyperparameters.

## Confrugration Selected:

       
     early_termination_policy = BanditPolicy(slack_factor = 0.1, evaluation_interval = 1, delay_evaluation=5)
     (allowable slack = 0.1, frequency for applying the policy = 1, First policy evaluation done after 5 intervals)
     ps = RandomParameterSampling(
     {
        "--C" :        choice(0.001,0.01,0.1, 0.5, 1,1.5,10,20,50,100,200,500,1000), (Reqularization Strength search space)

        "--max_iter" : choice(25,50,75,100,200,300) (Maximum iterations to converge search space)

        }
     )
     src = ScriptRunConfig(source_directory=script_folder,
                     script='train.py',
                     compute_target=compute_target, (Compute Target or the Cluster to run the task on is selected as compute_target)
                     environment=sklearn_env (Environment for Sklearn selected as sklearn_env)
            )
    hyperdrive_run_config =  HyperDriveConfig(
    hyperparameter_sampling = ps,  (Hyperparameters)
    primary_metric_name = 'Accuracy', (Primary Metric is selected as Accuracy)
    primary_metric_goal = PrimaryMetricGoal.MAXIMIZE,  (Primary Metric Goal is selected as to maximize Accuracy)
    max_total_runs = 100, (Maximum 100 Child Runs)
    max_concurrent_runs = 4, (Maximum 4 iterations should run concurrently)
    policy = early_termination_policy,
    run_config = src
    )


## Results
    With the above selected confugration the Hyperdrive Pipeline shows best results with --c = 1 and --max_iter = 50 giving accuracy of  0.8133333333333334(approximate).
        
     Screenshot 1: RunDetails Wizard Completed
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Hyperdrive/screenshot1.jpg"
     alt="RunDetails Wizard Completed"
     style="float: left; margin-right: 10px;" />

     Screenshot 2: Remote Run Completed and Hyper Parameters for Sample of Diffrent Runs
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Hyperdrive/screenshot2.jpg"
     alt="Remote Run Completed and Hyper Parameters for Sample of Diffrent Runs"
     style="float: left; margin-right: 10px;" />

     Screenshot 3: Best Run Completed (Visual 1)
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Hyperdrive/screenshot3.jpg"
     alt="Best Run Completed (Visual 1)"
     style="float: left; margin-right: 10px;" />

     Screenshot 4: Best Run Details Completed (Visual 2)
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Hyperdrive/screenshot4.jpg"
     alt="Best Run Details Completed (Visual 2)"
     style="float: left; margin-right: 10px;" />

     Screenshot 5: Best Run Details Completed (Visual 3)
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Hyperdrive/screenshot5.jpg"
     alt="Best Run Details Completed (Visual 3)"
     style="float: left; margin-right: 10px;" />

     Screenshot 6: Best Run Metrics (Visual 1)
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Hyperdrive/screenshot6.jpg"
     alt="Best Run Metrics (Visual 1)"
     style="float: left; margin-right: 10px;" />
     
     
## Improving Results
    In future we can improve the best model by choosing different primary metrics and different classification methods and calculating and comparing the values of mean_squared_error, to study how our predictions have deviated from actual values and, we study mean absolute percent error (MAPE) in detail. We can also study the impact of increasing number of clusters used to study to get faster results. All these could help us in reducing error in our model and help us to study the model much faster. We can also add more data to the model, or we can add more columns. Also, we can make new columns with existing ones with feature engineering and by applying our domain knowledge and obtain better results. Also, we can provide a more user-friendly user interface wile consuming the api’s and the swager documentation. Lot of steps run on command-line can be ran directly from Jyupiter notebook or could be automated in a single script. We do have room to select more hyperparameters to tune in addtion to the two selected and tuned.
    
## Model Deployment
    The best performing model was a model with acuuracy of aproximately .0.87934 with AUC_weight =  0.92554 using VotingEnsemble algoritim obtained via AutoML Method. The respective model was registered and deployed as rest endpoint via a scoring uri and the deployment was verified by sending two sets of data picked from the original dataset and sent as JSON data to the service and the service predicted the output correctly.
    
    Screenshot 1: Active Endpoint for Depoloyed Model
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Endpoints/screenshot1.jpg"
     alt="Active Endpoint"
     style="float: left; margin-right: 10px;" />

     Screenshot 2: Active Endpoint for Depoloyed Model Tested
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Hperdrive/screenshot2.jpg"
     alt="Active Endpoint for Depoloyed Model Tested"
     style="float: left; margin-right: 10px;" />

## Screen Recording
     Full Recording Without Voice: https://1drv.ms/u/s!AqbnW4s20s0s1hrJxYLm7kivaxgu?e=ZhlmqV
     Short Recording With Voice (18 Minitues): https://1drv.ms/v/s!AqbnW4s20s0s1h98Kpw-Wx_jus_B
     
## Standout Suggestions
    Convert Your Model to ONNX Format
    a) Hyperdrvive Method: The code to comvert and save the .joblib model to .onnx model was excecuted from the jupyter notebook.
    b) AutoML Method: The automl config was sent a setting to enable the onnx output via the jupyter code and respective ..onnx model was saved in .onnx format.
    
    Screenshot 1: Code excecution of .joblib model converstion to .onnx model for HyperDerive Method
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Standouts/screenshot1.jpg"
     alt="Image Showing Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />

     Screenshot 2: Code excecution of genration of .onnx model for AutoML Method
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Standouts/screenshot2.jpg"
     alt="Code excecution of genration of .onnx model for AutoML Method"
     style="float: left; margin-right: 10px;" />
    
    Enable logging in your deployed web app
    Respective sevrive logs were colleced and application insights was enabled as true via jupyter notebook code.
    
     Screenshot 3: Application Insights Were Enlabled
<img src="https://github.com/ashishsomvanshi/azuremlcapstone/blob/master/images/Standouts/screenshot3.jpg"
     alt="Application Insights Were Enlabled"
     style="float: left; margin-right: 10px;" />
