# Streamlit-stuff
This will be a container for streamlit resources and snippets.

## Quickstart: 

#### Basic requirements: 
  * streamlit
  * pandas
  * scikit-learn
  * matplotlib
  * plotly
  * numpy
  
#### Running a streamlit web-app: 
Use ```$ streamlit run '~/path_to_app/app.py' ``` 


 ## ML_fiddle

 ```ml_fiddle.py``` is a simple implementation to test various machine learning algorithms for classification on their chosen dataset. It allows the user to choose between various classification algorithms, fiddle with hyper-parameters to the model, and turn feature scaling on or off. It diplays information and statistics about the predictors and target variable, and estimates the out-of-sample accuracy using train-test-split, together with a confusion matrix. 

 #### Data
 The app accepts data in ```csv```-format, with separators ```,``` or ```;```. Feature columns must be numerical (non-categorical). The first row of the csv-file should contain the coulmn names. Decimal point should be ```.``` 

 #### Models 
 ```ml_fiddle``` currently suopport the following scikit-learn classification models:
  * MLPClassifier
  * KNeighborsClassifier
  * RandomForestClassifier
  * GradientBoostingClassifier

  To add more classifiers, edit the variable 
  ``` 
  MODEL_MAP = {'Neural netowrk': MLPClassifier,
             'Ranndom forest ' : RandomForestClassifier, 
             'K-nearest neighbors': KNeighborsClassifier, 
             'Gradient boosting ' : GradientBoostingClassifier, 
             'Logistic regression' : LogisticRegression} 
  ``` 
  
  The key is the name as it is displayed in the web-app. 
  
  ![image](https://user-images.githubusercontent.com/56155016/113124805-33bea500-9216-11eb-8689-336b8c6568d9.png)

