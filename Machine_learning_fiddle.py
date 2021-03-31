import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import inspect
import sys
import plotly.figure_factory as ff
import numbers 
import plotly.express as px
np.random.seed(1)
#Include models here - works with scikit-learn classifiers. 
MODEL_MAP = {'Neural netowrk': MLPClassifier,
             'Ranndom forest ' : RandomForestClassifier, 
             'K-nearest neighbors':KNeighborsClassifier, 
             'Gradient boosting ' : GradientBoostingClassifier, 
             'Logistic regression' : LogisticRegression}
scaler = StandardScaler()
label_encoder = LabelEncoder()

def load_data(path_to_dataset):
    try: 
        data = pd.read_csv(path_to_dataset, delimiter = delimiter)
        st.write('Dataset loaded.')
        return data
    except Exception as e:
        st.write('Exception: ' + str(e))
        return None

def pipeline(data, TRAIN_SPLIT = 0.7, TEST_SPLIT = 0.15, VAL_SPLIT = 0.15, scaling = True):
    data = data.dropna(axis = 0)
    N = data.shape[0]
    feature_column_names = np.array([data.columns[i] for i in feature_columns])
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    train_indices = indices[0:int(TRAIN_SPLIT*N)]
    #Splitting data into training, evaluation- and test-data. 
    val_indices = indices[train_indices.shape[0]+1:train_indices.shape[0]+1+int(VAL_SPLIT*N)]
    test_indices = indices[val_indices.shape[0]+1:]
    target = data[data.columns[target_column]]
    X_train = data.iloc[train_indices][feature_column_names].values
    X_test = data.iloc[test_indices][feature_column_names].values
    X_val = data.iloc[val_indices][feature_column_names].values
    #Feature scaling if chosen.
    if scale is True:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)
    st.write('# Statistics from trainig set (70%).')
    st.write('## Features')
    #Creating a dataframe of X_train and y_train to easily display statistics.
    feature_description_dataframe = pd.DataFrame(X_train)
    feature_description_dataframe.columns = feature_column_names
    feature_description_dataframe = feature_description_dataframe.describe()
    st.write(feature_description_dataframe)
    target_description = target.iloc[train_indices].describe()
    classes = set(target.values)
    n_classes = len(classes)
    target_description['N classes'] = n_classes
    st.write('## Description and distribution of target')
    for cl in classes: 
        cl_count = sum(np.where(target == cl, 1, 0))
        target_description[f'Classs {cl} (%)'] = 100*cl_count/target.shape[0]
    col1, col2 = st.beta_columns(2)
    col1.dataframe(target_description, height = 450)
    with col2:
        fig = px.histogram(target.iloc[train_indices], width=400, height=485)
        st.plotly_chart(fig)
    correlation_matrix = data.iloc[train_indices].corr()
    st.write('## Correlation matrix')
    st.write(correlation_matrix)
    fig = px.imshow(correlation_matrix)
    fig.update_xaxes(side='top')
    st.plotly_chart(fig)
    #If y is not binary (ie, more classes than 2), we apply one-hot encoding.
    y_train_encoded = label_encoder.fit_transform(target.iloc[train_indices])
    y_test = target.iloc[test_indices]
    y_val = target.iloc[val_indices]
    return X_train, X_test, X_val, y_train_encoded, y_test, y_val

def execute(model, model_kwargs, fit_kwargs):
    model = model(**model_kwargs)
    y_val = y_val_encoded
    y_test = y_test_encoded
    one_hot_encoded = True
    model.fit(X = X_train, y = y_train_encoded)
    st.header('Model information')
    st.write(f'Using model **{type(model).__name__}**  with arguments: {model_kwargs}.' )
    st.header('Results')
    st.write(f'Model was fitted with arguemnts {fit_kwargs}.')
    predicted_train = label_encoder.inverse_transform(model.predict(X_train))
    predicted_val = label_encoder.inverse_transform(model.predict(X_val))
    st.write(f'''   |Set                        | Accuracy                              |
                    |---------------------------|---------------------------------------| 
                    |Train                      | {accuracy_score(label_encoder.inverse_transform(y_train_encoded), predicted_train) : 2.4f}|                    
                    |Validation                 | {accuracy_score(y_val, predicted_val)  : 2.4f}                        | 
    #                 ''')
    calculated_confusion_matrix = confusion_matrix(y_val, predicted_val)
    confusion_dataframe = pd.DataFrame(calculated_confusion_matrix)
    confusion_dataframe.columns = [str(label) for label in np.unique(y_val)]
    confusion_dataframe.index = confusion_dataframe.columns.copy()
    st.write('## Confusion matrix')
    st.write('Rows are true classes, columns are predicted classes. \n\n')
    cols =  st.beta_columns(2)
    with cols[0]:
        st.write(confusion_dataframe)
    with cols[1]:
        fig = px.imshow(confusion_dataframe, labels = dict(x = 'True class', y = 'Predicted class'))
        fig.update_xaxes(side='top')
        st.plotly_chart(fig)
        
def get_default_args(funciton):
    signature = inspect.signature(funciton)
    numeric_kwargs =  { k: v.default for k, v in inspect.signature(funciton).parameters.items() if (v.default is not inspect.Parameter.empty) and (type(v.default) in [int, float]) }
    string_kwargs = { k: v.default for k, v in inspect.signature(funciton).parameters.items() if (v.default is not inspect.Parameter.empty) and not ( isinstance(v.default, numbers.Number))}
    return numeric_kwargs, string_kwargs

def argument_selector(function):
    numeric_kwargs, str_kwargs = get_default_args(function)
    st.sidebar.subheader(function.__name__)
    if len(numeric_kwargs) + len(str_kwargs) == 0:
        st.sidebar.write('Takes no arguments')
        return {}
    n_args = len(str_kwargs.keys()) + len(numeric_kwargs.keys())
    kwargs = {}
    for key in numeric_kwargs.keys():
        value = numeric_kwargs[key]
        if int(value) == 0: 
            max_value = 1
            min_value = 0
        else:
            max_value = 10*value
            min_value = max(0, -max_value)

        if type(value) == float:
            min_value = float(min_value)
            max_value = float(max_value)
        kwargs[key] = st.sidebar.slider(str(key), 
                                    value = numeric_kwargs[key],
                                    min_value = min_value,
                                    max_value = max_value)
    for key in str_kwargs.keys():
        value = str_kwargs[key]
        kwargs[key] = st.sidebar.text_input(key + '?', value = value)
    return kwargs
st.sidebar.header('Data (Only numerical features supported.) ')
path_to_dataset = st.sidebar.text_input('Path to dataset', value = '~/streamlit_stuff/dataset.csv')
delimiter = st.sidebar.selectbox(('Delimiter?'), [',', ';'])
feature_columns = eval(st.sidebar.text_input('Feature columns? (allows python expressions. E.g, np.arange(10) )', value = '(2, 3, 5, 7)'))
target_column = int(st.sidebar.text_input('Feature columns?', value = '11'))
st.sidebar.subheader('Feature scaling?')
scale = st.sidebar.selectbox((''), ['Yes', 'No'])
scale = scale == 'Yes'
load_dataset_button = st.sidebar.button('Load data')    
try:
    dataset = load_data(path_to_dataset)
    X_train, X_test, X_val, y_train_encoded, y_test_encoded, y_val_encoded = pipeline(dataset, scaling = scale)
except Exception as e:
    st.write('Encountered an exception: ' + str(e))
st.sidebar.header('Parameters')
st.sidebar.markdown('## Model')
model = st.sidebar.selectbox((''), list(MODEL_MAP.keys()), index = 1)
st.sidebar.markdown('## Hyperparameters')
model = MODEL_MAP[model]
model_kwargs =  argument_selector(model.__init__)
fit_kwargs = argument_selector(model().fit)
kwargs = dict(model_kwargs)
run = st.sidebar.button('Run model')
if run:
    for kwargs in [model_kwargs, fit_kwargs]:
        for key, value in kwargs.items():
            try:
                kwargs[key] = (eval(value))
            except Exception:
                try:
                    kwargs[key] = value
                except Exception as e:
                    print('Exception: ' + str(e))
    execute(model, model_kwargs, fit_kwargs)
