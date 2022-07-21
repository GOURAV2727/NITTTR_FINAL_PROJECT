# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:13:10 2022

@author: Gourav
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



st.title("PREDICTION GUI")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
#pre_name=st.sidebar.selectbox('Preprocessing:',['Scatter Plot','Heatmap'])    
clf_name = st.sidebar.selectbox('Select Classifier:',['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Decision Tree','KNN','Ada Boosting']) 
test_percent = st.sidebar.slider("Select Test Data Percentage:", 10, 30, 20, 5)\


def add_parameter_ui(clf_name):
    para = dict()
    if clf_name == "KNN":
        n_neighbors = st.sidebar.slider("n_neighbors", 1, 5,5,1)
        para["n_neighbors"] = n_neighbors
    
    return para

para = add_parameter_ui(clf_name)


def get_classifier(clf_name, para):
    if clf_name == "Logistic Regression":
        clf = LogisticRegression()
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier()
    elif clf_name == "Gradient Boosting":
        clf = GradientBoostingClassifier()        
    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = para["n_neighbors"])
    elif clf_name == "Decision Tree":
        clf= DecisionTreeClassifier()
    elif clf_name =="Ada Boosting":
        clf= AdaBoostClassifier()    
        
    return clf


clf = get_classifier(clf_name, para)


    

# Classsification
y = df.iloc[:,-1].values
X = df.iloc[:,:-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (test_percent/100), random_state=21)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) 

#acc = accuracy_score(y_test, y_pred)*100
#st.write(f""" ##### Accuracy = {acc}""") 
col1, col2 = st.columns(2)

with col1:
    acc = accuracy_score(y_test, y_pred)*100
    st.write(f""" 
             ##### Classifier = {clf_name}
             ##### Accuracy = {acc}
    """)
    
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(ax=ax, data=cm, annot=True)
    
    if st.button("Confusion Matrix"):
        st.pyplot(fig, figsize=(2, 2))
        
with col2:
    row, col = df.shape
    st.write(f"""
             ##### Number of Rows: {str(row)}
             ##### Number of Columns: {str(col)}
    """)
    
    if st.button("Data Description"):
        st.write(df.describe())

#'''feature = df.iloc[:,:-1] 

#feat_col = st.sidebar.selectbox("Select a feature to plot:", feature) 
#st.set_option('deprecation.showPyplotGlobalUse', False)
# Plot Features
#M = df[(df.iloc[:,:-1] )]
#B = df[(df.iloc[:,-1])]

#def plot_distribution(data_select, size_bin) : 
 #   x=df.iloc[:,3]
 # y=df.iloc[:,2]  
    
 #   group_labels = ['malignant', 'benign']
  #  colors = ['#FFD700', '#7EC0EE']

    # figu, ax = plt.subplots()
    # figu = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, bin_size = size_bin, curve_type='kde')
    # figu = plt.scattertter(x,y, color=colors)

# plott = plot_distribution(feat_col, .5)


# st.header("Plotting Graph:")
# if feat_col:
    # st.pyplot(plott)       



