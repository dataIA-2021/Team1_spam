#LIBRAIRY

import streamlit as st
import prog as pg
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, precision_score, recall_score
import seaborn as sns
import time
#THE GRAPHICS 
import plotly.graph_objects as go
#from matplotlib.pyplot import savefig
import matplotlib as mpl
import plotly.io as pio
import seaborn as sns


#THE FUNCTIONS OF THE APPS

def add_parameter_ui(clf_name,X_train, y_train):
    params = dict()
    if clf_name == 'KNN':
        n_neighbors = st.sidebar.slider('n_neighbors',1,15,2)
        params['n_neighbors'] = n_neighbors
        params = params['n_neighbors']
        #params = K['K']
        Q = st.sidebar.checkbox('AUTO')
        if Q:
           params = dict()
           clf = KNeighborsClassifier()
           k_range = list(range(1, 15,2))
           param_grid = dict(n_neighbors=k_range)
           grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', return_train_score=False,verbose=1)
           with st.spinner('GridSearch processing'):
               time.sleep(15)
           grid_search=grid.fit(X_train, y_train)
           Q=grid_search.best_params_
           params = Q['n_neighbors']
           print(params)
    elif clf_name == 'SVM':
        C = st.sidebar.slider('C', 1, 10)
        params['C'] = C
        params = params['C']
        Q = st.sidebar.checkbox('AUTO')
        if Q:
           params = dict()
           clf = SVC()
           k_range = list(range(1, 10,1))
           param_grid = dict(C=k_range)
           grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', return_train_score=False,verbose=1)
           with st.spinner('GridSearch processing'):
               time.sleep(30)
           grid_search=grid.fit(X_train, y_train)
           Q=grid_search.best_params_
           params = Q['C']
           print(params)
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
        Q = st.sidebar.checkbox('AUTO')
        if Q:
           params = dict()
           clf = RandomForestClassifier()
           k_range1 = list(range(2, 15,2))
           k_range2 = list(range(10, 100,10))
           param_grid = dict(n_estimators=k_range1,max_depth=k_range2)
           grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', return_train_score=False,verbose=1)
           with st.spinner('GridSearch processing'):               
               time.sleep(30)
           grid_search=grid.fit(X_train, y_train)
           Q=grid_search.best_params_
           params['max_depth'] = max_depth
           params['n_estimators'] = n_estimators 
         
             
    return params



def get_classifier(clf_name, params):
    
    if clf_name == 'KNN':
       clf = KNeighborsClassifier(n_neighbors=params)
    
    elif clf_name == 'SVM':
        clf = SVC(C=params)
        
    else:    
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],random_state=45)
    return clf

def get_dataset(dataset_name):
    if dataset_name == "spam":
        data = pd.read_csv('spam.csv',encoding= 'latin-1')
        data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
        data = data.rename(columns={'v1':'label', 'v2':'sms'})
        return data
    
    else: 
        data = pd.read_csv('spam.csv',encoding= 'latin-1')
        data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
        data = data.rename(columns={'v1':'label', 'v2':'sms'})
        return data
        data1 = st.text_input('enter your text')
        data_input = pd.DataFrame(columns = ['sms'])
        data_input.loc[0] = [data1]
        return data_input



#creation de l'application title

st.title("SPAM DETECTOR")

# st.write("""
# # Explore different classifier
         
#  """)
from PIL import Image
image = Image.open('correo-spam.jpg')

st.image(image, caption='Sunrise by the mountains')

metrics1 = st.sidebar.multiselect("What metrics to plot?",['Confusion Matrix',"Accuracy",'Recall', 'ROC Curve', 'Precision-Recall Curve'])

classifier_name = st.sidebar.selectbox("Select classifier", ("KNN","SVM", "Random Forest"))




#READ THE CSV AND GET THE FEATURES
data = pd.read_csv('spam.csv',encoding= 'latin-1')
data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
data = data.rename(columns={'v1':'label', 'v2':'sms'})
data = pg.get_features(data)


###############################MACHINE LEARNING#################################


#GET THE FEATURE AND THE LABEL FOR THE SPLIT


X,y = pg.get_X_y(data)

print(X)
print(y)


#SPLIT OF OUR DATASET


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10, stratify=(y))


#GET THE PARAMETERS FROM THE INTERFACE

params = add_parameter_ui(classifier_name,X_train,y_train)

#GET THE CLASSIFIER FROM THE INTERFACE

clf = get_classifier(classifier_name, params)
print(clf)
#FITING THE MODEL AND PREDICTING

model = clf.fit(X_train, y_train)
print(model)
y_pred = model.predict(X_test)
print('Y_PRED',y_pred)


#INTERFACE 

#THE USER INPUT

data1 = st.text_input('enter your text')
data_input = pd.DataFrame(columns = ['sms'])
data_input.loc[0] = [data1]
data_input = pg.get_features(data_input)
print(data_input)

#SHOW THE DATA FRAME
#st.write(data_input)


#PREDICTION OF THE INPUT CALCULATION
X_predict = data_input.drop(['sms'], axis = 1 )
print(X_predict)
y_pred_imput = model.predict(X_predict)
print(y_pred_imput)
y_proba = model.predict_proba(X_predict)
print(y_proba)
y_pred_imput = " ".join(y_pred_imput)# change into string 
y = y_proba.flatten() #changer la dimention du DF de 2d a 1d 



#SHOW THE PREDICTION
import time
if st.button('go'):
    with st.spinner('Wait for it...'):
        time.sleep(5)
        st.success('Done!')
        #st.write(data_input)
        st.write("It's ", y_pred_imput)        
        #st.write(y_proba)
        my_labels = 'ham','spam'
        plt.pie(y,labels=my_labels,autopct='%1.1f%%',colors=['green','red'],radius=1)
        plt.title('Probability')
        plt.axis('equal')
        st.pyplot()
else:
    st.write('wait for your message')


#SHOW THE ORIGINAL DATAFRAME
st.write(data)
#st.write(data_input)
#st.write(params)
# prediction de notre modele

#METRICS OF THE MODELS

#CONFUTION METRICS
#metrics1 = st.sidebar.multiselect("What metrics to plot?",['Confusion Matrix',"Accuracy",'ROC Curve', 'Precision-Recall Curve'])





def plot_metrics(metrics_list,y_test, y_pred):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if "Accuracy" in metrics_list:
        accuracy = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        st.write("Accuracy ", accuracy.round(2))
        import plotly.graph_objects as go
        fig1 = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = accuracy*100,
            domain = {'x': [1, 0], 'y': [1, 0]},
            title = {'text': "accuracy", 'font': {'size': 24}},
            delta = {'reference': 100, 'increasing': {'color': "RebeccaPurple"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': accuracy, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': 'red'},
                    {'range': [50, 100], 'color': 'green'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100}}))
        fig1. update_layout( autosize=False, width=100, height=300,)
        fig1.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
        st.plotly_chart(fig1, use_container_width=True)


    if "Recall" in metrics_list:
        recall = recall_score(y_test, y_pred, pos_label='spam').round(2)
        import plotly.graph_objects as go
        fig1 = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = recall*100,
            domain = {'x': [1, 0], 'y': [1, 0]},
            title = {'text': "Recall", 'font': {'size': 24}},
            delta = {'reference': 100, 'increasing': {'color': "RebeccaPurple"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': recall, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': 'red'},
                    {'range': [50, 100], 'color': 'green'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100}}))
        fig1. update_layout( autosize=False, width=100, height=300,)
        fig1.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
        st.plotly_chart(fig1, use_container_width=True)


    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix") 
        plot_confusion_matrix(clf, X_test, y_test)
        st.pyplot()
        
    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve") 
        plot_roc_curve(clf, X_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(clf, X_test, y_test)
        st.pyplot()




plot_metrics(metrics1,y_test, y_pred)














        
   
       



# # accuracy = clf.score(X_test, y_test)
# # y_pred = clf.predict(X_test)
# st.write("Accuracy ", accuracy.round(2))
# # st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
#st.write("Recall: ", recall_score(y_test, y_pred, pos_label='spam').round(2))


#ACCURACY
#acc = accuracy_score(y_test,y_pred)








##########################CREATION OF THE FIGURES########################
#GAUGE ACCURACY




#CONFUSION MATRIX









#SHOW THE FIGURES IN THE APP





#st.pyplot(fig)



































#IDEAS 

# st.write(f"classifier = {classifier_name}")
# st.write(f"accuracy= {acc}")



# pca = PCA(2)
# X_projected = pca.fit_transform(X)
# x1 = X_projected[:,0]
# x2 = X_projected[:,1]

# fig = plt.figure()
# plt.scatter(x1, x2, c='yellow', alpha=0.8, cmap="viridis")

# plt.xlabel('principal component1')
# plt.ylabel('principal component2')
# plt.colorbar()
# st.pyplot(fig)




