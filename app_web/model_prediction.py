import pandas as pd
import re

from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import sklearn.metrics as metrics

class prediction:
    
    
    def __init__(self, model):
        
        # Paramètre model correspond au modèle de 
        # prédiction choisi sur l'interface sur l'app web
        self.model = model
        self.data = []
    
    
    def load_data(self):
       
       # Chargement du CSV
       self.data = pd.read_csv("spam.csv", encoding='ISO-8859-1')
       self.data.rename(columns={'v1': 'Spam/Ham', 
                                 'v2': 'Message'}, 
                        inplace=True)
       
       # Supression des colonnes vides du CSV
       del self.data['Unnamed: 2']
       del self.data['Unnamed: 3']
       del self.data['Unnamed: 4']

       # Création des colonnes pour les features
       self.data=self.data.assign(Taille_Message=0)# voir taille message
       self.data=self.data.assign(Ponctuations=0)# voir présence ponctuations
       self.data=self.data.assign(Majuscules=0)# voir nombre Majuscules
       self.data=self.data.assign(Emoticônes=0)# voir présence émoticônes
       self.data=self.data.assign(URL=0)# voir présence URL
       self.data=self.data.assign(MotSpam=0)# voir présence mots 
       self.data=self.data.assign(Téléphone_5=0)# voir présence nombre au moins à 5 chiffres
    
    
    def modification_data(self, df):
        
        # Parcourt chaque ligne de la data
        for i in df.index:
            
            # indique la longueur du message
            df['Taille_Message'][i] = len(df["Message"][i])
            
            # indique par un 1 si présence de caractères spéciaux 
            list_spec = re.compile("!|\\$|\\£|\\?|\\#|\\%|_|\\*|\\=|\\&|\\-")
            list_spec_find = list_spec.findall(df["Message"][i])
            if len(list_spec_find) >= 1:
                df['Ponctuations'][i] = 1
            
            # indique le nombre de majuscules
            list_uppercase_characters = re.findall(r"[A-Z]", df["Message"][i])
            df['Majuscules'][i] = len(list_uppercase_characters)
            
            # indique par un 1 si présence d'émoticônes
            list_emoticone = re.compile(":\\)|:-\\)|;\\)|;-\\)|:\\(|:-\\(|:D|:-D|;D|;-D|:P|:-P|;P|;-P")
            list_emoticone_find = list_emoticone.findall(df["Message"][i])
            if len(list_emoticone_find) >= 1:
                df['Emoticônes'][i] = 1
            
            # indique par un 1 si présence d' url
            #r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|(([^\s()<>]+|(([^\s()<>]+)))))+(?:(([^\s()<>]+|(([^\s()<>]+))))|[^\s`!()[]{};:'".,<>?«»“”‘’]))"
            list_motif_url = re.compile("www|@|http|https")
            list_url_find = list_motif_url.findall(df["Message"][i])
            if len(list_url_find) >= 1:
                df['URL'][i] = 1
            
            # indique par un 1 si présence de mots souvent présents dans les spam
            list_motspam = re.compile("free|call|price|win|won|new|now|cash|text|txt|mobile|urgent|150p")
            list_motspam_find = list_motspam.findall(df["Message"][i])
            if len(list_motspam_find) >= 1:
                df['MotSpam'][i] = 1
            
            # indique par un 1 si présence de nombres avec au moins 5 chiffres
            list_telephone_5 = re.findall(r"\d{5}", df["Message"][i])
            if len(list_telephone_5) >= 1:
                df['Téléphone_5'][i] = 1
            
    
    
    
    def split(self):
        
        X = self.data.drop(['Spam/Ham', 'Message'], axis=1)
        Y = self.data['Spam/Ham']
        
        #Binary targets transform to 0 (ham) and 1 (spam)
        Y = label_binarize(Y, classes=['ham', 'spam'])
        
        # Split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
        
        return X_train, X_test, Y_train, Y_test
        
    
    
    def training(self, X_train, Y_train):
        
        if self.model == 'Multinomial Naive Bayes':
            self.classifier = MultinomialNB()
            # Declare model and parameter for Grid Search
            param_grid = {
                            #'model__alpha' : [0, 1e-08, 1e-06, 1e-04, 1e-02, 1],
                            'model__fit_prior': [True, False]
                            }
            
        elif self.model  == 'K Nearest Neighbors':
            self.classifier = KNeighborsClassifier()
            # Declare model and parameter for Grid Search
            param_grid = {
                            'model__n_neighbors': [3, 5, 7],
                            'model__weights': ['uniform', 'distance'],
                            #'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                            }
            
        elif self.model  == 'Logistic Regression':
            self.classifier = LogisticRegression()
            # Declare model and parameter for Grid Search
            param_grid = {
                            'model__penalty': ['l1', 'l2', 'elasticnet', 'none']
                            }
            
        elif self.model  == 'XGBoost':
            self.classifier = XGBClassifier()
            # Declare model and parameter for Grid Search
            param_grid = {
                            'model__booster': ['gbtree', 'gblinear', 'dart'],
                            #'model__tree_method': ['auto', 'exact', 'approx', 'hist', 'gpu_hist']
                            }
            
        elif self.model  == 'Random Forest':
            self.classifier = RandomForestClassifier()
            # Declare model and parameter for Grid Search
            param_grid = {
                            'model__criterion': ["gini", "entropy"],
                            'model__n_estimators': [90, 100, 115, 130],
                            #'model__min_samples_leaf': [1, 2, 5, 10, 15],
                            #'model__min_samples_split': [2, 5, 10, 15, 20],
                            #'model__max_features': ['auto', 'sqrt', 'log2'],
                            }
        
        
        # Declare the pipeline
        transformer_num = None
        '''
        transformer_num = ColumnTransformer(transformers=[
                                                        ('Scaling', RobustScaler(), ['Taille_Message', 'Majuscules'])
                                                        ]
                                            )
        '''
        pipe = Pipeline(steps=[
                               #('transformer', transformer_num),
                               ('smote', SMOTE(sampling_strategy=0.25)),
                               ('under',RandomUnderSampler(sampling_strategy=0.5)),
                               ('model', self.classifier)
                               ]
                        )
        
        # Declare the Grid Search method
        self.grid = GridSearchCV(pipe, param_grid, scoring = 'accuracy', cv=StratifiedKFold(n_splits = 5, shuffle = True, random_state = 123))
        
        # Fit the model
        self.grid.fit(X_train, Y_train)
        
        # Make predictions
        pred_on_train_data = self.grid.predict(X_train)
       
        # Matrice de confusion
        self.cf_matrix_training = confusion_matrix(Y_train, pred_on_train_data)
        
        # Recall Score
        #self.score_recall_training = recall_score(Y_train, pred_on_train_data, average='binary', pos_label='spam')
        self.score_recall_training = recall_score(Y_train, pred_on_train_data)
        self.score_recall_training = round(self.score_recall_training*100, 2)
        
        # Accuracy Score
        self.score_prediction_training = accuracy_score(Y_train, pred_on_train_data)
        self.score_prediction_training = round(self.score_prediction_training*100, 2)
        
        # Precision Score
        self.score_precision_training = precision_score(Y_train, pred_on_train_data)
        self.score_precision_training = round(self.score_precision_training*100, 2)
        
        # F1 Score
        self.score_f1_training = f1_score(Y_train, pred_on_train_data)
        self.score_f1_training = round(self.score_f1_training*100, 2)
        
        # ROC
        probs = self.grid.predict_proba(X_train)
        preds = probs[:,1]
        #self.fpr_training, self.tpr_training, threshold = metrics.roc_curve(Y_train, preds, pos_label='spam')
        self.fpr_training, self.tpr_training, threshold = metrics.roc_curve(Y_train, preds)
        self.roc_auc_training = metrics.auc(self.fpr_training, self.tpr_training)
        
        


    def testing(self, X_test, Y_test):
        
        # Make predictions
        pred_on_test_data = self.grid.predict(X_test)
        
        # Matrice de confusion
        self.cf_matrix_testing = confusion_matrix(Y_test, pred_on_test_data)
        
        # Recall Score
        #self.score_recall_testing = recall_score(Y_test, pred_on_test_data, average='binary', pos_label='spam')
        self.score_recall_testing = recall_score(Y_test, pred_on_test_data)
        self.score_recall_testing = round(self.score_recall_testing*100, 2)
        
        # Accuracy Score
        self.score_prediction_testing = accuracy_score(Y_test, pred_on_test_data)
        self.score_prediction_testing = round(self.score_prediction_testing*100, 2)
        
        # Precision Score
        self.score_precision_testing = precision_score(Y_test, pred_on_test_data)
        self.score_precision_testing = round(self.score_precision_testing*100, 2)
        
        # F1 Score
        self.score_f1_testing = f1_score(Y_test, pred_on_test_data)
        self.score_f1_testing = round(self.score_f1_testing*100, 2)
        
        
        # ROC
        probs = self.grid.predict_proba(X_test)
        preds = probs[:,1]
        #self.fpr_testing, self.tpr_testing, threshold = metrics.roc_curve(Y_test, preds, pos_label='spam')
        self.fpr_testing, self.tpr_testing, threshold = metrics.roc_curve(Y_test, preds)
        self.roc_auc_testing = metrics.auc(self.fpr_testing, self.tpr_testing)
        
        
    def creation_data_message(self, message):
        
        # Pour analyser le message de la prédiction simple
        # Création des colonnes pour les features
        self.data_message = pd.DataFrame(columns=['Message'])
        self.data_message.loc[0]=[message]
        self.data_message=self.data_message.assign(Taille_Message=0)# voir taille message
        self.data_message=self.data_message.assign(Ponctuations=0)# voir présence
        self.data_message=self.data_message.assign(Majuscules=0)# voir nombre Majuscules
        self.data_message=self.data_message.assign(Emoticônes=0)# voir présence émoticônes
        self.data_message=self.data_message.assign(URL=0)# voir présence URL
        self.data_message=self.data_message.assign(MotSpam=0)# voir présence mots 
        self.data_message=self.data_message.assign(Téléphone_5=0)# voir présence nombre au moins à 5 chiffres
        
