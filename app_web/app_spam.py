import ergonomie
import model_prediction

import streamlit as st
import time




# Configuration de la page principale
aff = ergonomie.affichage()
aff.config_page()

# Affichage sur l'app web de la sidebar à gauche et de l'introduction
model, action = aff.colonne_gauche()
aff.premier_paragraphe()

# Chargement du CSV
predict = model_prediction.prediction(model)
predict.load_data()

# Affichage sur l'app web de la data avec les colonnes feature renseignées 
predict.modification_data(predict.data)
aff.data(predict.data)
st.write('DataFrame réalisée en ajoutant les nouvelles colonnes caractéristiques et les données du fichier CSV.')

# Changement de la couleur de la progress bar
aff.couleur_progressbar()

# Si on choisit de faire des prédictions suivant la data du CSV
if action == 'Prédiction Data':
    
     # Initialisation de la barre de progression
     my_bar = st.progress(0)
     
     st.subheader('Prédiction Data:')
     if(st.button("Classify")):
         
         time.sleep(1)
             
         # Split
         X_train, X_test, Y_train, Y_test = predict.split()
         my_bar.progress(20)
             
         # ColumnTransformer() + Pipeline() + GridSearchCV() + fit() + predict()
         # + calcule de la Matrice de confusion + calcule du Recall Score + calcule du Accuracy Score 
         # + calcule du ROC
         predict.training(X_train, Y_train)
         st.success(f"Classifier utilisé :  {predict.classifier}")
         my_bar.progress(40)
             
         # Affichage des melleurs paramètres suite au fit du gridsearchcv()
         aff.parametres(predict.grid.best_params_)
         my_bar.progress(50)
             
         # predict() + calcule de la Matrice de confusion + calcule du Recall Score 
         # + calcule du Accuracy Score + calcule du ROC
         predict.testing(X_test, Y_test)
         my_bar.progress(60)
             
         with st.expander("Scores Training Data"):
             
             col1, col2 = st.columns(2)
                    
             # Affichage de la gauge du Accuracy Score et de la Matrice de confusion
             with col1:
                 aff.gauges(predict.score_prediction_training, 'Score de Prédiction Globale ( Accuracy )')
                 aff.gauges(predict.score_precision_training, 'Score de Prédiction des Vrais Spam ( Precision )')
                 aff.confusion_matrix(predict.cf_matrix_training)
                 
             # Affichage de la gauge du Recall Score et du graphique du ROC
             with col2:
                aff.gauges(predict.score_recall_training, 'Score de Prédiction des Vrais Spam ( Recall )')
                aff.gauges(predict.score_f1_training, 'F1 Score')
                aff.roc(predict.fpr_training, predict.tpr_training, predict.roc_auc_training)
                     
         my_bar.progress(80)  
             
         
         with st.expander("Scores Test Data"):
             
             col1, col2 = st.columns(2)

             # Affichage de la gauge du Accuracy Score, du score de presicion et de la Matrice de confusion
             with col1:
                 aff.gauges(predict.score_prediction_testing, 'Score de Prédiction Globale ( Accuracy )')
                 aff.gauges(predict.score_precision_testing, 'Score de Prédiction des Vrais Spam ( Precision )')
                 aff.confusion_matrix(predict.cf_matrix_testing)

             # Affichage de la gauge du Recall Score, du F1 score et du graphique du ROC
             with col2:
                 aff.gauges(predict.score_recall_testing, 'Score de Prédiction des Vrais Spam ( Recall )')
                 aff.gauges(predict.score_f1_testing, 'F1 Score')
                 aff.roc(predict.fpr_testing, predict.tpr_testing, predict.roc_auc_testing)

         my_bar.progress(100)             

         st.balloons()
   

# Si on choisit de faire une prédiction (spam ou ham) suivant le message renseigné        
else:
     
     # Initialisation de la barre de progression
     my_bar = st.progress(0)
     
     st.subheader('Prédiction Simple :')
     
     # Entrée du message pour la prédiction
     exemple_pred = st.text_input('Message :')
     if(st.button("Classify")):
         
         time.sleep(1)
         
         # Pour analyser le message de la prédiction simple
         # Création et renseignement d'une data avec des colonnes pour les features
         predict.creation_data_message(exemple_pred)
         predict.modification_data(predict.data_message)
         aff.data(predict.data_message)
         my_bar.progress(20)
             
         # Split
         X_train, X_test, Y_train, Y_test = predict.split()
         my_bar.progress(40)
             
         # ColumnTransformer() + Pipeline() + GridSearchCV() + fit() + predict()
         # + calcule de la Matrice de confusion + calcule du Recall Score + calcule du Accuracy Score 
         # + calcule du ROC
         predict.training(X_train, Y_train)
         st.success(f"Classifier utilisé :  {predict.classifier}")
         my_bar.progress(60)
             
         # Predict() (Ham ou Spam) avec la data comportant les colonnes features crées et renseignées
         prediction_exemple = predict.grid.predict(predict.data_message.drop(['Message'], axis=1))
         
         # Score de Probabilité de Prédiction (Ham ou Spam) avec la data 
         # comportant les colonnes features crées et renseignées
         proba_predict = predict.grid.predict_proba(predict.data_message.drop(['Message'], axis=1))
         my_bar.progress(80)
             
         # Comme les targets ont été labélisées, la prédiction renvoie un 1 ou 0
         if prediction_exemple[0] == 1:
             st.info(f"🎉 Résultat :  Spam à {round(proba_predict[0][1]*100, 2)} %")
         else:
             st.info(f"🎉 Résultat :  Ham à {round(proba_predict[0][0]*100, 2)} %")
                 
         my_bar.progress(100)
             
         st.balloons()










