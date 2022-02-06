import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

# Class pour affichage sur l'interface de l'app web
class affichage:
    
    # Configuration de la page principale
    def config_page(self):
        
        st.set_page_config(
        page_title="Spam App Web",
        layout="wide",
        initial_sidebar_state="expanded",
        )
    
    
    # Fonction pour afficher la colonne à gauche de l'interface de l'app web
    # Contient le choix des types de prédiction 
    # et le choix des modèles de prédiction 
    def colonne_gauche(self):
        
        st.sidebar.write(" # Steve et Jérémy")
        action = st.sidebar.radio("Action :",('Prédiction Data', 'Prédiction Simple'))
        model = st.sidebar.selectbox('Choix du Modèle de Classification :',
                                     ('Multinomial Naive Bayes','K Nearest Neighbors', 'Logistic Regression', 'Random Forest', 'XGBoost'))
        
        return model, action


    # Fonction pour afficher l'introduction
    def premier_paragraphe(self):
    
        st.write("""
                 # Projet : Spam Classifier
                 Le spam électronique désigne la réception d’un message non voulu à des fins publicitaires, commerciales ou malveillantes.
                """)
    
                
    # Fonction pour afficher la data
    def data(self, df):
        
        st.dataframe(df, 3000, 210)
    
    
    # Fonction pour afficher les gauges des scores 
    # de prédiction de accuracy et de recall
    def gauges(self, score, titre):
        
        fig = go.Figure(go.Indicator(
                                    mode = "gauge+number",
                                    gauge = {'axis': {'range': [None, 100]}, 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': score}},
                                    value = score,
                                    number = {'suffix': " %"},
                                    domain = {'row': 0, 'column': 0},
                                    title = {'text': titre}))
        '''
        fig.update_layout(
                            autosize=False,
                            width=250,
                            height=250,
                            margin=dict(
                                        l=20,
                                        r=20,
                                        b=10,
                                        #t=10,
                                        #pad=4
                                        ),
                            )
        '''
        st.plotly_chart(fig, use_container_width=True)
        
        
    # Fonction pour afficher la matrice de confusion
    def confusion_matrix(self, cf_matrix):
        
        fig, ax = plot_confusion_matrix(conf_mat=cf_matrix,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=True,
                                class_names=['Ham', 'Spam'],
                                cmap='Greens',
                                figsize=None)

        ax.set_title('Matrice de Confusion')
        
        st.pyplot(fig)
      
        
    # Fonction pour afficher les melleurs paramètres
    # suite au fit du gridsearchcv()
    def parametres(self, dico):
        
        for cle, valeur in dico.items():
            st.info(f" Pour un meilleur Score 'Accuracy'', pour le Paramètre : '{cle}',  la meilleur Valeur est :  {valeur}")


    # Fonction pour afficher le graphique du ROC
    def roc(self, fpr, tpr, roc_auc):
        
        fig, ax = plt.subplots()
        
        ax.set_title('Receiver Operating Characteristic')
        ax.plot(fpr, tpr, 'g', label = 'AUC = %0.2f' % roc_auc)
        ax.legend(loc = 'lower right')
        ax.plot([0, 1], [0, 1],'r--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        
        st.pyplot(fig)
        
    # Fonction pour changement de la couleur de la progress bar
    def couleur_progressbar(self):
        
        st.markdown("""
            <style>
                .stProgress > div > div > div > div {
                    background-image: linear-gradient(to right, #99ff99 , #00ccff);
                }
            </style>""",
            unsafe_allow_html=True)
    
    
        
        
       
        
