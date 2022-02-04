#importation des librairies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import streamlit as st
from string import digits
import matplotlib.pyplot as plt
import collections
import matplotlib.cm as cm
from matplotlib import rcParams
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
import webbrowser

data = pd.read_csv('spam.csv',encoding= 'latin-1')
data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
data = data.rename(columns={'v1':'label', 'v2':'sms'})

    


    
def count_url(txt):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,txt)
    return  len([x[0] for x in url])
    
#stopword
stopwords = STOPWORDS
stopwords.add('co uk')
stopwords.add('u')
stopwords.add('will')
stopwords.add('-')
stopwords.add('&')
stopwords.add('2')
stopwords.add('4')
    
def create_dictionary(txt):
        
    filtered_words = [word for word in txt.split() if word not in stopwords]
    counted_words = collections.Counter(filtered_words)

    words = []
    counts = []
    #counte les mots les plus communs
    for letter, count in counted_words.most_common(10):
        words.append(letter)
        counts.append(count)
    return words

def make_dico(txt):       
    spam = txt['sms'].loc[txt.label=='spam']
    spam1 = ' '.join(spam.str.lower())
    dico = create_dictionary(spam1)
    return dico

dico = make_dico(data)    

def count_words(txt):
    
    count = 0
    words = dico
    for i in range (0, len (txt)):
        txt.lower()
        words = dico
        #Checks whether given character is a punctuation mark  
        if txt[i] in (words) :  
            count = count + 1;  
            #print(count)
    return count
       
def len_number(txt):
    number_list1 = re.findall(r"\D(\d{5})\D", txt)
    number_list2 =  re.findall("(?<!\d)\d{11}(?!\d)", txt)
    number_list3 =  re.findall("(?<!\d)\d{10}(?!\d)", txt)
    return len(number_list1 + number_list2 + number_list3) 
        
        
def count_special_char(str):
    count = 0
    for i in range (0, len (str)):   
    #Checks whether given character is a punctuation mark  
        if str[i] in ('!' ,"\'" ,"\"","-" ,"$", "£","?","#", "%","_","*","=", "&", "'", ":","+"):  
            count = count + 1;  
    return count
         
def lengh_word_mean(word):
    words = word.split()
    average = sum(len(word) for word in words) / (len(words)+0.1)
    return average
    
def get_features(data):    
    v = data.sms.values.astype(str)
    v = v.view(np.uint8).reshape(len(data), -1)
    data['length'] = data['sms'].apply(len)
    data["avg_word_len"]=data.sms.apply(lengh_word_mean)
    data['Uppercase'] = ((v >= 65) & (v <= 90)).sum(1)
    data['Lowercase'] = ((v >= 97) & (v <= 122)).sum(1)
    data["special_char"]=data.sms.apply(count_special_char)    
    data["spam_word"]=data.sms.apply(count_words)
    data["amount_url"]=data.sms.apply(count_url)     
    data['phone_number'] = data.sms.apply(len_number)      
        
    return data



def make_DF(data):   
    v = data.sms.values.astype(str)
    v = v.view(np.uint8).reshape(len(data), -1)
    data['length'] = data['sms'].apply(len)
    data["avg_word_len"]=data.sms.apply(lengh_word_mean)
    data['Uppercase'] = ((v >= 65) & (v <= 90)).sum(1)
    data['Lowercase'] = ((v >= 97) & (v <= 122)).sum(1)
    data["special_char"]=data.sms.apply(count_special_char)    
    data["spam_word"]=data.sms.apply(count_words)
    data["amount_url"]=data.sms.apply(count_url)     
    data['phone_number'] = data.sms.apply(len_number)      
        
    return data




def get_X_y(data):
    y = data['label']
    X = data.drop(['label','sms'], axis = 1 )
    return X, y
       