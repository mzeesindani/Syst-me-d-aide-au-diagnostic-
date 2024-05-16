import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import pickle
import numpy as np
from flask import Flask, render_template, request,redirect, url_for

app=Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def prediction():
    if request.method=='POST':
        sexe= request.form['sexe']
        etat_civil= request.form['etat-civil']
        temperature = request.form['temperature']
        FR = request.form['fr']
        tensionD = request.form['tad']
        service = request.form['service']
        motif = request.form['motif']
        if sexe=='0':
            sexe_val='M'

        else:
            sexe_val='F'
        if etat_civil =='0':
            etat='Marié(e)'
        elif etat_civil=='1':
            etat='Célibataire'
        
        elif etat_civil=='2':
            etat='Veuf(ve)'
        
        else:
            etat='Divorcé(e)'

        
        donnee=[sexe_val,etat,motif,temperature,tensionD,FR,service]
        X = pd.DataFrame([motif])


        df2 = X


        #dfakim= np.where(len(df['Message'])==0)

        for item in range(len(df2)):   
            if item == 0:
                corpus = [df2.loc[item][0]]
                #print(corpus)
                tok = [word_tokenize(df2.loc[item][0])]
                wordset = set(tok[item][:])
        ##print(corpus)
            else:

                corpus.append(df2.loc[item])
                tok.append(word_tokenize(df2.loc[item]))


                wordset = set(tok[0][:]).union(set(tok[item][:]))


        tfidf = TfidfVectorizer().fit(corpus)
        count = tfidf.get_feature_names_out(corpus)

        vector = tfidf.transform(corpus)
        df_tfidf_vect = pd.DataFrame(data = vector.toarray(),columns=count)


        #df_tfidf_vect.to_csv('tfidf.csv')
        #print(df_tfidf_vect[df_tfidf_vect['keystone_authtoken'] > 0])
        

        # df['Motif de consultation'] =  df_tfidf_vect.sum(axis=1)

        # df['Motif de consultation']

        motif_encoded=df_tfidf_vect.sum(axis=1)[0]



        # pd.set_option('display.max_rows', 10)
        # reg = pd.read_csv('C:/Users\hp/Desktop/log/datasetVec3.csv')
        # print(reg)
        # print(reg.head())
        if (sexe and etat_civil and temperature and tensionD and FR and service and motif):
            
            if service == 'pediatrie generale':
                pediatrie=1
                gynecologie=0
                med_interne=0
                obstetrique=0
            elif service == 'gynecologie':
                pediatrie=0
                gynecologie=1
                med_interne=0
                obstetrique=0
            elif service == 'medecine interne':
                pediatrie=0
                gynecologie=0
                med_interne=1
                obstetrique=0
            elif service == 'Obstetrique':
                pediatrie=0
                gynecologie=0
                med_interne=0
                obstetrique=1
            else :
                pass
            sexe= int(sexe)
            etat_civil= int(etat_civil)
            temperature= int(temperature)
            tensionD= int(tensionD)
            FR= int(FR)
            data=[sexe,etat_civil,temperature,FR,tensionD,motif_encoded,gynecologie,med_interne,obstetrique,pediatrie]
          


            data = [data]
            cols = ['Sexe', 'Etat-civil', 'Température', 'FR', 'TA_Diastolique','Motif de consultation_', 'gynecologie', 'medecine interne','obstetrique', 'pediatrie generale']
            df2 = pd.DataFrame(np.array(data), columns=cols)
            df2
            diagnostics={0:'infection urinaire', 1:'syndrome infectieux', 2:'ist', 3:'infection urogénitale',4:'gastrite', 5:'grippe', 6:'paludisme', 7:"travail d'accouchement", 8:'sepsis',9:'hypertension artérielle',10:'fièvre typhoïde'}
            # model here
            model=pickle.load(open('model.pkl', 'rb'))
         
            prediction_v=model.predict(df2)
            prediction=prediction_v[0]
            for key in diagnostics:
                if key==prediction:
                    diagnostic=diagnostics[prediction]
                    print(prediction)
                    return render_template('index.html', prediction = diagnostic, data = donnee)
            # les donnees sont stocke
            # prediction='this is the prediction'
        else:
            return render_template('index.html',error='fill all the fields', data=data)
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)