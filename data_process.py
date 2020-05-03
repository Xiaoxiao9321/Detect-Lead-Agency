
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import re
from sklearn.linear_model import LogisticRegression
import spacy
import csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def preprocess_text(sen):

    sentence = re.sub('\sU.S.\s',' US ',sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence



if __name__ == '__main__':

    sp = spacy.load('en_core_web_sm')
    df = pd.read_csv('eis.csv')

    names= df['new_name'].tolist()
    agency = df['new_Agency'].tolist()
    agency = [item.lower() for item in agency]

    n_agency = df['normalized'].tolist()
    # n_agency = [item.lower() for item in n_agency]

    n_name_dict = dict(zip(agency,n_agency))

    dict_agency=dict(zip(names,n_agency))

    # print(n_name_dict)


    dict_content = {}
    folder_path = '/Users/xiaoxiaochen/Desktop/EIS_lead_agency/largest_PDF_texts/'

    for root, dir, files in os.walk(folder_path):
        for i in files:
            if i == '.DS_Store':
                continue

            f_path = os.path.join(root,i)

            f = open(f_path, 'r', encoding='utf-16')
            content= f.read().strip()
            new_content= preprocess_text(content)

            sentence = [str(word) for word in sp(new_content)][0:1000]

            # with open("Output.txt", "w") as text_file:
            #     text_file.write(new_content)

            print(sentence)



            # if len(sentence) < 1000:
            #     continue

            sent = ' '.join(sentence).lower()

            for key, value in n_name_dict.items():

                sent = sent.replace(key,value)

            dict_content[f_path.split('/')[-2]] = sent


    # content_list=[]
    # agency_list =[]
    dict_train={}

    for key, value in dict_content.items():
        for key1,value1 in dict_agency.items():
            if key == key1:
                dict_train[key]=[]
                #dict_train[tuple(value)]= value1
                dict_train[key].append(value)
                dict_train[key].append(value1)
                # content_list.append(value)
                # agency_list.append(value1)



    with open('data.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['document','content','target'])
        for key,value in dict_train.items():

            writer.writerow([key,value[0],value[1]])





    






