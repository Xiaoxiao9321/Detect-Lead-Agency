import pandas as pd
import spacy
import numpy as np
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction import text
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report


def split(dataset):
    train = dataset.sample(frac=0.7, random_state=0)
    test = dataset.drop(train.index)

    return train, test


def create(lst, voca_index, g_label):
    xp = []
    for key in lst:
        content = key[0].split()
        for i in content:
            if i not in voca_index.keys():
                continue
            position = voca_index[i]
            key[1][position] = 0

            if i in g_label:
                c_indexs = content.index(i)
                index_word3 = 300 - c_indexs
                position3 = voca_index[i]
                key[1][position3] = index_word3

        xp.append(key[1])

    return xp


if __name__=="__main__":
    sp = spacy.load('en_core_web_sm')

    # d_train = pd.read_csv('train.csv')
    # d_test = pd.read_csv('test.csv')

    df = pd.read_csv('data.csv')
    d_train, d_test = split(df)

    x_train = d_train['content'].tolist()
    y_train = d_train['target'].tolist()

    x_test = d_test['content'].tolist()
    y_test = d_test['target'].tolist()

    g_label = list(set(y_train))

    print(len(y_train))
    print(len(set(y_test)))

    additional_stopwords = ['action', 'chapter', 'eis', 'statement', 'appendix', 'environment', 'area', 'study',
                            'impact', 'scoping', 'feasibility', 'notice']

    sw = text.ENGLISH_STOP_WORDS.union(additional_stopwords)

    vect = CountVectorizer(analyzer='word', stop_words=sw)

    a_x_train = vect.fit(x_train)

    voca_index = a_x_train.vocabulary_

    new_x_train = vect.fit_transform(x_train)

    new_train = new_x_train.toarray()

    train_data = np.copy(new_train)

    train_lst = list(zip(x_train, new_train))

    train_instances = create(train_lst, voca_index, g_label)

    train_instances_array = np.array(train_instances)

    # concatenate position feature
    train_array = np.concatenate((train_data, train_instances_array), axis=1)

    # train logistic regression
    clf = LogisticRegression(solver='liblinear', multi_class='auto', max_iter=500)
    clf.fit(train_array, y_train)


    #test data
    new_x_test = vect.transform(x_test)
    new_test = new_x_test.toarray()

    test_data = np.copy(new_test)
    test_lst = list(zip(x_test, new_test))
    test_instances = create(test_lst, voca_index, g_label)
    test_instances_array = np.array(test_instances)

    test_array = np.concatenate((test_data, test_instances_array), axis=1)

    train_pre = clf.predict(train_array)
    test_pre = clf.predict(test_array)

    print("train_accuracy:", accuracy_score(y_train, train_pre))
    print("test_accuracy", accuracy_score(y_test, test_pre))



    # performance metrics
    targets = list(set(y_test))
    report = classification_report(y_test, test_pre, labels=targets)
    print(report)





