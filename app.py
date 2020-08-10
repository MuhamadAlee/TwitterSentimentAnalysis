import threading
from tkinter import Tk
import os
import pickle
import re
from nltk import word_tokenize, textwrap
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import string
import pandas as pd
import nltk
nltk.download('wordnet')
import csv
import nltk
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

from flask import Flask, render_template, request, redirect


def gui():
    from tkinter import messagebox
    messagebox.showinfo("Title", "a Tk MessageBox")


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/model', methods=['GET', 'POST'])
def models():
    lst=accuracy()
    return render_template('model.html', LST=lst)




@app.route('/recalls', methods=['GET', 'POST'])
def recalling():
    lst=recall()
    return render_template('recalls.html', LST=lst)


@app.route('/confusion', methods=['GET', 'POST'])
def confusion():
    lst=matrix()
    acr=lst[0]
    prc=lst[1]
    rcl=lst[2]
    return render_template('confusion.html', ACR=acr,PRC=prc, RCL=rcl)



@app.route('/precisions', methods=['GET', 'POST'])
def precisioning():
    lst=precision()
    return render_template('precisions.html', LST=lst)




@app.route('/input', methods=['GET', 'POST'])
def predict():
    return render_template('input.html')


@app.route('/trait', methods=['GET', 'POST'])
def traits():
    return render_template('traits.html')






def preprocessor(text):
    """ Return a cleaned version of text
    """
    # Remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    # Save emoticons for later appending
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove any non-word character and append the emoticons,
    # removing the nose character for standarization. Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))

    return text
stopwords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are",
                 "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
                 "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn",
                 "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
                 "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here",
                 "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it",
                 "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn",
                 "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on",
                 "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same",
                 "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such",
                 "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there",
                 "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was",
                 "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who",
                 "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's",
                 "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's",
                 "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's",
                 "when's", "where's", "who's", "why's", "would"]
def cleaning(user_input):
        review_with_no_special_character = re.sub('[^a-zA-Z]', ' ', str(user_input))
        review_in_lowercase = review_with_no_special_character.lower()
        review_in_tokens = word_tokenize(review_in_lowercase)
        review_with_no_stopwords = [word for word in review_in_tokens if not word in stopwords]
        return ' '.join(review_with_no_stopwords)



from nltk.stem import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


clf = pickle.load(open('data/LogisticRegression.pkl', 'rb'))
def LR(mytweet):
    mytweet=preprocessor(mytweet)
    twits=[mytweet]
    preds = clf.predict(twits)
    return preds[0]

def Current(mytweet):
    vectorizer = pickle.load(open("Data/vect.sav", "rb"))
    classifier_linear= pickle.load(open("Data/clas.sav", "rb"))
    mytweet = preprocessor(mytweet)
    review_vector = vectorizer.transform([mytweet])
    return classifier_linear.predict(review_vector)[0]


def accuracy():
    list=[]
    import pickle
    file = open('Accuracies/DT', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a=Accuracy["Accuracy"]
    a=format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/RF', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Accuracy"]
    a = format(a * 100, '.2f')
    list.append(a)
    import pickle
    file = open('Accuracies/NB', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Accuracy"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/LR', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Accuracy"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/SVM', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Accuracy"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/LSTM', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Accuracy"]
    a = format(a * 100, '.2f')
    list.append(a)

    return list



def matrix():
    list=[]
    import pickle
    file = open('Accuracies/DT', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a=Accuracy["Accuracy"]
    a=format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/RF', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Accuracy"]
    a = format(a * 100, '.2f')
    list.append(a)
    import pickle
    file = open('Accuracies/NB', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Accuracy"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/LR', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Accuracy"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/SVM', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Accuracy"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/LSTM', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Accuracy"]
    a = format(a * 100, '.2f')
    list.append(a)

    acr=list

    list = []
    import pickle
    file = open('Accuracies/DT', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Precision"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/RF', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Precision"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/NB', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Precision"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/LR', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Precision"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/SVM', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Precision"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/LSTM', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Precision"]
    a = format(a * 100, '.2f')
    list.append(a)

    prc=list

    list = []
    import pickle
    file = open('Accuracies/DT', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Recall"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/RF', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Recall"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/NB', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Recall"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/LR', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Recall"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/SVM', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Recall"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/LSTM', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Recall"]
    a = format(a * 100, '.2f')
    list.append(a)
    rcl=list


    mylist=[acr,prc,rcl]


    return mylist












def precision():
    list = []
    import pickle
    file = open('Accuracies/DT', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Precision"]
    a = format(a * 100, '.2f')
    list.append(a)



    import pickle
    file = open('Accuracies/RF', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Precision"]
    a = format(a * 100, '.2f')
    list.append(a)


    import pickle
    file = open('Accuracies/NB', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Precision"]
    a = format(a * 100, '.2f')
    list.append(a)


    import pickle
    file = open('Accuracies/LR', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Precision"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/SVM', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Precision"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/LSTM', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Precision"]
    a = format(a * 100, '.2f')
    list.append(a)

    return list













def recall():
    list = []
    import pickle
    file = open('Accuracies/DT', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Recall"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/RF', 'rb')
    Accuracy = pickle.load(file)
    file.close()
    a = Accuracy["Recall"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/NB', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Recall"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/LR', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Recall"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/SVM', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Recall"]
    a = format(a * 100, '.2f')
    list.append(a)

    import pickle
    file = open('Accuracies/LSTM', 'rb')
    Accuracy = pickle.load(file)
    file.close()

    a = Accuracy["Recall"]
    a = format(a * 100, '.2f')
    list.append(a)

    return list


def SVM(mytweet):
    current_vector = pickle.load(open("Data/vectorizer.sav", "rb"))
    clf_classifier = pickle.load(open("Data/classifier.sav", "rb"))
    user_input = mytweet
    user_input = preprocessor(user_input)
    user_input = cleaning(user_input)

    user_input = [user_input]

    user_input_vector = current_vector.transform(user_input)
    Sentiment_user_input = clf_classifier.predict(user_input_vector)

    return Sentiment_user_input[0]


import re
def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
    text = re.sub('#', '', text)  # Removing '#' hash tag

    text = re.sub('RT[\s]+', '', text)  # Removing RT
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
    text = text.lower()  # converting to lowercase
    text = re.sub(':', '', text)  # Removing
    return text

import string
string.punctuation
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def remove_whitespace(text):
    return  " ".join(text.split())

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)



@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        userDetails = request.form
        mytweet = userDetails['text']
        nm=""
        img=""
        trait=""
        hc=0
        bo=0
        bc=0
        bg=0
        bj=0
        rd=0
        mytweet=preprocessor(mytweet)
        mytweet = cleanTxt(mytweet)
        mytweet = remove_punct(mytweet)
        mytweet = remove_whitespace(mytweet)
        mytweet = remove_whitespace(mytweet)

        import random
        def uniqueid():
            seed = random.getrandbits(32)
            while True:
                yield seed
                seed += 1

        unique_sequence = uniqueid()
        id1 = next(unique_sequence)




        name = LR(mytweet)
        if name=="HillaryClinton":
            hc+=2
        elif name=="BarackObama":
            bo+= 2
        elif name == "JeremyCorbyn":
            bc+= 2
        elif name=="BillGates":
            bg+= 2
        elif name=="BorisJohnson":
            bj+=2
        elif name == "Random":
            rd +=2



        name = SVM(mytweet)
        if name == "HillaryClinton":
            hc += 2
        elif name == "BarackObama":
            bo += 2
        elif name == "JeremyCorbyn":
            bc += 2
        elif name == "BillGates":
            bg += 2
        elif name == "BorisJohnson":
            bj += 2
        elif name == "Random":
            rd +=2

        name = Current(mytweet)
        if name == "HillaryClinton":
            hc += 3
        elif name == "BarackObama":
            bo += 3
        elif name == "JeremyCorbyn":
            bc += 3
        elif name == "BillGates":
            bg += 3
        elif name == "BorisJohnson":
            bj += 3
        elif name == "Random":
            rd += 3


        mx=max(hc,bo,bc,bg,bj,rd)

        if mx==hc:
            img="static/img/hillary.jpg"
            nm="HillaryClinton"
            #trait="static/img/hillaryTraits.png"
        elif mx==bo:
            img = "static/img/barack.jpg"
            nm = "BarackObama"
            #trait="static/img/barackTraits.png"
        elif mx == bc:
            img = "static/img/corbyn.jpg"
            nm = "JeremyCorbyn"
            #trait="static/img/clintonTraits.png"
        elif mx == bg:
            img = "static/img/gates.jpg"
            nm = "BillGates"
            #trait="static/img/billTraits.png"
        elif mx == bj:
            img = "static/img/boris.jpg"
            nm = "BorisJohnson"
            #trait="static/img/borisTraits.png"
        elif mx == rd:
            img = "static/img/random.jpg"
            nm = "Random"
            # trait="static/img/borisTraits.png"

        from time import sleep

        mytweet= textwrap.shorten(mytweet, width=240, placeholder="...")

        return render_template('Input.html', image=img, tweet=mytweet, name=nm, trait=traits(mytweet,id1) )



def traits(txt,id1):
    import pandas as pd
    import nltk
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    File = pd.read_csv('Dataset/Emotion_Lexicon.csv', encoding='latin-1')
    anger = []
    anticipation = []
    disgust = []
    fear = []
    joy = []
    negative = []
    positive = []
    sadness = []
    surprise = []
    trust = []
    charged = []
    import csv

    with open('Dataset/Emotion_Lexicon.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        Words = []
        Anger = []
        Anticipation = []
        Disgust = []
        Fear = []
        Joy = []
        Negative = []
        Positive = []
        Sadness = []
        Surprise = []
        Trust = []
        Charged = []

        for row in readCSV:
            Wd = row[0]
            An = row[1]
            Ant = row[2]
            Di = row[3]
            Fe = row[4]
            Jo = row[5]
            Ne = row[6]
            Po = row[7]
            Sa = row[8]
            Su = row[9]
            tr = row[10]
            Ch = row[11]

            Words.append(Wd)
            Anger.append(An)
            Anticipation.append(Ant)
            Disgust.append(Di)
            Fear.append(Fe)
            Joy.append(Jo)
            Negative.append(Ne)
            Positive.append(Po)
            Sadness.append(Sa)
            Surprise.append(Su)
            Trust.append(tr)
            Charged.append(Ch)
        i = 0
        for item in Anger:
            if item == '1':
                anger.append(lemmatizer.lemmatize(Words[i]))

            i = i + 1

        i = 0

        for item in Anticipation:
            if item == '1':
                anticipation.append(lemmatizer.lemmatize(Words[i]))
            i = i + 1

        i = 0

        for item in Disgust:
            if item == '1':
                disgust.append(lemmatizer.lemmatize(Words[i]))
            i = i + 1

        i = 0

        for item in Fear:
            if item == '1':
                fear.append(lemmatizer.lemmatize(Words[i]))
            i = i + 1

        i = 0

        for item in Joy:
            if item == '1':
                joy.append(lemmatizer.lemmatize(Words[i]))
            i = i + 1

        i = 0

        for item in Negative:
            if item == '1':
                negative.append(lemmatizer.lemmatize(Words[i]))
            i = i + 1

        i = 0

        for item in Positive:
            if item == '1':
                positive.append(lemmatizer.lemmatize(Words[i]))
            i = i + 1

        i = 0

        for item in Sadness:
            if item == '1':
                sadness.append(lemmatizer.lemmatize(Words[i]))
            i = i + 1

        i = 0

        for item in Surprise:
            if item == '1':
                surprise.append(lemmatizer.lemmatize(Words[i]))
            i = i + 1

        i = 0

        for item in Trust:
            if item == '1':
                trust.append(lemmatizer.lemmatize(Words[i]))
            i = i + 1

        i = 0

        for item in Charged:
            if item == '1':
                charged.append(lemmatizer.lemmatize(Words[i]))
            i = i + 1

        phrase = []
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        from nltk.tokenize import word_tokenize

        text = txt
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        for item in tokens_without_sw:
            phrase.append(lemmatizer.lemmatize(item))

        angerCounter = 0
        anticipationCounter = 0
        disgustCounter = 0
        fearCounter = 0
        joyCounter = 0
        negativeCounter = 0
        positiveCounter = 0
        sadnessCounter = 0
        surpriseCounter = 0
        trustCounter = 0
        chargedCounter = 0
        arr = []
        for i in range(11):
            if i == 0:
                arr = anger
            elif i == 1:
                arr = anticipation
            elif i == 2:

                arr = disgust
            elif i == 3:

                arr = fear
            elif i == 4:

                arr = joy
            elif i == 5:

                arr = negative
            elif i == 6:

                arr = positive
            elif i == 7:

                arr = sadness
            elif i == 8:

                arr = surprise
            elif i == 9:

                arr = trust
            elif i == 10:

                arr = charged

            for item in phrase:
                for wd in arr:
                    if item == wd:
                        if i == 0:
                            angerCounter += 1
                        elif i == 1:
                            anticipationCounter += 1
                        elif i == 2:
                            disgustCounter += 1
                        elif i == 3:
                            fearCounter += 1
                        elif i == 4:
                            joyCounter += 1
                        elif i == 5:
                            negativeCounter += 1
                        elif i == 6:
                            positiveCounter += 1
                        elif i == 7:
                            sadnessCounter += 1
                        elif i == 8:
                            surpriseCounter += 1
                        elif i == 9:
                            trustCounter += 1
                        elif i == 10:
                            chargedCounter += 1

        if angerCounter == 0:
            angerCounter += 0.25

        if anticipationCounter == 0:
            anticipationCounter += 0.25

        if disgustCounter == 0:
            disgustCounter += 0.25

        if fearCounter == 0:
            fearCounter += 0.25

        if joyCounter == 0:
            joyCounter += 0.25

        if negativeCounter == 0:
            negativeCounter += 0.25

        if positiveCounter == 0:
            positiveCounter += 0.25
        if sadnessCounter == 0:
            sadnessCounter += 0.25

        if surpriseCounter == 0:
            surpriseCounter += 0.25

        if trustCounter == 0:
            trustCounter += 0.25

        if chargedCounter == 0:
            chargedCounter += 0.25

        import matplotlib.pyplot as plt
        plt.rcdefaults()
        import numpy as np
        import matplotlib.pyplot as plt

        plt.bar(10, angerCounter, 10, label="Anger")
        plt.bar(30, anticipationCounter, 10, label="Anticipation")
        plt.bar(50, disgustCounter, 10, label="Disgust")
        plt.bar(70, fearCounter, 10, label="Fear")
        plt.bar(90, joyCounter, 10, label="joy")
        plt.bar(110, negativeCounter, 10, label="Negative")
        plt.bar(130, positiveCounter, 10, label="Positive")
        plt.bar(150, sadnessCounter, 10, label="Sadness")
        plt.bar(170, surpriseCounter, 10, label="Surprise")
        plt.bar(190, trustCounter, 10, label="Trust")
        plt.bar(210, chargedCounter, 10, label="Charged")

        plt.legend()
        plt.ylabel('Frequency of Words')
        plt.title("Tweet's Emotions")



        tr="static/img/emotions/"+str(id1)+".png"


        plt.savefig(tr)
        plt.clf()
        plt.cla()
        plt.close()

        return tr








if __name__ == '__main__':
    app.run(debug=True)
