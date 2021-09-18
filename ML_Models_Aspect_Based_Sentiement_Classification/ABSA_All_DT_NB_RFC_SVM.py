#This code is used to Model Decision Tree, Random Forest, Naive Bayes and SVM for Aspect Based Sentiment Classification task.
#The code utilizes the dataset AspectBasedSentimentClassification_Dataset.json to train Decision Tree, Random Forest, Naive Bayes and SVM. 

from nltk.corpus import stopwords #stopwords function is used to obtain stopwords in english. Stopwords are commonly used words which have no sentiment involoved like "the", "a", "an", and "with".
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier        
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import json
import random
import re

#Function to train Decision Tree, Random Forest, Naive Bayes and SVM Models.
#Input: test_perc
#Output: trained models with accuracy, precision, recall, F1-score measures.  
def train_model(test_perc):
    
    #List used to dump the data from AspectBasedSentimentClassification_Dataset.json
    data = list()   
    
    #Read AspectBasedSentimentClassification_Dataset.json dataset and store it in a list 
    with open('AspectBasedSentimentClassification_Dataset.json' , 'r') as f:
        for l in f.readlines():
            if not l.strip():
                continue
            jd = json.loads(l)
            data.append(jd)

    f.close()
    
    #Shuffle the data
    random.shuffle(data)
    
    #List of stopwords which are there in stopwords of NLTK and we don't want to remove it.
    #These words are helpful to recognize cognitive aspects.
    aspect_stopwords= ['couldn', "couldn't", 'ours', "needn't", 'yours', 'ourselves', "aren't", 'he', 
                  "mightn't", 'mightn', 'yourself', "you'll", 'haven', 'herself', "weren't", "hasn't", 
                  'himself', "isn't", 'your', 'yourselves', "don't", 'her', 'weren', 'don', 'doesn', 
                  'mustn', "mustn't", 'not', "hadn't", 'hasn', 'were', 'nor', 'hers', 'being', "wasn't", 
                  'she', 'now', 'you', 'needn', 'once', 'hadn', 'further', 'isn', 'themselves',  
                  "haven't", "shouldn't", 'an', 'i', 'my', "didn't", 'no', "wouldn't",
                  'wouldn', 'didn', 'his', 'shouldn', "she's", 'ain', "doesn't", 'wasn', 'aren', 'our', "won't"]
    
    st_words=list(set(stopwords.words('english')))
    
    #Otaining updated stopwords. 
    #Updated stopwords = List of stopwords in NLTK - stopwords which are helpful to recognize cognitive aspects.    
    updated_stopwords=[x for x in st_words if x not in aspect_stopwords]
    
    corpus = []
    y=[]

    #Preprocessing STARTS
    for i in range(0, len(data)):
        review=data[i]['sentence']
        y.append(data[i]['label'])
        
        #Removing all the characters other than alphabets
        review = re.sub('[^a-zA-Z]', ' ', review)
        
        #Obtaining string in lowercase
        review = review.lower()
        
        #Splitting string into list of words
        review = review.split()
        
        ps = PorterStemmer()
        
        #Stemming and removing Stopwords
        review = [ps.stem(word) for word in review if not word in updated_stopwords]
        
        #Join list of words into string
        review = ' '.join(review)
        corpus.append(review)
    #Preprocessing ENDS


    #Vectorization
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    
    #Splitting dataset into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc)

    #Code to model Decision Tree STARTS
    model = DecisionTreeClassifier()    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)   
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='macro')   
    print("Decision Tree")
    print("Accuracy", accuracy, "Precision", precision, "Recall", recall,  "F1", f1_score)
    #Code to model Decision Tree ENDS
    
    #Code to model Random Forest STARTS
    model = RandomForestClassifier(n_estimators=501, criterion='entropy')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)    
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='macro')  
    print("Random Forest")
    print("Accuracy", accuracy, "Precision", precision, "Recall", recall,  "F1", f1_score)
    #Code to model Random Forest ENDS
  
    #Code to model Naive Bayes STARTS
    model = GaussianNB()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    y_pred = model.predict(X_test)  
    accuracy = accuracy_score(y_test, y_pred)  
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='macro')        
    print("Naive Bayes")
    print("Accuracy", accuracy, "Precision", precision, "Recall", recall,  "F1", f1_score)
    #Code to model Naive Bayes ENDS
    
    #Code to model SVM STARTS
    model = LinearSVC(C=0.01)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) 
    accuracy = accuracy_score(y_test, y_pred) 
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
    print("SVM")
    print("Accuracy", accuracy, "Precision", precision, "Recall", recall,  "F1", f1_score)
    #Code to model SVM ENDS

train_model(0.2)