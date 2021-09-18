Dataset Description:

The Cognitive Triad Dataset (CTD) comprises 5886 messages, 580 from Beyond Blue personal stories, 600 from the Time-to-Change blog, and 4706 from Twitter. The data were manually labeled by skilled annotators. This data is divided into six categories: self-positive, world-positive, future-positive, self-negative, world-negative, and future-negative. The Cognitive Triad Dataset was evaluated on two subtasks: aspect detection and sentiment classification on given aspects. The dataset will aid in the comprehension of Beck’s Cognitive Triad Inventory (CTI) items in a person’s social media posts.


Ready-to-Use Models:

In the preliminary work, Decision Tree, Random Forest, Naive Bayes, SVM and RNN-Capsule models are evaluated for aspect extraction and sentiment classification on the Cognitive Triad Dataset (CTD). The code for Decision Tree, Random Forest, Naive Bayes and SVM is provided. We run the CTD on RNN-Capsule code (https://github.com/thuwyq/WWW18-rnn-capsule).


Requirements:

The following are some general library needs for the project.
- nltk
- scikit-learn
- torch
- transformers


Preprocessing steps include:
1. Removing all the characters other than alphabets
2. Obtaining data in lowercase
3. Stemming
4. Removing Stopwords


Information about dataset files:

1. Dataset/AspectBasedSentimentClassification_Dataset.json: This Original Cognitive Triad Dataset (https://data.mendeley.com/datasets/wb2n39sgbp/1) is used to train model for aspect based sentiment classification tasks. The dataset has six classes to train: self-positive, world-positive, future-positive, self-negative, world-negative, and future-negative. 

	Label	Meaning      
	-------------------------
   	  0    	self-negative   
   
   	  1    	self-positive

   	  2    	future-negative
   
   	  3    	future-positive

   	  4    	world-negative

   	  5    	world-positive  

2. Dataset/AspectDatection_Dataset.json: This dataset is obtained by reducing the number of classes in the Original Cognitive Triad Dataset for aspect detection task. This dataset has three classes to train: self, future, and world.

	Label	Meaning      
	-------------------------
   	  0    	self   
   
   	  1    	future

   	  2    	world

3. Dataset/SentimentRecognition_Dataset.json: This dataset is obtained by reducing the number of classes in the Original Cognitive Triad Dataset for sentiment classification task. The dataset has two classes to train: negative, and positive.

	Label	Meaning      
	-------------------------
   	  0    	negative   
   
   	  1    	positive


Information about python files:

1. Dataset_Annotation_Code/Code_to_Obtain_Simulated_Data.py: This code is used to generate simulated data from existing real-time data. In the python file, comments are provided for the important steps.

2. ML_Models_Aspect_Based_Sentiement_Classification/ABSA_All_DT_NB_RFC_SVM.py: This code is used to Model Decision Tree, Random Forest, Naive Bayes and SVM for Aspect Based Sentiment Classification task. The code utilizes the dataset AspectBasedSentimentClassification_Dataset.json to train Decision Tree, Random Forest, Naive Bayes and SVM. In the python file, comments are provided for the important steps.

3. ML_Models_Aspect_Detection/AD_DT_NB_RFC_SVM.py: This code is used to Model Decision Tree, Random Forest, Naive Bayes and SVM for Aspect Detection task. The code utilizes the dataset All_Aspect_Dataset.json to train Decision Tree, Random Forest, Naive Bayes and SVM. In the python file, comments are provided for the important steps.

4. ML_Models_Sentiment_Classification/SA_All_DT_NB_RFC_SVM.py: This code is used to Model Decision Tree, Random Forest, Naive Bayes and SVM for Sentiment Classification task. The code utilizes the dataset All_Sentiments_Dataset.json to train Decision Tree, Random Forest, Naive Bayes and SVM. In the python file, comments are provided for the important steps.
