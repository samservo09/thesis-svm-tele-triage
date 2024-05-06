"""
1. Pre-processed text data is inputted. (Problem 1) 
2. Assign the input vector to an entity space with a complex dimensional feature. (Problem 3)
3. Find the best hyperplane position that maximizes the margin between instances of different classes, effectively splitting the feature space into regions corresponding to different classes. 
4. For handling nonlinear classification tasks, SVM maps the feature space from the input space to a higher-dimensional space using kernel tricks. (Problem 3) 
5. Use Multi-Class Classification Strategies [One-vs-All (OVA) or One-vs-One (OVO)] for multi-class classification.
6. SVM computes the position of a new instance relative to the decision boundary. (Problem 1)
7. SVM assigns class labels based on their position as a training result. (Problem 2)
8. Performance will be evaluated with the common four indicators of accuracy rate, precision rate, recall rate and F1 value.
"""
#necessary libraries
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

#set a random seed
np.random.seed(500)

#import the data
Corpus = pd.read_csv(r"data/corpus.csv",encoding='latin-1')

#data preprocessing
# Step - a : Remove blank rows if any.
Corpus['text'].dropna(inplace=True)
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
Corpus['text'] = [entry.lower() for entry in Corpus['text']]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Corpus['text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)

#prepare train and test datasets
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.3)

#encoding

#word vectorization

#use machine learning algorithms to predict the outcome