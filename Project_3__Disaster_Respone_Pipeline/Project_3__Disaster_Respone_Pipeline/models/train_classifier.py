# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import time 
import pickle

# Tokenisation/Lemmatisation
import re
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')  # Lemmatisation
from nltk.stem.porter import PorterStemmer      # Stemming
from nltk.stem.wordnet import WordNetLemmatizer # Lemmatising

# Classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression


def load_data(database_filepath):
    '''Loads the data from the local database
    Args:
        database_filepath (str) - path_to_database
    Returns:
        X (pd.DataFrame) - the messages to classify
        Y (pd.DataFrame) - the labelled classifications
        category_names (python list) - the categories of classifications
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response', con=engine)

    X = df['message']
    Y = df.iloc[:,3:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''Tokeniser for a single message
    Args:
        text (str) - the message
    Returns:
        (python list) - the list of tokens
    '''
    # lowercase, remove punc
    tokens = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
    # Split and remove stopwords
    #tokens = [w for w in word_tokenize(tokens) if w not in stopwords.words("english")]
    tokens = word_tokenize(tokens) # Without stopword removal
    # Stem
    tokens = [PorterStemmer().stem(t) for t in tokens]
    # Lemmatise nouns
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    # Lemmatise verbs
    tokens = [WordNetLemmatizer().lemmatize(t, pos='v').strip() for t in tokens]
    return tokens

def build_model():
    '''Creates the pipeline and initiates the model ready for fitting
    Args:
        None
    Returns:
        (sklearn Estimator)
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(SGDClassifier(loss ='modified_huber')))
                    ])

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''Creates the pipeline and initiates the model ready for fitting
    Args:
        model
        X_test (pd.DataFrame) - the test set messages
        Y_test (pd.DataFrame) - the test set labels
        category_names (python list) - the categories of classifications
    Returns:
        None
    '''
    Y_pred = model.predict(X_test)
    print('\tf1_score %0.2f' % f1_score(Y_pred, Y_test, average='weighted'))


def save_model(model, model_filepath):
    '''Pickles the model for saving
    Args:
        model (Sklearn Estimator)
        model_filepath (str) - filepath
    Returns:
        None
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump( model, f)


def main():
    '''Run, evaluate and save the model. Requiries 2 ordered arguments
    Args:
        path_to_data_DB (str) - the path to the local database
        pickle_filepath (str) - the path to the location to pickle the model to
    Returns:
        None
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()