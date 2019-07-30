import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
import warnings
import random
import nltk
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('df',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    categories = Y.columns
    return X, Y, categories


def tokenize(text):
    tokens = WhitespaceTokenizer().tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # clean tokens
    processed_tokens = []
    for token in tokens:
        token = lemmatizer.lemmatize(token).lower().strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        token = re.sub(r'\[[^.,;:]]*\]', '', token)
        
        # add token to compiled list if not empty
        if token != '':
            processed_tokens.append(token)
        
    return processed_tokens

def compute_text_length(data):
    """
    Compute the character length of texts
    """
    return np.array([len(text) for text in data]).reshape(-1, 1)


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([('text', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                                                     ('tfidf', TfidfTransformer()),
                                                     ])),
                                  ('length', Pipeline([('count', FunctionTransformer(compute_text_length, validate=False))]))]
                                 )),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # use GridSearch to tune model with optimal parameters
    parameters = {'features__text__vect__ngram_range':[(1,2),(2,2)],
            'clf__estimator__n_estimators':[50, 100]
             }
    model = GridSearchCV(pipeline, parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    # print scores
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), 
        target_names=category_names))



def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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