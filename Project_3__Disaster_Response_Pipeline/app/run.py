import json
import plotly
import pandas as pd
import sys
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''Tokeniser function to be used when the model is called
    Args:
        text (str) - string to be tokenised
    Returns:
        (python list) - list of tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def get_chart_object(df):
    '''This function prepares the three charts for use on the homepage
    Args:
        df (pd.DataFrame) - the dataframe to plot
    Returns:
        graphs (list of dicts) - the graphs list to be passed to plotly frontend
    '''
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(genre_counts.index.str.title())

    category_counts = df.sum(axis=0, numeric_only = True).sort_values()
    category_names = list(category_counts.index.str.replace('_', ' ').str.title())   
    
    # Dist plot
    df_hist = df[['genre','message']]
    df_hist['message_length'] = df_hist.message.str.split(' ').apply(len)
    colors = (['rgb(69,48,140)','rgb(139,59,156)','rgb(234,217,169)'])
    plot_order = df_hist.groupby('genre')['message'].count().sort_values(ascending=False).index
    x = [df_hist.loc[df_hist.genre==g,'message_length'].apply(lambda x: min(x,100)) for g in plot_order]
    
    hist_data = [[Histogram(
                    x=dat,
                    name= plot_order.str.title()[i],
                    opacity = j[1],
                    marker = {'color':colors[i]},
                    xbins=dict(size=1),
                    histnorm= j[0]
                ) for i,dat in enumerate(x)] for j in ((None,0.9), ('percent',0.6))]

    hist_layout =  [{
                'barmode':'overlay',
                'title': title,
                'xaxis': {
                    'range': [0,100],  
                    'title': "Message length (words)"
                },
                'yaxis': {
                    'visible': False,  
                },
                'legend' : {'x':0.85,'y':0.9}
            } for title in ['Distribution of Message Lengths','Distribution of Message Lengths (Normalised)']]
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    text=genre_counts,
                    hoverinfo= 'x+y',
                    textposition='outside',
                    cliponaxis = False,
                    marker={'color': genre_counts
                        , 'colorscale': 'Agsunset_r'}   
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'visible': False,  
                    #'title': "Count"
                },
                'xaxis': {
                    #'title': "Genre"
                }
            }
        },
        {
            'data': hist_data[0],

            'layout': hist_layout[0],
        },

        {
            'data': hist_data[1],

            'layout': hist_layout[1],
        },
        {
            'data': [
                Bar(
                    y=category_names,
                    x=category_counts,
                    orientation = 'h',
                    hoverinfo= 'x+y',
                    text=category_counts,
                    textposition='outside',
                    cliponaxis = False,
                    marker={'color':category_counts
                        , 'colorscale': 'Agsunset_r'}
                )
 

            ],

            'layout': {
                'title': 'Distribution of Message Categories<br>(note that certain categories are very sparce in the training set)',
                'xaxis': {
                    'visible': False,  
                    #'title': "Count"
                },
                'margin': {
                    'l': 200,                   
                }, 
                'height': 800,
            }
        }
    ]
    return graphs



# load data for local instance
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load data for Heroku (can't use SQLlite) - NOT CURRENTLY FUNCTIONING
#df = pd.read_json("/app/data/disasterresponse.json", orient='records') # Filepath is Heroku specific, change for local running

# load model
model = joblib.load("../models/classifier.pkl") 

@app.route('/')
@app.route('/index')
def index():
    '''Index webpage displays visuals and receives user input text for model'''
    graphs = get_chart_object(df)
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    '''Web page that handles user query and displays model results'''
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
    #pass

if __name__ == '__main__':
    main()