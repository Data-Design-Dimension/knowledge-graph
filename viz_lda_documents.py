import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import base64
from docx import Document
import io
from gensim import corpora  #, models
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
# import pyLDAvis.gensim_models as gensimvis
# import pyLDAvis
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud, STOPWORDS
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dcc.Store(id='data-store'),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True
    ),
    html.H5(id='heading'),
    html.Div(id='lda-visualization'),
])

def parse_contents(contents, filename):
    texts = []
    if ';base64,' in contents:
        content_type, content_string = contents.split(';base64,')
    else:
        print(f"Unexpected contents format for file {filename}: {contents}")
        return texts

    decoded = base64.b64decode(content_string)
    try:
        if 'docx' in filename:
            # Assume that the user uploaded a Word file
            doc = Document(io.BytesIO(decoded))

            # Extract text from all paragraphs
            text = ' '.join([para.text for para in doc.paragraphs])

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += ' ' + cell.text

            # Append the text as a single string, not a list
            texts.append(text)
    except Exception as e:
        print(e)

    return texts

@app.callback(Output('heading', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_heading(contents, filename):
    if contents is not None:
        # Join the filenames into a single string with filenames separated by commas
        heading_text = ','.join(filename)
        return heading_text

@app.callback(Output('data-store', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is None or list_of_names is None:
        raise PreventUpdate

    # Parse contents and preprocess each document separately
    documents = []
    id_to_filename = {}
    doc_index = 0
    for content, name in zip(list_of_contents, list_of_names):
        for doc in parse_contents(content, name):
            documents.append(preprocess_string(doc))
            id_to_filename[doc_index] = name
            doc_index += 1
    
    print(f"Length of documents: {range(len(documents))}")
    print(f"id_to_filename: {id_to_filename}")

    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(documents)
    print(f"Dictionary: {dictionary}")

    # Convert dictionary to a list of tuples (id, word) so that it is JSON serializable as required by the dcc.Store component
    dictionary_tuples = [(id, word) for word, id in dictionary.token2id.items()]

    # Convert each document into the bag-of-words (BoW) format
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Convert corpus to a list of lists of tuples so that it is JSON serializable
    corpus_tuples = [[(id, freq) for id, freq in doc] for doc in corpus]

    # Return the data to be stored
    return {'corpus': corpus_tuples, 'dictionary': dictionary_tuples, 'id_to_filename': id_to_filename, 'documents': documents}

# gensim.corpora.Dictionary does not provide a direct method to create dictionary from_tuples;
# This function will create a new dictionary and populate it with the data from the tuples (using tuples accomodates using dcc.Store which must serialize to JSON)
def dictionary_from_tuples(tuples):
    dictionary = corpora.Dictionary()
    dictionary.token2id = {word: id for id, word in tuples}
    dictionary.id2token = {id: word for id, word in tuples}
    return dictionary

@app.callback(Output('lda-visualization', 'children'),
              Input('data-store', 'data'))
def update_lda_visualization(data):
    if data is None:
        raise dash.exceptions.PreventUpdate
    
    explanatory_text = dcc.Markdown('''
                                    \n\nEach point in the chart is a document. The color tells us the main topic of the document. The closer the points, the more similar the documents.\n\nA topic modeling technique called LDA is used to classify the text in the documents. The horizontal and vertical axes are abstract representations from that modeling and are created by a technique called t-SNE.\n\nTogether, these methods help us visualize complex data like the words and meanings in these documents in a simpler 2D space. The labels on the horizontal and vertical axes don't represent specific characteristics of the documents. Instead, those numbers help us understand the relative similarities between documents by making a space in which all the documents are organized according to the topics found in the documents in the chart.
                                    ''')
    # Convert the list of lists back to a list of gensim vectors
    corpus = [[(id, freq) for id, freq in doc] for doc in data['corpus']]

    # Convert the list of tuples back to a gensim Dictionary
    dictionary = dictionary_from_tuples(data['dictionary'])

    # Get the id_to_filename dictionary and convert keys back to integers
    # because the dictionary has been converted to JSON in the dcc.Store component and JSON requires string keys,
    # but in order for the keys to be used to access the documents' filenames, those keys must be integers again
    id_to_filename = {int(key): value for key, value in data['id_to_filename'].items()}

    # Get the preprocessed documents
    documents = data['documents']

    print(f"Length of documents: {range(len(documents))}")
    print(f"id_to_filename: {id_to_filename}")
    # Get the filenames for each document
    filenames = [id_to_filename[i] for i in range(len(documents))]

    # Apply the LDA model
    lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=2)

    # Get the number of topics
    num_topics = lda_model.num_topics

    # Get the topic distribution for each document
    topic_dist = [[dict(lda_model[doc]).get(i, 0) for i in range(num_topics)] for doc in corpus]

    # Convert topic_dist to a 2D array
    topic_dist_array = np.array(topic_dist)

    # If number of documents is less than 5, return an empty graph with a message to upload more documents
    if len(corpus) < 5:
        return dcc.Graph(figure={'data': [], 'layout': {'title': 'Please upload 5 or more documents to visualize how topics are distributed across documents."'}})
    
    else:
        # Use t-SNE to reduce the topic vectors to 2 dimensions
        tsne = TSNE(n_components=2, perplexity=min(30, len(corpus)-1), random_state=0)
        topic_dist_2d = tsne.fit_transform(topic_dist_array)

        # Get the top words for each topic
        top_words = {i: [word for word, _ in lda_model.show_topic(i, topn=5)] for i in range(num_topics)}

        # Get the next top words for each topic for reference only
        additional_words = {i: [word for word, _ in lda_model.show_topic(i, topn=10)][5:] for i in range(num_topics)}


        # Get the most probable topic for each document
        most_probable_topic = np.argmax(topic_dist_array, axis=1)
        # Calculate the probability of the most probable topic for each document
        most_probable_topic_prob = np.max(topic_dist_array, axis=1)
        # Get the most probable topic's top words for each document
        most_probable_topic_words = [', '.join(top_words[topic]) for topic in most_probable_topic]
        # Get the most probable topic's additional words for each document
        most_probable_additional_words = [', '.join(additional_words[topic]) for topic in most_probable_topic]

        # Get the second most probable topic for each document
        second_most_probable_topic = np.argsort(topic_dist_array, axis=1)[:, -2]
        # Calculate the probability of the second most probable topic for each document
        second_most_probable_topic_prob = np.sort(topic_dist_array, axis=1)[:, -2]
        # Get the second most probable topic's top words for each document
        second_most_probable_topic_words = [', '.join(top_words[topic]) for topic in second_most_probable_topic]
        # Get the second most probable topic's additional words for each document
        second_most_probable_additional_words = [', '.join(additional_words[topic]) for topic in second_most_probable_topic]
   
    # Create a DataFrame
    df = pd.DataFrame({
        'x': topic_dist_2d[:, 0],
        'y': topic_dist_2d[:, 1],
        'Filename': filenames,
        'Most Probable Topic': most_probable_topic,
        'Probability of Topic': most_probable_topic_prob,
        # 'Top Words for Topic': most_probable_topic_words,
        'Additional Words for Topic': most_probable_additional_words,
        'Next Likely Topic': second_most_probable_topic,
        'Probability of Next Topic': second_most_probable_topic_prob,
        # 'Top Words for Next Topic': second_most_probable_topic_words,
        'Additional Words for Next Topic': second_most_probable_additional_words
    })
    # Modify the values of 'Most Probable Topic' to include the topic's top words
    # so that the values can be used as more concise informative labels in the scatter plot
    df['Most Probable Topic'] = df['Most Probable Topic'].apply(lambda x: f'Topic {x}: {top_words[x]}')
    df['Next Likely Topic'] = df['Next Likely Topic'].apply(lambda x: f'Topic {x}: {top_words[x]}')
   
    # Create a scatter plot
    fig = px.scatter(df, x='x', y='y', text='Filename',
                     color='Most Probable Topic',
                    #  size='Probability of Topic',
                     hover_data=['Filename',                             
                                 'Most Probable Topic', 
                                #  'Top Words for Topic',
                                 'Additional Words for Topic',
                                 'Probability of Topic',
                                 'Next Likely Topic',
                                #  'Top Words for Next Topic',
                                 'Additional Words for Next Topic',
                                 'Probability of Next Topic'
                                 ])

    fig.update_layout(
        title='Topic Distribution Across Documents',
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2'
    )
    fig.update_traces(textposition="top center",
                      marker={'size': 15}
                      )
    # Add an explanatory comment below the chart
    fig.add_annotation(
        x=0,
        y=-0.1,
        text='',
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=12
        )
    )
    # Return the Graph and Markdown components wrapped in a Div
    return html.Div([
        dcc.Graph(figure=fig),
        explanatory_text
    ])

if __name__ == '__main__':
    app.run_server(debug=True)