#===============================================================================
# Module mb_code.py
#===============================================================================

# Consolidation of some of my most used functions (and examples)

#===============================================================================
# This version: 08/10/2024
#===============================================================================
import pandas as pd
import numpy as np
pd.set_option('max_colwidth', 200)         # Altera largura máxima das colunas
pd.options.display.max_columns = None      # Exibe todas as colunas

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams["figure.figsize"] = (16,6)
plt.style.use('fivethirtyeight')

import seaborn as sns


import os
import sys
import time

import warnings
warnings.filterwarnings("ignore")


import datetime as DT
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pytz
from pytz import timezone

import csv

##====================================================================
# Where is my python executable
##====================================================================
import sys
import os
print(os.path.dirname(sys.executable))

##====================================================================
# How to pass API keys without showing them
##====================================================================
'''How to write python code that passes the api_key without 
showing in the code. For example, how to pass the 
open_api key using a jupyter notebook?'''

##====================================================================
# Set your API key as an environment variable:
##====================================================================
import os
os.environ['OPENAPI_KEY'] = 'your-api-key-here'

# Retrieve the API key in your Python code:
api_key = os.environ.get('OPENAPI_KEY')

# Use the API key in your API requests:
'''This will add the API key to the Authorization header of your API request.
 Make sure to replace https://api.example.com with the actual URL 
 of the API that you're using.
 
 By setting the API key as an environment variable, you can easily 
 pass it to your Python code without having to hardcode it 
 in your Jupyter notebook. This also makes it easy to switch 
 between different API keys without having to modify your code.
 '''
#### UNCOMMENT BELOW WHEN RUNNING
#import requests
#headers = {'Authorization': f'Bearer {api_key}'}
#response = requests.get('https://api.example.com', headers=headers)


##====================================================================
# Create subdirectory from current_directory
##====================================================================
def create_subdirectory(subdirectory_name):
    ''' create subdirectory from current directory 
    '''
    import os
    current_directory = os.getcwd()
    new_directory = os.path.join(current_directory, subdirectory_name)

    # Check if the directory already exists
    if not os.path.exists(new_directory):
        os.makedirs(new_directory, exist_ok = True)
        print(f"Created subdirectory: {subdirectory_name}")
    else:
        print(f"Subdirectory already exists: {subdirectory_name}")

# Example usage
subdirectory_name = "my_subdirectory"
create_subdirectory(subdirectory_name)


#============================================================================================
# Function to obtain the top n values for a categorical variable
# Extends the top5 function below
#============================================================================================
def top_values(df, n = 5):
    """Given dataframe, generate top n (default n = 5)
    unique values for non-numeric data"""
    columns = df.select_dtypes(include=['object', 'category']).columns
    for col in columns:
        print("Top n unique values of " + col)
        print(df[col].value_counts().reset_index().rename(columns={"index": col, col: "Count"})[
              :min(n, len(df[col].value_counts()))])
        print(" ")

#============================================================================================
# Function plot_series_v2
#============================================================================================
def plot_series_v2(df_to_plot, date_column, cols_to_plot, title_text='  ', plt_graph_style='ggplot'):
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.style.use(plt_graph_style)
    colors_toplot = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c']
    # list of markers for each series
    markers_toplot = ['o', 's', '^', 'd'] 
    plt.rcParams["figure.figsize"] = (16, 5)
    
    fig, ax = plt.subplots()
    df_to_plot.set_index(date_column, inplace=True)
    
    # Loop through cols_to_plot and plot each series with a different color and marker
    for i, col in enumerate(cols_to_plot):
        df_to_plot[col].plot(linewidth=2.5, color=colors_toplot[i], marker=markers_toplot[i], markersize=8, label=col, ax=ax)
    
    plt.title( title_text, size=16)  
    plt.xlabel('Data', fontsize=12)
    
    ax.xaxis.set_tick_params(which='major', labelsize=8, length=10, rotation = -45)
    ax.xaxis.set_tick_params(which='minor', labelsize=8, length=10, rotation = -45)
    
    plt.legend(loc='best') # add legend with label for each series
    plt.show()
    df_to_plot.reset_index(inplace=True)

#===============================================================================
# ***Function to plot histogram between two specified percentiles***
#===============================================================================
def plot_hist(df, coluna, percentil1 = 0.05, percentil2 = 0.95, color = 'navy', num_bins = 30):
    """percentil1 e percentil2  são os percentis entre os quais o histograma será mostrado,
    escritos como decimais
    coluna é o nome da coluna (entre aspas simples)
    """
    p1 = df[coluna].quantile(percentil1)
    p2 = df[coluna].quantile(percentil2)
    g = sns.displot(data = df, x = coluna, kind = 'hist', height = 6, aspect = 1.5, color = color,
        bins = num_bins, log_scale = False, binrange = (p1, p2))
    plt.title('Histograma ' + str(coluna) + '\n', fontsize = 14)
    plt.figure(figsize=(12,5));
    #sns.set(rc={'figure.figsize':(12,5)});
    plt.show();
    return 
    
#===============================================================================
# Crosstabs - heatmap
#===============================================================================
import seaborn as sns
result_crosstab = pd.crosstab(data["Company Name"], data["Employee Work Location"])
sns.heatmap(result_crosstab, annot=True)


#===============================================================================
# Radar plot
#===============================================================================
import plotly.graph_objects as go 
import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df =pd.DataFrame(dict(
    categorias = ['Compaixão', 'Flexibilidade', 'Honestidade', 'Lealdade', 'Persistência', 'Resiliência' ], \
    scores = [4, 5, 2, 3, 3, 1]))

# Create a color map for each dimension
blue_9_colors = ['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#08519c','#08306b']

def cria_radar(lista_categorias, lista_scores):
    ''' Cria radar plot'''
    import pandas as pd 
    import plotly.graph_objects as go 
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r = lista_scores,
                                theta =lista_categorias,
                                fill = "toself",
                                name = "Personalidade",
                                line=dict(color = '#08519c'),
                                fillcolor = '#2171b5',
                                showlegend=True))
    fig.show()

#===============================================================================
# Heatmap das dimensões
#===============================================================================
def cria_heatmap(lista_categorias, lista_scores):
    '''Cria um heapmap das dimensões da personalidade
    '''
    import pandas as pd 
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    cmap = blue_9_colors
    x_axis_labels = lista_categorias #df['categorias'].tolist() # labels for x-axis
    plt.figure(figsize=(6,6)) #15,6))
    sns.heatmap(lista_scores.values.reshape(1, -1), cmap=cmap, cbar=False, annot=True, fmt="d", \
            xticklabels=x_axis_labels, yticklabels = ' ', linewidths=2, linecolor="k", square= True)
#sns.heatmap(df['scores'].values.reshape(1, -1), cmap=cmap, cbar=False, annot=True, fmt="d", \
            #xticklabels=x_axis_labels, yticklabels = ' ', linewidths=2, linecolor="k", square= True)
    plt.show()

#===============================================================================
# Excuta funções cria_radar e cria_heatmap
#===============================================================================
#fig1 = cria_radar(df['categorias'], df['scores'])
#fig2 = cria_heatmap(df['categorias'], df['scores'])

#fig1.show()
#fig2.show()


#===============================================================================
#===============================================================================
# EDA functions
# Source: https://gist.github.com/jiahao87/c97214065f996b76ab8fe4ca1964b2b5
#===============================================================================
#===============================================================================
def top5(df):
    """Given dataframe, generate top 5 unique values for non-numeric data"""
    columns = df.select_dtypes(include=['object', 'category']).columns
    for col in columns:
        print("Top 5 unique values of " + col)
        print(df[col].value_counts().reset_index().rename(columns={"index": col, col: "Count"})[
              :min(5, len(df[col].value_counts()))])
        print(" ")
    
    
def categorical_eda(df, hue=None):
    """Given dataframe, generate EDA of categorical data"""
    print("\nTo check: \nUnique count of non-numeric data\n")
    print(df.select_dtypes(include=['object', 'category']).nunique())
    top5(df)
    # Plot count distribution of categorical data
    for col in df.select_dtypes(include='category').columns:
        fig = sns.catplot(x=col, kind="count", data=df, hue=hue)
        fig.set_xticklabels(rotation=90)
        plt.show()
        
def numeric_eda(df, hue=None):
    """Given dataframe, generate EDA of numeric data"""
    print("\nTo check: \nDistribution of numeric data")
    display(df.describe().T)
    columns = df.select_dtypes(include=np.number).columns
    figure = plt.figure(figsize=(20, 10))
    figure.add_subplot(1, len(columns), 1)
    for index, col in enumerate(columns):
        if index > 0:
            figure.add_subplot(1, len(columns), index + 1)
        sns.boxplot(y=col, data=df, boxprops={'facecolor': '#6baed6'})  #'None'
    figure.tight_layout()
    plt.show()
    
    if len(df.select_dtypes(include='category').columns) > 0:
        for col_num in df.select_dtypes(include=np.number).columns:
            for col in df.select_dtypes(include='category').columns:
                fig = sns.catplot(x=col, y=col_num, kind='violin', data=df, height=5, aspect=2)
                fig.set_xticklabels(rotation=90)
                plt.show()
    
    # Plot the pairwise joint distributions
    print("\nTo check pairwise joint distribution of numeric data")
    if hue==None:
        sns.pairplot(df.select_dtypes(include=np.number))
    else:
        sns.pairplot(df.select_dtypes(include=np.number).join(df[[hue]]), hue=hue)
    plt.show()

#===============================================================================
# ***** Bar plot examples  *******
# TODO: CONVERT TO FUNCTIONS
#===============================================================================
#df_born_year = df_query.year.value_counts().to_frame()
#df_born_year = df_born_year.reset_index()
#df_born_year.columns = ['Ano_nasc', 'Freq']

# df_born_year.Ano_nasc = df_born_year.Ano_nasc.astype(int)

# sns.set(rc={'figure.figsize':(17,12)})
# df_born_year.sort_values('Freq',inplace=True)
# ax = df_born_year.plot.barh(x = 'Ano_nasc', y = 'Freq', color = '#bd0026', title = 'Top Anos Nascimento\n')

# sns.set(rc={'figure.figsize':(17,8)})
# df_born_year.sort_values('Ano_nasc')#,inplace=True)
# ax = df_born_year.plot.bar(x = 'Ano_nasc', y = 'Freq', color = '#bd0026', title = 'Distribuição por Anos Nascimento\n')

# ax = df_born_year[df_born_year['Ano_nasc'] > 1940].sort_values('Ano_nasc'). plot.bar(x = 'Ano_nasc', y = 'Freq', color = '#bd0026', title = 'Distribuição por Anos Nascimento ( > 1940) \n')

# # Bar plot - Distribuição por mês de nascimento
# df_born_month = df_query.month.value_counts().to_frame()
# df_born_month = df_born_month.reset_index()
# df_born_month.columns = ['Mes_nasc', 'Freq']

# df_born_month.Mes_nasc = df_born_month.Mes_nasc.astype(int)

# sns.set(rc={'figure.figsize':(17,6)})
# df_born_month.sort_values('Mes_nasc',inplace=True)
# ax = df_born_month.plot.bar(x = 'Mes_nasc', y = 'Freq', color = '#bd0026', title = 'Distribuição Mês Nascimento\n')

#===============================================================================
# Wordclouds
#===============================================================================
def cria_wordcloud(df_input, col_to_analyze, titulo = ' '):  #stopwords = CLEANING_LST, 
    ''' TODO: CRIEI ESTA FUNÇÃO A PARTIR DE CÓDIGO ANTIGO, VERIFICAR SE FUNCIONA DIREITO
    CLEANING_LST é uma lista de stopwords customizada'''
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    import pandas as pd
    
    df = df_input.copy()
    print('')
    print('Versão Wordcloud :', wordcloud.__version__)
    
    # Cria bloco de texto a partir da informação da coluna selecionada
    text_name = " ".join(linha for linha in df.col_to_analyze)
    
    linha = None
    wordcloud_name = WordCloud(stopwords=CLEANING_LST, background_color="white", colormap = 'nipy_spectral', #colormap = 'copper',
                        max_words = 250, width=1600, collocations = False, min_word_length = 3, normalize_plurals = False,
                        min_font_size = 15,
                        max_font_size = 120, height=900, random_state=123).generate(text_name)
    
    plt.figure(figsize = (16,9)) 
    plt.imshow(wordcloud_name, interpolation='bilinear') 
    plt.title( titulo +'\n', fontsize = 25, color = 'navy')
    plt.axis("off") 
    plt.show()

#===============================================================================
# Heatmaps com cara legal e mask
# ****** ATENCAO _ PRECISA MUDAR df ABAIXO PARA df.corr()******
#===============================================================================
# ***** Exemplo 1 - Fonte: projeto BBB ******
def heatmap_with_mask(df_input, titulo = ' ', annot = False, colors = 'Blues', mask = True):
    ''' Quite general implementation of heatmap with/without mask
        For green/red use, for example, the following list of colors:
    colors = [ '#b10026', '#e31a1c',  '#fc4e2a', '#fd8d3c','#feb24c', '#fed976', '#ffffb2', '#e5f5e0', '#41ae76','#238b45', '#005824'] 
    '''
    import pandas as pd 
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    sns.set(rc={'figure.figsize':(16,6)})
    #colors = ['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
    #colors = 'YlOrRd_r'  # colormap in reverse order
    
    # Vermelho, amarelo, verde
    df = df_input.copy()   # ******MUDAR PARA df_input.corr().copy()
    if mask:   # If heatmap to be drawn WITH mask
        # define the mask to set the values in the upper triangle to True
        mascara = np.triu(np.ones_like(df.corr(), dtype= np.dtype(bool)))  # check it new
        if annot:
            sns.heatmap(df, cmap = colors, annot = True, fmt = '.1f', annot_kws={"size":8}, linewidth = 1, mask = mascara); # format 0.0  
            plt.title(titulo+'\n', fontdict = {'fontsize' : 16})
            plt.show()
        else:
            sns.heatmap(df, cmap = colors, annot = False, linewidth = 1, mask = mascara); # NO annotations  
            plt.title(titulo+'\n', fontdict = {'fontsize' : 16})
            plt.show()
    else:  # Heatmap to be drwan WITHOUT mask (i.e, entire matrix)
        if annot:
            sns.heatmap(df, cmap = colors, annot = True, fmt = '.1f', annot_kws={"size":8}, linewidth = 1); # format 0.0  
            plt.title(titulo+'\n', fontdict = {'fontsize' : 16})
            plt.show()
        else:
            sns.heatmap(df, cmap = colors, annot = False, linewidth = 1); # NO annotations  
            plt.title(titulo+'\n', fontdict = {'fontsize' : 16})
            plt.show()

#===============================================================================
# FilterText
# source: https://medium.com/pythoneers/10-impressive-automation-scripts-you-need-to-try-using-python-bc9bc7563633
#===============================================================================
import re

def filter_text(text):
    # Filter email addresses
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)

    # Filter mentions (Twitter-style)
    mentions = re.findall(r'@\w+', text)

    # Filter hashtags
    hashtags = re.findall(r'#\w+', text)

    # Filter links (HTTP/HTTPS)
    links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

    # Filter HTML tags
    html_tags = re.findall(r'<[^>]+>', text)

    return {
        'emails': emails,
        'mentions': mentions,
        'hashtags': hashtags,
        'links': links,
        'html_tags': html_tags,
    }

if __name__ == "__main__":
    # Example text with HTML tags
    example_text = """
    For more information, contact <a href="support@example.com">support@example.com</a>.
    Follow us on Twitter: @example_user. Visit our website: https://www.example.com
    Join the conversation with #PythonProgramming.
    Connect with John Doe at john.doe@example.com.
    I love using Python for <b>natural language processing</b> and sentiment analysis!
    """
    # Filter information from the text
    filtered_info = filter_text(example_text)

    # Display the filtered information
    print("Emails:", filtered_info['emails'])
    print("Mentions:", filtered_info['mentions'])
    print("Hashtags:", filtered_info['hashtags'])
    print("Links:", filtered_info['links'])
    print("HTML Tags:", filtered_info['html_tags'])


#===============================================================================
# TrendyStocks - uses PyTrends
# source: https://medium.com/pythoneers/10-impressive-automation-scripts-you-need-to-try-using-python-bc9bc7563633
#===============================================================================
#!pip install pytrends
from pytrends.request import TrendReq
import matplotlib.pyplot as plt

# Function to get Google Trends data
def get_google_trends_data(keywords, timeframe='today 3-m', geo='US'):
    pytrends = TrendReq(hl='en-US', tz=360)

    # Build the payload
    pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop='')

    # Get interest over time
    interest_over_time_df = pytrends.interest_over_time()

    return interest_over_time_df

# Example keywords related to your article
STOCKS = ["AMZN", "MSFT", "NVDA", "AAPL", "GOOG"]

# Fetch Google Trends data
trends_data = get_google_trends_data(STOCKS)

# Plot the data
plt.figure(figsize=(20, 12))
trends_data.plot(title='Google Trends for STOCKS')
plt.xlabel('Date')
plt.ylabel('Interest Over Time')
plt.show()

#===============================================================================
#
# source: https://medium.com/pythoneers/10-impressive-automation-scripts-you-need-to-try-using-python-bc9bc7563633
#===============================================================================
#===============================================================================
# FUNCTION TO TRANSFORM DAY OF YEAR INTO DATE
#===============================================================================
def day_of_year_to_date(day_of_year, year):
    from datetime import datetime, timedelta
    date_object = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    return date_object.strftime('%Y-%m-%d')

# Example usage:
#day_of_year = 100
#year = 2024
#print(day_of_year_to_date(day_of_year, year))  # Output: 2024-04-09


#===============================================================================
# FUNCTION TO SCALE DATA 
#===============================================================================
def padronizacao_dados(df_input,method = 'standard'):
    '''Função para padronizar os dados
    Input, Output = dataframes
    Permite usar 3 tipos de scaling, Standard, MinmaxScaler (entre 0 e 1ou entre 0 e 100) e RobustScaler
    '''
    
    df = df_input.copy()  
    
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    
    #===============================================================================
    # Seleciona apenas as colunas numéricas do dataframe
    # Objetivo: evitar erro quando coluna 'Programa' existe
    #===============================================================================
    df = df.select_dtypes(include=[int, float, 'number'])
    #df_select = df.drop(cat_columns, axis=1)
    
    if method == 'standard':
        df_normalized = (df - df.mean())/df.std()
    elif method == 'robust':
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(df)
        #df_normalized = pd.DataFrame(scaled_data, columns = list(df.columns))
        df_normalized = pd.DataFrame(scaled_data, columns = list(df.columns), index = df.index)
    elif method == 'minmax':
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        #df_normalized = pd.DataFrame(scaled_data, columns = list(df.columns))
        df_normalized = pd.DataFrame(scaled_data, columns = list(df.columns), index = df.index)
    elif method == 'minmax_100':
        scaler = MinMaxScaler(feature_range = (0,100))
        scaled_data = scaler.fit_transform(df)
        #df_normalized = pd.DataFrame(scaled_data, columns = list(df.columns))
        df_normalized = pd.DataFrame(scaled_data, columns = list(df.columns), index = df.index)    
    
    return df_normalized


#===============================================================================
# FUNÇÃO PARA CALCULAR VIF = VARIANCE INFLATION FACTOR
#===============================================================================
def calculate_vif(data):
    import pandas as pd
    import numpy as np
    #import sklearn
    from sklearn.linear_model import LinearRegression
    vif_df = pd.DataFrame()
    vif_df["variables"] = data.columns
    vif_df["VIF"] = [1 / (1 - LinearRegression().fit(data.drop([var], axis=1), data[var]).score(data.drop([var], axis=1), data[var])) for var in data.columns]
    return vif_df


#===============================================================================
#
#===============================================================================


#===============================================================================
#
#===============================================================================


#===============================================================================
#
#===============================================================================


#===============================================================================
#
#===============================================================================


#===============================================================================
#
#===============================================================================



