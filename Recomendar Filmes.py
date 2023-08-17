#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


notas = pd.read_csv('rating.csv')


# In[3]:


notas.isnull().sum()


# In[4]:


notas


# In[5]:


filmes = pd.read_csv('movie.csv')


# In[6]:


filmes


# In[7]:


filmesv2 = filmes[filmes['genres'] != '(no genres listed)']


# In[8]:


filmesv2.reset_index(inplace = True,drop= True)


# In[9]:


filmesv2.info()


# In[10]:


generos = filmesv2['genres'].str.split('|',expand = True)
generos


# In[11]:


filmesv2.drop('genres', axis = 1, inplace = True)


# In[12]:


filmesv2


# In[13]:



filmesv3 = pd.concat([filmesv2, generos], axis=1)


# In[14]:


filmesv3


# In[15]:


filmesv3.isna().sum()


# In[16]:


filmesv3.drop(columns = [2,3,4,5,6,7,8,9],axis = 1, inplace = True)


# In[17]:


filmesv3


# In[18]:


genome_scores = pd.read_csv('genome_scores.csv')
genome_scores


# In[19]:


notas_filme = notas.groupby('movieId')[['rating']].mean()


# In[20]:


notas_filme


# In[21]:


avaliacoes = pd.merge(filmesv3, notas_filme, how = 'inner', on = 'movieId')


# In[22]:


avaliacoes


# In[23]:


avaliacoes = avaliacoes.rename(columns = {0:'Genero1', 1:'Genero2'})


# In[24]:


avaliacoes.info()


# In[25]:


from sklearn.preprocessing import OneHotEncoder


# In[26]:


ohe = OneHotEncoder(dtype=int)
colunas_ohe = ohe.fit_transform(avaliacoes[['Genero1']]).toarray()
avaliacoes2 = avaliacoes.drop('Genero1', axis=1)

dados_musicas_dummies = pd.concat([avaliacoes2, pd.DataFrame(colunas_ohe, columns=ohe.get_feature_names_out(['Genero1']))], axis=1)
dados_musicas_dummies


# In[27]:


dados_musicas_dummies


# In[28]:


ohe = OneHotEncoder(dtype=int)
colunas_ohe = ohe.fit_transform(dados_musicas_dummies[['Genero2']]).toarray()
avaliacoes3 = dados_musicas_dummies.drop('Genero2', axis=1)

dados_musicas_dummies2 = pd.concat([avaliacoes3, pd.DataFrame(colunas_ohe, columns=ohe.get_feature_names_out(['Genero2']))], axis=1)
dados_musicas_dummies2


# In[29]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
SEED = 1224
np.random.seed(1224)


pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=0.7, random_state=SEED))])


music_embedding_pca = pca_pipeline.fit_transform(dados_musicas_dummies2.drop(['movieId','title'],axis=1))
projection_m = pd.DataFrame(data=music_embedding_pca)


# In[30]:


projection_m 


# In[31]:


pca_pipeline[1].n_components_


# In[32]:


from sklearn.cluster import KMeans

SEED = 12
kmeans_pca_pipeline = KMeans(n_clusters=5, verbose=False, random_state=SEED)

kmeans_pca_pipeline.fit(projection_m)

avaliacoes['cluster_pca'] = kmeans_pca_pipeline.predict(projection_m)
projection_m['cluster_pca'] = kmeans_pca_pipeline.predict(projection_m)


# In[33]:


projection_m['title'] = avaliacoes['title']


# In[34]:


projection_m


# In[35]:


import plotly.express as px


# In[36]:


pca_pipeline[1].explained_variance_ratio_.sum()


# In[37]:


pca_pipeline[1].explained_variance_.sum()


# In[38]:


nome_filme = 'The Pirates (2014)'


# In[39]:


from pandas.core.dtypes.cast import maybe_upcast
from sklearn.metrics.pairwise import euclidean_distances

def recomendar(filme):
    cluster = list(projection_m[projection_m['title']== filme]['cluster_pca'])[0]
    filmes_recomendados = projection_m[projection_m['cluster_pca']== cluster][[0, 1, 'title']]
    x_filme = list(projection_m[projection_m['title']== filme][0])[0]
    y_filme = list(projection_m[projection_m['title']== filme][1])[0]

    #distâncias euclidianas
    distancias = euclidean_distances(filmes_recomendados[[0, 1]], [[x_filme, y_filme]])
    filmes_recomendados['movieId'] = avaliacoes['movieId']
    filmes_recomendados['distancias']= distancias
    recomendada = filmes_recomendados.sort_values('distancias').head(10)
    recomedacoes = recomendada.drop(columns = [0,1,'distancias'])
    
    return recomedacoes


# In[40]:


recomendar('Innocence (2014)')


# In[41]:


avaliacoes.sample(20)


# In[42]:


recomendar('Return of Dracula, The (1958)')


# In[65]:


from ipywidgets import widgets, HBox, VBox
from IPython.display import display

filme = widgets.Text(description="Filme")
ano = widgets.Text(description="Ano")

botao = widgets.Button(description="Recomendar")

left = VBox([filme])
right = VBox([ano])
inputs = HBox([left, right])


def simulador(sender):
    filme_value = str(filme.value) if filme.value else 0
    ano_value = str(ano.value) if ano.value else 0

    # Realizando a previsão usando o modelo (defina a variável "model" antes desta etapa)
    entrada = str(filme_value + ' ' + '(' + ano_value + ')')
    print(entrada)
    #print(recomendar(entrada))
    print(recomendar(entrada))
    
# Atribuindo a função "simulador" ao evento click do botão
botao.on_click(simulador)


# In[66]:


display(inputs, botao)

