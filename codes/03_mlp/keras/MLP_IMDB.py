# -*- coding: utf-8 -*-
"""SI 2 - MLP_IMDB.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X8NzgPPkkidt03fLFUm6ix1qVLLf0tfy
"""

# numpy é usada para manipulação de arrays numéricos
import numpy as np

#pandas é usados para manipulação de data frames
import pandas as pd

# Tensorflow é a implementação padrão dos modelos de DL, ele é base do Keras
import tensorflow as tf
from tensorflow import keras

# IMDB é o dataset textual que vamos manipular
from keras.datasets import imdb

# objetos do keras para criar arquiteturas de DL
from keras.models import Sequential
from keras.layers import Dense, Input

# para geração de gráficos
import matplotlib.pyplot as plt

# Carregar o dataset do IMDB. Dataset composto de comentários sobre filmes/séries
# categorizados como positivos (1) ou negativos (0).
# O dataset no keras já foi preprocessando, e cada palavra é substituída por um índice inteiro.
# O comentários são representados por sequências de inteiros variáveis.

# Definindo a quantidade de palavras distintas que vamos considerar (vocabulário)
nb_words = 10000
# carregando o dataset via keras
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=nb_words)

# Explorando os dados

# concatenando os atributos descritivos do treino e teste
data    = np.concatenate((x_train, x_test), axis=0)

# Numero de palavras distintas (únicas) em todos os textos
print("Number of unique words:", len(np.unique(np.hstack(data))))

# concatenando os atributos preditivos do treino e teste
targets = np.concatenate((y_train, y_test), axis=0)
# quantidade de categorias, labels
print("Categories:", np.unique(targets))

# Tamanho medio e desvio padrão dos textos
length = [len(i) for i in data]
print("Average Review length:", np.mean(length))
print("Standard Deviation:", round(np.std(length)))

# Dimensões do dataset [treino e teste]
# Treino = (x_train, y_train)
print("* x_train: " + str(type(x_train)) + " com " + str(x_train.shape))
print("* y_train: " + str(type(y_train)) + " com " + str(y_train.shape))

# Teste = (x_test, y_test)
print("* x_test:  " + str(type(x_test)) + " com " + str(x_test.shape))
print("* y_test:  " + str(type(y_test)) + " com " + str(y_test.shape))

# Olhando os exemplos
print(x_train[0], "length:", len(x_train[0]), "class:", y_train[0])
print(x_train[1], "length:", len(x_train[1]), "class:", y_train[1])
print(x_train[2], "length:", len(x_train[2]), "class:", y_train[2])
print(x_train[3], "length:", len(x_train[3]), "class:", y_train[3])

# Engenharia reversa para descobrir os textos originais

# tabela de palavras|indices
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# decodificador das mensagens
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
print(decoded_review)

# A MLP network expects input data to be of fixed size, so we convert the integer sequences into
# vectors of length nb_words. Each element in the vectors corresponds to a specific word in
# the vocabulary: the element is "1" if the words appears in the review and "0" otherwise.

# Como os vetores dos comentários possuem tamanhos diferentes, a MLP vai falhar
# miseravelmente para poder executar. Nós devemos padronizar e tornar todos os
# exemplos com uma mesma quantidade de características.
# Vamos "vetorizar" os exemplos, todos com tamanho nb_words (10000)
# Cada elemento do vetor corresponde a uma palavra específica no vocabulário:
# se a palavra existe, a posição do vetor é preenchida com "1". Caso contrário,
# recebe "0".

def vectorize(sequences, dimension = nb_words):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1
  return results

# Vetorizamos os conjuntos descritivos (treino e teste)
X_train = vectorize(x_train) # X_train = x_train modificado
X_test  = vectorize(x_test)  # X_test  = x_test modificado

# Converter os labels de inteiros para floats (treino e teste)
y_train = np.asarray(y_train).astype('float32')
y_test  = np.asarray(y_test).astype('float32')

# Verificando as dimensões dos conjuntos
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

# Vendo como ficou um exemplo codificado
print(X_train[0], "length:", len(X_train[0]), "class:", y_train[0])

# Vendo o dataset como um data frame
pd.DataFrame(X_train)

# Definindo uma MLP

#MLP é um modelo de rede neura sequencial
mlpModel = Sequential()

# Adicionando as Camadas Ocultas
# Teremos uma única camada oculta, com 10 neurônios, todos com ativação sigmoidal
# input_shape especifica qual é a dimensão do sinal de entrada, que são as
# nb_words palavras
mlpModel.add(Dense(10, activation = "sigmoid", input_shape=(nb_words, )))

# Adicionando a Camada de Saída
# camada com um único neurônio, com ativação sigmoidal
mlpModel.add(Dense(1, activation = "sigmoid"))

# imprime o modelo, para verificarmos a arquitetura
mlpModel.summary()

# Aqui especificamos como será o processo de treinamento do modelo
# Nós vamos usar:
#    - SGD: algoritmo de gradiente descendente para treinar a rede
#    - binary crossentropy: entropia binária cruzada como medida de erro (loss),
#         que vai ser minimizada entre as épocas
#.   - accuracy: acurácia do modelo em cada época
mlpModel.compile(
    loss='binary_crossentropy',
    optimizer='SGD',
    metrics=['accuracy']
)

# Esse é o setup experimental para execução do treinamento da rede
# o método fit chama o treinamento da rede neural
#     - X_train: é o conjunto de treinamento
#     - y_train: são os rótulos do conjunto de treinamento
#     - epochs: quantidade de épocas que a rede irá treinar
#     - batch_size: quantidade de exemplos treinados em lote, para gerar um ajuste de pesos

history = mlpModel.fit(
    X_train, y_train,
    epochs = 10,
    batch_size = 512,
    verbose=2
)

#Visualizando as curvas de erro e acurácia
plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['loss'])
plt.title('training loss')

plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['accuracy'])
plt.title('training accuracy');

# Avaliando o modelo treinado no conjunto de testes
scores = mlpModel.evaluate(X_test, y_test, verbose=2)
print("%s: %.2f%%" % (mlpModel.metrics_names[1], scores[1]*100))