import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Carregando os dados
dataset = pd.read_csv('Case_cobranca.csv') 

dataset['ALVO']   = [0 if np.isnan(x) or x > 90 else 1 for x in dataset['TEMP_RECUPERACAO']]
dataset['PRE_IDADE']        = [18 if np.isnan(x) or x < 18 else x for x in dataset['IDADE']] 
dataset['PRE_IDADE']        = [1. if x > 76 else (x-18)/(76-18) for x in dataset['PRE_IDADE']] 
dataset['PRE_QTDE_DIVIDAS'] = [0.  if np.isnan(x) else x/16. for x in dataset['QTD_DIVIDAS']]    
dataset['PRE_NOVO']         = [1 if x=='NOVO'                      else 0 for x in dataset['TIPO_CLIENTE']]    
dataset['PRE_TOMADOR_VAZIO']= [1 if x=='TOMADOR' or str(x)=='nan'  else 0 for x in dataset['TIPO_CLIENTE']]                        
dataset['PRE_CDC']          = [1 if x=='CDC'                       else 0 for x in dataset['TIPO_EMPRESTIMO']]
dataset['PRE_PESSOAL']      = [1 if x=='PESSOAL'                   else 0 for x in dataset['TIPO_EMPRESTIMO']]
dataset['PRE_SEXO_M']       = [1 if x=='M'                         else 0 for x in dataset['CD_SEXO']]

y = dataset['ALVO']
X = dataset.iloc[:, 8:15].values 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 325) ## Os três ultimos digitos do meu RA

from sklearn.neighbors import KNeighborsClassifier

gx = []
gy = []

Classifier_kNN = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='brute', p=2)
Classifier_kNN.fit(X_train, y_train)
y_pred_test_KNN    = Classifier_kNN.predict(X_test)
Erro_KNN_Classificacao = np.mean(np.absolute(y_pred_test_KNN - y_test))
print('---------------------------------------------------------------')
print('k', 'Erro de Classificação')
print('1',Erro_KNN_Classificacao)
gx.append(1)
gy.append(Erro_KNN_Classificacao)

for k in range(5, 201, 5):
    Classifier_kNN = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='brute', p=2)
    Classifier_kNN.fit(X_train, y_train)
    y_pred_test_KNN    = Classifier_kNN.predict(X_test)

    Erro_KNN_Classificacao = np.mean(np.absolute(y_pred_test_KNN - y_test))
    print(k,Erro_KNN_Classificacao)
    
    gx.append(k)
    gy.append(Erro_KNN_Classificacao)
print('---------------------------------------------------------------')


print('------------------------- Gráfico -----------------------------')
print()
plt.plot(gx,gy)
plt.plot(gx,gy, 'bo') 
plt.title('Escolha do Melhor k')
plt.ylabel('Erro de Classificação')
plt.xlabel('Valor de k')
plt.show()
