import numpy as np
import pandas as pd

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

from sklearn.linear_model import LogisticRegression
LogisticReg = LogisticRegression()
LogisticReg.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
Classifier_kNN = KNeighborsClassifier(n_neighbors=30, weights='uniform', algorithm='brute', p=2)
Classifier_kNN.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0,
                       min_samples_leaf=100, min_samples_split=200,
                       min_weight_fraction_leaf=0.0,
                       random_state=0, splitter='best')
dtree.fit(X_train, y_train)

y_pred_test_RLog  = LogisticReg.predict(X_test)
y_pred_test_RNA_P  = LogisticReg.predict_proba(X_test)

y_pred_test_KNN    = Classifier_kNN.predict(X_test)

y_pred_test_DT = dtree.predict(X_test)

Erro_RLog_Classificacao = np.mean(np.absolute(y_pred_test_RLog - y_test))
Erro_KNN_Classificacao = np.mean(np.absolute(y_pred_test_KNN - y_test))
Erro_DT_Classificacao = np.mean(np.absolute(y_pred_test_DT - y_test))

print()
print('---------------------------------------------')
print('Regressão Logística  - Erro de Classificação:',Erro_RLog_Classificacao)
print('Regressão Logística  - Acurácia:',1-Erro_RLog_Classificacao)
print('KNN                  - Erro de Classificação:',Erro_KNN_Classificacao)
print('KNN                  - Acurácia:',1-Erro_KNN_Classificacao)
print('Árvore               - Erro de Classificação:',Erro_DT_Classificacao)
print('Árvore               - Acurácia:',1-Erro_DT_Classificacao)
print('----------------------------------------------')
