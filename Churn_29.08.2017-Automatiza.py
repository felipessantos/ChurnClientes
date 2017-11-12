
# coding: utf-8

# In[1]:

import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import csv
import unidecode 
import pandas.core.algorithms as algos
from scipy.stats import kendalltau   
from funcoes_uteis import *
from dateutil.relativedelta import relativedelta

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder

from dateutil.relativedelta import relativedelta

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[38]:

def diff_month(d1, d2):
    return (d1.year - d2.year)*12 + d1.month - d2.month

def periodicidade(x):
    if x == 'Trienal':
        return 36    
    elif x == 'Anual':
        return 12
    elif x == 'Semestral':
        return 6
    elif x == 'Trimestral':
        return 3
    else: 
        return 1

def marca_lista(lista, tamanho, variavel):
    i = 0
    for i in range(tamanho):
        if variavel == lista[i]: return i
                
def cria_curva(percentiles, variavel):
    Perc = list()
    for i in range(len(percentiles)):
        Perc.append(np.percentile(variavel, percentiles[i]))
    return Perc

def marca_base(Perc, x):
    if x >= Perc[(len(Perc)-1)]:
        return len(Perc) +1
    else:
        for i in range(len(Perc)):
            if x < Perc[i]:
                return i + 1
            
def aux_nome_data(data):
    y = str(data.year)
    m = str(data.month)
    d = str(data.day)
    if data.day < 9:
        d = '0'+str(data.day)
    if data.month < 9:
        m = '0'+str(data.month)
    return y+m+d


# In[39]:

def ArrumaBase_Churn(fim_janela_feature, janela_churn, df_base):
    fim_janela_churn = fim_janela_feature+ relativedelta(months=janela_churn)
    inicio_janela_churn = fim_janela_feature
    df_treino = df_base[(df_base.Instalacao_AnoMes < fim_janela_feature) & 
                        (df_base.MesesParaChurn == 0) &
                        (df_base.Status == 'ativo')].copy()
    df_treino_fut = df_base[(df_base.Instalacao_AnoMes < fim_janela_feature) & 
                        (df_base.Data_churn > fim_janela_churn)].copy()
    df_treino_fut['FlagChurn'] = 0
    df_treino_churn = df_base[(df_base.Data_churn >= inicio_janela_churn) & 
                              (df_base.Data_churn < fim_janela_churn)].copy()
    df_treino = pd.concat([df_treino, df_treino_churn, df_treino_fut])
    agg_dict = {'Provisioning' : 'count'}
    aux = df_base[df_base.Instalacao_AnoMes < fim_janela_feature].groupby(['cd_ChaveCliente']).agg(agg_dict).copy()
    aux.reset_index(inplace= True)
    aux.rename(columns= {'Provisioning': 'Qtd_Provisioning'}, inplace= True)
    df_treino = pd.merge(df_treino, aux, on='cd_ChaveCliente')
    agg_dict = {'nr_PrecoMensal' : 'sum'}
    aux = df_base[df_base.Instalacao_AnoMes < fim_janela_feature].groupby(['cd_ChaveCliente']).agg(agg_dict).copy()
    aux.reset_index(inplace= True)
    aux.rename(columns= {'nr_PrecoMensal': 'Valor_cliente_Mes'}, inplace= True)
    df_treino = pd.merge(df_treino, aux, on='cd_ChaveCliente')
    df_treino['Periodicidade_Meses'] = [periodicidade(x) for x in df_treino.ds_Periodicidade]
    df_treino['idade_prov'] = [diff_month(fim_janela_feature, Inst) for Inst in df_treino.Instalacao_AnoMes]
    df_treino['idade_cli'] = [diff_month(fim_janela_feature, Inst) for Inst in df_treino.Primeiro_Servico_LW_AnoMes]
    df_treino['quantidade_renovacoes'] = [int(id_prov/peri_mes) for 
                                          id_prov, peri_mes in zip (df_treino.idade_prov, df_treino.Periodicidade_Meses)]
    df_treino['Qtd_meses_P_renovacoes'] = [peri_mes-(idade_prov-qtd_renov*peri_mes) for
                                           peri_mes,idade_prov,qtd_renov in 
                                           zip(df_treino.Periodicidade_Meses, 
                                               df_treino.idade_prov, 
                                               df_treino.quantidade_renovacoes)]
    df_treino['mes_Proxima_renovacao'] = [datetime((d + timedelta((qtd_renovacoes*periodicidade+mdelta)*365/12)).year,
                                                   (d + timedelta((qtd_renovacoes*periodicidade+mdelta)*365/12)).month,1) 
                                          for d, qtd_renovacoes, periodicidade, mdelta in 
                                          zip (df_treino.Instalacao_AnoMes, 
                                               df_treino.quantidade_renovacoes,
                                               df_treino.Periodicidade_Meses,
                                               df_treino.Qtd_meses_P_renovacoes)]
    percentiles= [10,20,30,40,50,60,70,80,90]
    Perc = cria_curva(percentiles, df_treino.nr_PrecoMensal)
    df_treino['nr_PrecoMensal_curva'] = [marca_base(Perc, x) for x in df_treino.nr_PrecoMensal]
    Perc = cria_curva(percentiles, df_treino.Valor_cliente_Mes)
    df_treino['Valor_cliente_Mes_curva'] = [marca_base(Perc, x) for x in df_treino.Valor_cliente_Mes]
    return df_treino


# In[40]:

def CriaRandomForest_Churn(df_treino):
    # shuffle rows
    X = df_treino.sample(frac = 1)
    # create X and y matrices
    y = X.FlagChurn.values
#    Colunas_Modelo = X.reset_index(drop=True).drop(['FlagChurn'], axis = 1).columns
    X = X.reset_index(drop=True).drop(['FlagChurn'], axis = 1).values
    # scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=2)
    auc_list = []
    k = 1
    for train, valid in skf.split(X, y):
#        print('Fold #', k)
#        print("train indices: %s\nvalidation indices %s" % (train, valid))
        clf = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=10)
        clf.fit(X[train], y[train])
        y_pred = clf.predict_proba(X[valid])
        auc = roc_auc_score(y[valid], y_pred[:,1])
        auc_list.append(auc)
#        print('AUC on fold #', k, ':', auc, '\n')
        k += 1
#    print('Average AUC on', k-1, 'folds:', np.mean(auc_list))
    return clf


# In[41]:

def CriaCluster_Churn(clf, df_treino, Colunas_Modelo):
    X = df_treino[Colunas_Modelo].copy()
    X = X.reset_index(drop=True).drop(['FlagChurn'], axis = 1).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_pred = clf.predict_proba(X)
    y_pred = pd.DataFrame(data=y_pred[:,1])
    y_pred.rename(columns= {0: 'Prob_Churn'}, inplace= True)
    df_treino = pd.concat([df_treino, y_pred], axis = 1)
    percentiles = list()
    inicio = 0
    fim = 100
    salto = 0.25
    aux = round((fim - inicio)/salto)
    for i in range(aux):
        percentiles.append((inicio +i*salto))
    Perc = cria_curva(percentiles, df_treino.Prob_Churn)
    df_treino['Prob_Churn_Grupo'] = [marca_base(Perc, x) for x in df_treino.Prob_Churn]
    dict_lista_aux = {'Provisioning' : 'count',
                      'Prob_Churn' : 'min',
                      'FlagChurn' : 'mean'}
    RESUMO = df_treino.groupby('Prob_Churn_Grupo').agg(dict_lista_aux)
    RESUMO.sort_values(['Prob_Churn'], ascending= 0 ,inplace=True)
    RESUMO.Prob_Churn = round(RESUMO.Prob_Churn, ndigits = 2)
    RESUMO['FlagChurn_aux'] = [p*q for p, q in zip (RESUMO.FlagChurn, RESUMO.Provisioning)]
    RESUMO['acumulado'] = RESUMO.Provisioning.cumsum()
    RESUMO['Prob_acumulado'] = RESUMO.FlagChurn_aux.cumsum()
    RESUMO['FlagChurn_acumulado'] = [p/q for p, q in zip (RESUMO.Prob_acumulado, RESUMO.acumulado)]
    RESUMO['Prob_Chrun_aux'] = [p*q for p, q in zip (RESUMO.Prob_Churn, RESUMO.Provisioning)]
    RESUMO['acumulado'] = RESUMO.Provisioning.cumsum()
    RESUMO['Prob_acumulado'] = RESUMO.Prob_Chrun_aux.cumsum()
    RESUMO['Prob_Chrun_acumulado'] = [p/q for p, q in zip (RESUMO.Prob_acumulado, RESUMO.acumulado)]
    RESUMO.reset_index(inplace= True)
    lista = RESUMO.FlagChurn_acumulado
    Prob_aux1 = RESUMO.loc[0].FlagChurn_acumulado
    curva_indice = []
    for i in range(len(lista)):
        Prob_aux2 = lista[i]
        if len(curva_indice) < 9 and Prob_aux1/Prob_aux2 >= 1.25:
            Prob_aux1 = Prob_aux2 
            curva_indice.append(i)
    Perc = []
    for i in range(len(curva_indice)):
        Perc.append(RESUMO.FlagChurn_acumulado.loc[curva_indice[i]])
    Perc = Perc[::-1]
    RESUMO['Cluster'] = [marca_base(Perc, x) for x in RESUMO.FlagChurn_acumulado]
    RESUMO.reset_index(inplace= True, drop= True)
    lista = [RESUMO[RESUMO.Cluster == 1].Prob_Churn_Grupo.max(),
             RESUMO[RESUMO.Cluster == 2].Prob_Churn_Grupo.max(),
             RESUMO[RESUMO.Cluster == 3].Prob_Churn_Grupo.max(),
             RESUMO[RESUMO.Cluster == 4].Prob_Churn_Grupo.max(),
             RESUMO[RESUMO.Cluster == 5].Prob_Churn_Grupo.max(),
             RESUMO[RESUMO.Cluster == 6].Prob_Churn_Grupo.max(),
             RESUMO[RESUMO.Cluster == 7].Prob_Churn_Grupo.max(),
             RESUMO[RESUMO.Cluster == 8].Prob_Churn_Grupo.max(),
             RESUMO[RESUMO.Cluster == 9].Prob_Churn_Grupo.max()]
    df_treino['Cluster'] = [marca_base(lista, x) for x in df_treino.Prob_Churn_Grupo]
    return df_treino


# In[59]:

def ComparaCluster_DadoCli(df_treino, texto):
    fim_janela_feature = df_treino.Instalacao_AnoMes.max()
    aux_Cluster = pd.DataFrame(data=df_treino.Cluster.value_counts()) 
    aux_Cluster.sort_index(inplace= True)
    aux_Cluster['Perc'] = aux_Cluster.Cluster/aux_Cluster.Cluster.sum()
    dict_lista_aux = {'Provisioning' : 'count',
                      'Prob_Churn' : 'min',
                      'FlagChurn' : 'mean',
                      'nr_PrecoMensal' : 'mean',
                      'fl_ServicoPai' : 'mean',
                      'fl_Dev' : 'mean',
                      'fl_GerenteConta' : 'mean',
                      'Valor_cliente_Mes' : 'mean',
                      'Periodicidade_Meses' : 'mean',
                      'idade_prov' : 'mean',
                      'idade_cli' : 'mean',
                      'quantidade_renovacoes' : 'mean',
                      'Qtd_meses_P_renovacoes' : 'mean'}
    RESUMO = df_treino.groupby('Cluster').agg(dict_lista_aux)
    RESUMO['QtdChurn'] = RESUMO.FlagChurn*RESUMO.Provisioning
    NomeCSV = 'RelarotioClusterChurn'+texto+aux_nome_data(fim_janela_feature)+'.csv'
    RESUMO.to_csv(NomeCSV)
    return aux_Cluster

def Compara_ClusterProduto_QtdProv(df_treino):
    fim_janela_feature = df_treino.Instalacao_AnoMes.max()
    agg_dict_heatmap = {'Provisioning' : 'count'}
    heatmap = df_treino.groupby(['Cluster','Servico']).agg(agg_dict_heatmap).copy()
    heatmap.reset_index(inplace= True)
    heatmap = heatmap.pivot('Cluster','Servico', 'Provisioning').copy()
    heatmap.fillna(0, inplace= True)
    heatmap['TOTAL'] = heatmap[heatmap.columns].sum(axis = 1)    
    heatmap = heatmap.T
    heatmap['TOTAL'] = heatmap[heatmap.columns].sum(axis = 1)
    heatmap.sort('TOTAL', ascending= 0, inplace= True)
    heatmap.drop('TOTAL', axis= 1)
    NomeCSV = 'RelarotioServicoClusterQtdProvs'+aux_nome_data(fim_janela_feature)+'.csv'
    heatmap.to_csv(NomeCSV)    
    
    agg_dict_heatmap = {'FlagChurn' : 'mean'}
    heatmap = df_treino.groupby(['Cluster','Servico']).agg(agg_dict_heatmap).copy()
    heatmap.reset_index(inplace= True)
    heatmap = heatmap.pivot('Cluster','Servico', 'FlagChurn').copy()
    heatmap.fillna(0, inplace= True)
    heatmap = heatmap.T
    NomeCSV = 'RelarotioServicoClusterChurn'+aux_nome_data(fim_janela_feature)+'.csv'
    heatmap.to_csv(NomeCSV)    
    return heatmap


# In[55]:

def ArrumaBase_Churn_AplicaAlgoritmo(fim_janela_feature, janela_churn, df_base, aux_Cluster, Colunas_Modelo, clf, lista_perfil, lista_classificacao, lista_servico):
    fim_janela_churn = fim_janela_feature+ relativedelta(months=janela_churn)
    inicio_janela_churn = fim_janela_feature
    df_treino = df_base[(df_base.Instalacao_AnoMes < fim_janela_feature) & 
                        (df_base.MesesParaChurn == 0) &
                        (df_base.Status == 'ativo')].copy()
    df_treino_fut = df_base[(df_base.Instalacao_AnoMes < fim_janela_feature) & 
                        (df_base.Data_churn > fim_janela_churn)].copy()
    df_treino_fut['FlagChurn'] = 0
    df_treino_churn = df_base[(df_base.Data_churn >= inicio_janela_churn) & 
                              (df_base.Data_churn < fim_janela_churn)].copy()
    df_treino = pd.concat([df_treino, df_treino_churn, df_treino_fut])
    
    agg_dict = {'Provisioning' : 'count'}
    aux = df_base[df_base.Instalacao_AnoMes < fim_janela_feature].groupby(['cd_ChaveCliente']).agg(agg_dict).copy()
    aux.reset_index(inplace= True)
    aux.rename(columns= {'Provisioning': 'Qtd_Provisioning'}, inplace= True)
    df_treino = pd.merge(df_treino, aux, on='cd_ChaveCliente')
    
    agg_dict = {'nr_PrecoMensal' : 'sum'}
    aux = df_base[df_base.Instalacao_AnoMes < fim_janela_feature].groupby(['cd_ChaveCliente']).agg(agg_dict).copy()
    aux.reset_index(inplace= True)
    aux.rename(columns= {'nr_PrecoMensal': 'Valor_cliente_Mes'}, inplace= True)
    df_treino = pd.merge(df_treino, aux, on='cd_ChaveCliente')
    
    df_treino['Periodicidade_Meses'] = [periodicidade(x) for x in df_treino.ds_Periodicidade]
    df_treino['idade_prov'] = [diff_month(fim_janela_feature, Inst) for Inst in df_treino.Instalacao_AnoMes]
    df_treino['idade_cli'] = [diff_month(fim_janela_feature, Inst) for Inst in df_treino.Primeiro_Servico_LW_AnoMes]
    df_treino['quantidade_renovacoes'] = [int(id_prov/peri_mes) for 
                                          id_prov, peri_mes in zip (df_treino.idade_prov, df_treino.Periodicidade_Meses)]
    df_treino['Qtd_meses_P_renovacoes'] = [peri_mes-(idade_prov-qtd_renov*peri_mes) for
                                           peri_mes,idade_prov,qtd_renov in 
                                           zip(df_treino.Periodicidade_Meses, 
                                               df_treino.idade_prov, 
                                               df_treino.quantidade_renovacoes)]
    df_treino['mes_Proxima_renovacao'] = [datetime((d + timedelta((qtd_renovacoes*periodicidade+mdelta)*365/12)).year,
                                                   (d + timedelta((qtd_renovacoes*periodicidade+mdelta)*365/12)).month,1) 
                                          for d, qtd_renovacoes, periodicidade, mdelta in 
                                          zip (df_treino.Instalacao_AnoMes, 
                                               df_treino.quantidade_renovacoes,
                                               df_treino.Periodicidade_Meses,
                                               df_treino.Qtd_meses_P_renovacoes)]
    ################################################################################################
    tamanho = len(lista_perfil) 
    df_treino['Perfil_indice'] = [marca_lista(lista_perfil, tamanho, variavel)
                                   for variavel in df_treino.Perfil]
    tamanho = len(lista_classificacao) 
    df_treino['classificacao_indice'] = [marca_lista(lista_classificacao, tamanho, variavel)
                                         for variavel in df_treino.classificacao]
    tamanho = len(lista_servico) 
    df_treino['Servico_indice'] = [marca_lista(lista_servico, tamanho, variavel)
                                   for variavel in df_treino.Servico]
    
    percentiles= [10,20,30,40,50,60,70,80,90]
    Perc = cria_curva(percentiles, df_treino.nr_PrecoMensal)
    df_treino['nr_PrecoMensal_curva'] = [marca_base(Perc, x) for x in df_treino.nr_PrecoMensal]

    Perc = cria_curva(percentiles, df_treino.Valor_cliente_Mes)
    df_treino['Valor_cliente_Mes_curva'] = [marca_base(Perc, x) for x in df_treino.Valor_cliente_Mes]
    df_treino.fillna(0, inplace= True)
    X = df_treino[Colunas_Modelo].copy()
    X = X.reset_index(drop=True).drop(['FlagChurn'], axis = 1).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_pred = clf.predict_proba(X)
    y_pred = pd.DataFrame(data=y_pred[:,1])
    y_pred.rename(columns= {0: 'Prob_Churn'}, inplace= True)
    df_treino = pd.concat([df_treino, y_pred], axis = 1)
    df_treino.sort_values(['Prob_Churn'], ascending= 1 ,inplace=True)
    df_treino.reset_index(inplace= True, drop= True)
    df_treino.reset_index(inplace= True)
    df_treino.rename(columns= {'index': 'aux_cluster'}, inplace= True)
    lista = [round(aux_Cluster[aux_Cluster.index == 1].Perc.max()*df_treino.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 2].Perc.max()*df_treino.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 3].Perc.max()*df_treino.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 4].Perc.max()*df_treino.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 5].Perc.max()*df_treino.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 6].Perc.max()*df_treino.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 7].Perc.max()*df_treino.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 8].Perc.max()*df_treino.aux_cluster.max(),0),
             round(aux_Cluster[aux_Cluster.index == 9].Perc.max()*df_treino.aux_cluster.max(),0)]
    lista[1] = lista[1] + lista[0]
    lista[2] = lista[2] + lista[1]
    lista[3] = lista[3] + lista[2]
    lista[4] = lista[4] + lista[3]
    lista[5] = lista[5] + lista[4]
    lista[6] = lista[6] + lista[5]
    lista[7] = lista[7] + lista[6]
    lista[8] = lista[8] + lista[7]
    df_treino['Cluster'] = [marca_base(lista, x) for x in df_treino.aux_cluster]
    return df_treino


# In[8]:

df_base = pd.read_csv('/home/felipe/Algoritmos_todosProdutos/Churn_Consumo_Recomendacao.csv'
                      , error_bad_lines = False
                      , sep=';'
                      , dtype= {7: str}
                      , encoding='latin-1'
                      , header = 0)
df_base['Status'] = ['ativo' if s in ['Ativo', 'Atendido', 'Em ativação',
                                      'Aguardando ativação'] else 'inativo'
                     for s in df_base.Status]
df_base['fl_Dev'] = df_base['fl_Dev'].astype(float,)
df_base = df_base[(df_base.nr_PrecoMensal > '0,00')].copy()
df_base.sort_values(['Instalacao'], ascending= 1 ,inplace=True)
df_base.drop_duplicates(['Provisioning'], keep='last', inplace= True)
col_datas = ['Data_Desativacao', 'Data_Fim', 'Instalacao', 'dt_Reativacao', 'Primeiro_Servico_LW']
converte_datetime(df_base, col_datas)
df_base['Data_Fim_flag'] = [1 if d_fim != datetime(1900,1,1) else 0 for d_fim in df_base.Data_Fim]
df_base['Data_Desativacao_flag'] = [1 if f_des != datetime(1900,1,1) else 0 for f_des in df_base.Data_Desativacao]
df_base['Data_Desativacao_flag'] = [1 if f_des != datetime(1900,1,1) else 0 for f_des in df_base.Data_Desativacao]
df_base['Data_churn'] = df_base[['Data_Fim', 'Data_Desativacao']].min(axis = 1).astype('datetime64[ns]')
df_base['Data_churn'] = [d_fim if ((d_des < d_rea < d_fim) & f_fim & f_des) else d_chu
                              for d_des, d_rea, d_fim, f_fim, f_des, d_chu in 
                              zip(df_base.Data_Desativacao, df_base.dt_Reativacao, 
                                  df_base.Data_Fim, df_base.Data_Fim_flag, 
                                  df_base.Data_Desativacao_flag, df_base.Data_churn)]
df_base['Data_churn_flag'] = [1 if d > datetime(1900, 1, 1) else 0 for d in df_base.Data_churn]
#######################################
df_base = df_base[df_base.Primeiro_Servico_LW >= datetime(1990,1,1)].copy()  #
#######################################
df_base['Primeiro_Servico_LW_AnoMes'] = [datetime(d.year, d.month, 1) for d in df_base.Primeiro_Servico_LW]
df_base['Instalacao_AnoMes'] = [datetime(d.year, d.month, 1) for d in df_base.Instalacao]
df_base['Data_churn_AnoMes'] = [datetime(d.year, d.month, 1) for d in df_base.Data_churn]
df_base['fl_ServicoPai'] = [1 if s in ['SIM', 'Sim', 'sim'] else 0 for s in df_base.fl_ServicoPai]
df_base['fl_GerenteConta'] = [0 if s== 1 else 1 for s in df_base.id_GerenteConta]
df_base['nr_PrecoMensal'] = [x.replace(',', '.') for x in df_base.nr_PrecoMensal]
df_base['nr_PrecoMensal'] = df_base.nr_PrecoMensal.astype(float)
df_base['MesesParaChurn'] = [diff_month(ch, ins) if ch > datetime(1900, 1, 1) else 0
                           for ch, ins in  zip(df_base.Data_churn, df_base.Instalacao)]
df_base['FlagChurn'] = [1 if d!= 0 else 0 for d in df_base.MesesParaChurn]


# # 

# In[9]:

fim_janela_feature_1 = df_base.Instalacao_AnoMes.max()+ relativedelta(months=-6)
fim_janela_feature_2 = df_base.Instalacao_AnoMes.max()+ relativedelta(months=-8)
fim_janela_feature_3 = df_base.Instalacao_AnoMes.max()+ relativedelta(months=-10)
fim_janela_feature_4 = df_base.Instalacao_AnoMes.max()+ relativedelta(months=-12)
janela_churn = 1

result = pd.concat([ArrumaBase_Churn(fim_janela_feature_1, janela_churn, df_base),
                    ArrumaBase_Churn(fim_janela_feature_2, janela_churn, df_base),
                    ArrumaBase_Churn(fim_janela_feature_3, janela_churn, df_base), 
                    ArrumaBase_Churn(fim_janela_feature_4, janela_churn, df_base)], 
                   keys=[fim_janela_feature_1 + relativedelta(months=janela_churn),
                         fim_janela_feature_2 + relativedelta(months=janela_churn),
                         fim_janela_feature_3 + relativedelta(months=janela_churn),
                         fim_janela_feature_4 + relativedelta(months=janela_churn)])

lista_perfil = result.Perfil.unique()
tamanho = len(lista_perfil) 
result['Perfil_indice'] = [marca_lista(lista_perfil, tamanho, variavel)
                           for variavel in result.Perfil]

lista_classificacao = result.classificacao.unique()
tamanho = len(lista_classificacao) 
result['classificacao_indice'] = [marca_lista(lista_classificacao, tamanho, variavel)
                                  for variavel in result.classificacao]

lista_servico = result.Servico.unique()
tamanho = len(lista_servico) 
result['Servico_indice'] = [marca_lista(lista_servico, tamanho, variavel)
                            for variavel in result.Servico]

result.fillna(0, inplace= True)
colunas= ['nr_PrecoMensal', 'fl_ServicoPai', 'fl_Dev', 'fl_GerenteConta', 'Qtd_Provisioning',
          'Valor_cliente_Mes', 'Periodicidade_Meses', 'idade_prov', 'idade_cli', 'quantidade_renovacoes',
          'Qtd_meses_P_renovacoes', 'Perfil_indice', 'classificacao_indice', 'Servico_indice', 
          'nr_PrecoMensal_curva', 'Valor_cliente_Mes_curva', 'FlagChurn']
clf = CriaRandomForest_Churn(result[colunas])


# # 

# # Cria Cluster

# In[60]:

fim_janela_feature = df_base.Instalacao_AnoMes.max()+ relativedelta(months=-6)
janela_churn = 1
df_treino = ArrumaBase_Churn(fim_janela_feature, janela_churn, df_base)

tamanho = len(lista_perfil) 
df_treino['Perfil_indice'] = [marca_lista(lista_perfil, tamanho, variavel)
                              for variavel in df_treino.Perfil]

tamanho = len(lista_classificacao) 
df_treino['classificacao_indice'] = [marca_lista(lista_classificacao, tamanho, variavel)
                                     for variavel in df_treino.classificacao]

tamanho = len(lista_servico) 
df_treino['Servico_indice'] = [marca_lista(lista_servico, tamanho, variavel)
                               for variavel in df_treino.Servico]

df_treino.fillna(0, inplace= True)
df_treino = CriaCluster_Churn(clf, df_treino, colunas)
aux_Cluster = ComparaCluster_DadoCli(df_treino, 'BaseTreino')
Compara_ClusterProduto_QtdProv = Compara_ClusterProduto_QtdProv(df_treino)


# # 

# # Aplicando na Base

# In[61]:

fim_janela_feature = df_base.Instalacao_AnoMes.max()
janela_churn = 10 #sem importancia

df_treino = ArrumaBase_Churn_AplicaAlgoritmo(fim_janela_feature, janela_churn, df_base, aux_Cluster, colunas, clf, lista_perfil, lista_classificacao, lista_servico)
aux_Cluster = ComparaCluster_DadoCli(df_treino, 'BaseAtiva')

NomeCSV = 'ProvsProbChurnCluster'+aux_nome_data(fim_janela_feature)+'.csv'
colunas_interessantes = ['Provisioning', 'Prob_Churn', 'Cluster']
df_treino[colunas_interessantes].to_csv(NomeCSV)    


# # 

# # 

# In[62]:

auxiliar = pd.read_csv('./RelarotioClusterChurnBaseAtiva20170801.csv'
                       , error_bad_lines = False
                       , sep=','
                       , encoding='latin-1')
auxiliar


# In[63]:

auxiliar = pd.read_csv('./RelarotioClusterChurnBaseTreino20170801.csv'
                       , error_bad_lines = False
                       , sep=','
                       , encoding='latin-1')
auxiliar


# In[64]:

auxiliar = pd.read_csv('./RelarotioServicoClusterChurn20170801.csv'
                       , error_bad_lines = False
                       , sep=','
                       , encoding='latin-1')
auxiliar


# In[65]:

auxiliar = pd.read_csv('./RelarotioServicoClusterQtdProvs20170801.csv'
                       , error_bad_lines = False
                       , sep=','
                       , encoding='latin-1')
auxiliar


# In[66]:

auxiliar = pd.read_csv('./ProvsProbChurnCluster20170801.csv'
                       , error_bad_lines = False
                       , sep=','
                       , encoding='latin-1')
auxiliar


# In[ ]:



