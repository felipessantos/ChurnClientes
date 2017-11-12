import pandas as pd
import numpy as np

# coding: utf-8

# In[ ]:

def conserta_colunas(um_df):
    um_df.columns = [s.replace(' ','_') for s in um_df.columns]
    um_df.columns = [s.replace('.','') for s in um_df.columns]
    um_df.columns = [s.replace('\xc2\xba','') for s in um_df.columns]
    um_df.columns = [s.replace('\xc3\xa7\xc3\xa3','ca') for s in um_df.columns]
    um_df.columns = [s.replace('\xc3\xa7','c') for s in um_df.columns]
    um_df.columns = [s.replace('\xc3\xa3','a') for s in um_df.columns]
    um_df.columns = [s.replace('\xc3\x9a','u') for s in um_df.columns]
    um_df.columns = [s.replace('\xc3\xaa','e') for s in um_df.columns]
    
def remover_acentos_manual(um_s):
    um_s = um_s.replace('\xc3\xa7\xc3\xa3','ca')
    um_s = um_s.replace('\xc3\xa7','c')
    um_s = um_s.replace('\xc3\xa3','a')
    um_s = um_s.replace('\xe3', 'a')
    return um_s
#     um_s = um_s.replace()


def converte_datetime(um_df, cols):
    for c in cols:
        um_df[c] = pd.to_datetime(um_df[c], errors='coerce')    

# Retorna o elemento mais comum de um pandas Series
def mais_comum(l):
    return l.value_counts().idxmax()

# Retorna uma tupla com todos os elementos do array
def retorna_tuple(um_arr):
    um_arr.sort_values(inplace=True)
    return tuple(um_arr)

# Retirando as hierarquias das colunas
def retira_hierarquias(um_df):
    um_df.columns = ['_'.join(col).strip() for col in um_df.columns.values]

def visao_geral(um_df):
    print ('shape do dataframe:', um_df.shape)
#     print um_df.head()
    print 
    print('=' * 38)
    contagem = um_df.count()
    for c in um_df.columns:
        print
        print (c)
        val_uniq = len(um_df[c].unique())
        print (' ',val_uniq, 'valores unicos')
        if um_df[c].dtype == '<M8[ns]':
            print ('  datetime:', um_df[c].min(), 'ate', um_df[c].max())
#             print (um_df[c].value_counts().sort_index()[0])
        if val_uniq < 9:
            vc = pd.concat([um_df[c].value_counts(), 100 * um_df[c].value_counts(True)], axis = 1)
            vc.columns = ['qtd', '%']
            vc['bar'] = ['*' * int(perc/2) for perc in vc['%'] ]
            vc['%'] = ['{:.1f}'.format(perc) for perc in vc['%']]
            print (vc)
        qty_nan = len(um_df) - contagem[c]
        print ('  Prop NaN:', qty_nan, '/', len(um_df), '(', (100 * qty_nan)/len(um_df), '% )')
        print
        print('-'*58)


# In[ ]:

from unicodedata import normalize
from unidecode import unidecode
import re

def remover_acentos_e_preprocessar(txt, codif='utf-8', remove_acentos = False, minuscula=False, 
                                   remove_hifen=False, remove_abre_chaves=False, 
                                   substitui_fecha_chaves=False):
    txt_retorno = txt
    if remove_acentos:
#         txt_retorno = normalize('NFKD', txt_retorno.decode(codif)).encode('ASCII','ignore')
        txt_retorno = normalize('NFKD', txt_retorno).encode('ASCII','ignore')
    if minuscula:
        txt_retorno = txt_retorno.lower()
    if remove_hifen:
        txt_retorno = txt_retorno.replace('-','')
    if remove_abre_chaves: 
        txt_retorno = txt_retorno.replace('[','')
    if substitui_fecha_chaves:
        txt_retorno = txt_retorno.replace(']',' ')
    return txt_retorno
    
def remover_acentos_passar_minusculas(txt, codif='utf-8'):
    return normalize('NFKD', txt.decode(codif)).encode('ASCII','ignore').lower()

# def remover_acentos_latin(txt, codif='latin-1'):
#     return normalize('NFKD', txt.decode(codif)).encode('ASCII','ignore')

def sep_tokens_trans_minusculas(txt):
    return txt.lower().split('_')

def trans_minusculas_repoe_sep(txt):
    return re.sub(r"[_-]", " " , txt.lower())

