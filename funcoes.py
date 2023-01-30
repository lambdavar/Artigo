import datetime as dt
import wget
import os
import pandas as pd
import numpy as np
from zipfile import ZipFile
from timeit import default_timer as timer
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
from keras.utils import load_img
from keras.utils import save_img
from keras.utils import img_to_array
from keras.utils import array_to_img
from multiprocessing.pool import Pool
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor

def background(f):
    def wrapped(*args, **kwargs):
        #executor = ProcessPoolExecutor()
        executor = None
        return asyncio.get_event_loop().run_in_executor(executor, f, *args, **kwargs)

    return wrapped

# cria janela movel de t-lookback pra cada linha de dados do menor timeframe
def cria_janela(linha, dfs , lookback):
    # cria janela vazia. precisa ter 1 coluna a mais que a qtd de timeframes
    df_janela = np.full(shape= (lookback, len(dfs)+1), fill_value=np.nan, dtype="float32")
    # coloca o preco de fechamento de 1m na primeira coluna e o close time de 1m na ultima coluna
    df_janela[:, 0] = dfs[0][linha:linha+lookback, 0]
    df_janela[:, -1] = dfs[0][linha:linha+lookback, -1]
    
    # define se a janela é long ou short
    # long se o preço de fechamento é maior no minuto seguinte, short se não, none se igual
    long = None
    if dfs[0][linha, 0] < dfs[0][linha-1, 0]:
        long = "long"
    elif dfs[0][linha, 0] > dfs[0][linha-1, 0]:
        long = "short"

    i = 1
    while i < len(dfs):
        # pega o numero da linha o primeiro closetime do timeframe é igual ou menor que o da janela atual
        index = (dfs[i][:,1] <= df_janela[0,-1]).argmax()
        # coloca as 20 (lookback) linhas subsequentes dentro da janela
        df_janela[:, i] = dfs[i][index:index+lookback, 0]
        i += 1
    # deleta coluna de close time e retorna
    return(df_janela[:,:-1], long)

# usando tambem markov
# ler o artigo https://arxiv.org/pdf/1506.00327.pdf
# nesse artigo ele explica o uso de GAF, GDF e MTF ao mesmo tempo em 3 canais
# também usa uma SVM no final da CNN dele
def cria_gaf(dft, quantis):
    # cria uma imagem gaf RGB
    # precisa da df_janela transposta pra funcionar
    gasf = GramianAngularField(method='summation')#, image_size=0.5)
    gadf = GramianAngularField(method='difference')#, image_size=0.5)
    mtf = MarkovTransitionField(n_bins=quantis)#, image_size=0.5)# , 
    gadf = gadf.transform(dft)
    gasf = gasf.transform(dft)
    mtf = mtf.transform(dft)
    
    # une os 3 metodos em uma unica imagem com 3 canais (RGB) com np.stack
    # depois une as 9(timeframes) matrizes geradas  horizontalmente com np.vstack
    return np.hstack(np.stack((gasf,gadf, mtf), axis=-1))

@background
def gera_imagem(linha, dfs, lookback, quantis):
    df_janela, decisao = cria_janela(linha, dfs, lookback)
    # se decisao não é long nem short ou tem um buraco nos dados no meio da janela, pula a iteração
    if decisao == None or (0 in df_janela):
        return
    # cria gaf image
    img_array = cria_gaf(df_janela.T, quantis)
    # salva como imagem na pasta long ou short
    save_img(f'./Dados/Imagens/{decisao}/{linha}.png', array_to_img(img_array), scale=False)
    return

def roda_async(linhas, dfs, lookback, quantis):
    # rodar em paralelo
    loop = asyncio.get_event_loop()                                       # Have a new event loop
    # pra todas as linhas usa a funcao gera_imagem
    looper = asyncio.gather(*[gera_imagem(linha, dfs, lookback, quantis) for linha in range(linhas)])         # Run the loop
    # espera tudo acabar antes de continuar                  
    results = loop.run_until_complete(looper)  # Wait until finish
    
def gera_imagem2(linha, dfs, lookback, quantis):
    df_janela, decisao = cria_janela(linha, dfs, lookback)
    # se decisao não é long nem short ou tem um buraco nos dados no meio da janela, pula a iteração
    if decisao == None or (0 in df_janela):
        return
    # cria gaf image
    img_array = cria_gaf(df_janela.T, quantis)
    # salva como imagem na pasta long ou short
    save_img(f'./Dados/Imagens/{decisao}/{linha}.png', array_to_img(img_array), scale=False)
    return

def roda_paralelo(linhas, dfs, lookback, quantis):
    p = Pool(os.cpu_count())
    lista = []

    for linha in range(linhas):
        lista.append(p.apply_async(gera_imagem2, args=(linha, dfs, lookback, quantis)))

    output = [p.get() for p in lista]
