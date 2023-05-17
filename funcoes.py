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
from multiprocessing.pool import ThreadPool
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

def background(f):
    def wrapped(*args, **kwargs):
        #executor = ProcessPoolExecutor(max_workers=3)
        #executor = ThreadPoolExecutor()
        executor = None
        return asyncio.get_event_loop().run_in_executor(executor, f, *args, **kwargs)

    return wrapped

# só extrai o arquivo dado como parâmetro para a pasta ./Dados/temp
def extrai_arquivo(arq):
    try:
        ZipFile(arq, 'r').extractall('./Dados/temp/')
    except:
        print(f"erro ao extrair {arq}")
        
# Adiciona e preenche linhas faltantes nos arquivos com 0
def corrige_arquivos(timeframes):
    # só corrige se houve um item em timeframes
    if timeframes:
        freq = [item.replace("m", "T") for item in timeframes]
        for timeframe, freqs in zip(timeframes, freq):
            df = (pd.read_csv(f"./Dados/Processados/BTCUSDT-{timeframe}.csv", index_col ="Close time"))
            df.index = pd.to_datetime(df.index, unit="ms")
            novo_index= pd.date_range(start = df.index[0], end=df.index[-1], freq=freqs)
            df = df.reindex(novo_index, fill_value=0)
            # faz virar unix de volta
            df.index = (df.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
            df.reset_index(inplace=True)
            df.rename(columns={"index": "Close time"}, inplace=True)
            df.to_csv(f"./Dados/Processados/BTCUSDT-{timeframe}.csv", index=False)
    else:
        print("Nada corrigido")
        
# baixa dados, concatena em um dataframe só e salva em um .csv
# limpa depois
def baixa_e_concatena(ticker, timeframe, ano_inicial):
    ano_corrente, mes_corrente, dia_corrente = [dt.date.today().year, dt.date.today().month, dt.date.today().day]
    
    # baixa dados da binance conforme ticker e timeframe selecionados para a pasta ./Dados/
    # timeframes disponiveis: 12h 15m 1d 1h 1m 1mo 1s 1w 2h 30m 3d 3m 4h 5m 6h 8h
    # tickers disponiveis: https://data.binance.vision/?prefix=data/spot/monthly/klines/
    url = "https://data.binance.vision/data/spot/monthly/klines/"
    if not os.path.exists(f"./Dados/Processados/{ticker}-{timeframe}.csv"):
        for ano in range(ano_inicial, ano_corrente+1):
            for mes in range(1,12+1):
                mes = str(mes).zfill(2)
                if not ((os.path.exists(f"./Dados/temp/{ticker}-{timeframe}-{ano}-{mes}.zip"))):
                    try:
                        wget.download(f"{url}{ticker}/{timeframe}/{ticker}-{timeframe}-{ano}-{mes}.zip"
                                      , out = f"./Dados/temp/")
                        pass
                    except:
                        print(f"\nFalha ao baixar {url}{ticker}/{timeframe}/{ticker}-{timeframe}-{ano}-{mes}.zip")
                else:
                    print(f"{ano}/{mes} já baixado")
    else:
        print(f"{ticker}-{timeframe} já processado")
        return
    
    # cria uma lista de arquivos do ticker e timeframe selecionado
    lista_arquivos = os.listdir("./Dados/temp/")
    lista_arquivos = [x for x in lista_arquivos if x.startswith(f"{ticker}-{timeframe}")]
    lista_arquivos[-5:]
    
    # cria um dataframe vazio pra colocar todos os dados dentro
    nomes = ["Open time","Open","High","Low","Close","Volume","Close time","Quote asset volume"
                                 ,"Number of trades","Taker buy base asset volume","Taker buy quote asset volume","Ignore"]
    df = pd.DataFrame(columns = nomes)
    
    # concatena tudo em um CSV e deixa na pasta ./Dados/Processados/
    for arq in lista_arquivos:
        extrai_arquivo(f"./Dados/temp/{arq}")
        df = pd.concat([df, pd.read_csv(f'./Dados/temp/{arq[:-4]}.csv', sep=',',decimal='.'
                                   , encoding='latin1', names=nomes, header=None)], ignore_index=True, copy=False)
        os.remove(f"./Dados/temp/{arq[:-4]}.csv")
    df.drop("Ignore", inplace=True, axis=1)
    df.set_index("Open time", inplace=True)
    df.to_csv(f"./Dados/Processados/{ticker}-{timeframe}.csv")
    
    print(f"./Dados/Processados/{ticker}-{timeframe}.csv")
    
    # deleta tudo que é temporario e já foi processado
    for arq in lista_arquivos:
        os.remove(f"./Dados/temp/{arq}")
    
    
    return(timeframe)


# cria lista de qtd de minutos ou segundos por timeframe, dependendo de qual for o primeiro timeframe utilizado
# uso isso somente pra pegar a ultima linha que devemos iterar para criação de janelas
def timeframes_mesma_unidade(timeframes):
    lista = []
    for timeframe in timeframes:
        qtd = int(timeframe[:-1])
        unidade = timeframe[-1]
        if timeframes[0] == "1s":
            lista.append(int(pd.to_timedelta(qtd, unit=unidade).total_seconds()))
        elif timeframes[0] == "1m":
            lista.append(int(pd.to_timedelta(qtd, unit=unidade).total_seconds()/60))
        else:
            print("Timeframe inicial não é de 1 minuto ou 1 segundo.")
    return(lista)



# cria janela movel de t-lookback pra cada linha de dados do menor timeframe
def cria_janela(linha, dfs , lookback):
    # cria janela vazia. precisa ter 1 coluna a mais que a qtd de timeframes
    df_janela = np.full(shape= (lookback, len(dfs)+1), fill_value=np.nan, dtype="float64")
    # coloca o preco de fechamento de 1m na primeira coluna e o close time de 1m na ultima coluna
    df_janela[:, 0] = dfs[0][linha:linha+lookback, 0]
    df_janela[:, -1] = dfs[0][linha:linha+lookback, -1]
    
    # define se a janela é long ou short
    # long se o preço de fechamento é maior no minuto seguinte, short se não, none se igual
    long = None
    if dfs[0][linha, 0] < dfs[0][linha-1, 0]:
        long = "long0"
    elif dfs[0][linha, 0] > dfs[0][linha-1, 0]:
        long = "short1"

    i = 1
    while i < len(dfs):
        # pega o numero da linha o primeiro closetime do timeframe é igual ou menor que o da janela atual
        index = (dfs[i][:,1] <= df_janela[0,-1]).argmax()
        # coloca as 20 (lookback) linhas subsequentes dentro da janela
        df_janela[:, i] = dfs[i][index:index+lookback, 0]
        i += 1
    # deleta coluna de close time e retorna
    return(df_janela[:,:-1], long, int(df_janela[0,-1]))


# usando tambem markov
# ler o artigo https://arxiv.org/pdf/1506.00327.pdf
# nesse artigo ele explica o uso de GAF, GDF e MTF ao mesmo tempo em 3 canais
# também usa uma SVM no final da CNN dele
def cria_gaf(dft, quantis, img_size):
    # cria uma imagem gaf RGB
    # precisa da df_janela transposta pra funcionar
    gasf = GramianAngularField(method='summation', image_size=img_size)
    gadf = GramianAngularField(method='difference', image_size=img_size)
    mtf = MarkovTransitionField(n_bins=quantis, image_size=img_size)# , 
    gadf = gadf.transform(dft)
    gasf = gasf.transform(dft)
    mtf = mtf.transform(dft)
    
    # une os 3 metodos em uma unica imagem com 3 canais (RGB) com np.stack
    # depois une as 9(timeframes) matrizes geradas  horizontalmente com np.vstack
    return np.hstack(np.stack((gasf,gadf, mtf), axis=-1))

#@background
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

#@background 
def gera_imagem2(linha, dfs_close, dfs_volume, lookback, quantis, pasta, img_size):
    df_janela, decisao, timestamp = cria_janela(linha, dfs_close, lookback)
    
    # se decisao não é long nem short ou tem um buraco nos dados no meio da janela, pula a iteração
    if decisao == None or (0 in df_janela):
        return
    
    df_janela_volume = cria_janela(linha, dfs_volume, lookback)[0]
    
    # cria gaf images
    img_array = cria_gaf(df_janela.T, quantis, img_size)
    img_array = np.vstack((img_array, cria_gaf(df_janela_volume.T, quantis, img_size)))
    
    # salva como imagem na pasta (treino ou teste)/(long ou short)
    
    save_img(f'./Dados/Imagens/{pasta}/{decisao}/{timestamp}.png', array_to_img(img_array), scale=False)
    # converte pra imagem usando pil
    #from PIL import Image as im
    #img_array = (((img_array+1)/2)*255).astype("uint8")
    #im.fromarray(img_array, mode="RGB").save(f'./Dados/Imagens/{pasta}/{decisao}/{timestamp}.png')
    return

def roda_paralelo(linhas_treino, linhas_teste, dfs_close, dfs_volume, lookback, quantis):
    #p = Pool(2)
    p = ThreadPool()
    lista = []
    
    for linha in linhas_teste:
        lista.append(p.apply_async(gera_imagem2, args=(linha, dfs_close, dfs_volume, lookback, quantis, "teste")))
    
    for linha in linhas_treino:
        lista.append(p.apply_async(gera_imagem2, args=(linha, dfs_close, dfs_volume, lookback, quantis, "treino")))
    
    for p in tqdm(lista):
        p.get()

def roda_async2(linhas_treino, linhas_teste, dfs_close, dfs_volume, lookback, quantis):
    # rodar em paralelo
    loop = asyncio.get_event_loop()                                       # Have a new event loop
    # pra todas as linhas usa a funcao gera_imagem
    looper1 = asyncio.gather(*[gera_imagem2(linha, dfs_close, dfs_volume, lookback, quantis, "teste") for linha in linhas_teste])
    looper2 = asyncio.gather(*[gera_imagem2(linha, dfs_close, dfs_volume, lookback, quantis, "treino") for linha in linhas_treino])
    loopers = asyncio.gather(looper1, looper2)
    # Run the loop
    # espera tudo acabar antes de continuar                  
    results = loop.run_until_complete(loopers)  # Wait until finish

def image_shape():
    # pega as dimensões da imagem, que é a dimensão do input
    if os.path.isfile('./Dados/Imagens/teste/long0/1655745299999.png'):
        img = load_img('./Dados/Imagens/teste/long0/1655745299999.png')
    elif os.path.isfile('./Dados/Imagens/teste/short1/1655745299999.png'):
        img = load_img('./Dados/Imagens/teste/short1/1655745299999.png')
    else:
        print("sem arquivos de imagem para pegar o tamanho")
        return
    img = img_to_array(img)
    return(img.shape)
