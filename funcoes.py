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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam
from keras.optimizers import Adagrad
from keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

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

def pega_files_labels(path):
    list_subfolders = [f.name for f in os.scandir(path) if f.is_dir()]
    list_subfolders

    df = pd.DataFrame()
    
    i=0
    for subfolder in list_subfolders:
        path2 = path + str(subfolder)
        # pega arquivos .npz
        arquivos = [item.path for item in os.scandir(path2) if item.name.endswith(".tfrecords")]
        # poe num df temporario que iremos juntar com o principal
        df_temp = pd.DataFrame(arquivos, columns=["arquivos"])
        df_temp["labels"] = i
        # junta com o principal
        df = pd.concat([df, df_temp], axis=0, ignore_index=True)
        i += 1
    # retorna lista de arquivos e labels
    return df["arquivos"].to_numpy(), df["labels"].to_numpy(dtype=bool)

def decode_fn(tfrecord, img_shape=(40, 100, 3), label=True):
    '''
    Decodifica arquivos .tfrecords para que seja acessado como um dicionário pelo tensorflow
    
    Args:
    tfrecord: Arquivo tfrecord lido com a função tf.data.TFRecordDataset
    img_shape: shape de numpy array
    '''
    data =  tf.io.parse_single_example(
      # Data
      tfrecord,

      # Schema
      {
        #'timestamp': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature(img_shape, tf.float32)},
    )
    
    if label==True:
        return data["image"], data["label"]
    elif label==False:
        return data["image"]
    else:
        print("valor de label inválido. Precisa ser true ou false")
        return

def cria_batches(batch_size, img_shape, valid_data=False, test_data=False, train_data=False):
    # se for treino precisa embaralhar antes
    if train_data == True:
        print("Criando dados de treino")
        pasta = "treino"
        path = f"./Dados/TFRecords/{pasta}/"
        # pega nomes e labels
        filenames, labels = pega_files_labels(path)
        # embaralha
        np.random.shuffle(filenames)
        # lê arquivos
        data = tf.data.TFRecordDataset(
            filenames, compression_type="ZLIB", num_parallel_reads=tf.data.AUTOTUNE).map(
            lambda x: decode_fn(x, img_shape, label=True), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return data
    # se for validacao não embaralha
    elif valid_data == True:
        print("Criando dados de validacao")
        pasta = "validacao"
        path = f"./Dados/TFRecords/{pasta}/"
        # pega nomes e labels
        filenames, labels = pega_files_labels(path)
        filenames = tf.data.Dataset.from_tensor_slices(filenames)
        # le a array
        data = tf.data.TFRecordDataset(
            filenames, compression_type="ZLIB", num_parallel_reads=tf.data.AUTOTUNE).map(
            lambda x: decode_fn(x, img_shape, label=True), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return data
    # Se for teste não tem labels
    elif test_data == True:
        print("Criando dados de teste")
        pasta = "teste"
        path = f"./Dados/TFRecords/{pasta}/"
        # pega nomes e labels
        filenames, labels = pega_files_labels(path)
        filenames = tf.data.Dataset.from_tensor_slices(filenames)
        # le a array
        data = tf.data.TFRecordDataset(
            filenames, compression_type="ZLIB").map(
            lambda x: decode_fn(x, img_shape, label=True)).batch(batch_size)
            #pega_imagem).batch(batch_size)
        # pega batch
        #data_batch = data.batch(batch_size)
        return data
    else:
        print("Chame a função corretamente")
        return
    
# rede CNN
def CNN(img_shape):
    model = keras.Sequential()
    model.add(layers.Input(shape=img_shape))
    
    # reescala os dados pra entre 0 e 1
    #model.add(layers.Rescaling(1./255))
    
    # convolucao
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(1, 2)))
    model.add(layers.Dropout(0.15))
    
    # classificação
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(8, activation="linear"))
    model.add(layers.Dense(1, activation="sigmoid"))
    
    opt = Adagrad(learning_rate=0.1)
    model.compile(loss = "binary_crossentropy", optimizer=opt, metrics=["accuracy",
                                                                     keras.metrics.Precision(),
                                                                     keras.metrics.Recall(), 
                                                                     keras.metrics.AUC()])

    return model


def treina_modelo_CNN(num_modelo, img_shape, batch_size):
    model = CNN(img_shape)
    # precisa definir o checkpoint antes de começar cada CNN. Se não acaba usando um do outro
    checkpoint = ModelCheckpoint(f"./modelos/upgrade/modelo CNN {num_modelo}", monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
    # para de treinar se a acuracia de treino parar de aumentar por 3 epochs
    es = EarlyStopping(monitor='val_accuracy', patience=4, mode="max")
    tb = TensorBoard(f"./modelos/upgrade/logs")
    callbacks_list = [checkpoint, es, tb]
    # treina    
    model.fit(cria_batches(batch_size, img_shape, train_data=True),  epochs=10, use_multiprocessing=True, callbacks=callbacks_list, validation_data=cria_batches(batch_size, img_shape, valid_data=True)) #verbose=0
    model = keras.models.load_model(f"./modelos/upgrade/modelo CNN {num_modelo}")
    return model.evaluate(cria_batches(batch_size, img_shape, valid_data=True))
