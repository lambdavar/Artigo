import funcoes as f
import numpy as np

if __name__ == "__main__":
    
    # define o ticker do ativo que vou usar pra treinar o modelo
    ticker = "BTCUSDT"
    # timeframes precisa estar em ordem crescente e começar em 1s ou 1m
    timeframes = ("1m", "5m", "15m", "30m", "1h")#, "2h", "4h", "8h", "1d")
    # quantos períodos vamos olhar pro passado
    lookback = 20
    # numero de quantis pra usar no markov transition field. Precisa tunar.
    quantis = 3
    # pega número de timeframes por minuto ou segundo
    timeframes_padronizado = tuple(f.timeframes_mesma_unidade(timeframes))
    # tamanho das batches de treinamento
    batch_size = 1024#512
    # set seed pros resultados não variarem
    #seed = np.random.randint(99999)
    #seed = 777
    #tf.keras.utils.set_random_seed(seed)
    #np.random.seed(seed)
    # pega tamanho da imagem (input)
    img_shape = (40, 100, 3)#f.image_shape()
    # porcentagem dos dados que vou usar pra treino e teste
    pct_imagens_teste = 0.1
    pct_imagens_validacao = 0.1
    # numero de modelos pro ensemble
    n_modelos=5
    # proporção pra diminuir as imagens no momento da criação 1 é imagem inteira
    img_size = 1.0
    # investimento inicial em dolares
    investimento_inicial = 1000
    # fee em %
    fee = 0.1
    
    resultado_CNN = np.empty((n_modelos, 5))
    for i in range(n_modelos):
        resultado_CNN[i] = f.treina_modelo_CNN(i, img_shape, batch_size)