# -*- coding: cp1252 -*-
import csv
import numpy as np

def hiperbolicDerivada(A):
    fx = tangenteHiberbolica(A)
    return np.multiply((1 - fx),fx)


def tangenteHiberbolica(A):
    numerador = np.exp(A) - np.exp(-A)
    denominador = np.exp(A) + np.exp(-A)
    result = numerador/ denominador
    return result

def sigmoid(A):
    return 1/(1+np.exp(-A))

def sigmoidDerivada(A):
    fx = sigmoid(A)
    return np.multiply((1 - fx),fx)

def mlp(atrasos):
    # np.seterr(over = 'raise')
    M = np.genfromtxt('fechamentos.csv', delimiter=',')
    tam = 0.7 # proporção do tamanho do conjunto de treinamento na abordagem holdout
    alfa = 0.01 # alfa usado na atualização dos pesos
    x = 10 # numero de neuronios na camada interna
    ns = 3 #número de neurônios na camada de saida

    train_data = M[:tam * len(M), :]
    test_data = M[tam * len(M) + 1:-1, :]

    Yd = train_data[:, [-1]]
    # arrumando as saidas para caso eu queria usar a abordagem com 3 saidas (comentar isso caso contrário)
    saidas = []
    for i in Yd:
        if i == -1:
            saidas.append([1, 0, 0])
        elif i == 0:
            saidas.append([0, 1, 0])
        else:
            saidas.append([0, 0, 1])
    Yd = np.matrix(saidas)

    X = train_data[:, :len(train_data)]
    [N, ne] = np.shape(X)
    uns = np.ones((N, 1))
    X = np.concatenate((uns, X), 1) # colocando o bias

    # modificar: atrasos não são as entradas anteriores, mas sim a saida de ontem
    # (ne + atrasos)
    A = np.random.rand(x, (ne*atrasos) + 1) / 10 # se tiver mais de um atraso, precisa de mais arestas
    B = np.random.rand(ns, x + 1) / 10

    # batelada, não é necessário em padrão a padrão
    '''
    Zin = np.dot(X, np.transpose(A))
    Z = np.concatenate((uns, sigmoid(Zin)), 1)
    Yin = np.dot(Z, np.transpose(B))
    Y = sigmoid(Yin)
    e = Y - Yd

    Et = sum(sum(np.multiply(e, e))) * (1.0 / N)
    erro = Et
    '''
    nepmax = 1000
    epsilon = 0.1
    nep = 0

    results = []

    # Batelada
    '''
    while nep < nepmax:
        nep = nep +1
        #training
        Zin = np.dot(X, np.transpose(A))
        Z = np.concatenate((uns, sigmoid(Zin)), 1)
        Yin = np.dot(Z, np.transpose(B))
        Y = sigmoid(Yin)
        e = Y - Yd

        Et = sum(sum(np.multiply(e, e))) * (1.0 / N)
        #erro = np.asarray(Et)[0][0]
        erro = np.sum(Et) / len(Et) #media dos erros das classes, não sei se é valido
        results.append(erro)
        if (erro < epsilon):
            break

        dJdB = np.dot(np.transpose(np.multiply(e, sigmoidDerivada(Yin))),Z)
        dJdZ = np.dot(np.multiply(e,sigmoidDerivada(Yin)),B[:,1:])
        dJdA = np.dot(np.transpose(np.multiply(dJdZ, sigmoidDerivada(Zin))),X)

        B = B - alfa*dJdB
        A = A - alfa*dJdA
    '''

    # Padrão a Padrão
    while nep < nepmax:
        nep = nep + 1
        # training
        erros = []
        # atualizando padrão a padrão
        for i in range(len(X)):
            if i - 1 < 0:  # fiz para considerar o passado (24 atrás)
                padrao = [np.array([0] * len(X[i]))]
            else:
                padrao = [X[i - 1]]  # padrão do indice i


            # parte quando tem mais de um atraso
            for k in range(2,atrasos+1):
                '''
                if k == 2:
                    indiceAtraso = i - 2  # 48 horas de atraso
                elif k == 3:
                    indiceAtraso = i - 7  # 1 semana de atraso
                elif k == 4:
                    indiceAtraso = i - 30  # 30 dias de atraso
                '''
                indiceAtraso = i - k
                if indiceAtraso <0 :
                    padraoAux = [np.array([0] * (len(X[i])-1))] #-1 porque não preciso colocar o bias de novo
                else:
                    padraoAux = [X[indiceAtraso][1:]] # tirando o bias dos atrasos
                padrao = np.concatenate((padrao, padraoAux), 1)

            yd = Yd[i]  # seu resultado esperado
            Zin = np.dot(padrao, np.transpose(A))
            Z = np.insert(sigmoid(Zin), 0, 1.0) # adicionando o bias ao Z

            Teste = [np.asarray(Z)] # acertando seu valor
            Z = np.array(Teste) # como um numpy array

            Yin = np.dot(Z, np.transpose(B)) #calculando o valor da saída
            Y = sigmoid(Yin) # atualizando esse valor com a função de ativação
            e = Y - yd # calculando o erro desse saída
            erros.append(e)# e adicionando o vetor de erros

            # Calculando o valor das derivadas de B e A para atualização dos pesos
            dJdB = np.dot(np.transpose(np.multiply(e, sigmoidDerivada(Yin))), Z)
            dJdZ = np.dot(np.multiply(e, sigmoidDerivada(Yin)), B[:, 1:])
            dJdA = np.dot(np.transpose(np.multiply(dJdZ, sigmoidDerivada(Zin))), padrao)

            # atualizando os pesos
            B = B - alfa * dJdB
            A = A - alfa * dJdA

        # calculando o MSE da rede neural
        Et = sum(sum(np.multiply(erros, erros))) * (1.0 / len(X))
        erro = np.sum(Et) / len(Et)  # media dos erros das classes, não sei se é valido
        results.append(erro)
        if (erro < epsilon):
            break

    testErrors = []
    for i in range(len(test_data)):
        if i - 1 < 0:  # fiz para considerar o passado
            padrao = [np.array([0] * len(X[i]))]  # esta certo mesmo sendo X[i], só quero o tamanho
        else:
            padrao = [np.insert(test_data[i - 1, :len(test_data)], 0, 1.0)]

        #parte dos atrasos
        for k in range(2, atrasos+1):
            '''
            if k ==2:
                indiceAtraso = i-2 #48 horas de atraso
            elif k == 3:
                indiceAtraso = i- 7 # 1 semana de atraso
            elif k == 4:
                indiceAtraso = i - 30 #30 dias de atraso
            '''
            indiceAtraso = i -k
            if indiceAtraso< 0:
                padraoAux = [np.array([0] * (len(X[i]) - 1))]  # -1 porque não preciso colocar o bias de novo
            else:
                #padraoAux = [np.insert(test_data[indiceAtraso, :len(test_data)], 0, 1.0)]
                padraoAux = [test_data[indiceAtraso, :len(test_data)]]
            padrao = np.concatenate((padrao, padraoAux), 1)


        yd = test_data[i, -1] # pegando o valor de saída esperado
        # caso tenha mais de uma saída
        if yd == -1:
            yd = [1, 0, 0]
        elif yd == 0:
            yd = [0, 1, 0]
        else:
            yd = [0, 0, 1]

        Zin = np.dot(padrao, np.transpose(A))
        Z = np.insert(sigmoid(Zin), 0, 1.0) #calculando a saida e adicionando um valor de bias

        #acertando o formato do Z como numpy array
        Teste = [np.asarray(Z)]
        Z = np.array(Teste)


        Yin = np.dot(Z, np.transpose(B)) #calculando o valor de Yin
        Y = sigmoid(Yin) #aplicando a função de ativação
        e = Y - yd # e o erro da saida
        testErrors.append(e) # adicionando o valor da saída ao vetor de erros de testes

    Et = sum(sum(np.multiply(testErrors, testErrors))) * (1.0 / len(test_data))
    erro = np.sum(Et) / len(Et)  # media dos erros das classes, não sei se é valido
    print "Resultados treinamento " + str(results[0]) + " " + str(results[-1])
    print "Resultado teste: " + str(erro)

if __name__ == "__main__":#método main do arquivo, aqui que o algoritmo é chamado
    mlp(30)

