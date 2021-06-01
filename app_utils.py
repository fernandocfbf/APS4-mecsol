from cmath import *
from math import *
from numpy import linalg
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from tqdm import tqdm

def distancia_entre_pontos(x1, x2, y1, y2):
    """
    função que calcula a distância entre dois pontos
    recebe: coordenadas de dois pontos [inteiro]
    retorna: distância entre os pontos [inteiro]
    """
    
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def matriz_conectividade(numero_do_membro, incidencia, num_nos):
    """
    função que calcula a matriz de conectividade de um elemento específico
    recebe: número do elemento [inteiro] e a matriz de incidência lida do excel [matriz]
    retorna: conectividade [lista]
    """
    
    conectividade = num_nos*[0]
    
    # O numero do membro-1 é a linha a matriz que eu tenho que utilizar
    no_1 = int(incidencia[numero_do_membro-1, 0])
    no_2 = int(incidencia[numero_do_membro-1, 1])
    
    conectividade[no_1-1] = -1
    conectividade[no_2-1] = 1

    return conectividade

def conec_global_T(incidencia, num_membros, num_nos):
    """
    função responsável por devolver a matriz de conectividade global
    recebe: número de nós [inteiro], matriz de incidencia [matriz]
    retorna: matriz onde cada linha é uma conectividade de um membro [matriz]
    """
  
    C = []
    for i in range(num_membros):
        C.append(matriz_conectividade(i+1, incidencia, num_nos)) #repare que i começa em 0
    return np.array(C).T


def calculate_Se(n_membro, m_incidencia, m_nos, m_membros):
    """ 
    Função responsável por calcular o valor de Se para um dado elemento
    recebe: número do membro [inteiro], matriz de incidência [matriz], matriz dos nós [matriz], matriz dos membros [matriz]
    retorna: valor de Se para o elemento escolhido [matriz/inteiro]
    """
    
    E = m_incidencia[n_membro-1, 2] #elemento linear elástico
    A = m_incidencia[n_membro-1, 3] #área da senção transversal
    
    no_1 = int(m_incidencia[n_membro-1, 0])
    no_2 = int(m_incidencia[n_membro-1, 1])
    
    # Pegar a coordenada do no_1
    x_no1 = m_nos[0, no_1-1]
    y_no1 = m_nos[1, no_1-1]
    
    # Pegar a coordenada do no_2
    x_no2 = m_nos[0, no_2-1]
    y_no2 = m_nos[1, no_2-1]
    
    l = distancia_entre_pontos(x_no2, x_no1, y_no2, y_no1)
    
    cordeenadas_membro = m_membros[:, n_membro-1] #pega as coordenadas do membro
    
    coornadas_matriz = np.array([cordeenadas_membro]) #transforma em matriz
    coornadas_matriz_T = coornadas_matriz.T #calcula a transposta
    
    me = sum(i**2 for i in cordeenadas_membro)

    
    segunda_parte = (np.dot(coornadas_matriz_T, coornadas_matriz))/me
    
    return ((E*A)/l)*segunda_parte

def  calculate_K(n_membro, m_incidencia, m_nos, m_membros, num_nos):
    """
    função responsável por calcular a matriz K para um elemento
    recebe: número do membro [inteiro], matriz incidência, matriz de nós e matriz de membros
    retorna: matriz K para o dado elemento
    """
    
    MC = np.array([matriz_conectividade(n_membro, m_incidencia, num_nos)]).T #vetor (9,1)
    MC_T = MC.T #vetor (1,9)
    Se = calculate_Se(n_membro, m_incidencia, m_nos, m_membros)#(2,2)
    dot = np.dot(MC, MC_T) 

    return np.kron(dot, Se) 

def matriz_global(num_nos, num_membros, m_incidencia, m_nos, m_membros):
    """
    função responsável por realizar a somatória das matrizes K de cada elementos
    recebe: número de membros [inteiro], matriz incidência, matriz de nós e matriz de membros
    retorna: somatória das matrizes K de cada elemento [matriz]
    """
    
    get_shape = calculate_K(1, m_incidencia, m_nos, m_membros, num_nos) 
    
    x = get_shape.shape[0] #linhas
    y = get_shape.shape[1] #colunas
    
    kg = np.zeros((x, y)) #sempre vai ser num_membros*2
    for i in range(num_membros ):
        kg += calculate_K(i, m_incidencia, m_nos, m_membros, num_nos)        
    return kg

def MR_para_solucao(matriz, v_rest):
    """
    função responsável por excluir as colunas/linhas desejadas de uma matriz
    recebe: matriz
    retorna: matriz com as linhas e colunas desejadas apagadas
    """
    
    v_rest = v_rest[:,0].tolist() #transforma v_rest em lista
    v_rest_int = [int(item) for item in v_rest] #cast dos valores para inteiro
    
    matriz_resp = matriz.copy() #copia a matriz recebida
    matriz_resp = np.delete(matriz_resp, v_rest_int, axis=0) #deleta as linhas
    matriz_resp = np.delete(matriz_resp, v_rest_int, axis=1) #deleta as colunas
    
    return matriz_resp

def completa_u(matriz_u, vet_rest):
    #faz a matriz virar lista
    lista=[]
    for e in matriz_u[:,0]:
        lista.append(e)
        
    ## Inserir zeros no u que não tem para assim complentar
    for i in vet_rest[:,0]:
        lista.insert(int(i), 0)
        
    # Transforma em vetor de novo
    matriz=np.array([lista])
    matriz=matriz.T
    
    return matriz

def calcula_deslocamentos(matriz_rigidez, matriz_força):
    L,U = scipy.linalg.lu(matriz_rigidez, permute_l=True)
    y = scipy.linalg.solve(L, matriz_força)
    x = scipy.linalg.solve_triangular(U, y)
    return x

def vetor_global_de_forcas(num_rest, v_rest, v_carregamento):
    """
    para perguntar: para Pg considera todos os nós com força, ou so os
    que tem forças de apoio e que tem F ext?
    """
    
    matriz_resp = v_carregamento.copy() #copia a matriz carregamento
    
    v_rest = v_rest[:,0].tolist()
    v_rest_int = [int(item) for item in v_rest]
    
    matriz_resp = np.delete(matriz_resp, v_rest_int, axis=0) #deleta a linha do índice
        
    return matriz_resp

def calculate_force(matriz_k, u, linha_number):
    """
    função responsável por calcular a força para um dado elemento
    recebe: matriz K do elemento, matriz u (completo), número da linha do elemento
    """
    
    linha = matriz_k[linha_number,:] #pega a linha desejada na matriz
    return np.dot(linha, u) #retorna multiplicação de linha X coluna

def calculate_force_complete(KG, u_completo, vet_rest):
    lista_forcas = []
    for i in vet_rest[:,0]:
        lista_forcas.append(calculate_force(KG, u_completo, int(i)))
        
    return lista_forcas

def tensao_e_deformacao(n_elemento, n_de_membros, matriz_u, m_incidencia, m_nos, A):
    """
    função responsável por calcular a tensão e deformação para cada membro
    recebe: número do membro desejado [inteiro], número total de membros [inteiro], matriz u calculada (completa), matriz de incidencia, matriz de nós.
    retorna: tensão [inteiro] e deformação [inteiro] calculadas para o membro desejado
    """
   
    no_1 = int(m_incidencia[n_elemento-1, 0])
    no_2 = int(m_incidencia[n_elemento-1, 1])     
    
    matriz_aux = np.array((
            matriz_u[(no_1-1)*2],       #deslocamento em x (nó 1)
            matriz_u[(no_1-1)*2 +1],    #deslocamento em y (nó 1)
            matriz_u[(no_2-1)*2],       #deslocamento em x (nó 2)
            matriz_u[(no_2-1)*2 +1]))   #deslocamento em y (nó 2)
    
    
    E =  m_incidencia[n_elemento-1, 2]  
    
    # Pegar a coordenada do no_1
    x_no1 = m_nos[0, no_1-1]
    y_no1 = m_nos[1, no_1-1]
    
    # Pegar a coordenada do no_2
    x_no2 = m_nos[0, no_2-1]
    y_no2 = m_nos[1, no_2-1]
    
    #calcula distância entre os nós
    l = sqrt((x_no2-x_no1)**2+(y_no2-y_no1)**2) 
    
    sen = (y_no2-y_no1)/l #calcula seno do elemento
    cos = (x_no2-x_no1)/l #calcula coss do elemento
    
    c = np.array(([-cos, -sen, cos, sen]))
    
    tensao = (E/l) * np.dot(c, matriz_aux)
    forca = (E/l) * np.dot(c, matriz_aux)*A
    deformacao = (1/l) * np.dot(c, matriz_aux)
    
    return tensao[0], deformacao[0], forca[0]

def compara_solucoes(array1, array2):
    return max( abs( (s2 - s1)/s2) for s1,s2 in zip(array1, array2))

def solucao_gauss(k, F, ite, tol=1e-3):
    """
    função responsável por calcular a solução de Gauss para um sistema de equações
    recebe: matriz k, matriz de forças, número de iterações [inteiro], tolerância [float]
    retorna: solução do sistema de equações através da teoria de Gauss [matriz]
    """
    
    matriz_x = np.zeros((F.shape[0], 1)) #cria uma matriz nx1
    matriz_compare = matriz_x.copy() #salva os valores da iteração anterior
    
    for iteracao in tqdm(range(ite)):
        for indice in range(matriz_x.shape[0]):
            b = F[indice]
            ax = sum(a*x for a,x in zip(k[indice, :], matriz_x[:,0])) - k[indice, indice]*matriz_x[indice,0]            
            matriz_x[indice] = (b - ax)/k[indice, indice]
        
        if iteracao > 1:
            erro = compara_solucoes(matriz_compare[:,0], matriz_x[:,0])
        
        if iteracao > 1 and erro < tol:
            print("Convergiu na {0}º iteracao".format(iteracao))
            break
        else:
            matriz_compare = matriz_x.copy() #atualiza valores 
        
    return matriz_x

def solucao_jacobi(k, F, ite, tol):
    """
    função responsável por calcular a solução de Gauss para um sistema de equações
    recebe: matriz k, matriz de forças, número de iterações [inteiro], tolerância [float]
    retorna: solução do sistema de equações através da teoria de Gauss [matriz]
    """
    
    matriz_x = np.zeros((F.shape[0], 1)) #cria uma matriz nx1
    matriz_x_auxiliar = np.zeros((F.shape[0], 1))
    
    for iteracao in range(ite):
        for indice in range(matriz_x.shape[0]):
            b = F[indice]
            ax = sum(a*x for a,x in zip(k[indice, :], matriz_x[:,0])) - k[indice, indice]*matriz_x[indice,0]            
            matriz_x_auxiliar[indice] = (b - ax)/k[indice, indice]
            
        matriz_x = matriz_x_auxiliar
        
        #print(matriz_x)
        #print("--------------") #Com esse print é possivel perceber que a de Gauss converte antes
            
    return matriz_x