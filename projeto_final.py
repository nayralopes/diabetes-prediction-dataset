import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
Descrição do Trabalho:
Este estudo analisa a relação entre idade e prevalência de diabetes com base em um conjunto de dados médicos e 
demográficos. Os dados incluem informações sobre idade e status de diabetes dos pacientes (positivo ou negativo). 
O objetivo principal é visualizar a distribuição percentual de diabetes em diferentes faixas etárias, permitindo uma 
melhor compreensão dos grupos de risco. 

Através de gráficos de barras empilhadas, podemos identificar padrões e tendências que podem ser úteis para 
profissionais de saúde e pesquisadores na identificação de fatores de risco e no desenvolvimento de estratégias de 
prevenção e tratamento do diabetes.
"""

arquivo = "diabetes_prediction_dataset.csv"

def carga_dados(arquivo):
    # Importar o arquivo
    dados = pd.read_csv(arquivo, sep=",")
    return dados

dados = carga_dados(arquivo)

print(dados)
print(dados.info())
print(dados.head())
print(dados.tail())
print(dados.describe())

def preparo_dados(dados):
    # converter tipo de coluna para inteiro
    dados['age'] = dados['age'].astype(int)
    return dados

dados = preparo_dados(dados)

print(dados.info())

def visualizacao_histograma(dados):
    """
    Plota um histograma que mostra a frequência das idades
    :param dados:
    :return:
    """
    # Tamanho da figura
    plt.figure(figsize=(10, 5))
    # Plotagem com 30 colunas, na cor azul claro, borda na cor preta com 70% de transparência
    plt.hist(dados['age'], bins=30, color='royalblue', edgecolor='black', alpha=0.7)
    # Rótulos do gráfico
    plt.xlabel("Idade")
    plt.ylabel("Frequência")
    plt.title("Distribuição de Idades")
    # Linha tracejada no grid apenas no eixo 'y' com 60% de transparência
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

visualizacao_histograma(dados)

def visualizacao_faixas_etarias(dados):
    """
    Plota um gráfico de barras por categorias de idade(crianças, adolescentes, adultos e idosos)
    :param dados:
    :return:
    """
    # Definindo categorias
    categorias = {
        "Crianças (0-9)": np.sum((dados['age'] >= 0) & (dados['age'] <= 9)),
        "Adolescentes (10-19)": np.sum((dados['age'] >= 10) & (dados['age'] <= 19)),
        "Adultos (20-59)": np.sum((dados['age'] >= 20) & (dados['age'] <= 59)),
        "Idosos (60+)": np.sum(dados['age'] >= 60)
    }

    # Criando gráfico de barras
    plt.figure(figsize=(8, 5))
    plt.bar(categorias.keys(), categorias.values(), color=['blue', 'green', 'orange', 'red'], alpha=0.7)

    # Adicionando rótulos
    plt.xlabel("Faixa Etária")
    plt.ylabel("Quantidade de Pessoas")
    plt.title("Distribuição das Idades por Faixa Etária")

    # Exibir valores nas barras
    for i, v in enumerate(categorias.values()):
        # Valor a 100 pontos acima da barra, valor para string, centralizado e fonte do texto 12
        plt.text(i, v + 100, str(v), ha='center', fontsize=12)

    # Exibir gráfico
    plt.show()

visualizacao_faixas_etarias(dados)

def visualizacao_barras_empilhadas(dados):
    """
    Plota um gráfico de barras empilhadas por categorias de idade e ocorrências de diabete.
    :param dados:
    :return:
    """
    # Definir categorias etárias
    faixas = {
        "Crianças (0-9)": (dados['age'] <= 9),
        "Adolescentes (10-19)": (dados['age'] >= 10) & (dados['age'] <= 19),
        "Adultos (20-59)": (dados['age'] >= 20) & (dados['age'] <= 59),
        "Idosos (60+)": (dados['age'] >= 60)
    }

    # Contar quantos possuem e não possuem diabetes em cada faixa etária
    categorias = {k: [np.sum(v & (dados['diabetes'] == 0)), np.sum(v & (dados['diabetes'] == 1))] for k, v in faixas.items()}

    # Criar gráfico de barras empilhadas
    labels = list(categorias.keys())
    sem_diabetes = [v[0] for v in categorias.values()]
    com_diabetes = [v[1] for v in categorias.values()]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, sem_diabetes, color='royalblue', label="Sem Diabetes", alpha=0.7)
    plt.bar(labels, com_diabetes, bottom=sem_diabetes, color='red', label="Com Diabetes", alpha=0.7)

    # Adicionando rótulos
    plt.xlabel("Faixa Etária")
    plt.ylabel("Quantidade de Pessoas")
    plt.title("Distribuição das Idades por Faixa Etária e Diabetes")
    plt.legend()

    # Exibir valores nas barras
    for i in range(len(labels)):
        total = sem_diabetes[i] + com_diabetes[i]
        plt.text(i, total + 100, str(total), ha='center', fontsize=12)

    # Exibir gráfico
    plt.show()

visualizacao_barras_empilhadas(dados)

def visualizacao_barras_empilhadas_percentual(dados):
    """
    Plota um gráfico de barras empilhadas por categorias de idade e ocorrências de diabete, em valor percentual.
    :param dados:
    :return:
    """
    # Definir categorias etárias
    faixas = {
        "Crianças (0-9)": (dados['age'] <= 9),
        "Adolescentes (10-19)": (dados['age'] >= 10) & (dados['age'] <= 19),
        "Adultos (20-59)": (dados['age'] >= 20) & (dados['age'] <= 59),
        "Idosos (60+)": (dados['age'] >= 60)
    }

    # Contar quantos possuem e não possuem diabetes em cada faixa etária
    categorias = {k: [np.sum(v & (dados['diabetes'] == 0)), np.sum(v & (dados['diabetes'] == 1))] for k, v in faixas.items()}

    # Converter valores absolutos para percentuais
    totais = [sum(v) for v in categorias.values()]
    sem_diabetes = [v[0] / t * 100 for v, t in zip(categorias.values(), totais)]
    com_diabetes = [v[1] / t * 100 for v, t in zip(categorias.values(), totais)]

    # Criar gráfico de barras empilhadas em percentual
    labels = list(categorias.keys())
    plt.figure(figsize=(8, 5))
    plt.bar(labels, sem_diabetes, color='royalblue', label="Sem Diabetes", alpha=0.7)
    plt.bar(labels, com_diabetes, bottom=sem_diabetes, color='red', label="Com Diabetes", alpha=0.7)

    # Adicionando rótulos
    plt.xlabel("Faixa Etária")
    plt.ylabel("Percentual (%)")
    plt.title("Distribuição Percentual das Idades por Faixa Etária e Diabetes")
    plt.legend()

    # Exibir valores nas barras
    for i in range(len(labels)):
        plt.text(i, sem_diabetes[i] / 2, f"{sem_diabetes[i]:.1f}%", ha='center', fontsize=10, color='white')
        plt.text(i, sem_diabetes[i] + com_diabetes[i] / 2, f"{com_diabetes[i]:.1f}%", ha='center', fontsize=10, color='white')

    # Exibir gráfico
    plt.show()

visualizacao_barras_empilhadas_percentual(dados)