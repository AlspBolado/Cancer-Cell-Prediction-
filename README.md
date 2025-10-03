# Breast-Cancer-Cell-Prediction

## Sobre o Projeto

Este projeto foi desenvolvido como solução para um desafio do Kaggle, cujo objetivo é criar um algoritmo capaz de discernir entre células cancerígenas malignas e benignas de mama a partir de uma base de dados limitada.

A metodologia foi dividida em duas etapas principais:

    -Tratamento e Análise dos Dados: Realizado em Python para explorar e preparar um novo dataset otimizado.
    -Implementação do Algoritmo: O modelo de classificação foi desenvolvido em C com foco em eficiência.

O objetivo final era atingir uma acurácia de 90%, precisão de 90% e, crucialmente, um recall de 95%, minimizando a ocorrência de falsos negativos, pior caso possível para o diagnóstico de câncer.

## Tecnologias Utilizadas

    -Linguagem do Algoritmo: C;
    -Análise e Tratamento de Dados: Python;
    -Modelo: Regressão Logística.

## Metodologia

Para a análise dos dados, foi utilizado Python devido à sua agilidade para manipular e visualizar os dados do arquivo .csv. A abordagem inicial envolveu a criação de um dataset reduzido drasticamente, contendo metade das features originais, para comparar a performance com o modelo treinado no dataset completo. Assim, daria uma noção de maneira rápida se usar um dataset reduzido e sem redundâncias seria benéfica.

O modelo de classificação foi a Regressão Logística Binária, escolhida por sua simplicidade, interpretabilidade e forte aplicabilidade em problemas de classificação binária (Positivo vs. Negativo). A implementação em C foi uma decisão para buscar maior performance e controle sobre a manipulação do que em Python, por exemplo.

Ambos algoritmos foram executados com seeds aleatórias e testados 1000 vezes, para, além de garantir uma confiabilidade maior às métricas de precisão, recall e acurácia, também ser possível averiguar quais células eram as diagnosticadas incorretamente com maior frequência.

## Resultados

O modelo treinado com base no dataset inteiro teve a seguinte performance: 
