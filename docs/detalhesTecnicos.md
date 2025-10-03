# Detalhes Técnicos do Projeto

Link para o dataset do kaggle e desafio: https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data

## Metodologia no Código do Algoritmo

Além de retirar algumas características da análise, foi também normalizado todos os dados para manter um desvio padrão igual à 1 e a média em 0 para evitar com que outliers afetassem muito os pesos do algoritmo. Além disso, 20% do dataset foi utilizado para treino enquanto os 80% restante foi utilizado para o teste, em torno de 130 casos eram usados então para treino, um número relativamente baixo.

## Análise dos Dados

Os diagnósticos, à medida que são feitos, são comparados com o resultado real, contabilizados como certos ou incorretos e armazenados em um arquivo .csv, além disso as IDs são pegas em caso de erro para uma análise de porque houve a falha.

## Versão do Código Disponibilizada

A versão disponível não contém 100% das implementações, apenas as que acho serem realmente relevantes para visualização da execução do projeto e entendimento do seu funcionamento.

## Como Executar o Algoritmo

O algoritmo é facilmente executado, basta ter o dataset bruto para execução do código em Python para obter gráficos e informações sobre os dados, como estão agrupados e suas características além do dataset revisado. Ou se desejar, bastar baixar o arquivo .csv disponibilizado e executar o algoritmo em C.
