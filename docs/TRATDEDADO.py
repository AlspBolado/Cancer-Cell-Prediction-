#Modulos Importados
import pandas as pd
import numpy as np
import seaborn as sea
from seaborn import FacetGrid
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from imblearn.over_sampling import SMOTE


#Leitura do Arquivo
df = pd.read_csv(r"D:\Comp2\PrjComp2\Breast-cancer.csv")

df.head()
df.info()
df.describe()


#Gráfico do Raio Médio VS Diagnóstico
sea.histplot(data=df, x='radius_mean', hue='diagnosis', kde=True, palette='tab10', bins=30)
plt.title("Distribuição do Raio Medio por Diagnostico")
plt.show()

#Aqui selecionei algumas características para analisar melhor (1o a média, 2o o pior)
caracteristicas_selecionadas = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
sea.pairplot(df[caracteristicas_selecionadas + ['diagnosis']], hue='diagnosis', palette='Set1')
plt.show()

caracteristicas_selecionadas = ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst']
sea.pairplot(df[caracteristicas_selecionadas + ['diagnosis']], hue='diagnosis', palette='Set1')
plt.show()

#Como suspeitava que a área era um fator importante, fiz outro gráfico para comprovar
sea.violinplot(x='diagnosis', y='area_mean',data=df,hue='diagnosis', palette='muted')
plt.title("Gráfico violino da área pelo diagnostico")
plt.show()

sea.scatterplot(data=df, x='radius_mean', y='area_mean', hue='diagnosis', palette='Set1')
plt.title('Gráfico Raio Médio vs Área Média')
plt.show()

sea.jointplot(data=df, x='radius_mean', y='perimeter_mean', hue='diagnosis', kind='scatter', palette='coolwarm')
plt.show()

#Aqui fiz uma matriz Heatmap para demonstrar visualmente as correlações
numerical_df = df.select_dtypes(include=[np.number])
corr = numerical_df.corr()
plt.figure(figsize=(20, 15))
sea.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap de Correlação')
plt.show()

#Aqui transformei a matriz em uma tabela mais fácil de ler (como era poucos objetos, utilizei o quicksort do maior para o menor)
df_numeric = df.drop(columns=['diagnosis'])
corr_matrix = df_numeric.corr()
corr_pairs = corr_matrix.unstack()
sorted_corr_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
sorted_corr_pairs = sorted_corr_pairs[sorted_corr_pairs != 1]
sorted_corr_pairs = sorted_corr_pairs.drop_duplicates()
print(sorted_corr_pairs)

#Como perímetro médio e pior e raio médio e pior estão quase perfeitamente correlacionados, além do perímetro, área e raio são extremamente correlacionadas
#dropei 2 para evitar redundância durante o treinamento da IA
df.drop(['perimeter_mean','area_mean'],axis=1,inplace=True)
df.shape

df.drop(['perimeter_worst','area_worst'],axis=1,inplace=True)
df.shape

df.drop(['perimeter_se','area_se'],axis=1,inplace=True)
df.shape

#Aqui refiz a matriz Heatmap para demonstrar visualmente as correlações
numerical_df = df.select_dtypes(include=[np.number])
corr = numerical_df.corr()
plt.figure(figsize=(20, 15))
sea.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap de Correlação')
plt.show()

#Aqui transformei a matriz em uma tabela mais fácil de ler (como era poucos objetos, utilizei o quicksort do maior para o menor)
df_numeric = df.drop(columns=['diagnosis'])
corr_matrix = df_numeric.corr()
corr_pairs = corr_matrix.unstack()
sorted_corr_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
sorted_corr_pairs = sorted_corr_pairs[sorted_corr_pairs != 1]
sorted_corr_pairs = sorted_corr_pairs.drop_duplicates()
print(sorted_corr_pairs)

#Aqui fiz os gráficos entre os valores médios e o pior para visualizar caso seria melhor dropar ou manter as variáveis
sea.set(style="whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

features = [('radius_mean', 'radius_worst'), ('texture_mean', 'texture_worst'), ('smoothness_mean', 'smoothness_worst'),
            ('compactness_mean', 'compactness_worst'), ('concavity_mean', 'concavity_worst'),
            ('concave points_mean', 'concave points_worst'),
            ('symmetry_mean', 'symmetry_worst')]

colors = ['red', 'blue', 'green']

for i, (x_feature, y_feature) in enumerate(features):
    row, col = divmod(i, 3)
    sea.scatterplot(
        x=x_feature,
        y=y_feature,
        data=df,
        ax=axes[row, col],
        color=colors[i % len(colors)]
    )
    axes[row, col].set_title(f' {x_feature} vs {y_feature}')
    axes[row, col].set_xlabel(x_feature)
    axes[row, col].set_ylabel(y_feature)

if len(features) < 9:
    for i in range(len(features), 9):
        row, col = divmod(i, 3)
        axes[row, col].axis('off')
plt.tight_layout()
plt.show()

#Aqui removi parâmetros muito correlacionados para evitar redundância na IA
df.drop(['radius_worst','concave points_mean','texture_worst','smoothness_worst','concavity_worst',
               'compactness_mean'],axis=1,inplace=True)
df.shape

df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
print(df['diagnosis'].value_counts())

sea.countplot(x='diagnosis', data=df, palette='Set2')
plt.title('Diagnosis Distribution')
plt.xlabel('Diagnosis (0 = Malignant, 1 = Benign)')
plt.ylabel('Count')
plt.show()

x = df.drop(columns=['diagnosis'])
y = df['diagnosis']

resampler = SMOTE()
x_resampled, y_resampled = resampler.fit_resample(x, y)

counts = y_resampled.value_counts()
plt.bar(counts.index, counts.values)
plt.title('Distribuição do Diagnóstico')
plt.xlabel('Diagnóstico')
plt.ylabel('Contagem')
plt.show()

df_resampled = pd.concat([pd.DataFrame(x_resampled), pd.DataFrame(y_resampled, columns=['diagnosis'])], axis=1)

df_resampled.to_csv(r"D:\Comp2\PrjComp2\Breast-cancer-resampled.csv", index=False)