#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>


#define MAX_LINE_LENGTH 1024
#define MAX_FEATURES 18


void lerCSV(const char* nomeArquivo, double** dados, int* linhas, int* colunas) {
    FILE* arquivo;
    errno_t err = fopen_s(&arquivo, nomeArquivo, "r");
    if (err != 0) {
        perror("Erro ao abrir o arquivo");
        exit(EXIT_FAILURE);
    }

    char linha[MAX_LINE_LENGTH];
    int linhaAtual = 0;
    char* contexto;

    while (fgets(linha, MAX_LINE_LENGTH, arquivo)) {
        if (linhaAtual == 0) {
            char* token = strtok_s(linha, ",", &contexto);
            while (token) {
                (*colunas)++;
                token = strtok_s(NULL, ",", &contexto);
            }
        }
        else {
            dados[linhaAtual - 1] = (double*)malloc((*colunas) * sizeof(double));
            char* token = strtok_s(linha, ",", &contexto);
            int colunaAtual = 0;
            while (token) {
                dados[linhaAtual - 1][colunaAtual] = atof(token);
                token = strtok_s(NULL, ",", &contexto);
                colunaAtual++;
            }
        }
        linhaAtual++;
    }

    *linhas = linhaAtual - 1;
    fclose(arquivo);
}

void normalizarDados(double** dados, int linhas, int colunas) {
    for (int j = 0; j < colunas - 1; j++) {
        double media = 0.0, desvioPadrao = 0.0;
        for (int i = 0; i < linhas; i++) {
            media += dados[i][j];
        }
        media /= linhas;
        for (int i = 0; i < linhas; i++) {
            desvioPadrao += pow(dados[i][j] - media, 2);
        }
        desvioPadrao = sqrt(desvioPadrao / linhas);
        for (int i = 0; i < linhas; i++) {
            dados[i][j] = (dados[i][j] - media) / desvioPadrao;
        }
    }
}

void dividirDados(double** dados, int linhas, int colunas, double** treino, double** teste, int* linhasTreino, int* linhasTeste, double proporcaoTreino) {
    int treinoCount = (int)(linhas * proporcaoTreino);
    int testeCount = linhas - treinoCount;
    *linhasTreino = treinoCount;
    *linhasTeste = testeCount;

    srand(time(NULL));
    for (int i = 0; i < linhas; i++) {
        int j = rand() % linhas;
        double* temp = dados[i];
        dados[i] = dados[j];
        dados[j] = temp;
    }

    for (int i = 0; i < treinoCount; i++) {
        treino[i] = dados[i];
    }
    for (int i = 0; i < testeCount; i++) {
        teste[i] = dados[treinoCount + i];
    }
}

double sigmoide(double z) {
    return 1.0 / (1.0 + exp(-z));
}

void treinarModelo(double** dados, int linhas, int colunas, double* pesos, double taxaAprendizado, int iteracoes) {
    for (int iter = 0; iter < iteracoes; iter++) {
        for (int i = 0; i < linhas; i++) {
            double z = 0.0;
            for (int j = 0; j < colunas - 1; j++) {
                z += pesos[j] * dados[i][j];
            }
            double h = sigmoide(z);
            double erro = dados[i][colunas - 1] - h;
            for (int j = 0; j < colunas - 1; j++) {
                pesos[j] += taxaAprendizado * erro * dados[i][j];
            }
        }
    }
}

int prever(double* entrada, double* pesos, int colunas) {
    double z = 0.0;
    for (int j = 0; j < colunas - 1; j++) {
        z += pesos[j] * entrada[j];
    }
    return sigmoide(z) >= 0.5 ? 1 : 0;
}

double calcularPrecisao(double** dados, int linhas, int colunas, double* pesos) {
    int acertos = 0;
    for (int i = 0; i < linhas; i++) {
        int previsao = prever(dados[i], pesos, colunas);
        if (previsao == (int)dados[i][colunas - 1]) {
            acertos++;
        }
    }
    return (double)acertos / linhas;
}

void calcularMatrizConfusao(double** dados, int linhas, int colunas, double* pesos, int* TP, int* TN, int* FP, int* FN) {
    *TP = *TN = *FP = *FN = 0;

    for (int i = 0; i < linhas; i++) {
        int previsao = prever(dados[i], pesos, colunas);
        int verdadeiro = (int)dados[i][colunas - 1];

        if (previsao == 1 && verdadeiro == 1) (*TP)++;
        else if (previsao == 0 && verdadeiro == 0) (*TN)++;
        else if (previsao == 1 && verdadeiro == 0) (*FP)++;
        else if (previsao == 0 && verdadeiro == 1) (*FN)++;
    }
}

void exibirDiagnosticos(double** dados, int linhas, int colunas, double* pesos) {
    printf("Caso\tPrevisto\tReal\tStatus\n");
    for (int i = 0; i < linhas; i++) {
        int previsao = prever(dados[i], pesos, colunas);
        int real = (int)dados[i][colunas - 1];
        const char* status = (previsao == real) ? "Correto" : "Incorreto";

        printf("%d\t%d\t\t%d\t%s\n", i + 1, previsao, real, status);
    }
}

int main() {
    const char* nomeArquivo = "Breast-cancer1.csv";
    int linhas = 0, colunas = 0;
    double** dados = (double**)malloc(2000 * sizeof(double*));
    if (!dados) {
        perror("Erro ao alocar memória para dados");
        exit(EXIT_FAILURE);
    }

    lerCSV(nomeArquivo, dados, &linhas, &colunas);

    normalizarDados(dados, linhas, colunas);

    int linhasTreino = 0, linhasTeste = 0;
    double** treino = (double**)malloc(linhas * sizeof(double*));
    double** teste = (double**)malloc(linhas * sizeof(double*));
    dividirDados(dados, linhas, colunas, treino, teste, &linhasTreino, &linhasTeste, 0.8);

    double* pesos = (double*)malloc((colunas - 1) * sizeof(double));
    for (int i = 0; i < colunas - 1; i++) {
        pesos[i] = 0.0;
    }

    double taxaAprendizado = 0.01;
    int iteracoes = 1000;
    treinarModelo(treino, linhasTreino, colunas, pesos, taxaAprendizado, iteracoes);

    double precisao = calcularPrecisao(teste, linhasTeste, colunas, pesos);
    printf("Precisão do modelo: %.2f%%\n", precisao * 100);

    int TP, TN, FP, FN;
    calcularMatrizConfusao(teste, linhasTeste, colunas, pesos, &TP, &TN, &FP, &FN);

    printf("Matriz de Confusão:\n");
    printf("TP: %d, FP: %d\n", TP, FP);
    printf("FN: %d, TN: %d\n", FN, TN);

    printf("\nAnálise caso a caso no conjunto de teste:\n");
    exibirDiagnosticos(teste, linhasTeste, colunas, pesos);

    free(dados);
    free(treino);
    free(teste);
    free(pesos);

    return 0;
}