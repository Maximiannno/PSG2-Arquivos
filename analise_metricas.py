#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para analisar o arquivo metrics_simulation_summary.csv gerado
pela simulação de métodos de reconstrução com censura.

O que ele faz:

1. Lê o CSV com as métricas resumidas por:
   - dist_type (gamma, lognormal)
   - censor_level (nível de censura)
   - method (parametric, nonparametric, lod_half, lod_sqrt2)

2. Gera gráficos em PNG para cada:
   - tipo de distribuição (dist_type)
   - métrica (err_mean_rel, err_sd_rel, mae_quant_rel, ks, kl)
   mostrando como a métrica varia com o nível de censura para cada método.

3. Gera tabelas de resumo em CSV:
   - summary_by_method.csv
        * médias das métricas para cada dist_type + method
        * serve para ver qual método é, em média, melhor para cada distribuição

   - best_method_by_censor.csv
        * para cada dist_type + censor_level, escolhe o método com menor KS
        * guarda as métricas desse método para aquele nível de censura
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# =============== CONFIGURAÇÕES BÁSICAS ===============

# Nome do arquivo de entrada com o resumo das simulações.
INPUT_CSV = "metrics_simulation_summary.csv"

# Diretório onde serão salvos:
# - gráficos PNG
# - tabelas de resumo (CSV)
OUTPUT_DIR = "relatorio_metrics"

# Lista das métricas numéricas que vamos:
# - plotar (gráficos)
# - resumir (médias por método)
METRICS = ["err_mean_rel", "err_sd_rel", "mae_quant_rel", "ks", "kl"]


def load_data(path: str) -> pd.DataFrame:
    """
    Carrega o CSV de métricas para um DataFrame do pandas.

    Parâmetros
    ----------
    path : str
        Caminho para o arquivo CSV de entrada (ex.: 'metrics_simulation_summary.csv').

    Retorno
    -------
    df : pd.DataFrame
        DataFrame contendo as colunas:
        - dist_type       (str)
        - censor_level    (float)
        - method          (str)
        - err_mean_rel    (float)
        - err_sd_rel      (float)
        - mae_quant_rel   (float)
        - ks              (float)
        - kl              (float)
        (e eventualmente outras colunas vindas do CSV).

    Observações
    -----------
    - Faz uma checagem simples de existência de arquivo.
    - Garante que a coluna 'censor_level' seja tratada como float,
      o que é importante para grafar em ordem correta.
    """
    if not os.path.exists(path):
        # Se o arquivo não existir, interrompe com um erro claro.
        raise FileNotFoundError(f"Arquivo {path} não encontrado.")

    # Lê o CSV em um DataFrame
    df = pd.read_csv(path)

    # Garante que o nível de censura seja float (pode ter vindo como string)
    df["censor_level"] = df["censor_level"].astype(float)

    return df


def plot_metric_vs_censor(df: pd.DataFrame):
    """
    Gera gráficos de cada métrica em função do nível de censura, separando por método.

    Para cada dist_type (gamma, lognormal, etc.) e para cada métrica em METRICS:
      - Eixo X: nível de censura (%)  [censor_level * 100]
      - Eixo Y: valor médio da métrica (já agregada pela simulação)
      - Uma curva por método (method)

    Resultado:
      - Cria arquivos PNG dentro de OUTPUT_DIR, com nome:
        f"{dist}_{metric}.png"
        Ex.: "gamma_err_mean_rel.png", "lognormal_ks.png", etc.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame carregado de metrics_simulation_summary.csv,
        contendo pelo menos as colunas:
        - 'dist_type'
        - 'censor_level'
        - 'method'
        - colunas de métricas em METRICS
    """
    # Garante que o diretório de saída exista (não dá erro se já existir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Lista de tipos de distribuição presentes (por ex.: ['gamma', 'lognormal'])
    dist_types = df["dist_type"].unique()

    # Lista dos métodos (parametric, nonparametric, lod_half, lod_sqrt2, etc.)
    methods = df["method"].unique()

    # Loop sobre cada tipo de distribuição
    for dist in dist_types:
        # Filtra o DataFrame para um único dist_type
        df_dist = df[df["dist_type"] == dist]

        # Para cada métrica desejada, um gráfico
        for metric in METRICS:
            # Cria uma nova figura
            plt.figure(figsize=(8, 5))

            # Desenha uma linha para cada método
            for method in methods:
                # Subconjunto apenas daquele método dentro daquele dist_type
                df_sub = df_dist[df_dist["method"] == method].sort_values("censor_level")

                # Se não há dados para esse método, pula
                if df_sub.empty:
                    continue

                # Plot:
                #   X = nível de censura em porcentagem
                #   Y = valor da métrica
                plt.plot(
                    df_sub["censor_level"] * 100.0,  # converte p/ porcentagem
                    df_sub[metric],
                    marker="o",
                    label=method,
                )

            # Rótulos dos eixos e título
            plt.xlabel("Nível de censura (%)")
            plt.ylabel(metric)
            plt.title(f"{dist} – {metric} vs. censura")

            # A métrica KL (divergência de Kullback-Leibler) pode variar em várias ordens
            # de grandeza, por isso escala logarítmica pode dar uma visualização melhor.
            if metric == "kl":
                plt.yscale("log")

            # Grade leve no fundo
            plt.grid(True, alpha=0.3)

            # Legenda com nomes dos métodos
            plt.legend()

            # Ajusta para evitar cortes de rótulos
            plt.tight_layout()

            # Caminho do arquivo de saída PNG
            out_path = os.path.join(OUTPUT_DIR, f"{dist}_{metric}.png")

            # Salva a figura em alta resolução (300 dpi)
            plt.savefig(out_path, dpi=300)

            # Fecha a figura para liberar memória
            plt.close()


def make_summary_tables(df: pd.DataFrame):
    """
    Cria e salva tabelas de resumo a partir do DataFrame de métricas.

    Gera dois arquivos CSV dentro de OUTPUT_DIR:

    1) summary_by_method.csv
       - Agrega as métricas (METRICS) por dist_type + method.
       - Calcula a média de cada métrica para cada par (dist_type, method).
       - Útil para comparar o desempenho médio dos métodos em cada distribuição.

    2) best_method_by_censor.csv
       - Para cada dist_type + censor_level:
           * procura o método que possui o menor valor de KS.
           * guarda a linha correspondente (com todas as métricas).
       - Isso permite ver, para cada nível de censura, qual método se destaca
         em termos de Kolmogorov-Smirnov (criticando a forma da distribuição).

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas:
        - 'dist_type'
        - 'method'
        - 'censor_level'
        - e todas as métricas em METRICS (pelo menos).
    """
    # Garante que o diretório exista
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- 1) Resumo por método (média das métricas) ----------

    # groupby(["dist_type", "method"]):
    #   agrupa por tipo de distribuição e por método
    # [METRICS].mean():
    #   calcula a média das métricas numéricas dentro de cada grupo
    summary_by_method = (
        df.groupby(["dist_type", "method"])[METRICS]
        .mean()
        .reset_index()
        # Ordena de forma que, dentro de cada dist_type, apareçam primeiro
        # os métodos com menor KS médio (melhor ajuste em termos de forma).
        .sort_values(["dist_type", "ks"])
    )

    # Caminho do CSV de resumo por método
    summary_path = os.path.join(OUTPUT_DIR, "summary_by_method.csv")

    # Salva CSV (sem índice)
    summary_by_method.to_csv(summary_path, index=False)

    # ---------- 2) Melhor método por censura (usando KS como critério principal) ----------

    best_rows = []

    # Loop por tipo de distribuição
    for dist in df["dist_type"].unique():
        df_dist = df[df["dist_type"] == dist]

        # Para cada nível de censura dentro desse dist_type
        for censor in sorted(df_dist["censor_level"].unique()):
            df_cens = df_dist[df_dist["censor_level"] == censor]

            # idx da linha com menor KS dentro desse sub-dataframe
            best_idx = df_cens["ks"].idxmin()

            # Copia essa linha (para não amarrar a referência ao df original)
            row = df_cens.loc[best_idx].copy()
            best_rows.append(row)

    # DataFrame com uma linha por (dist_type, censor_level),
    # sempre com o método que teve menor KS.
    best_by_censor = pd.DataFrame(best_rows).reset_index(drop=True)

    # Caminho do CSV de "melhor método por censura"
    best_path = os.path.join(OUTPUT_DIR, "best_method_by_censor.csv")

    # Salva o CSV
    best_by_censor.to_csv(best_path, index=False)


def main():
    """
    Função principal do script.

    Passos executados:
    ------------------
    1. Lê o arquivo de entrada (metrics_simulation_summary.csv) com load_data().
    2. Gera gráficos de cada métrica vs nível de censura para cada dist_type,
       salvando-os no diretório OUTPUT_DIR através de plot_metric_vs_censor().
    3. Gera e salva tabelas de resumo (summary_by_method.csv e
       best_method_by_censor.csv) com make_summary_tables().
    4. Imprime mensagens simples no console para indicar progresso e
       informar onde os resultados foram salvos.
    """
    print(f"Lendo dados de {INPUT_CSV}...")
    df = load_data(INPUT_CSV)

    print("Gerando gráficos métricas vs censura...")
    plot_metric_vs_censor(df)
    print(f"Gráficos salvos em: {OUTPUT_DIR}/")

    print("Gerando tabelas de resumo...")
    make_summary_tables(df)
    print("Tabelas salvas em:")
    print(f" - {os.path.join(OUTPUT_DIR, 'summary_by_method.csv')}")
    print(f" - {os.path.join(OUTPUT_DIR, 'best_method_by_censor.csv')}")


if __name__ == "__main__":
    # Se o script for executado diretamente (e não importado como módulo),
    # chama a função main() para rodar toda a análise.
    main()
