#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monte Carlo para avaliar reconstrução de distribuições com censura.

- Simula distribuições verdadeiras (gamma ou lognormal) com parâmetros aleatórios.
- Aplica censura em níveis: 0.20, 0.40, 0.60, 0.80, 0.95, 0.98, 0.99.
- Métodos de reconstrução avaliados:

    * parametric  : melhor dos dois (gamma MLE com censura OU lognormal MLE com censura)
    * nonparametric: reconstrução piecewise uniform (versão Python do R)
    * lod_half    : substitui censurados por LOD/2
    * lod_sqrt2   : substitui censurados por LOD/sqrt(2)

- Para cada run calcula:
    * média, desvio padrão, quantis (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    * KS (Kolmogorov–Smirnov)
    * KL (Kullback–Leibler, via histograma em log-escala)

- Ao final gera:
    * metrics_simulation_all.csv  -> todas as simulações
    * metrics_simulation_summary.csv -> médias das métricas por:
        dist_type x censor_level x method
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize

import warnings
# Suprime todos os RuntimeWarning (inclui o "invalid value encountered in subtract")
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------- CONFIG GERAL ---------------- #

# N_REPS: número de repetições da simulação Monte Carlo.
# Cada repetição sorteia uma "distribuição verdadeira" e aplica todos os níveis de censura.
N_REPS = 500          # número de simulações (ajuste: 100, 500, 1000, ...)

# N_ORIG: tamanho da amostra "verdadeira" (sem censura) gerada em cada repetição.
N_ORIG = 5000         # tamanho da amostra original em cada simulação

# CENSOR_LEVELS: níveis de censura expressos como quantis da distribuição verdadeira.
# Ex.: 0.20 significa que o LOD será o quantil de ordem 0.20, gerando ~20% de censura.
CENSOR_LEVELS = (0.20, 0.40, 0.60, 0.80, 0.95, 0.98, 0.99)

# TOTSAMP_NONPARAM: tamanho aproximado da amostra reconstruída pelo método não paramétrico.
TOTSAMP_NONPARAM = 20000  # tamanho da reconstrução não-paramétrica

# RANDOM_SEED: semente fixa para reprodutibilidade da simulação.
RANDOM_SEED = 123

# QUANTILES: quantis que serão usados para avaliar a qualidade da reconstrução.
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


# ---------------- FUNÇÕES AUXILIARES: SIMULAÇÃO ---------------- #

def sample_random_gamma_params():
    """
    Sorteia parâmetros "razoáveis" para uma distribuição gamma.

    Retorna:
        shape, scale:
            - shape ~ U(0.5, 5)
            - scale ~ U(0.1, 5)

    A ideia é gerar gammas com formatos e escalas variadas,
    mas evitando valores extremos (shape muito pequeno ou scale ~0).
    """
    shape = np.random.uniform(0.5, 5.0)
    scale = np.random.uniform(0.1, 5.0)
    return shape, scale


def sample_random_lognorm_params():
    """
    Sorteia parâmetros para uma lognormal em termos de meanlog e sdlog.

    Retorna:
        meanlog, sdlog:
            - meanlog ~ U(-2, 2)
            - sdlog   ~ U(0.3, 1.5)

    Esses intervalos geram distribuições com variabilidade moderada
    e médias não muito extremas em escala log.
    """
    meanlog = np.random.uniform(-2.0, 2.0)
    sdlog = np.random.uniform(0.3, 1.5)
    return meanlog, sdlog


def simulate_distribution(dist_type, n, params):
    """
    Gera uma amostra de tamanho n da distribuição especificada.

    Parâmetros
    ----------
    dist_type : str
        'gamma' ou 'lognormal'.
    n : int
        Tamanho da amostra a ser gerada.
    params : tuple
        Se dist_type == 'gamma'   -> (shape, scale)
        Se dist_type == 'lognormal' -> (meanlog, sdlog)

    Retorna
    -------
    np.ndarray
        Vetor com n amostras geradas da distribuição escolhida.
    """
    if dist_type == "gamma":
        shape, scale = params
        dist = stats.gamma(a=shape, scale=scale)
    elif dist_type == "lognormal":
        meanlog, sdlog = params
        # Em scipy, lognorm é parametrizada por s = sdlog e scale = exp(meanlog)
        dist = stats.lognorm(s=sdlog, scale=np.exp(meanlog))
    else:
        raise ValueError("dist_type deve ser 'gamma' ou 'lognormal'.")
    return dist.rvs(size=n)


# ---------------- FUNÇÕES AUXILIARES: NÃO PARAMÉTRICA (piecewise uniform) ---------------- #

def build_df_aux_piecewise(conc_obs, cens):
    """
    Constrói o dataframe auxiliar de frequências para a reconstrução não paramétrica
    (piecewise uniform), em formato análogo ao código R.

    Parâmetros
    ----------
    conc_obs : array-like
        Concentrações observadas, onde os valores censurados já foram substituídos por LOD.
    cens : array-like (bool)
        Vetor booleano: True se o valor original foi censurado, False caso contrário.

    Lógica
    ------
    - Cria um DataFrame com:
        conc   : valor observado (LOD ou valor medido)
        notcens: indicador 1 se não censurado, 0 se censurado
    - Faz um groupby por (conc, notcens) para contar quantas vezes cada par ocorre.
      Isso gera uma tabela de frequências com colunas:
        ['conc', 'notcens', 'freq']

    Retorna
    -------
    pd.DataFrame
        DataFrame df_aux com as colunas 'conc', 'notcens', 'freq',
        ordenado por conc e notcens.
    """
    notcens = (~cens).astype(int)
    df = pd.DataFrame({"conc": conc_obs, "notcens": notcens})
    # tabela de frequências por conc x notcens
    df_aux = (
        df.groupby(["conc", "notcens"])
          .size()
          .reset_index(name="freq")
          .sort_values(["conc", "notcens"])
          .reset_index(drop=True)
    )
    return df_aux


def simulate_concentrations_piecewise(df_aux, totsamp, conclb=1e-6, uniftype="logunif"):
    """
    Versão Python de simulate_concentrations do R (reconstrução piecewise uniform).

    Ideia geral
    -----------
    Para cada "bloco" (conc, notcens, freq) em df_aux, criamos um número de pontos
    simulados proporcional à frequência (freq) e com distribuição uniforme:

    - Se notcens == 1:
        * Sorteia valores uniformes (em escala normal ou log) entre prev_conc e conc
          (para representar o intervalo entre o valor anterior e o atual) e
        * Completa com alguns pontos exatamente em conc, se necessário.
    - Se notcens == 0 (censurado):
        * Sorteia valores uniformes no intervalo [conclb, conc], representando que
          os valores verdadeiros podem estar em qualquer lugar abaixo do LOD.

    Parâmetros
    ----------
    df_aux : pd.DataFrame
        Deve conter colunas 'conc', 'notcens', 'freq' já ordenadas por conc.
    totsamp : int
        Tamanho alvo da amostra simulada (aproximado).
    conclb : float
        Lower bound (limite inferior) para simulação de valores censurados.
    uniftype : str
        'unif'   -> uniformemente em escala linear.
        'logunif'-> uniformemente em escala log (mais coerente com dados positivos).

    Retorna
    -------
    np.ndarray
        Vetor com os valores simulados (ordenados).
    """
    df_aux = df_aux.copy()
    df_aux = df_aux.sort_values(["conc", "notcens"]).reset_index(drop=True)

    total_freq = df_aux["freq"].sum()
    if total_freq <= 0:
        # não há dados para reconstruir
        return np.array([])

    # B é um fator de multiplicação: cada ocorrência real gera B pontos simulados.
    B = int(round(totsamp / total_freq))
    if B < 1:
        B = 1
    curTotSamp = B * total_freq

    sim = np.empty(curTotSamp, dtype=float)
    idx = 0

    # prev_conc guarda o "limite inferior" do intervalo para dados não censurados
    prev_conc = df_aux["conc"].iloc[0]

    for _, row in df_aux.iterrows():
        conc = float(row["conc"])
        notcens = int(row["notcens"])
        freq = int(row["freq"])

        # quantidade de pontos simulados para esse bloco
        qt_sim_conc = B * freq

        if conc < conclb:
            # Em teoria não deveria acontecer se conclb for pequeno o suficiente.
            raise ValueError(
                f"CONCLB={conclb} maior que conc atual={conc} na não paramétrica."
            )

        if notcens == 1:
            # dado não censurado: mistura uniforme entre prev_conc e conc
            # + possibilidade de pontos fixos em conc.
            if uniftype.lower() == "unif":
                base = np.random.uniform(prev_conc, conc, size=B)
            else:
                # uniforme em escala log entre prev_conc e conc
                base = np.exp(
                    np.random.uniform(
                        np.log(prev_conc), np.log(conc), size=B
                    )
                )
            if qt_sim_conc > B:
                # se precisamos de mais que B pontos, o excesso é preenchido com conc.
                rem = qt_sim_conc - B
                block = np.concatenate([base, np.full(rem, conc)])
            else:
                # caso contrário, usamos apenas parte dos B pontos
                block = base[:qt_sim_conc]
        else:
            # dado censurado: valores possíveis entre conclb e conc
            if uniftype.lower() == "unif":
                block = np.random.uniform(conclb, conc, size=qt_sim_conc)
            else:
                block = np.exp(
                    np.random.uniform(np.log(conclb), np.log(conc), size=qt_sim_conc)
                )

        # coloca esse bloco na posição correta do vetor sim
        sim[idx: idx + qt_sim_conc] = block
        idx += qt_sim_conc
        prev_conc = conc

    # ordena a amostra final
    sim = np.sort(sim)
    return sim


def reconstruct_nonparametric(conc_obs, cens, totsamp=TOTSAMP_NONPARAM):
    """
    Wrapper para a reconstrução não paramétrica.

    Parâmetros
    ----------
    conc_obs : array-like
        Concentrações observadas (censurados já substituídos por LOD).
    cens : array-like (bool)
        Vetor booleano indicando quais valores foram censurados.
    totsamp : int
        Tamanho alvo da amostra simulada.

    Retorna
    -------
    np.ndarray
        Amostra reconstruída pela abordagem piecewise uniform.
    """
    df_aux = build_df_aux_piecewise(conc_obs, cens)
    if df_aux.empty:
        # Se por algum motivo não houver dados, devolve uma cópia das observações
        return conc_obs.copy()
    sim = simulate_concentrations_piecewise(df_aux, totsamp, conclb=1e-6,
                                            uniftype="logunif")
    return sim


# ---------------- FUNÇÕES AUXILIARES: MLE CENSURADO ---------------- #

def neg_loglik_gamma(params, x, cens, lod):
    """
    Função de log-verossimilhança negativa para uma distribuição Gamma
    com censura à esquerda em LOD.

    Parâmetros
    ----------
    params : (shape, scale)
        Parâmetros da Gamma.
    x : np.ndarray
        Dados verdadeiros (sem substituição de censura).
    cens : np.ndarray (bool)
        True se a observação é censurada (x < LOD).
    lod : float
        Limite de detecção (LOD) usado na censura.

    Lógica
    ------
    - Para dados não censurados: usa pdf(x).
    - Para dados censurados: usa P(X <= LOD) = F(LOD).

    Retorna
    -------
    float
        Valor da log-verossimilhança negativa (para ser minimizada).
    """
    shape, scale = params
    # Condições de validade dos parâmetros
    if shape <= 0 or scale <= 0:
        return np.inf
    dist = stats.gamma(a=shape, scale=scale)

    # Parte não censurada
    x_det = x[~cens]
    ll = 0.0
    if x_det.size > 0:
        ll += np.sum(dist.logpdf(x_det))

    # Parte censurada: contribuição via cdf no LOD
    n_cens = np.sum(cens)
    if n_cens > 0:
        F_l = dist.cdf(lod)
        if F_l <= 0:
            # evitar log(0)
            return np.inf
        ll += n_cens * np.log(F_l)

    # retorna negativo porque o otimizador minimiza
    return -ll


def neg_loglik_lognorm(params, x, cens, lod):
    """
    Log-verossimilhança negativa para distribuição Lognormal com censura à esquerda.

    Parâmetros
    ----------
    params : (meanlog, sdlog)
        Parâmetros no espaço log.
    x : np.ndarray
        Dados verdadeiros.
    cens : np.ndarray (bool)
        Vetor de censura (True se censurado).
    lod : float
        Limite de detecção.

    Lógica
    ------
    - Dados não censurados: contribuição via logpdf.
    - Dados censurados: P(X <= LOD) = F(LOD).

    Retorna
    -------
    float
        Log-verossimilhança negativa.
    """
    meanlog, sdlog = params
    if sdlog <= 0:
        return np.inf
    dist = stats.lognorm(s=sdlog, scale=np.exp(meanlog))

    x_det = x[~cens]
    ll = 0.0
    if x_det.size > 0:
        ll += np.sum(dist.logpdf(x_det))

    n_cens = np.sum(cens)
    if n_cens > 0:
        F_l = dist.cdf(lod)
        if F_l <= 0:
            return np.inf
        ll += n_cens * np.log(F_l)

    return -ll


def fit_gamma_censored(x, cens, lod):
    """
    Ajusta uma distribuição Gamma a dados censurados à esquerda em LOD,
    via MLE (Maximum Likelihood Estimation).

    Parâmetros
    ----------
    x : np.ndarray
        Dados verdadeiros (sem substituições).
    cens : np.ndarray (bool)
        Vetor de censura.
    lod : float
        Limite de detecção.

    Lógica
    ------
    - Usa somente dados não censurados para obter chute inicial de média e variância.
    - Calcula shape0 e scale0 a partir de (m, v) da Gamma:
        shape0 = m^2 / v
        scale0 = v / m
    - Otimiza a log-verossimilhança negativa usando L-BFGS-B.
    - Retorna (shape, scale) se convergir, caso contrário retorna None.
    """
    x_det = x[~cens]
    if x_det.size < 2:
        # poucos dados observados, impossível ajustar com segurança
        return None
    m = np.mean(x_det)
    v = np.var(x_det, ddof=1)
    if m <= 0 or v <= 0:
        # fallback se média ou variância derem problema
        shape0, scale0 = 1.0, max(m, 1e-3)
    else:
        shape0 = m * m / v
        scale0 = v / m

    bounds = [(1e-6, None), (1e-9, None)]
    res = minimize(
        neg_loglik_gamma,
        x0=[shape0, scale0],
        args=(x, cens, lod),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not res.success:
        return None
    shape, scale = res.x
    if shape <= 0 or scale <= 0:
        return None
    return float(shape), float(scale)


def fit_lognorm_censored(x, cens, lod):
    """
    Ajusta uma distribuição Lognormal a dados censurados à esquerda.

    Parâmetros
    ----------
    x : np.ndarray
        Dados verdadeiros.
    cens : np.ndarray (bool)
        Vetor de censura.
    lod : float
        Limite de detecção.

    Lógica
    ------
    - Usa somente os dados não censurados e > 0 para obter o chute inicial
      de meanlog e sdlog.
    - Executa otimização L-BFGS-B da log-verossimilhança negativa.
    """
    x_det = x[~cens]
    x_det = x_det[x_det > 0]
    if x_det.size < 2:
        return None
    logs = np.log(x_det)
    meanlog0 = np.mean(logs)
    sdlog0 = np.std(logs, ddof=1)
    if sdlog0 <= 0:
        sdlog0 = 0.5

    bounds = [(None, None), (1e-6, None)]
    res = minimize(
        neg_loglik_lognorm,
        x0=[meanlog0, sdlog0],
        args=(x, cens, lod),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not res.success:
        return None
    meanlog, sdlog = res.x
    if sdlog <= 0:
        return None
    return float(meanlog), float(sdlog)


# ---------------- MÉTRICAS: SUMÁRIO, KS, KL ---------------- #

def summary_stats(x, quantiles):
    """
    Calcula estatísticas de resumo para um vetor de dados.

    Parâmetros
    ----------
    x : array-like
        Dados.
    quantiles : list[float]
        Lista de quantis desejados (entre 0 e 1).

    Retorna
    -------
    mean : float
        Média dos dados.
    sd : float
        Desvio padrão amostral (ddof=1).
    qs : np.ndarray
        Quantis nas posições especificadas em 'quantiles'.
    """
    x = np.asarray(x)
    mean = np.mean(x)
    sd = np.std(x, ddof=1)
    qs = np.quantile(x, quantiles)
    return mean, sd, qs


def compute_kl(x_true, x_est, n_bins=200):
    """
    Computa a divergência de Kullback-Leibler KL(P_true || P_est)
    usando histogramas em escala log dos dados verdadeiros e estimados.

    Parâmetros
    ----------
    x_true : array-like
        Dados da distribuição verdadeira (referência).
    x_est : array-like
        Dados da distribuição estimada/reconstruída.
    n_bins : int
        Número de bins no histograma log-espacial.

    Lógica
    ------
    - Combina x_true e x_est para definir um range [lo, hi].
    - Constrói bins log-espaciais entre lo e hi.
    - Estima densidades p_hist, q_hist em cada bin.
    - Converte densidades em probabilidades (multiplicando pela largura dos bins).
    - Adiciona pequeno epsilon para evitar zeros.
    - Usa stats.entropy(p, q) para obter KL(P||Q).
    """
    x_true = np.asarray(x_true)
    x_est = np.asarray(x_est)

    data = np.concatenate([x_true, x_est])
    lo = np.min(data)
    hi = np.max(data)
    if lo <= 0:
        lo = 1e-9
    if hi <= lo:
        hi = lo * 10.0

    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)

    p_hist, _ = np.histogram(x_true, bins=edges, density=True)
    q_hist, _ = np.histogram(x_est, bins=edges, density=True)
    widths = np.diff(edges)

    p = p_hist * widths
    q = q_hist * widths

    eps = 1e-12
    p = p + eps
    q = q + eps

    p /= p.sum()
    q /= q.sum()

    return float(stats.entropy(p, q))


# ---------------- DETECÇÃO DE QUANTIS (para resumo) ---------------- #

def detect_quantile_columns(df):
    """
    Detecta automaticamente quais colunas do DataFrame correspondem a quantis
    'verdadeiros' e 'estimados'.

    Convenção
    ---------
    - Colunas com quantis verdadeiros terminam com '_true', ex.: 'q05_true'.
    - Colunas com quantis estimados terminam com '_est', ex.: 'q05_est'.

    Retorna
    -------
    quant_names : list[str]
        Lista dos nomes-base dos quantis (sem sufixo), ex.: ['q05', 'q10', ...].
    map_true : dict
        Mapeia nome-base -> nome da coluna "_true".
    map_est : dict
        Mapeia nome-base -> nome da coluna "_est".
    """
    true_cols = [c for c in df.columns if c.endswith("_true")]
    est_cols = [c for c in df.columns if c.endswith("_est")]

    quant_names = []
    map_true = {}
    map_est = {}

    for ct in true_cols:
        prefix = ct[:-5]  # remove "_true"
        ce = prefix + "_est"
        if ce in est_cols:
            quant_names.append(prefix)
            map_true[prefix] = ct
            map_est[prefix] = ce

    quant_names = sorted(quant_names)
    return quant_names, map_true, map_est


def add_relative_errors(df):
    """
    Adiciona colunas de erro relativo ao DataFrame de resultados.

    Colunas adicionadas
    -------------------
    - err_mean_rel : |mean_est - mean_true| / |mean_true|
    - err_sd_rel   : |sd_est   - sd_true|   / |sd_true|
    - mae_quant_rel: média do erro relativo absoluto em todos os quantis disponíveis.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas 'mean_true', 'mean_est', 'sd_true', 'sd_est'
        e colunas de quantis *_true e *_est.

    Retorna
    -------
    pd.DataFrame
        Mesmo DataFrame com as colunas de erro adicionadas.
    """
    eps = 1e-12  # evita divisão por zero
    df["err_mean_rel"] = (df["mean_est"] - df["mean_true"]).abs() / (
        df["mean_true"].abs() + eps
    )
    df["err_sd_rel"] = (df["sd_est"] - df["sd_true"]).abs() / (
        df["sd_true"].abs() + eps
    )

    # Detecta quais colunas são quantis
    qnames, map_true, map_est = detect_quantile_columns(df)
    if not qnames:
        df["mae_quant_rel"] = np.nan
        return df

    errs = []
    for q in qnames:
        ct = map_true[q]
        ce = map_est[q]
        err_q = (df[ce] - df[ct]).abs() / (df[ct].abs() + eps)
        errs.append(err_q.values)

    # errs: matriz (n_linhas, n_quantis)
    errs = np.vstack(errs).T
    # MAE médio sobre todos os quantis
    df["mae_quant_rel"] = errs.mean(axis=1)
    return df


# ---------------- LOOP PRINCIPAL DE SIMULAÇÃO ---------------- #

def run_simulation(
    n_reps=N_REPS,
    n_orig=N_ORIG,
    censor_levels=CENSOR_LEVELS,
    totsamp_nonparam=TOTSAMP_NONPARAM,
    random_seed=RANDOM_SEED,
):
    """
    Executa toda a simulação Monte Carlo.

    Parâmetros
    ----------
    n_reps : int
        Número de repetições independentes.
    n_orig : int
        Tamanho da amostra verdadeira em cada repetição.
    censor_levels : iterable[float]
        Níveis de censura (quantis) usados para definir LOD.
    totsamp_nonparam : int
        Tamanho alvo para reconstrução não paramétrica.
    random_seed : int ou None
        Semente de aleatoriedade.

    Lógica de alto nível
    --------------------
    Para cada repetição:
      1. Sorteia o tipo da distribuição verdadeira (gamma ou lognormal)
         e seus parâmetros.
      2. Gera a amostra verdadeira x_true.
      3. Calcula estatísticas "true".
      4. Para cada nível de censura:
         a) Define LOD como quantil da amostra verdadeira (x_true).
         b) Cria vetor de censura (x < LOD).
         c) Aplica os quatro métodos de reconstrução:
            - parametric (melhor entre gamma/lognormal MLE)
            - nonparametric (piecewise uniform)
            - lod_half  (substitui censurados por LOD/2)
            - lod_sqrt2 (substitui censurados por LOD/sqrt(2))
         d) Calcula métricas (média, sd, quantis, KS, KL) para cada método.
      5. Armazena tudo em uma lista de dicionários.

    Retorna
    -------
    pd.DataFrame
        DataFrame com uma linha por combinação (rep, censor_level, method).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    rows = []

    # total de passos (cada combinação rep x nível de censura)
    total_steps = n_reps * len(censor_levels)
    step = 0

    for rep in range(1, n_reps + 1):
        # escolhe tipo da distribuição verdadeira
        dist_type = np.random.choice(["gamma", "lognormal"])
        if dist_type == "gamma":
            params = sample_random_gamma_params()
        else:
            params = sample_random_lognorm_params()

        # simula amostra original (verdadeira)
        x_true = simulate_distribution(dist_type, n_orig, params)
        x_true = np.asarray(x_true)

        # estatísticas da distribuição verdadeira
        mean_true, sd_true, qs_true = summary_stats(x_true, QUANTILES)

        for censor_level in censor_levels:
            # atualiza e mostra progresso no terminal
            step += 1
            print(
                f"Progresso: {step}/{total_steps} simulações (rep={rep}, censura={int(censor_level*100)}%)",
                end="\r",
                flush=True,
            )

            # define LOD como quantil da amostra verdadeira
            lod = float(np.quantile(x_true, censor_level))
            cens = x_true < lod  # True se estiver abaixo de LOD

            # reconstrução não paramétrica (piecewise)
            # conc_obs: substitui valores censurados pelo LOD
            conc_obs = np.where(cens, lod, x_true)
            x_np = reconstruct_nonparametric(conc_obs, cens, totsamp=totsamp_nonparam)

            # métodos LOD/2 e LOD/sqrt(2)
            x_lod2 = np.where(cens, lod / 2.0, x_true)
            x_lod_s2 = np.where(cens, lod / np.sqrt(2.0), x_true)

            # fits paramétricos com censura (Gamma e Lognormal)
            gamma_fit = fit_gamma_censored(x_true, cens, lod)
            logn_fit = fit_lognorm_censored(x_true, cens, lod)

            # lista de candidatos paramétricos (amostra gerada a partir dos parâmetros ajustados)
            candidates = []
            if gamma_fit is not None:
                shape, scale = gamma_fit
                distg = stats.gamma(a=shape, scale=scale)
                x_pg = distg.rvs(size=n_orig)  # amostra gerada da Gamma ajustada
                candidates.append(("gamma_mle", x_pg))
            if logn_fit is not None:
                meanlog, sdlog = logn_fit
                distl = stats.lognorm(s=sdlog, scale=np.exp(meanlog))
                x_pl = distl.rvs(size=n_orig)  # amostra gerada da Lognormal ajustada
                candidates.append(("lognormal_mle", x_pl))

            # escolhe o melhor paramétrico com base no KS em relação à x_true
            best_param_sample = None
            best_param_name = None
            best_param_ks = np.inf
            best_param_kl = np.nan

            for name, x_est in candidates:
                ks_stat = stats.ks_2samp(x_true, x_est).statistic
                kl_val = compute_kl(x_true, x_est)
                if ks_stat < best_param_ks:
                    best_param_ks = ks_stat
                    best_param_kl = kl_val
                    best_param_name = name
                    best_param_sample = x_est

            def add_row(method_name, x_est, extra_info=None):
                """
                Função interna para adicionar uma linha de resultados à lista 'rows'.

                Parâmetros
                ----------
                method_name : str
                    Nome do método ("parametric", "nonparametric", "lod_half", "lod_sqrt2").
                x_est : array-like
                    Amostra estimada/reconstruída pelo método.
                extra_info : dict ou None
                    Informações extras para guardar na linha (por exemplo, qual
                    família paramétrica foi escolhida).

                Essa função:
                    - calcula média, sd, quantis de x_est;
                    - calcula KS e KL comparando x_est com x_true;
                    - guarda as estatísticas em um dicionário e o acrescenta à lista rows.
                """
                if x_est is None or len(x_est) == 0:
                    return
                mean_est, sd_est, qs_est = summary_stats(x_est, QUANTILES)
                ks_stat = stats.ks_2samp(x_true, x_est).statistic
                kl_val = compute_kl(x_true, x_est)

                row = {
                    "rep": rep,
                    "dist_type": dist_type,
                    "censor_level": censor_level,
                    "method": method_name,
                    "mean_true": mean_true,
                    "mean_est": mean_est,
                    "sd_true": sd_true,
                    "sd_est": sd_est,
                    "ks": ks_stat,
                    "kl": kl_val,
                }

                # quantis verdadeiros
                for q, v in zip(QUANTILES, qs_true):
                    row[f"q{int(q * 100):02d}_true"] = v
                # quantis estimados
                for q, v in zip(QUANTILES, qs_est):
                    row[f"q{int(q * 100):02d}_est"] = v

                if extra_info:
                    row.update(extra_info)

                rows.append(row)

            # paramétrico (melhor entre gamma/lognormal)
            if best_param_sample is not None:
                add_row("parametric", best_param_sample,
                        extra_info={"param_choice": best_param_name})

            # não paramétrica
            add_row("nonparametric", x_np)

            # LOD/2
            add_row("lod_half", x_lod2)

            # LOD/sqrt(2)
            add_row("lod_sqrt2", x_lod_s2)

    # quebra de linha depois do último print de progresso
    print()

    df = pd.DataFrame(rows)
    return df


def main():
    """
    Função principal do script.

    Passos
    ------
    1. Executa a simulação completa (run_simulation).
    2. Salva o DataFrame com todas as linhas em 'metrics_simulation_all.csv'.
    3. Calcula erros relativos (média, sd, quantis) e adiciona ao DataFrame.
    4. Faz um resumo (média das métricas) por:
       - dist_type
       - censor_level
       - method
    5. Salva o resumo em 'metrics_simulation_summary.csv'.
    6. Imprime mensagens básicas de conclusão.
    """
    df_all = run_simulation()

    # salva o bruto
    df_all.to_csv("metrics_simulation_all.csv", index=False)

    # adiciona erros relativos e MAE de quantis
    df_all = add_relative_errors(df_all)

    # resumo: média das métricas por dist_type x censor_level x method
    group_cols = ["dist_type", "censor_level", "method"]
    metrics_cols = [
        "err_mean_rel",
        "err_sd_rel",
        "mae_quant_rel",
        "ks",
        "kl",
    ]
    df_summary = (
        df_all.groupby(group_cols)[metrics_cols]
        .mean()
        .reset_index()
        .sort_values(group_cols)
    )

    df_summary.to_csv("metrics_simulation_summary.csv", index=False)

    print("Simulação concluída.")
    print("Arquivo detalhado: metrics_simulation_all.csv")
    print("Resumo (médias por dist_type x censor_level x method): metrics_simulation_summary.csv")


if __name__ == "__main__":
    # Executa a função principal apenas se o script for chamado diretamente.
    main()
