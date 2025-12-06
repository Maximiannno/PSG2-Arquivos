import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm as lognorm_dist
from scipy.stats import gaussian_kde, ks_2samp
from scipy.optimize import minimize


# ============================================================
#  Versão Python do simulate_concentrations (arquivo R)
# ============================================================

def simulate_concentrations_piecewise(df_aux: pd.DataFrame,
                                      totsamp: int,
                                      conclb: float,
                                      uniftype: str):
    """
    Implementa o simulate_concentrations do R:

    - df_aux com colunas: ['conc', 'notcens', 'freq']
    - totsamp = TOTSAMP
    - conclb = CONCLB
    - uniftype = 'unif' ou 'logunif'
    """
    uniftype = uniftype.lower()
    if uniftype not in ("unif", "logunif"):
        raise ValueError("UNIFTYPE must be 'unif' or 'logunif'")

    qtconc = df_aux.shape[0]
    total_freq = df_aux["freq"].sum()

    if total_freq <= 0:
        raise ValueError("Total frequency must be positive.")

    B = int(round(totsamp / total_freq))
    cur_tot_samp = int(B * total_freq)

    sim_conc = np.empty(cur_tot_samp, dtype=float)
    cumm_index = 0

    conc_vals = df_aux["conc"].values.astype(float)
    notcens_vals = df_aux["notcens"].values.astype(int)
    freq_vals = df_aux["freq"].values.astype(int)

    for i in range(qtconc):
        conc_i = conc_vals[i]
        notcens_i = notcens_vals[i]
        freq_i = freq_vals[i]

        qt_sim_conc = B * freq_i
        if qt_sim_conc <= 0:
            continue

        if conc_i < conclb:
            raise ValueError(
                f"Lower bound for simulated concentration ({conclb}) "
                f"is higher than the current concentration ({conc_i})."
            )

        # R: lastindex = ifelse(i == 1, 1, i - 1)
        if i == 0:
            last_index = 0
        else:
            last_index = i - 1

        conc_last = conc_vals[last_index]

        if notcens_i == 1:
            # Dados não censurados
            if uniftype == "unif":
                unif_part = np.random.uniform(low=conc_last, high=conc_i, size=B)
                remaining = np.full(qt_sim_conc - B, conc_i, dtype=float)
            else:  # "logunif"
                unif_part = np.exp(
                    np.random.uniform(
                        low=np.log(conc_last),
                        high=np.log(conc_i),
                        size=B,
                    )
                )
                remaining = np.full(qt_sim_conc - B, conc_i, dtype=float)

            sim_chunk = np.concatenate([unif_part, remaining])
        else:
            # Dados censurados
            if uniftype == "unif":
                sim_chunk = np.random.uniform(
                    low=conclb,
                    high=conc_i,
                    size=qt_sim_conc,
                )
            else:  # "logunif"
                sim_chunk = np.exp(
                    np.random.uniform(
                        low=np.log(conclb),
                        high=np.log(conc_i),
                        size=qt_sim_conc,
                    )
                )

        sim_conc[cumm_index: cumm_index + qt_sim_conc] = sim_chunk
        cumm_index += qt_sim_conc

    sim_conc = sim_conc[:cumm_index]
    sim_conc.sort()
    return sim_conc


# ============================================================
#  Construção de df_aux e reconstrução não paramétrica
# ============================================================

def nonparametric_reconstruction(conc: np.ndarray,
                                 cens: np.ndarray,
                                 totsamp: int = 20000,
                                 conclb: float = 1e-6,
                                 uniftype: str = "logunif"):
    """
    Versão simplificada do process_combination para um único "grupo",
    usando conc e cens, conforme os arquivos R.

    Retorna:
      sim_conc, percentiles, quant_sim_conc
    """
    vsel = pd.DataFrame({"conc": conc, "cens": cens.astype(int)})
    vsel["notcens"] = 1 - vsel["cens"]
    vsel = vsel.sort_values(["conc", "notcens"]).reset_index(drop=True)

    # xtabs(~ conc + notcens)
    freq_table = (
        vsel.groupby(["conc", "notcens"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    # garantir colunas 0 e 1
    for col in [0, 1]:
        if col not in freq_table.columns:
            freq_table[col] = 0
    freq_table = freq_table[[0, 1]]

    uniqueconc = freq_table.index.values.astype(float)

    # R: matr_conc_nocens = cbind(rep(uniqueconc, each=2), rep(c(0,1), times=nrow))
    conc_rep = np.repeat(uniqueconc, 2)
    notcens_rep = np.tile(np.array([0, 1], dtype=int), len(uniqueconc))

    # R: vectfreq = as.vector(t(auxtabfreq))
    vectfreq = freq_table.to_numpy().ravel(order="C")

    df_aux = pd.DataFrame(
        {"conc": conc_rep, "notcens": notcens_rep, "freq": vectfreq}
    )
    df_aux = df_aux[df_aux["freq"] > 0].reset_index(drop=True)

    sim_conc = simulate_concentrations_piecewise(
        df_aux=df_aux,
        totsamp=totsamp,
        conclb=conclb,
        uniftype=uniftype,
    )

    cur_tot_samp = len(sim_conc)

    # R: percentiles = seq(0,1, by = 1/curTotSamp * 5)
    step = 1.0 / cur_tot_samp * 5.0
    if step <= 0:
        step = 1.0
    percentiles = np.arange(0.0, 1.0 + step / 2.0, step)
    percentiles = np.clip(percentiles, 0.0, 1.0)

    # quantis da reconstrução não paramétrica
    quant_sim_conc = np.quantile(sim_conc, percentiles, method="linear")

    return sim_conc, percentiles, quant_sim_conc


# ============================================================
#  Ajustes Gamma e Lognormal via MLE com censura (tipo fitdistcens)
# ============================================================

def fit_gamma_mle_censored(conc_obs: np.ndarray,
                           cens: np.ndarray,
                           L: float):
    """
    Ajusta Gamma (shape, rate) por máxima verossimilhança com censura à esquerda.
    Usa média/variância da amostra censurada apenas como chute inicial interno.
    """
    conc_obs = np.asarray(conc_obs, dtype=float)
    cens = np.asarray(cens, dtype=int)

    x_uncens = conc_obs[cens == 0]
    n_cens = int((cens == 1).sum())

    if x_uncens.size == 0:
        raise ValueError("Não há dados não censurados para ajuste Gamma MLE.")

    x_uncens = np.clip(x_uncens, 1e-12, None)
    L = float(max(L, 1e-12))

    # chute inicial simples via momentos da amostra censurada (uso interno)
    m = float(conc_obs.mean())
    s = float(conc_obs.std(ddof=1))
    if s <= 0:
        s = max(1e-3, 0.1 * m)
    shape_init = m**2 / (s**2 + 1e-12)
    rate_init = m / (s**2 + 1e-12)

    theta0 = np.log([max(shape_init, 1e-6), max(rate_init, 1e-6)])

    def neg_loglik(log_params):
        log_shape, log_rate = log_params
        shape = np.exp(log_shape)
        rate = np.exp(log_rate)
        scale = 1.0 / rate

        pdf_vals = gamma_dist.pdf(x_uncens, a=shape, scale=scale)
        if np.any(pdf_vals <= 0):
            return 1e20
        ll_uncens = np.log(pdf_vals).sum()

        if n_cens > 0:
            F_L = gamma_dist.cdf(L, a=shape, scale=scale)
            if F_L <= 0 or F_L >= 1:
                return 1e20
            ll_cens = n_cens * np.log(F_L)
        else:
            ll_cens = 0.0

        return -(ll_uncens + ll_cens)

    res = minimize(neg_loglik, theta0, method="L-BFGS-B")
    if not res.success:
        print("[WARN] Gamma MLE não convergiu, usando chute inicial.")
        log_shape_hat, log_rate_hat = theta0
    else:
        log_shape_hat, log_rate_hat = res.x

    shape_hat = float(np.exp(log_shape_hat))
    rate_hat = float(np.exp(log_rate_hat))
    return shape_hat, rate_hat


def fit_lnorm_mle_censored(conc_obs: np.ndarray,
                           cens: np.ndarray,
                           L: float):
    """
    Ajusta Lognormal (meanlog, sdlog) por máxima verossimilhança com censura à esquerda.
    Usa média e desvio da amostra censurada apenas como chute inicial interno.
    """
    conc_obs = np.asarray(conc_obs, dtype=float)
    cens = np.asarray(cens, dtype=int)

    x_uncens = conc_obs[cens == 0]
    n_cens = int((cens == 1).sum())

    if x_uncens.size == 0:
        raise ValueError("Não há dados não censurados para ajuste Lognormal MLE.")

    x_uncens = np.clip(x_uncens, 1e-12, None)
    L = float(max(L, 1e-12))

    # chute inicial via momentos da amostra censurada (uso interno)
    m = float(conc_obs.mean())
    s = float(conc_obs.std(ddof=1))
    if s <= 0:
        s = max(1e-3, 0.1 * m)
    v = s**2
    s2 = np.log(v / (m**2) + 1.0)
    sdlog_init = np.sqrt(s2)
    meanlog_init = np.log(m) - 0.5 * s2

    theta0 = np.array([meanlog_init, np.log(max(sdlog_init, 1e-6))], dtype=float)

    def neg_loglik(theta):
        meanlog, log_sdlog = theta
        sdlog = np.exp(log_sdlog)

        pdf_vals = lognorm_dist.pdf(x_uncens, s=sdlog, scale=np.exp(meanlog))
        if np.any(pdf_vals <= 0):
            return 1e20
        ll_uncens = np.log(pdf_vals).sum()

        if n_cens > 0:
            F_L = lognorm_dist.cdf(L, s=sdlog, scale=np.exp(meanlog))
            if F_L <= 0 or F_L >= 1:
                return 1e20
            ll_cens = n_cens * np.log(F_L)
        else:
            ll_cens = 0.0

        return -(ll_uncens + ll_cens)

    res = minimize(neg_loglik, theta0, method="L-BFGS-B")
    if not res.success:
        print("[WARN] Lognormal MLE não convergiu, usando chute inicial.")
        meanlog_hat, log_sdlog_hat = theta0
    else:
        meanlog_hat, log_sdlog_hat = res.x

    sdlog_hat = float(np.exp(log_sdlog_hat))
    return float(meanlog_hat), sdlog_hat


# ============================================================
#  Métricas: médias, quantis, KS e KL
# ============================================================

def compute_summary_stats(x: np.ndarray):
    """
    Retorna dict com mean, sd e quantis.
    """
    x = np.asarray(x, dtype=float)
    q_probs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    quantiles = np.quantile(x, q_probs)
    stats = {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
    }
    for p, q in zip(q_probs, quantiles):
        stats[f"q{int(p*100):02d}"] = float(q)
    return stats


def estimate_kl_divergence(sample_p: np.ndarray,
                           sample_q: np.ndarray,
                           grid: np.ndarray):
    """
    Estima D_KL(P||Q) onde P tem amostra sample_p e Q tem amostra sample_q,
    usando KDE e integração numérica em 'grid'.
    """
    eps = 1e-12
    kde_p = gaussian_kde(sample_p)
    kde_q = gaussian_kde(sample_q)

    p_vals = kde_p(grid)
    q_vals = kde_q(grid)

    p_vals = np.maximum(p_vals, eps)
    q_vals = np.maximum(q_vals, eps)

    integrand = p_vals * np.log(p_vals / q_vals)
    kl = np.trapz(integrand, grid)
    return float(kl)


def compute_metrics_all_models(dist_type: str,
                               censor_frac: float,
                               x_true_sample: np.ndarray,
                               sim_np: np.ndarray,
                               gamma_mle_sample: np.ndarray,
                               lnorm_mle_sample: np.ndarray,
                               sub_L2_sample: np.ndarray,
                               sub_Lsqrt2_sample: np.ndarray):
    """
    Calcula:
      - médias, sd, quantis
      - KS (2-sample) vs verdadeira
      - KL (P_true || P_model) via KDE

    para:
      - 'true'
      - 'nonparam_piecewise'
      - 'nonparam_L2'
      - 'nonparam_Lsqrt2'
      - 'gamma_mle'
      - 'lnorm_mle'
    """
    models = {
        "true": x_true_sample,
        "nonparam_piecewise": sim_np,
        "nonparam_L2": sub_L2_sample,
        "nonparam_Lsqrt2": sub_Lsqrt2_sample,
        "gamma_mle": gamma_mle_sample,
        "lnorm_mle": lnorm_mle_sample,
    }

    # grid para KL: baseado em todos os dados
    all_samples = np.concatenate(list(models.values()))
    xmin = max(1e-8, np.quantile(all_samples, 0.001))
    xmax = np.quantile(all_samples, 0.999)
    grid = np.linspace(xmin, xmax, 400)

    metrics_rows = []
    for name, sample in models.items():
        stats = compute_summary_stats(sample)

        if name == "true":
            ks_stat = 0.0
            kl_div = 0.0
        else:
            ks_stat = float(
                ks_2samp(x_true_sample, sample, alternative="two-sided").statistic
            )
            kl_div = estimate_kl_divergence(x_true_sample, sample, grid)

        row = {
            "dist_type": dist_type,
            "censor_frac": censor_frac,
            "model": name,
            "ks": ks_stat,
            "kl": kl_div,
        }
        row.update(stats)
        metrics_rows.append(row)

    return pd.DataFrame(metrics_rows)


# ============================================================
#  Pipeline principal (simulação + censura + reconstrução)
# ============================================================

def run_pipeline_for_distribution(dist_type: str,
                                  n: int = 500,
                                  censor_fraction: float = 0.2):
    """
    Pipeline para Gamma ou Lognormal:

      1) simula distribuição verdadeira (parâmetros aleatórios);
      2) aplica censura à esquerda;
      3) reconstrói:
         - não paramétrica (piecewise-uniform)
         - substituições L/2 e L/√2
         - Gamma via MLE censurada
         - Lognormal via MLE censurada
      4) plota CDFs
      5) gera amostras grandes de cada modelo e calcula:
         - mean, sd, quantis
         - KS
         - KL
      6) retorna DataFrame de métricas
    """

    rng = np.random.default_rng()

    # 1) Simulação verdadeira
    if dist_type.lower() == "gamma":
        shape_true = rng.uniform(0.5, 5.0)
        rate_true = rng.uniform(0.1, 2.0)
        scale_true = 1.0 / rate_true

        x_true = gamma_dist.rvs(a=shape_true, scale=scale_true, size=n, random_state=rng)
        params_true = (shape_true, rate_true)

        print(f"[{dist_type}] true shape={shape_true:.3f}, rate={rate_true:.3f}")

    elif dist_type.lower() == "lognormal":
        meanlog_true = rng.uniform(-2.0, 2.0)
        sdlog_true = rng.uniform(0.2, 1.5)

        x_true = lognorm_dist.rvs(s=sdlog_true, scale=np.exp(meanlog_true),
                                  size=n, random_state=rng)
        params_true = (meanlog_true, sdlog_true)

        print(f"[{dist_type}] true meanlog={meanlog_true:.3f}, sdlog={sdlog_true:.3f}")

    else:
        raise ValueError("dist_type must be 'gamma' or 'lognormal'")

    # 2) Censura à esquerda
    L = np.quantile(x_true, censor_fraction)
    cens = (x_true < L).astype(int)
    conc_obs = np.where(cens == 1, L, x_true)

    # 2a) Outras abordagens não paramétricas: substituições L/2 e L/√2
    conc_sub_L2 = conc_obs.copy()
    conc_sub_L2[cens == 1] = L / 2.0

    conc_sub_Lsqrt2 = conc_obs.copy()
    conc_sub_Lsqrt2[cens == 1] = L / np.sqrt(2.0)

    print(f"[{dist_type}] censoring at {censor_fraction*100:.1f}%-quantile L={L:.4f}")
    print(f"[{dist_type}] censored fraction (empírico) = {cens.mean():.3f}")

    # 3) Não-paramétrica (piecewise)
    sim_conc_np, percentiles_np, quants_np = nonparametric_reconstruction(
        conc=conc_obs,
        cens=cens,
        totsamp=20000,
        conclb=1e-6,
        uniftype="unif",   # em vez de "logunif"
    )

    # 4) Gamma & Lognormal via MLE censurada
    shape_mle, rate_mle = fit_gamma_mle_censored(conc_obs, cens, L)
    meanlog_mle, sdlog_mle = fit_lnorm_mle_censored(conc_obs, cens, L)

    print(f"[{dist_type}] MLE Gamma (cens): shape={shape_mle:.3f}, rate={rate_mle:.3f}")
    print(f"[{dist_type}] MLE Lognormal (cens): meanlog={meanlog_mle:.3f}, sdlog={sdlog_mle:.3f}")

    # 5) Grid e CDF verdadeira (para visualização)
    x_min = 0.0
    x_max = np.quantile(x_true, 0.999)
    x_grid = np.linspace(x_min, x_max, 600)

    if dist_type.lower() == "gamma":
        true_cdf = gamma_dist.cdf(x_grid, a=params_true[0], scale=1.0 / params_true[1])
    else:
        true_cdf = lognorm_dist.cdf(x_grid, s=params_true[1], scale=np.exp(params_true[0]))

    # CDF empírica dos dados censurados
    xs_cens_sorted = np.sort(conc_obs)
    ys_cens = np.linspace(0, 1, len(xs_cens_sorted), endpoint=True)

    # CDF empírica das substituições L/2 e L/√2
    xs_L2_sorted = np.sort(conc_sub_L2)
    ys_L2 = np.linspace(0, 1, len(xs_L2_sorted), endpoint=True)

    xs_Lsqrt2_sorted = np.sort(conc_sub_Lsqrt2)
    ys_Lsqrt2 = np.linspace(0, 1, len(xs_Lsqrt2_sorted), endpoint=True)

    # CDFs paramétricas (MLE)
    gamma_cdf_mle = gamma_dist.cdf(x_grid, a=shape_mle, scale=1.0 / rate_mle)
    lnorm_cdf_mle = lognorm_dist.cdf(x_grid, s=sdlog_mle, scale=np.exp(meanlog_mle))

    # 6) Plot CDFs
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_grid, true_cdf, "k-", lw=2, label="Verdadeira")

    ax.step(xs_cens_sorted, ys_cens, where="post",
            color="gray", alpha=0.7, label="Dados censurados")

    # Substituições não paramétricas simples
    ax.step(xs_L2_sorted, ys_L2, where="post",
            color="tab:orange", alpha=0.9, linestyle="-",
            label="Substituição L/2")
    ax.step(xs_Lsqrt2_sorted, ys_Lsqrt2, where="post",
            color="tab:green", alpha=0.9, linestyle="--",
            label="Substituição L/√2")

    # Reconstrução piecewise-uniform
    ax.plot(quants_np, percentiles_np, lw=2,
            color="tab:blue", label="Não paramétrica (piecewise)")

    # Ajustes paramétricos (somente MLE)
    ax.plot(x_grid, gamma_cdf_mle, lw=2, linestyle="-.",
            color="tab:red", alpha=0.7, label="Gamma (MLE cens.)")
    ax.plot(x_grid, lnorm_cdf_mle, lw=2,
            linestyle=(0, (3, 5, 1, 5)), color="tab:purple", alpha=0.7,
            label="Lognormal (MLE cens.)")

    ax.set_xlabel("Concentração")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"CDF reconstruída a partir de dados censurados "
        f"({dist_type}, {int(censor_fraction*100)}% cens.)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    cens_label = int(round(censor_fraction * 100))
    out_name = f"CDF_{dist_type.lower()}_cens{cens_label}.png"
    fig.savefig(out_name, dpi=300, bbox_inches="tight")
    print(f"[{dist_type}] Saved CDF plot to {out_name}")

    plt.show()
    plt.close(fig)

    # 7) Amostras grandes de cada modelo para métricas
    n_eval = 20000
    if dist_type.lower() == "gamma":
        true_sample = gamma_dist.rvs(a=params_true[0], scale=1.0 / params_true[1],
                                     size=n_eval, random_state=rng)
    else:
        true_sample = lognorm_dist.rvs(s=params_true[1], scale=np.exp(params_true[0]),
                                       size=n_eval, random_state=rng)

    gamma_mle_sample = gamma_dist.rvs(a=shape_mle, scale=1.0 / rate_mle,
                                      size=n_eval, random_state=rng)
    lnorm_mle_sample = lognorm_dist.rvs(s=sdlog_mle, scale=np.exp(meanlog_mle),
                                        size=n_eval, random_state=rng)

    # Amostras grandes baseadas nas distribuições empíricas de substituição
    sub_L2_sample = rng.choice(conc_sub_L2, size=n_eval, replace=True)
    sub_Lsqrt2_sample = rng.choice(conc_sub_Lsqrt2, size=n_eval, replace=True)

    # 8) Métricas
    metrics_df = compute_metrics_all_models(
        dist_type=dist_type,
        censor_frac=censor_fraction,
        x_true_sample=true_sample,
        sim_np=sim_conc_np,
        gamma_mle_sample=gamma_mle_sample,
        lnorm_mle_sample=lnorm_mle_sample,
        sub_L2_sample=sub_L2_sample,
        sub_Lsqrt2_sample=sub_Lsqrt2_sample,
    )
    return metrics_df


# ============================================================
#  Rodar múltiplos níveis de censura (20, 40, 60, 80, 95%)
# ============================================================

def run_multiple_censoring_levels(dist_type: str,
                                  n: int = 5000,
                                  censor_levels=(0.60, 0.80, 0.95, 0.98, 0.99)):
    """
    Roda o pipeline para cada nível de censura e consolida as métricas em CSV.
    """
    all_metrics = []
    for c in censor_levels:
        print("\n" + "=" * 60)
        print(f"Executando {dist_type} com censura = {int(c * 100)}%")
        print("=" * 60)
        metrics_df = run_pipeline_for_distribution(dist_type, n=n, censor_fraction=c)
        all_metrics.append(metrics_df)

    if all_metrics:
        res = pd.concat(all_metrics, ignore_index=True)
        out_csv = f"metrics_{dist_type.lower()}.csv"
        res.to_csv(out_csv, index=False, sep=";")
        print(f"[{dist_type}] métricas salvas em {out_csv}")


# ============================================================
#  Execução principal
# ============================================================

if __name__ == "__main__":
    # Gamma
    run_multiple_censoring_levels("gamma")

    # Lognormal
    run_multiple_censoring_levels("lognormal")
