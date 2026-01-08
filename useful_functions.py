import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu


def get_response_time(web_logs, begin_date, end_date) -> pd.DataFrame:
    """
    Возвращает load_time пользователей за указанный период:
    типа begin_date = pd.to_datetime("23.03.2022", dayfirst=True).date()
         end_date   = pd.to_datetime("30.03.2022", dayfirst=True).date()
    """
    mask = (web_logs["date"].dt.date >= begin_date) & (web_logs["date"].dt.date < end_date)
    result = web_logs.loc[mask, ["date", "user_id", "load_time"]]

    return result


def get_revenue_web(sales, web_logs, begin_date, end_date) -> pd.DataFrame:
    """
    Выручка пользователей, которые заходили на сайт в указанный период
    """
    mask_logs = (web_logs["date"].dt.date >= begin_date) & (web_logs["date"].dt.date < end_date)
    users = web_logs.loc[mask_logs, "user_id"].unique()

    mask_sales = (sales["date"].dt.date >= begin_date) & (sales["date"].dt.date < end_date)
    sales_period = sales.loc[mask_sales]

    revenue = (
        sales_period
        .groupby("user_id", as_index=False)["price"].sum()
        .rename(columns ={"price": "total_revenue_web"}))

    result = pd.DataFrame({"user_id": users}) \
        .merge(revenue, on="user_id", how="left") \
        .fillna(0)

    return result


def get_revenue_all(sales, web_logs, begin_date, end_date) -> pd.DataFrame:
    """
    Выручка всех пользователей, которые заходили 
    и не захходили на сайт до end_date
    """
    users = web_logs.loc[web_logs["date"].dt.date < end_date, "user_id"].unique()
    mask_sales = (sales["date"].dt.date >= begin_date) & (sales["date"].dt.date < end_date)

    sales_period = sales.loc[mask_sales]

    revenue = (
        sales_period
        .groupby("user_id", as_index=False)["price"]
        .sum()
        .rename(columns={"price": "total_revenue"})
        .reset_index())
    
    result = pd.DataFrame({"user_id": users}) \
        .merge(revenue, on="user_id", how="left") \
        .fillna(0)

    return result

def get_data_subset(df, begin_date, end_date, user_ids=None, columns=None) -> pd.DataFrame:
    """
    Возвращает подмножество данных, необходимых для выполнения тестирования
    begin_date = pd.to_datetime("23.03.2022", dayfirst=True).date()
    end_date   = pd.to_datetime("30.03.2022", dayfirst=True).date()
    """
    df_res = df.copy()

    if begin_date is not None:
        df_res = df_res[df_res["date"].dt.date >= begin_date]

    if end_date is not None:
        df_res = df_res[df_res["date"].dt.date < end_date]

    if user_ids is not None:
        df_res = df_res[df_res["user_id"].isin(user_ids)]

    if columns is not None:
        df_res = df_res[columns]

    df_res["date"] = df_res["date"].dt.date

    return df_res


def get_avg_load_time_per_user(web_logs, begin_date, end_date) -> pd.DataFrame:
    """
    Средний load_time на пользователя за период.
    """
    logs = get_response_time(web_logs, begin_date, end_date) 
    out = (logs.groupby("user_id", as_index=False)["load_time"]
                .mean()
                .rename(columns={"load_time": "avg_load_time"}))
    return out


def get_orders_per_user(sales, begin_date, end_date) -> pd.DataFrame:
    """
    Число заказов (sale_id) на пользователя за период.
    """
    df = get_data_subset(sales, begin_date, end_date, columns=["user_id", "sale_id", "date"])
    out = (df.groupby("user_id", as_index=False)["sale_id"]
             .nunique()
             .rename(columns={"sale_id": "orders_cnt"}))
    return out


def build_experiment_metrics(users, sales, web_logs, begin_date, end_date) -> pd.DataFrame:
    """
    Собирает таблицу уровня user_id для анализа эксперимента:
    user_id, pilot, total_revenue, total_revenue_web, orders_cnt, avg_load_time
    """
    # Метрики за период эксперимента:
    rev_all = get_revenue_all(sales, web_logs, begin_date, end_date)     # основная метрика  
    rev_web = get_revenue_web(sales, web_logs, begin_date, end_date)     # вспомогательная метрика   
    orders = get_orders_per_user(sales, begin_date, end_date)            # вспомогательная метрика
    load = get_avg_load_time_per_user(web_logs, begin_date, end_date)   # контрольная метрика

    df = users[["user_id", "pilot"]].copy()

    df = df.merge(rev_all, on="user_id", how="left").fillna({"total_revenue": 0})
    df = df.merge(rev_web, on="user_id", how="left").fillna({"total_revenue_web": 0})
    df = df.merge(orders, on="user_id", how="left").fillna({"orders_cnt": 0})
    df = df.merge(load, on="user_id", how="left") 
    df["avg_load_time"] = df["avg_load_time"].fillna(0)

    return df


def ttest_pvalue(x_control, x_pilot):
    x_control = np.asarray(x_control)
    x_pilot = np.asarray(x_pilot)
    x_control = x_control[~np.isnan(x_control)]
    x_pilot = x_pilot[~np.isnan(x_pilot)]
    _, p = stats.ttest_ind(x_control, x_pilot, equal_var=False)
    return float(p)


def calculate_theta(metric_control, metric_pilot, cov_control, cov_pilot):
    """
    Theta = cov(metric, cov) / var(cov) по объединённым данным.
    """
    m = pd.concat([metric_control, metric_pilot], ignore_index=True)
    c = pd.concat([cov_control, cov_pilot], ignore_index=True)

    var_c = np.var(c, ddof=1)
    if var_c == 0 or np.isnan(var_c):
        return 0.0
    return float(np.cov(m, c, ddof=1)[0, 1] / var_c)

from datetime import date

def build_covariates_for_users(users, sales, web_logs, exp_begin, exp_end, cov_days=(7, 28)):
    """
    Возвращает df: user_id, pilot, cov_7d, cov_28d (выручка all за пред-периоды).
    """
    base = users[["user_id", "pilot"]].copy()

    for d in cov_days:
        cov_begin = (pd.to_datetime(exp_begin) - pd.Timedelta(days=d)).date()
        cov_end = pd.to_datetime(exp_begin).date()

        cov_df = get_revenue_all(sales, web_logs, cov_begin, cov_end)[["user_id", "total_revenue"]].copy()
        cov_df = cov_df.rename(columns={"total_revenue": f"cov_{d}d"})
        base = base.merge(cov_df, on="user_id", how="left").fillna({f"cov_{d}d": 0})

    return base


def cuped_pvalue(df, metric_col, cov_col, group_col="pilot"):
    """
    CUPED t-test:
      metric_cuped = metric - theta * cov
    """
    df2 = df[[group_col, metric_col, cov_col]].dropna()

    control = df2[df2[group_col] == 0]
    pilot = df2[df2[group_col] == 1]

    theta = calculate_theta(
        control[metric_col].reset_index(drop=True),
        pilot[metric_col].reset_index(drop=True),
        control[cov_col].reset_index(drop=True),
        pilot[cov_col].reset_index(drop=True),
        )

    m_c = control[metric_col] - theta * control[cov_col]
    m_p = pilot[metric_col] - theta * pilot[cov_col]

    return ttest_pvalue(m_c, m_p)


def plot_pvalue_ecdf_hist(pvalues, title=None):
    pvalues = np.asarray(pvalues)
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=pvalues,
            histnorm="probability density",
            nbinsx=20,
            name="p-value гистограмма",
            opacity=0.7
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[1, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Uniform density"
        )
    )

    fig.update_layout(
        title=title or "p-value распределение",
        xaxis_title="p-value",
        yaxis_title="Density",
        bargap=0.1,
        height=400,
        width=700
    )
    fig.show()


def plot_pvalue_ecdf_prob(pvalues, title=None):
    pvalues = np.sort(np.asarray(pvalues))
    y = np.arange(1, len(pvalues) + 1) / len(pvalues)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pvalues,
            y=y,
            mode="lines",
            name="ECDF"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="y = x"
        )
    )

    fig.update_layout(
        title=title or "p-value ECDF",
        xaxis_title="p-value",
        yaxis_title="Probability",
        height=400,
        width=700
    )
    fig.show()

def aa_pvalues(metric_series: pd.Series, n_iter=1000, seed=42):
    x = metric_series.dropna().to_numpy()
    n = len(x)

    if n < 20:
        raise ValueError(f"Слишком мало наблюдений: {n}")

    rng = np.random.default_rng(seed)
    pvalues = []

    half = n // 2
    for _ in range(n_iter):
        idx = rng.permutation(n)
        a = x[idx[:half]]
        b = x[idx[half:2 * half]]

        _, p = ttest_ind(a, b, equal_var=False)
        pvalues.append(p)

    return np.array(pvalues)


def describe_metric(s: pd.Series, name: str) -> pd.DataFrame:
    s = s.dropna()

    desc = pd.DataFrame({
        "metric": [name],
        "n_users": [s.shape[0]],
        "mean": [s.mean()],
        "std": [s.std(ddof=1)],
        "median": [s.median()],
        "p75": [s.quantile(0.75)],
        "p90": [s.quantile(0.90)],
        "p95": [s.quantile(0.95)],
        "p99": [s.quantile(0.99)],
        "share_zero": [(s == 0).mean() if (s == 0).any() else 0.0]
    })

    return desc


def plot_metric_hist(s: pd.Series, title: str, log1p=False):
    x = s.dropna().to_numpy()
    if log1p:
        x = np.log1p(x)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, nbinsx=50, opacity=0.8))
    fig.update_layout(title=title, xaxis_title=("log1p(metric)" if log1p else "metric"),
                      yaxis_title="count", height=380, width=720)
    fig.show()

def get_mde(std, sample_size_per_group, alpha=0.05, beta=0.2, two_sided=True):
    alpha_eff = alpha / 2 if two_sided else alpha
    z_alpha = norm.ppf(1 - alpha_eff)
    z_beta = norm.ppf(1 - beta)
    mde = (z_alpha + z_beta) * (np.sqrt(2) * std) / np.sqrt(sample_size_per_group)
    return float(mde)

def check_ttest(a, b, alpha=0.05):
    _, p = ttest_ind(a, b, equal_var=False)
    return p < alpha

def run_experiment_pvalues(
    baseline: pd.Series,
    n_per_group: int,
    n_iter: int = 2000,
    alpha: float = 0.05,
    effect: float = 0.0,              
    transform: str = "none",          # "none" | "log1p"
    test: str = "ttest",              # "ttest" | "mannwhitney"
    alternative: str = "two-sided",   # для mannwhitney: "two-sided", "less", "greater"
    seed: int = 42,
    return_pvalues: bool = True):
    x = baseline.dropna().to_numpy()

    if transform == "log1p":
        x = np.log1p(x)
    elif transform != "none":
        raise ValueError("transform должен быть 'none' или 'log1p'")

    n = len(x)
    if n < 2 * n_per_group:
        pass

    rng = np.random.default_rng(seed)
    pvalues = []
    rejects = 0

    for _ in range(n_iter):
        a = rng.choice(x, size=n_per_group, replace=True)
        b = rng.choice(x, size=n_per_group, replace=True) + effect

        if test == "ttest":
            _, p = ttest_ind(a, b, equal_var=False)
        elif test == "mannwhitney":
            _, p = mannwhitneyu(a, b, alternative=alternative)

        pvalues.append(p)
        rejects += (p < alpha)

    pvalues = np.array(pvalues)
    reject_rate = rejects / n_iter

    if return_pvalues:
        return pvalues, reject_rate
    return reject_rate
