import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path

BASE = Path("/Users/aniket/github/synthetic_data_generator/generated_data")

CHART_FONT = dict(size=15)


def discover_groups(base):
    groups = {}
    for folder in sorted(base.rglob("*")):
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.parquet"))
        if files:
            groups[folder.name] = {f.stem: f for f in files}
    return groups


@st.cache_data
def parquet_time_col(path):
    schema = pq.read_schema(str(path))
    timestamp_fields = [f.name for f in schema if pa.types.is_timestamp(f.type)]
    preferred = ["time", "timestamp"]
    for name in preferred:
        if name in timestamp_fields:
            return name
    return timestamp_fields[0] if timestamp_fields else None


@st.cache_data
def time_bounds(path, col):
    s = pd.read_parquet(path, columns=[col])[col]
    return s.min(), s.max()


@st.cache_data
def load(path, time_col=None, start=None, end=None):
    if time_col and start and end:
        filters = [(time_col, ">=", pd.Timestamp(start)), (time_col, "<=", pd.Timestamp(end))]
        return pd.read_parquet(path, filters=filters)
    return pd.read_parquet(path)


def chart(fig, height=500):
    fig.update_layout(height=height, font=CHART_FONT, legend=dict(font=CHART_FONT))
    st.plotly_chart(fig, use_container_width=True)


def table_height(n_rows, max_height=600):
    return min(max_height, 40 * n_rows + 46)


def numeric_cols(df):
    return df.select_dtypes("number").columns.tolist()


def categorical_cols(df):
    return df.select_dtypes(["object", "category", "bool"]).columns.tolist()


def datetime_cols(df):
    return df.select_dtypes("datetime").columns.tolist()


def view_overview(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Numeric", len(numeric_cols(df)))
    c4.metric("Null rate", f"{df.isnull().mean().mean() * 100:.1f}%")

    st.dataframe(
        pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "nulls": df.isnull().sum(),
            "null %": (df.isnull().mean() * 100).round(1),
            "unique": df.nunique(),
        }),
        use_container_width=True,
        height=table_height(len(df.columns)),
    )


def view_sample(df):
    n = st.slider("Rows", 5, 100, 10)
    sample = df.head(n)
    st.dataframe(sample, use_container_width=True, height=table_height(len(sample)))


def view_distributions(df):
    nums = numeric_cols(df)
    cats = categorical_cols(df)

    if nums:
        col = st.selectbox("Numeric column", nums)
        chart(px.histogram(df, x=col, nbins=50, template="simple_white"))

    if cats:
        col = st.selectbox("Categorical column", cats)
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        chart(px.bar(counts, x=col, y="count", template="simple_white"))


def view_nulls(df):
    summary = pd.DataFrame({
        "null_count": df.isnull().sum(),
        "null_pct": (df.isnull().mean() * 100).round(1),
    }).query("null_count > 0")

    if summary.empty:
        st.success("No nulls.")
        return

    chart(px.bar(summary.reset_index(), x="index", y="null_pct", template="simple_white"))
    st.dataframe(summary, use_container_width=True, height=table_height(len(summary)))


def view_correlations(df):
    nums = numeric_cols(df)
    if len(nums) < 2:
        st.info("Need at least two numeric columns.")
        return
    corr = df[nums].corr()
    h = max(600, 60 * len(nums))
    chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template="simple_white"), height=h)


def view_time_series(df):
    dt_cols = datetime_cols(df)
    nums = numeric_cols(df)
    if not dt_cols or not nums:
        st.info("No datetime/numeric column pair found.")
        return

    time_col = st.selectbox("Time column", dt_cols)
    group_col = st.selectbox("Group by", ["None"] + categorical_cols(df))

    if group_col != "None":
        all_values = sorted(df[group_col].dropna().unique().tolist())
        selected = st.multiselect("Filter", all_values, default=all_values)
        df = df[df[group_col].isin(selected)]

    c1, c2 = st.columns(2)
    y1_col = c1.selectbox("Left axis", nums, index=0)
    y2_col = c2.selectbox("Right axis", ["None"] + nums, index=0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    groups = df[group_col].unique() if group_col != "None" else [None]
    colors_left = px.colors.qualitative.Plotly
    colors_right = px.colors.qualitative.Pastel

    for i, grp in enumerate(groups):
        subset = df[df[group_col] == grp] if grp is not None else df
        name = str(grp) if grp is not None else y1_col

        fig.add_trace(
            go.Scatter(x=subset[time_col], y=subset[y1_col],
                       name=f"{name} ({y1_col})",
                       line=dict(color=colors_left[i % len(colors_left)])),
            secondary_y=False,
        )
        if y2_col != "None":
            fig.add_trace(
                go.Scatter(x=subset[time_col], y=subset[y2_col],
                           name=f"{name} ({y2_col})",
                           line=dict(color=colors_right[i % len(colors_right)])),
                secondary_y=True,
            )

    fig.update_layout(height=700, font=CHART_FONT, legend=dict(font=CHART_FONT), template="simple_white")
    fig.update_yaxes(title_text=y1_col, secondary_y=False)
    if y2_col != "None":
        fig.update_yaxes(title_text=y2_col, secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


def view_scatter(df):
    nums = numeric_cols(df)
    if len(nums) < 2:
        st.info("Need at least two numeric columns.")
        return

    x_col = st.selectbox("X", nums)
    y_col = st.selectbox("Y", nums, index=min(1, len(nums) - 1))
    color_col = st.selectbox("Color by", ["None"] + categorical_cols(df))

    chart(px.scatter(df, x=x_col, y=y_col, color=color_col if color_col != "None" else None, opacity=0.5, template="simple_white"))


def view_stats(df):
    st.dataframe(df.describe(include="all").T, use_container_width=True, height=table_height(len(df.columns)))


VIEWS = {
    "Overview": view_overview,
    "Sample": view_sample,
    "Distributions": view_distributions,
    "Nulls": view_nulls,
    "Correlations": view_correlations,
    "Time series": view_time_series,
    "Scatter": view_scatter,
    "Stats": view_stats,
}


def sidebar_date_filter(path):
    schema = pq.read_schema(str(path))
    time_cols = [f.name for f in schema if pa.types.is_timestamp(f.type)]
    if not time_cols:
        return None, None, None

    st.divider()
    time_col = st.selectbox("Filter by", time_cols)
    mn, mx = time_bounds(path, time_col)

    default_start = (mn - pd.Timedelta(days=1)).date()
    default_end = (mx + pd.Timedelta(days=1)).date()

    st.divider()
    st.caption(f"Filter by `{time_col}`")
    date_range = st.date_input(
        "Date range",
        value=(default_start, default_end),
        min_value=default_start,
        max_value=default_end,
    )

    if len(date_range) != 2:
        return time_col, None, None

    start, end = date_range
    return time_col, start, end


def main():
    st.set_page_config(page_title="Dataset Explorer", layout="wide")
    st.title("Dataset Explorer")
    st.markdown("""
        <style>
            [data-testid="stDataFrame"] * { font-size: 18px !important; }
            [data-testid="stSidebar"] * { font-size: 18px !important; }
            [data-testid="stSidebar"] label { font-size: 18px !important; }
        </style>
    """, unsafe_allow_html=True)

    groups = discover_groups(BASE)

    with st.sidebar:
        st.header("Select dataset")
        group = st.selectbox("Group", list(groups.keys()), format_func=lambda k: k.replace("_", " "))
        dataset = st.selectbox("Dataset", list(groups[group].keys()), format_func=lambda k: k.replace("_", " "))
        st.divider()
        view = st.radio("View", list(VIEWS.keys()))

        path = groups[group][dataset]
        time_col, start, end = sidebar_date_filter(path)

    df = load(path, time_col, start, end)
    st.caption(f"{group} / {dataset}" + (f" | {start} to {end}" if time_col else ""))
    VIEWS[view](df)


main()