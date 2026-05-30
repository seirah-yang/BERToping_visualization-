"""RBQM topic modeling, gap, network, and text-mining analysis.

The script reads the CSV outputs from the RBQM topic-modeling pipeline and
exports all derived tables to a csv folder and all figures to a png folder.
It is written so the same functions can be used from a notebook.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact


REQUIRED_FILES = {
    "topic_info": "rbqm_bertopic_topic_info.csv",
    "document_node_membership": "rbqm_document_node_membership.csv",
    "documents": "rbqm_documents_with_topics.csv",
    "edge_temporal": "rbqm_edge_temporal_evolution.csv",
    "node_growth": "rbqm_node_growth_summary.csv",
    "node_temporal": "rbqm_node_temporal_evolution.csv",
    "semantic_edges": "rbqm_semantic_edges.csv",
    "semantic_nodes": "rbqm_semantic_nodes.csv",
    "topic_interpretation": "rbqm_topic_interpretation.csv",
    "topic_temporal": "rbqm_topic_temporal_evolution.csv",
    "vosviewer": "vosviewer_cooccurrence.csv",
}

BLOCK_ORDER = ["Before 2015", "2015-2019", "2020-2023", "2024-current"]
PRE_BLOCKS = {"Before 2015", "2015-2019"}
POST_BLOCKS = {"2020-2023", "2024-current"}
PERIOD_ORDER = ["Pre-COVID", "Post-COVID"]

CATEGORY_COLORS = {
    "RBQM": "#2F6F73",
    "AI": "#7B4EA3",
    "Data": "#D97904",
    "Governance": "#3F5F9F",
    "Monitoring": "#0F8B8D",
    "Clinical_Operations": "#6F8F3A",
    "Risk": "#B23A48",
    "DCT": "#8A6F3D",
    "Statistics": "#555555",
    "Other": "#8F8F8F",
}

STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "among",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "can",
    "clinical",
    "could",
    "data",
    "did",
    "does",
    "doing",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "into",
    "its",
    "itself",
    "may",
    "more",
    "most",
    "not",
    "now",
    "off",
    "only",
    "other",
    "our",
    "out",
    "over",
    "own",
    "quality",
    "same",
    "should",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "trial",
    "trials",
    "under",
    "until",
    "upon",
    "use",
    "used",
    "using",
    "was",
    "were",
    "when",
    "where",
    "which",
    "while",
    "who",
    "will",
    "with",
    "within",
}


@dataclass
class ArtifactRegistry:
    output_root: Path
    csv_dir: Path
    png_dir: Path
    zip_dir: Path
    tables: list[Path] = field(default_factory=list)
    figures: list[Path] = field(default_factory=list)
    zips: list[Path] = field(default_factory=list)


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 260,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def slugify(name: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", str(name), flags=re.UNICODE).strip("_")
    return name or "output"


def default_data_candidates() -> list[Path]:
    home = Path.home()
    return [
        Path.cwd() / "Topic Modeling output table",
        Path.cwd() / "rbqm_semantic_outputs",
        home / "Downloads" / "Network_Gap Analysis of Oversight" / "Topic Modeling output table",
        home / "Downloads" / "Topic Modeling output table",
        Path("/content/drive/MyDrive/AI용 Prompt/Network Analysis/rbqm_semantic_outputs"),
    ]


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    candidates = [Path(data_dir)] if data_dir else default_data_candidates()
    for candidate in candidates:
        if candidate.exists() and all((candidate / name).exists() for name in REQUIRED_FILES.values()):
            return candidate
    checked = "\n".join(f"- {path}" for path in candidates)
    missing = ", ".join(REQUIRED_FILES.values())
    raise FileNotFoundError(
        "Could not locate the topic-modeling CSV folder.\n"
        f"Checked:\n{checked}\nRequired files: {missing}"
    )


def make_registry(output_root: str | Path | None = None) -> ArtifactRegistry:
    root = Path(output_root) if output_root else Path.cwd() / "rbqm_topicmodeling_analysis_outputs"
    registry = ArtifactRegistry(
        output_root=root,
        csv_dir=root / "csv",
        png_dir=root / "png",
        zip_dir=root / "zip",
    )
    registry.csv_dir.mkdir(parents=True, exist_ok=True)
    registry.png_dir.mkdir(parents=True, exist_ok=True)
    registry.zip_dir.mkdir(parents=True, exist_ok=True)
    return registry


def read_csv_safely(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return pd.read_csv(path)


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {key: read_csv_safely(data_dir / filename) for key, filename in REQUIRED_FILES.items()}


def save_table(df: pd.DataFrame | pd.Series, name: str, registry: ArtifactRegistry) -> Path:
    if isinstance(df, pd.Series):
        df = df.reset_index()
    path = registry.csv_dir / f"{slugify(name)}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    if path not in registry.tables:
        registry.tables.append(path)
    return path


def save_figure(fig: plt.Figure, name: str, registry: ArtifactRegistry) -> Path:
    path = registry.png_dir / f"{slugify(name)}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    registry.figures.append(path)
    return path


def create_zip(source_dir: Path, output_path: Path) -> Path:
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))
    return output_path


def zip_outputs(registry: ArtifactRegistry) -> None:
    registry.zips.append(create_zip(registry.csv_dir, registry.zip_dir / "csv_outputs.zip"))
    registry.zips.append(create_zip(registry.png_dir, registry.zip_dir / "png_outputs.zip"))


def copy_source_csvs(data_dir: Path, registry: ArtifactRegistry) -> Path:
    target = registry.csv_dir / "source_inputs"
    target.mkdir(exist_ok=True)
    for path in sorted(data_dir.glob("*.csv")):
        shutil.copy2(path, target / path.name)
    return target


def covid_period_from_block(block: Any) -> str:
    block_text = str(block)
    if block_text in PRE_BLOCKS:
        return "Pre-COVID"
    if block_text in POST_BLOCKS:
        return "Post-COVID"
    return "Excluded"


def covid_period_from_year(year: Any) -> str:
    try:
        year_int = int(year)
    except (TypeError, ValueError):
        return "Unknown"
    return "Pre-COVID" if year_int <= 2019 else "Post-COVID"


def add_period_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "covid_period" in out.columns:
        out["covid_period"] = out["covid_period"].fillna("Unknown")
    elif "time_block" in out.columns:
        out["covid_period"] = out["time_block"].map(covid_period_from_block)
    elif "period" in out.columns:
        out["covid_period"] = out["period"]
    elif "year" in out.columns:
        out["covid_period"] = out["year"].map(covid_period_from_year)
    else:
        out["covid_period"] = "Unknown"

    if "period" in out.columns:
        valid = out["period"].isin(PERIOD_ORDER)
        out.loc[valid, "covid_period"] = out.loc[valid, "period"]
    if "year" in out.columns:
        unknown = ~out["covid_period"].isin(PERIOD_ORDER)
        out.loc[unknown, "covid_period"] = out.loc[unknown, "year"].map(covid_period_from_year)
    return out


def ordered_blocks(values: pd.Series) -> list[str]:
    present = set(values.dropna().astype(str))
    ordered = [block for block in BLOCK_ORDER if block in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def build_file_inventory(data_dir: Path, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for key, filename in REQUIRED_FILES.items():
        df = tables[key]
        path = data_dir / filename
        records.append(
            {
                "table_key": key,
                "file_name": filename,
                "rows": len(df),
                "columns": len(df.columns),
                "file_size_bytes": path.stat().st_size,
                "columns_list": "; ".join(map(str, df.columns)),
            }
        )
    return pd.DataFrame(records)


def build_data_dictionary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for table_name, df in tables.items():
        for col in df.columns:
            non_null = int(df[col].notna().sum())
            records.append(
                {
                    "table": table_name,
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "non_null": non_null,
                    "missing": int(len(df) - non_null),
                    "example": "" if df[col].dropna().empty else str(df[col].dropna().iloc[0])[:160],
                }
            )
    return pd.DataFrame(records)


def get_period_totals(documents: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    docs = add_period_columns(documents)
    period_totals = docs[docs["covid_period"].isin(PERIOD_ORDER)].groupby("covid_period").size().to_dict()
    block_totals = docs.groupby("time_block").size().to_dict() if "time_block" in docs.columns else {}
    return (
        {period: int(period_totals.get(period, 0)) for period in PERIOD_ORDER},
        {block: int(block_totals.get(block, 0)) for block in BLOCK_ORDER},
    )


def safe_share(count: float, total: float) -> float:
    return float(count) / float(total) if total else 0.0


def bh_adjust(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    n = len(valid)
    if n == 0:
        return q
    ranked = valid * n / np.arange(1, n + 1)
    ranked = ranked[::-1].cummin()[::-1].clip(upper=1.0)
    q.loc[ranked.index] = ranked
    return q


def fisher_pvalue(a_count: int, a_total: int, b_count: int, b_total: int) -> float:
    a_other = max(a_total - a_count, 0)
    b_other = max(b_total - b_count, 0)
    if a_total == 0 or b_total == 0:
        return np.nan
    try:
        _, p_value = fisher_exact([[a_count, a_other], [b_count, b_other]], alternative="two-sided")
    except ValueError:
        return np.nan
    return float(p_value)


def classify_gap(row: pd.Series) -> str:
    if row["is_new_after_covid"]:
        return "new_post_covid"
    if row["is_disappeared_after_covid"]:
        return "pre_only_or_decreased_to_zero"
    if row["q_value"] <= 0.10 and row["absolute_gap"] > 0:
        return "statistical_upward_signal"
    if row["q_value"] <= 0.10 and row["absolute_gap"] < 0:
        return "statistical_downward_signal"
    if row["large_upward_shift"]:
        return "large_upward_shift"
    if row["large_downward_shift"]:
        return "large_downward_shift"
    return "stable_or_small_change"


def build_period_gap(
    temporal_df: pd.DataFrame,
    id_cols: list[str],
    period_totals: dict[str, int],
    value_col: str = "n_docs",
) -> pd.DataFrame:
    df = add_period_columns(temporal_df)
    df = df[df["covid_period"].isin(PERIOD_ORDER)].copy()
    if df.empty:
        empty_cols = (
            id_cols
            + [
                "pre_n_docs",
                "post_n_docs",
                "pre_total_docs",
                "post_total_docs",
                "pre_share",
                "post_share",
                "absolute_gap",
                "absolute_gap_abs",
                "relative_ratio",
                "log2_ratio_smoothed",
                "is_new_after_covid",
                "is_disappeared_after_covid",
                "large_upward_shift",
                "large_downward_shift",
                "p_value_fisher",
                "q_value",
                "signal_label",
            ]
        )
        return pd.DataFrame(columns=empty_cols)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0).astype(int)

    grouped = df.groupby(id_cols + ["covid_period"], dropna=False, as_index=False)[value_col].sum()
    pivot = grouped.pivot_table(index=id_cols, columns="covid_period", values=value_col, fill_value=0)
    pivot = pivot.reindex(columns=PERIOD_ORDER, fill_value=0).reset_index()
    pivot.columns.name = None

    pivot = pivot.rename(columns={"Pre-COVID": "pre_n_docs", "Post-COVID": "post_n_docs"})
    pivot["pre_total_docs"] = period_totals.get("Pre-COVID", 0)
    pivot["post_total_docs"] = period_totals.get("Post-COVID", 0)
    pivot["pre_share"] = pivot["pre_n_docs"].map(lambda x: safe_share(x, period_totals.get("Pre-COVID", 0)))
    pivot["post_share"] = pivot["post_n_docs"].map(lambda x: safe_share(x, period_totals.get("Post-COVID", 0)))
    pivot["absolute_gap"] = pivot["post_share"] - pivot["pre_share"]
    pivot["absolute_gap_abs"] = pivot["absolute_gap"].abs()
    pivot["relative_ratio"] = (pivot["post_share"] + 0.001) / (pivot["pre_share"] + 0.001)
    pivot["log2_ratio_smoothed"] = np.log2(pivot["relative_ratio"])
    pivot["is_new_after_covid"] = (pivot["pre_n_docs"] == 0) & (pivot["post_n_docs"] > 0)
    pivot["is_disappeared_after_covid"] = (pivot["pre_n_docs"] > 0) & (pivot["post_n_docs"] == 0)

    positive_gaps = pivot.loc[pivot["absolute_gap"] > 0, "absolute_gap"]
    negative_gaps = pivot.loc[pivot["absolute_gap"] < 0, "absolute_gap_abs"]
    upward_threshold = positive_gaps.quantile(0.75) if not positive_gaps.empty else np.inf
    downward_threshold = negative_gaps.quantile(0.75) if not negative_gaps.empty else np.inf
    pivot["large_upward_shift"] = (pivot["absolute_gap"] > 0) & (pivot["absolute_gap_abs"] >= upward_threshold)
    pivot["large_downward_shift"] = (pivot["absolute_gap"] < 0) & (pivot["absolute_gap_abs"] >= downward_threshold)

    pivot["p_value_fisher"] = pivot.apply(
        lambda row: fisher_pvalue(
            int(row["pre_n_docs"]),
            int(row["pre_total_docs"]),
            int(row["post_n_docs"]),
            int(row["post_total_docs"]),
        ),
        axis=1,
    )
    pivot["q_value"] = bh_adjust(pivot["p_value_fisher"])
    pivot["signal_label"] = pivot.apply(classify_gap, axis=1)
    return pivot.sort_values(["absolute_gap_abs", "post_share"], ascending=False).reset_index(drop=True)


def build_adjacent_block_changes(
    temporal_df: pd.DataFrame,
    id_cols: list[str],
    block_totals: dict[str, int],
    value_col: str = "n_docs",
) -> pd.DataFrame:
    df = temporal_df.copy()
    df = df[df["time_block"].isin(BLOCK_ORDER)].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0).astype(int)
    grouped = df.groupby(id_cols + ["time_block"], dropna=False, as_index=False)[value_col].sum()
    pivot = grouped.pivot_table(index=id_cols, columns="time_block", values=value_col, fill_value=0)
    pivot = pivot.reindex(columns=BLOCK_ORDER, fill_value=0).reset_index()

    records = []
    for _, row in pivot.iterrows():
        entity = {col: row[col] for col in id_cols}
        for earlier, later in zip(BLOCK_ORDER[:-1], BLOCK_ORDER[1:]):
            earlier_total = int(block_totals.get(earlier, 0))
            later_total = int(block_totals.get(later, 0))
            earlier_count = int(row.get(earlier, 0))
            later_count = int(row.get(later, 0))
            earlier_share = safe_share(earlier_count, earlier_total)
            later_share = safe_share(later_count, later_total)
            records.append(
                {
                    **entity,
                    "earlier_block": earlier,
                    "later_block": later,
                    "earlier_n_docs": earlier_count,
                    "later_n_docs": later_count,
                    "earlier_total_docs": earlier_total,
                    "later_total_docs": later_total,
                    "earlier_share": earlier_share,
                    "later_share": later_share,
                    "share_delta": later_share - earlier_share,
                    "share_delta_abs": abs(later_share - earlier_share),
                    "new_in_later_block": earlier_count == 0 and later_count > 0,
                    "disappeared_in_later_block": earlier_count > 0 and later_count == 0,
                    "p_value_fisher": fisher_pvalue(earlier_count, earlier_total, later_count, later_total),
                }
            )
    out = pd.DataFrame(records)
    out["q_value"] = bh_adjust(out["p_value_fisher"])
    out["exploratory_signal"] = (
        (out["q_value"] <= 0.10)
        | (out["share_delta_abs"] >= 0.20)
        | out["new_in_later_block"]
        | out["disappeared_in_later_block"]
    )
    return out.sort_values(["exploratory_signal", "share_delta_abs"], ascending=False).reset_index(drop=True)


def build_category_gap_from_membership(
    membership: pd.DataFrame,
    documents: pd.DataFrame,
    period_totals: dict[str, int],
) -> pd.DataFrame:
    docs = add_period_columns(documents)
    doc_period = docs[["doc_uid", "covid_period"]].drop_duplicates()
    work = membership.merge(doc_period, on="doc_uid", how="left")
    work = work[work["covid_period"].isin(PERIOD_ORDER)].copy()
    work = work.drop_duplicates(["doc_uid", "category", "covid_period"])
    grouped = work.groupby(["category", "covid_period"], as_index=False).size()
    grouped = grouped.rename(columns={"size": "n_docs"})
    return build_period_gap(grouped, ["category"], period_totals, value_col="n_docs")


def enrich_gap_tables(tables: dict[str, pd.DataFrame], gaps: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    node_meta_cols = [
        "node",
        "category",
        "layer",
        "type",
        "doc_frequency",
        "mention_frequency",
        "degree_strength",
        "degree_centrality",
        "betweenness_centrality",
        "eigenvector_centrality",
        "community",
    ]
    node_meta = tables["semantic_nodes"][[c for c in node_meta_cols if c in tables["semantic_nodes"].columns]].drop_duplicates("node")
    if not node_meta.empty:
        gaps["node_gap"] = gaps["node_gap"].merge(node_meta, on="node", how="left")

    edge_meta_cols = ["source", "target", "weight", "first_year", "last_year", "source_category", "target_category"]
    edge_meta = tables["semantic_edges"][[c for c in edge_meta_cols if c in tables["semantic_edges"].columns]].drop_duplicates(
        ["source", "target"]
    )
    if not edge_meta.empty:
        gaps["edge_gap"] = gaps["edge_gap"].merge(edge_meta, on=["source", "target"], how="left")

    topic_meta_cols = [
        "topic",
        "semantic_label",
        "topic_words",
        "top_nodes",
        "top_categories",
        "year_min",
        "year_max",
        "example_titles",
    ]
    topic_meta = tables["topic_interpretation"][
        [c for c in topic_meta_cols if c in tables["topic_interpretation"].columns]
    ].drop_duplicates("topic")
    if not topic_meta.empty:
        gaps["topic_gap"] = gaps["topic_gap"].merge(topic_meta, on="topic", how="left")
    return gaps


def build_network_metrics(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    graph = nx.Graph()
    for _, row in nodes.iterrows():
        graph.add_node(row["node"], **row.to_dict())
    for _, row in edges.iterrows():
        weight = float(row.get("weight", 1) or 1)
        attrs = row.to_dict()
        attrs.pop("weight", None)
        graph.add_edge(row["source"], row["target"], weight=weight, **attrs)

    if graph.number_of_nodes() == 0:
        return pd.DataFrame()

    degree_centrality = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, weight=None)
    try:
        eigenvector = nx.eigenvector_centrality_numpy(graph, weight="weight")
    except Exception:
        eigenvector = {node: np.nan for node in graph.nodes}
    strength = dict(graph.degree(weight="weight"))

    try:
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
        community_lookup = {node: idx for idx, community in enumerate(communities) for node in community}
    except Exception:
        community_lookup = {node: np.nan for node in graph.nodes}

    records = []
    for node, attrs in graph.nodes(data=True):
        records.append(
            {
                "node": node,
                "category": attrs.get("category", "Other"),
                "doc_frequency": attrs.get("doc_frequency", np.nan),
                "mention_frequency": attrs.get("mention_frequency", np.nan),
                "degree_strength_recomputed": strength.get(node, 0),
                "degree_centrality_recomputed": degree_centrality.get(node, np.nan),
                "betweenness_centrality_recomputed": betweenness.get(node, np.nan),
                "eigenvector_centrality_recomputed": eigenvector.get(node, np.nan),
                "community_recomputed": community_lookup.get(node, np.nan),
            }
        )
    return pd.DataFrame(records).sort_values("degree_strength_recomputed", ascending=False).reset_index(drop=True)


def split_terms(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value)
    if text.strip().startswith("["):
        try:
            parsed = ast.literal_eval(text)
            return [str(item).strip() for item in parsed if str(item).strip()]
        except (SyntaxError, ValueError):
            pass
    delimiter = ";" if ";" in text else ","
    return [part.strip().strip("'\"") for part in text.split(delimiter) if part.strip()]


def build_topic_keyword_long(topic_interpretation: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in topic_interpretation.iterrows():
        for field in ["topic_words", "top_nodes", "top_categories"]:
            for rank, term in enumerate(split_terms(row.get(field)), start=1):
                records.append(
                    {
                        "topic": row.get("topic"),
                        "semantic_label": row.get("semantic_label"),
                        "field": field,
                        "rank": rank,
                        "term": term,
                    }
                )
    return pd.DataFrame(records)


def build_topic_node_matrix(membership: pd.DataFrame) -> pd.DataFrame:
    if not {"topic", "node", "doc_uid"}.issubset(membership.columns):
        return pd.DataFrame()
    work = membership.drop_duplicates(["topic", "node", "doc_uid"])
    matrix = pd.pivot_table(work, index="topic", columns="node", values="doc_uid", aggfunc="count", fill_value=0)
    return matrix.reset_index()


def tokenize(text: Any) -> list[str]:
    if pd.isna(text):
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z_]{2,}", str(text).lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def build_text_mining_tables(documents: pd.DataFrame, period_totals: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    docs = add_period_columns(documents)
    text_col = "doc_processed" if "doc_processed" in docs.columns else "combined_text"
    docs["tokens"] = docs[text_col].map(tokenize)

    rows = []
    top_rows = []
    for period in PERIOD_ORDER:
        period_docs = docs[docs["covid_period"] == period]
        token_counter: Counter[str] = Counter()
        doc_counter: Counter[str] = Counter()
        for tokens in period_docs["tokens"]:
            token_counter.update(tokens)
            doc_counter.update(set(tokens))
        for term, count in token_counter.most_common(50):
            top_rows.append(
                {
                    "covid_period": period,
                    "term": term,
                    "token_count": int(count),
                    "doc_count": int(doc_counter.get(term, 0)),
                    "doc_share": safe_share(doc_counter.get(term, 0), period_totals.get(period, 0)),
                }
            )

    all_terms = sorted({row["term"] for row in top_rows})
    for term in all_terms:
        record: dict[str, Any] = {"term": term}
        for period in PERIOD_ORDER:
            period_docs = docs[docs["covid_period"] == period]
            token_count = 0
            doc_count = 0
            for tokens in period_docs["tokens"]:
                token_count += tokens.count(term)
                doc_count += int(term in set(tokens))
            record[f"{period}_token_count"] = token_count
            record[f"{period}_doc_count"] = doc_count
            record[f"{period}_doc_share"] = safe_share(doc_count, period_totals.get(period, 0))
        record["doc_share_gap"] = record["Post-COVID_doc_share"] - record["Pre-COVID_doc_share"]
        record["log2_ratio_smoothed"] = math.log2((record["Post-COVID_doc_share"] + 0.001) / (record["Pre-COVID_doc_share"] + 0.001))
        record["p_value_fisher"] = fisher_pvalue(
            int(record["Pre-COVID_doc_count"]),
            period_totals.get("Pre-COVID", 0),
            int(record["Post-COVID_doc_count"]),
            period_totals.get("Post-COVID", 0),
        )
        rows.append(record)

    gap = pd.DataFrame(rows)
    if not gap.empty:
        gap["q_value"] = bh_adjust(gap["p_value_fisher"])
        gap = gap.sort_values("doc_share_gap", ascending=False).reset_index(drop=True)
    return gap, pd.DataFrame(top_rows)


def wrap_label(text: Any, width: int = 22) -> str:
    words = str(text).replace("_", " ").split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def plot_document_year_counts(documents: pd.DataFrame, registry: ArtifactRegistry) -> Path:
    docs = add_period_columns(documents)
    counts = docs.groupby(["year", "covid_period"], as_index=False).size().rename(columns={"size": "n_docs"})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=counts, x="year", y="n_docs", hue="covid_period", palette=["#4C78A8", "#F58518"], ax=ax)
    ax.set_title("Document Counts by Year and COVID Period")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Documents")
    ax.legend(title="")
    return save_figure(fig, "document_counts_by_year", registry)


def plot_topic_heatmap(topic_temporal: pd.DataFrame, topic_interpretation: pd.DataFrame, registry: ArtifactRegistry) -> Path:
    data = topic_temporal.copy()
    pivot = data.pivot_table(index="topic", columns="time_block", values="doc_share", fill_value=0)
    pivot = pivot.reindex(columns=[block for block in BLOCK_ORDER if block in pivot.columns])
    labels = topic_interpretation.set_index("topic")["semantic_label"].to_dict()
    pivot.index = [f"T{idx}: {wrap_label(labels.get(idx, idx), 24)}" for idx in pivot.index]
    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.55 * len(pivot) + 2)))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"label": "doc share"}, ax=ax)
    ax.set_title("Topic Temporal Evolution")
    ax.set_xlabel("Time block")
    ax.set_ylabel("Topic")
    return save_figure(fig, "topic_temporal_heatmap", registry)


def plot_gap_bar(
    gap_df: pd.DataFrame,
    label_col: str,
    title: str,
    file_name: str,
    registry: ArtifactRegistry,
    top_n: int = 18,
) -> Path:
    if gap_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        return save_figure(fig, file_name, registry)
    plot_df = gap_df.nlargest(top_n, "absolute_gap_abs").copy()
    plot_df["label"] = plot_df[label_col].map(lambda x: wrap_label(x, 28))
    plot_df = plot_df.sort_values("absolute_gap")
    colors = np.where(plot_df["absolute_gap"] >= 0, "#C44E52", "#4C78A8")
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.38)))
    ax.barh(plot_df["label"], plot_df["absolute_gap"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Post-COVID share minus Pre-COVID share")
    ax.set_ylabel("")
    return save_figure(fig, file_name, registry)


def plot_node_scatter(node_gap: pd.DataFrame, registry: ArtifactRegistry) -> Path:
    df = node_gap.copy()
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    categories = sorted(df.get("category", pd.Series(["Other"])).fillna("Other").unique())
    for category in categories:
        part = df[df.get("category", "Other").fillna("Other") == category]
        ax.scatter(
            part["pre_share"],
            part["post_share"],
            s=70 + 400 * part["absolute_gap_abs"].clip(upper=0.5),
            alpha=0.78,
            label=category,
            color=CATEGORY_COLORS.get(category, "#8F8F8F"),
            edgecolor="white",
            linewidth=0.6,
        )
    lim = max(float(df[["pre_share", "post_share"]].max().max()), 0.05) + 0.05
    ax.plot([0, lim], [0, lim], "--", color="#777777", linewidth=1)
    highlight = df[df["signal_label"].ne("stable_or_small_change")].nlargest(12, "absolute_gap_abs")
    for _, row in highlight.iterrows():
        ax.annotate(wrap_label(row["node"], 16), (row["pre_share"], row["post_share"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.set_title("Node Share: Pre-COVID vs Post-COVID")
    ax.set_xlabel("Pre-COVID document share")
    ax.set_ylabel("Post-COVID document share")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    return save_figure(fig, "node_pre_post_scatter", registry)


def plot_category_gap(category_gap: pd.DataFrame, registry: ArtifactRegistry) -> Path:
    df = category_gap.sort_values("absolute_gap").copy()
    colors = np.where(df["absolute_gap"] >= 0, "#C44E52", "#4C78A8")
    fig, ax = plt.subplots(figsize=(8.5, max(4, len(df) * 0.42)))
    ax.barh(df["category"].map(wrap_label), df["absolute_gap"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Category-Level Gap by COVID Period")
    ax.set_xlabel("Post-COVID share minus Pre-COVID share")
    ax.set_ylabel("")
    return save_figure(fig, "category_pre_post_gap", registry)


def plot_network(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    registry: ArtifactRegistry,
    name: str,
    title: str,
    weight_col: str = "weight",
    max_edges: int = 45,
) -> Path:
    edge_df = edges.copy()
    if edge_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No edges matched this filter", ha="center", va="center")
        ax.axis("off")
        return save_figure(fig, name, registry)

    if weight_col not in edge_df.columns:
        edge_df[weight_col] = edge_df.get("post_share", 1)
    edge_df = edge_df.sort_values(weight_col, ascending=False).head(max_edges)
    node_meta = nodes.set_index("node").to_dict("index") if "node" in nodes.columns else {}
    graph = nx.Graph()
    for _, row in edge_df.iterrows():
        graph.add_edge(row["source"], row["target"], weight=float(row.get(weight_col, 1) or 1))
    for node in graph.nodes:
        graph.nodes[node].update(node_meta.get(node, {}))

    fig, ax = plt.subplots(figsize=(11, 8))
    pos = nx.spring_layout(graph, seed=42, k=0.72)
    weights = np.array([graph[u][v].get("weight", 1.0) for u, v in graph.edges])
    if len(weights):
        edge_widths = 0.8 + 4.0 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    else:
        edge_widths = 1.0

    node_colors = [
        CATEGORY_COLORS.get(str(graph.nodes[node].get("category", "Other")), "#8F8F8F")
        for node in graph.nodes
    ]
    node_sizes = [
        280 + 80 * float(graph.nodes[node].get("doc_frequency", graph.degree(node)) or graph.degree(node))
        for node in graph.nodes
    ]
    nx.draw_networkx_edges(graph, pos, ax=ax, width=edge_widths, edge_color="#A0A0A0", alpha=0.65)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_sizes, node_color=node_colors, edgecolors="white", linewidths=1)
    nx.draw_networkx_labels(graph, pos, labels={node: wrap_label(node, 15) for node in graph.nodes}, font_size=8, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    return save_figure(fig, name, registry)


def plot_centrality(network_metrics: pd.DataFrame, registry: ArtifactRegistry) -> Path:
    df = network_metrics.nlargest(15, "degree_strength_recomputed").sort_values("degree_strength_recomputed")
    fig, ax = plt.subplots(figsize=(8.5, max(4, len(df) * 0.38)))
    ax.barh(df["node"].map(lambda x: wrap_label(x, 24)), df["degree_strength_recomputed"], color="#2F6F73")
    ax.set_title("Top Nodes by Weighted Degree Strength")
    ax.set_xlabel("Weighted degree strength")
    ax.set_ylabel("")
    return save_figure(fig, "network_top_degree_strength", registry)


def plot_term_gap(term_gap: pd.DataFrame, registry: ArtifactRegistry) -> Path:
    if term_gap.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No term gap data available", ha="center", va="center")
        ax.axis("off")
        return save_figure(fig, "term_pre_post_gap", registry)
    positive = term_gap.nlargest(12, "doc_share_gap")
    negative = term_gap.nsmallest(12, "doc_share_gap")
    df = pd.concat([negative, positive]).drop_duplicates("term").sort_values("doc_share_gap")
    colors = np.where(df["doc_share_gap"] >= 0, "#C44E52", "#4C78A8")
    fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.34)))
    ax.barh(df["term"].map(lambda x: wrap_label(x, 24)), df["doc_share_gap"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Text-Mining Term Gap by COVID Period")
    ax.set_xlabel("Post-COVID document-share gap")
    ax.set_ylabel("")
    return save_figure(fig, "term_pre_post_gap", registry)


def plot_adjacent_signals(changes: pd.DataFrame, label_col: str, registry: ArtifactRegistry, name: str, title: str) -> Path:
    df = changes[changes["exploratory_signal"]].nlargest(18, "share_delta_abs").copy()
    if df.empty:
        df = changes.nlargest(18, "share_delta_abs").copy()
    df["label"] = df[label_col].map(lambda x: wrap_label(x, 22)) + "\n" + df["earlier_block"] + " -> " + df["later_block"]
    df = df.sort_values("share_delta")
    colors = np.where(df["share_delta"] >= 0, "#C44E52", "#4C78A8")
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.42)))
    ax.barh(df["label"], df["share_delta"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Later block share minus earlier block share")
    ax.set_ylabel("")
    return save_figure(fig, name, registry)


def build_summary_table(gaps: dict[str, pd.DataFrame], documents: pd.DataFrame) -> pd.DataFrame:
    docs = add_period_columns(documents)
    records = [
        {"metric": "documents_total", "value": len(docs)},
        {"metric": "documents_pre_covid", "value": int((docs["covid_period"] == "Pre-COVID").sum())},
        {"metric": "documents_post_covid", "value": int((docs["covid_period"] == "Post-COVID").sum())},
    ]
    for name, df in gaps.items():
        if df.empty or "signal_label" not in df.columns:
            continue
        records.extend(
            [
                {"metric": f"{name}_rows", "value": len(df)},
                {"metric": f"{name}_new_post_covid", "value": int((df["signal_label"] == "new_post_covid").sum())},
                {"metric": f"{name}_large_upward_shift", "value": int(df["signal_label"].isin(["large_upward_shift", "statistical_upward_signal"]).sum())},
            ]
        )
    return pd.DataFrame(records)


def run_analysis(
    data_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    copy_sources: bool = True,
) -> dict[str, Any]:
    setup_style()
    resolved_data_dir = resolve_data_dir(data_dir)
    registry = make_registry(output_root)
    tables = load_tables(resolved_data_dir)

    if copy_sources:
        copy_source_csvs(resolved_data_dir, registry)

    period_totals, block_totals = get_period_totals(tables["documents"])

    save_table(build_file_inventory(resolved_data_dir, tables), "input_file_inventory", registry)
    save_table(build_data_dictionary(tables), "data_dictionary", registry)

    docs_with_period = add_period_columns(tables["documents"])
    save_table(
        docs_with_period.groupby(["year", "covid_period"], as_index=False).size().rename(columns={"size": "n_docs"}),
        "document_counts_by_year_period",
        registry,
    )
    save_table(
        docs_with_period.groupby(["time_block", "covid_period"], as_index=False).size().rename(columns={"size": "n_docs"}),
        "document_counts_by_time_block_period",
        registry,
    )

    gaps = {
        "node_gap": build_period_gap(tables["node_temporal"], ["node"], period_totals),
        "edge_gap": build_period_gap(tables["edge_temporal"], ["source", "target"], period_totals),
        "topic_gap": build_period_gap(tables["topic_temporal"], ["topic"], period_totals),
        "category_gap": build_category_gap_from_membership(
            tables["document_node_membership"], tables["documents"], period_totals
        ),
    }
    gaps = enrich_gap_tables(tables, gaps)

    for name, df in gaps.items():
        save_table(df, name, registry)

    save_table(
        gaps["node_gap"][gaps["node_gap"]["signal_label"].ne("stable_or_small_change")],
        "highlight_nodes_pre_post_gap",
        registry,
    )
    save_table(
        gaps["edge_gap"][gaps["edge_gap"]["signal_label"].ne("stable_or_small_change")],
        "highlight_edges_pre_post_gap",
        registry,
    )
    save_table(
        gaps["topic_gap"][gaps["topic_gap"]["signal_label"].ne("stable_or_small_change")],
        "highlight_topics_pre_post_gap",
        registry,
    )

    node_changes = build_adjacent_block_changes(tables["node_temporal"], ["node"], block_totals)
    edge_changes = build_adjacent_block_changes(tables["edge_temporal"], ["source", "target"], block_totals)
    topic_changes = build_adjacent_block_changes(tables["topic_temporal"], ["topic"], block_totals)
    save_table(node_changes, "node_adjacent_block_changes", registry)
    save_table(edge_changes, "edge_adjacent_block_changes", registry)
    save_table(topic_changes, "topic_adjacent_block_changes", registry)

    network_metrics = build_network_metrics(tables["semantic_nodes"], tables["semantic_edges"])
    save_table(network_metrics, "network_metrics_recomputed", registry)

    topic_keyword_long = build_topic_keyword_long(tables["topic_interpretation"])
    topic_node_matrix = build_topic_node_matrix(tables["document_node_membership"])
    save_table(topic_keyword_long, "topic_keyword_long", registry)
    save_table(topic_node_matrix, "topic_node_matrix", registry)

    term_gap, top_terms = build_text_mining_tables(tables["documents"], period_totals)
    save_table(term_gap, "text_mining_term_pre_post_gap", registry)
    save_table(top_terms, "text_mining_top_terms_by_period", registry)

    summary = build_summary_table(gaps, tables["documents"])
    save_table(summary, "analysis_run_summary", registry)

    plot_document_year_counts(tables["documents"], registry)
    plot_topic_heatmap(tables["topic_temporal"], tables["topic_interpretation"], registry)
    plot_gap_bar(gaps["topic_gap"], "topic", "Topic Gap: Post-COVID vs Pre-COVID", "topic_pre_post_gap", registry, top_n=12)
    plot_gap_bar(gaps["node_gap"], "node", "Node Gap: Post-COVID vs Pre-COVID", "node_pre_post_gap_top", registry)
    plot_gap_bar(
        gaps["edge_gap"].assign(edge_label=lambda x: x["source"] + " -> " + x["target"]),
        "edge_label",
        "Edge Gap: Post-COVID vs Pre-COVID",
        "edge_pre_post_gap_top",
        registry,
    )
    plot_node_scatter(gaps["node_gap"], registry)
    plot_category_gap(gaps["category_gap"], registry)
    plot_network(
        tables["semantic_edges"],
        tables["semantic_nodes"],
        registry,
        "semantic_network_overall",
        "Overall Semantic Network",
        weight_col="weight",
    )
    new_edges = gaps["edge_gap"][gaps["edge_gap"]["is_new_after_covid"]].copy()
    plot_network(
        new_edges,
        tables["semantic_nodes"],
        registry,
        "new_post_covid_edges_network",
        "New Post-COVID Semantic Relationships",
        weight_col="post_share",
        max_edges=60,
    )
    plot_centrality(network_metrics, registry)
    plot_term_gap(term_gap, registry)
    plot_adjacent_signals(node_changes, "node", registry, "node_adjacent_block_signals", "Node Changes Between Adjacent Time Blocks")
    plot_adjacent_signals(
        topic_changes.assign(topic=lambda x: "Topic " + x["topic"].astype(str)),
        "topic",
        registry,
        "topic_adjacent_block_signals",
        "Topic Changes Between Adjacent Time Blocks",
    )

    manifest_path = registry.csv_dir / "artifact_manifest.csv"
    planned_zips = [registry.zip_dir / "csv_outputs.zip", registry.zip_dir / "png_outputs.zip"]
    csv_paths = list(registry.tables)
    if manifest_path not in csv_paths:
        csv_paths.append(manifest_path)
    manifest = pd.DataFrame(
        {
            "artifact_type": ["csv"] * len(csv_paths) + ["png"] * len(registry.figures) + ["zip"] * len(planned_zips),
            "path": [str(path) for path in csv_paths + registry.figures + planned_zips],
        }
    )
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    if manifest_path not in registry.tables:
        registry.tables.append(manifest_path)
    zip_outputs(registry)

    return {
        "data_dir": resolved_data_dir,
        "output_root": registry.output_root,
        "csv_dir": registry.csv_dir,
        "png_dir": registry.png_dir,
        "zip_dir": registry.zip_dir,
        "tables": tables,
        "gaps": gaps,
        "node_changes": node_changes,
        "edge_changes": edge_changes,
        "topic_changes": topic_changes,
        "network_metrics": network_metrics,
        "term_gap": term_gap,
        "registry": registry,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RBQM topic-modeling analysis exports.")
    parser.add_argument("--data-dir", type=str, default=None, help="Folder containing the RBQM CSV files.")
    parser.add_argument("--output-root", type=str, default=None, help="Output folder. Defaults to ./rbqm_topicmodeling_analysis_outputs.")
    parser.add_argument("--no-copy-sources", action="store_true", help="Do not copy input CSV files into csv/source_inputs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_analysis(
        data_dir=args.data_dir,
        output_root=args.output_root,
        copy_sources=not args.no_copy_sources,
    )
    print(f"Data directory: {result['data_dir']}")
    print(f"CSV output: {result['csv_dir']}")
    print(f"PNG output: {result['png_dir']}")
    print(f"ZIP output: {result['zip_dir']}")
