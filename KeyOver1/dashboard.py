import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.db import get_engine

ACTION_LABELS = {
    1000000: "Visualize",
    1000001: "Create",
    1000002: "Edit",
    1000003: "Delete",
    1000004: "Copy",
    1000005: "Share",
}

ELEMENT_LABELS = {
    1: "FVG",
    2: "AMCO",
    3: "VETTING",
    4: "RHODENSE",
    5: "PAPARDO",
    6: "PULEJO",
}

ACTION_COLORS = {
    "Visualize": "#4A90D9",
    "Create":    "#27AE60",
    "Edit":      "#F39C12",
    "Delete":    "#E74C3C",
    "Copy":      "#8E44AD",
    "Share":     "#16A085",
}

CSS = """
<style>
/* Fuente y fondo general — tamaño base aumentado para mejor legibilidad */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #e0e0e0;
    font-size: 18px !important;
}

/* Texto general de la app */
p, div, span, li, td, th, label {
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
}

/* Título principal */
h1 {
    letter-spacing: -0.5px;
    font-size: 2.4rem !important;
    margin-bottom: 0.3rem !important;
}

/* Métricas: borde sutil + fondo oscuro */
[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 26px 28px 22px 28px;
}
[data-testid="metric-container"] label {
    color: #8a8fa8 !important;
    font-size: 1rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* Divisor */
hr { border-color: #2a2d3a !important; }

/* Subheaders */
h2, h3 {
    color: #c5c9d6;
    font-weight: 600;
    font-size: 1.35rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
}

/* Tablas de datos */
[data-testid="stDataFrame"] table {
    font-size: 1.05rem !important;
}
[data-testid="stDataFrame"] th {
    font-size: 1rem !important;
    padding: 10px 14px !important;
}
[data-testid="stDataFrame"] td {
    font-size: 1.05rem !important;
    padding: 10px 14px !important;
}

/* Caption */
.caption-text {
    color: #4a4f63;
    font-size: 0.95rem;
    margin-top: 1.5rem;
}
</style>
"""


@st.cache_data(ttl=60)
def load_stats() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engine = get_engine()

    action_df = pd.read_sql(
        """
        SELECT
            u.user_id,
            u.name || ' ' || u.surname AS usuario,
            al.action_id,
            COUNT(*) AS total
        FROM activity_log al
        JOIN users u ON al.user_id = u.user_id
        GROUP BY u.user_id, u.name, u.surname, al.action_id
        ORDER BY u.user_id, total DESC
        """,
        engine,
    )

    element_df = pd.read_sql(
        """
        SELECT
            al.element_id,
            al.action_id,
            u.name || ' ' || u.surname AS usuario,
            COUNT(*) AS total
        FROM activity_log al
        JOIN users u ON al.user_id = u.user_id
        GROUP BY al.element_id, al.action_id, u.name, u.surname
        ORDER BY al.element_id, al.action_id, total DESC
        """,
        engine,
    )

    session_df = pd.read_sql(
        """
        SELECT
            al.user_id,
            ll.login_log_id,
            COUNT(*) AS acciones_en_sesion
        FROM activity_log al
        JOIN login_log ll
            ON al.user_id = ll.user_id
            AND al.logged_at BETWEEN ll.logged_at AND COALESCE(ll.logout_at, NOW())
        WHERE ll.result = TRUE
        GROUP BY al.user_id, ll.login_log_id
        """,
        engine,
    )

    return action_df, session_df, element_df


def build_summary(action_df: pd.DataFrame, session_df: pd.DataFrame) -> pd.DataFrame:
    if action_df.empty:
        return pd.DataFrame()

    action_df["action_label"] = action_df["action_id"].map(ACTION_LABELS).fillna(
        action_df["action_id"].astype(str)
    )

    rows = []
    for uid, grp in action_df.groupby("user_id"):
        grp_sorted = grp.sort_values("total", ascending=False)
        usuario = grp_sorted["usuario"].iloc[0]
        total_acciones = int(grp_sorted["total"].sum())

        accion_mas   = grp_sorted.iloc[0]["action_label"]
        accion_mas_n = int(grp_sorted.iloc[0]["total"])
        accion_menos   = grp_sorted.iloc[-1]["action_label"]
        accion_menos_n = int(grp_sorted.iloc[-1]["total"])

        dist = {row["action_label"]: int(row["total"]) for _, row in grp_sorted.iterrows()}
        pct_top = round(accion_mas_n / total_acciones * 100, 1) if total_acciones else 0

        user_sessions = session_df[session_df["user_id"] == uid]
        num_sesiones  = int(user_sessions["login_log_id"].nunique()) if not user_sessions.empty else 0
        avg_por_sesion = (
            round(user_sessions["acciones_en_sesion"].mean(), 1) if not user_sessions.empty else 0.0
        )

        rows.append({
            "Usuario":               usuario,
            "Total acciones":        total_acciones,
            "Acción top":            accion_mas,
            "N top":                 accion_mas_n,
            "% acción top":          pct_top,
            "Acción menos":          accion_menos,
            "N menos":               accion_menos_n,
            "Sesiones":              num_sesiones,
            "Media acc/sesión":      avg_por_sesion,
            **{f"#{k}": v for k, v in dist.items()},
        })

    return pd.DataFrame(rows)


def render_kpis(df: pd.DataFrame) -> None:
    total_users   = len(df)
    total_actions = int(df["Total acciones"].sum())
    total_sessions = int(df["Sesiones"].sum())
    avg_session   = round(df["Media acc/sesión"].mean(), 1) if total_users else 0
    top_user      = df.loc[df["Total acciones"].idxmax(), "Usuario"] if total_users else "—"

    cols = st.columns(5)
    cols[0].metric("Usuarios",          total_users)
    cols[1].metric("Acciones totales",  f"{total_actions:,}")
    cols[2].metric("Sesiones totales",  total_sessions)
    cols[3].metric("Media acc/sesión",  avg_session)
    cols[4].metric("Usuario más activo", top_user)


def render_summary_table(df: pd.DataFrame) -> None:
    display = df[[
        "Usuario", "Total acciones", "Acción top", "% acción top",
        "Acción menos", "Sesiones", "Media acc/sesión",
    ]].copy()

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Usuario":          st.column_config.TextColumn("Usuario", width="medium"),
            "Total acciones":   st.column_config.NumberColumn("Total acciones", format="%d"),
            "Acción top":       st.column_config.TextColumn("Acción top", width="small"),
            "% acción top":     st.column_config.ProgressColumn(
                                    "% top", format="%.1f%%", min_value=0, max_value=100
                                ),
            "Acción menos":     st.column_config.TextColumn("Menos frecuente", width="small"),
            "Sesiones":         st.column_config.NumberColumn("Sesiones", format="%d"),
            "Media acc/sesión": st.column_config.NumberColumn("Media/sesión", format="%.1f"),
        },
    )


def render_action_chart(df: pd.DataFrame) -> None:
    detail_cols = [c for c in df.columns if c.startswith("#")]
    if not detail_cols:
        return

    chart_df = df[["Usuario"] + detail_cols].set_index("Usuario")
    chart_df.columns = [c.lstrip("#") for c in chart_df.columns]
    chart_df = chart_df.fillna(0).astype(int)

    st.bar_chart(chart_df, use_container_width=True, height=380)


def render_detail_table(df: pd.DataFrame) -> None:
    detail_cols = [c for c in df.columns if c.startswith("#")]
    if not detail_cols:
        return

    detail_df = df[["Usuario"] + detail_cols].set_index("Usuario")
    detail_df.columns = [c.lstrip("#") for c in detail_df.columns]

    st.dataframe(
        detail_df,
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(col, format="%d")
            for col in detail_df.columns
        },
    )


def render_element_action_table(element_df: pd.DataFrame) -> None:
    if element_df.empty:
        return

    df = element_df.copy()
    df["elemento"] = df["element_id"].map(ELEMENT_LABELS).fillna(df["element_id"].astype(str))
    df["accion"]   = df["action_id"].map(ACTION_LABELS).fillna(df["action_id"].astype(str))

    pivot = df.pivot_table(
        index=["elemento", "accion"],
        columns="usuario",
        values="total",
        aggfunc="sum",
        fill_value=0,
    ).astype(int)

    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values(["elemento", "Total"], ascending=[True, False])
    pivot = pivot.reset_index()
    pivot.columns.name = None

    user_cols = [c for c in pivot.columns if c not in ("elemento", "accion", "Total")]
    col_config = {
        "elemento": st.column_config.TextColumn("Elemento", width="small"),
        "accion":   st.column_config.TextColumn("Acción",   width="small"),
        "Total":    st.column_config.NumberColumn("Total",  format="%d"),
        **{u: st.column_config.NumberColumn(u, format="%d") for u in user_cols},
    }

    st.dataframe(pivot, use_container_width=True, hide_index=True, column_config=col_config)


def main():
    st.set_page_config(
        page_title="KeyOver1 — Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    st.title("KeyOver1")
    st.markdown("<span style='color:#8a8fa8;font-size:1.1rem;'>Actividad de usuarios · actualizado cada 60 s</span>", unsafe_allow_html=True)

    st.divider()

    action_df, session_df, element_df = load_stats()
    df = build_summary(action_df, session_df)

    if df.empty:
        st.warning("No hay datos de actividad en la base de datos.")
        return

    render_kpis(df)
    st.divider()

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.subheader("Resumen por usuario")
        render_summary_table(df)

    with col_right:
        st.subheader("Distribución de acciones")
        render_action_chart(df)

    st.divider()
    st.subheader("Detalle por acción")
    render_detail_table(df)

    st.divider()
    st.subheader("Acciones por elemento y usuario")
    render_element_action_table(element_df)

    st.markdown(
        "<p class='caption-text'>KeyOver1 Dashboard · streamlit run dashboard.py</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
