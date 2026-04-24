import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.db import get_engine

TRANSLATIONS: dict[str, dict] = {
    "es": {
        "subtitle": "Actividad de usuarios · actualizado cada 60 s",
        "no_data": "No hay datos de actividad en la base de datos.",
        "summary_header": "Resumen por usuario",
        "actions_dist_header": "Distribución de acciones",
        "action_detail_header": "Detalle por acción",
        "element_action_header": "Acciones por elemento y usuario",
        "kpi_users": "Usuarios",
        "kpi_total_actions": "Acciones totales",
        "kpi_total_sessions": "Sesiones totales",
        "kpi_avg_session": "Media acc/sesión",
        "kpi_top_user": "Usuario más activo",
        "col_usuario": "Usuario",
        "col_total_acciones": "Total acciones",
        "col_accion_top": "Acción top",
        "col_pct_top": "% top",
        "col_accion_menos": "Menos frecuente",
        "col_sesiones": "Sesiones",
        "col_media_sesion": "Media/sesión",
        "col_elemento": "Elemento",
        "col_accion": "Acción",
        "col_total": "Total",
        "footer": "KeyOver1 Dashboard · streamlit run dashboard.py",
        "actions": {
            1000000: "Visualizar",
            1000001: "Crear",
            1000002: "Editar",
            1000003: "Eliminar",
            1000004: "Copiar",
            1000005: "Compartir",
        },
    },
    "it": {
        "subtitle": "Attività utenti · aggiornato ogni 60 s",
        "no_data": "Nessun dato di attività nel database.",
        "summary_header": "Riepilogo per utente",
        "actions_dist_header": "Distribuzione delle azioni",
        "action_detail_header": "Dettaglio per azione",
        "element_action_header": "Azioni per elemento e utente",
        "kpi_users": "Utenti",
        "kpi_total_actions": "Azioni totali",
        "kpi_total_sessions": "Sessioni totali",
        "kpi_avg_session": "Media az./sessione",
        "kpi_top_user": "Utente più attivo",
        "col_usuario": "Utente",
        "col_total_acciones": "Totale azioni",
        "col_accion_top": "Azione top",
        "col_pct_top": "% top",
        "col_accion_menos": "Meno frequente",
        "col_sesiones": "Sessioni",
        "col_media_sesion": "Media/sessione",
        "col_elemento": "Elemento",
        "col_accion": "Azione",
        "col_total": "Totale",
        "footer": "KeyOver1 Dashboard · streamlit run dashboard.py",
        "actions": {
            1000000: "Visualizza",
            1000001: "Crea",
            1000002: "Modifica",
            1000003: "Elimina",
            1000004: "Copia",
            1000005: "Condividi",
        },
    },
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
    "Visualizar": "#4A90D9",
    "Crear":      "#27AE60",
    "Editar":     "#F39C12",
    "Eliminar":   "#E74C3C",
    "Copiar":     "#8E44AD",
    "Compartir":  "#16A085",
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

/* Botones de idioma */
[data-testid="stButton"] button {
    font-size: 1.5rem !important;
    padding: 4px 10px !important;
    line-height: 1 !important;
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


def build_summary(
    action_df: pd.DataFrame, session_df: pd.DataFrame, t: dict
) -> pd.DataFrame:
    if action_df.empty:
        return pd.DataFrame()

    action_labels = t["actions"]
    adf = action_df.copy()
    adf["action_label"] = adf["action_id"].map(action_labels).fillna(
        adf["action_id"].astype(str)
    )

    rows = []
    for uid, grp in adf.groupby("user_id"):
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
            "Usuario":           usuario,
            "Total acciones":    total_acciones,
            "Acción top":        accion_mas,
            "N top":             accion_mas_n,
            "% acción top":      pct_top,
            "Acción menos":      accion_menos,
            "N menos":           accion_menos_n,
            "Sesiones":          num_sesiones,
            "Media acc/sesión":  avg_por_sesion,
            **{f"#{k}": v for k, v in dist.items()},
        })

    return pd.DataFrame(rows)


def render_kpis(df: pd.DataFrame, t: dict) -> None:
    total_users    = len(df)
    total_actions  = int(df["Total acciones"].sum())
    total_sessions = int(df["Sesiones"].sum())
    avg_session    = round(df["Media acc/sesión"].mean(), 1) if total_users else 0
    top_user       = df.loc[df["Total acciones"].idxmax(), "Usuario"] if total_users else "—"

    cols = st.columns(5)
    cols[0].metric(t["kpi_users"],         total_users)
    cols[1].metric(t["kpi_total_actions"], f"{total_actions:,}")
    cols[2].metric(t["kpi_total_sessions"], total_sessions)
    cols[3].metric(t["kpi_avg_session"],   avg_session)
    cols[4].metric(t["kpi_top_user"],      top_user)


def render_summary_table(df: pd.DataFrame, t: dict) -> None:
    display = df[[
        "Usuario", "Total acciones", "Acción top", "% acción top",
        "Acción menos", "Sesiones", "Media acc/sesión",
    ]].copy()

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Usuario":          st.column_config.TextColumn(t["col_usuario"],        width="medium"),
            "Total acciones":   st.column_config.NumberColumn(t["col_total_acciones"], format="%d"),
            "Acción top":       st.column_config.TextColumn(t["col_accion_top"],     width="small"),
            "% acción top":     st.column_config.ProgressColumn(
                                    t["col_pct_top"], format="%.1f%%", min_value=0, max_value=100
                                ),
            "Acción menos":     st.column_config.TextColumn(t["col_accion_menos"],   width="small"),
            "Sesiones":         st.column_config.NumberColumn(t["col_sesiones"],     format="%d"),
            "Media acc/sesión": st.column_config.NumberColumn(t["col_media_sesion"], format="%.1f"),
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


def render_detail_table(df: pd.DataFrame, t: dict) -> None:
    detail_cols = [c for c in df.columns if c.startswith("#")]
    if not detail_cols:
        return

    detail_df = df[["Usuario"] + detail_cols].set_index("Usuario")
    detail_df.columns = [c.lstrip("#") for c in detail_df.columns]
    detail_df.index.name = t["col_usuario"]

    st.dataframe(
        detail_df,
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(col, format="%d")
            for col in detail_df.columns
        },
    )


def render_element_action_table(element_df: pd.DataFrame, t: dict) -> None:
    if element_df.empty:
        return

    df = element_df.copy()
    df["elemento"] = df["element_id"].map(ELEMENT_LABELS).fillna(df["element_id"].astype(str))
    df["accion"]   = df["action_id"].map(t["actions"]).fillna(df["action_id"].astype(str))

    pivot = df.pivot_table(
        index=["elemento", "accion"],
        columns="usuario",
        values="total",
        aggfunc="sum",
        fill_value=0,
    ).astype(int)

    pivot[t["col_total"]] = pivot.sum(axis=1)
    pivot = pivot.sort_values(["elemento", t["col_total"]], ascending=[True, False])
    pivot = pivot.reset_index()
    pivot.columns.name = None

    user_cols = [c for c in pivot.columns if c not in ("elemento", "accion", t["col_total"])]
    col_config = {
        "elemento":      st.column_config.TextColumn(t["col_elemento"], width="small"),
        "accion":        st.column_config.TextColumn(t["col_accion"],   width="small"),
        t["col_total"]:  st.column_config.NumberColumn(t["col_total"],  format="%d"),
        **{u: st.column_config.NumberColumn(u, format="%d") for u in user_cols},
    }

    st.dataframe(pivot, use_container_width=True, hide_index=True, column_config=col_config)


def render_lang_selector() -> None:
    lang = st.session_state.get("lang", "es")
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "🇪🇸",
            use_container_width=True,
            type="primary" if lang == "es" else "secondary",
            key="btn_es",
        ):
            st.session_state.lang = "es"
            st.rerun()
    with c2:
        if st.button(
            "🇮🇹",
            use_container_width=True,
            type="primary" if lang == "it" else "secondary",
            key="btn_it",
        ):
            st.session_state.lang = "it"
            st.rerun()


def main():
    st.set_page_config(
        page_title="KeyOver1 — Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    if "lang" not in st.session_state:
        st.session_state.lang = "es"

    t = TRANSLATIONS[st.session_state.lang]

    title_col, lang_col = st.columns([9, 1])
    with title_col:
        st.title("KeyOver1")
        st.markdown(
            f"<span style='color:#8a8fa8;font-size:1.1rem;'>{t['subtitle']}</span>",
            unsafe_allow_html=True,
        )
    with lang_col:
        render_lang_selector()

    st.divider()

    action_df, session_df, element_df = load_stats()
    df = build_summary(action_df, session_df, t)

    if df.empty:
        st.warning(t["no_data"])
        return

    render_kpis(df, t)
    st.divider()

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.subheader(t["summary_header"])
        render_summary_table(df, t)

    with col_right:
        st.subheader(t["actions_dist_header"])
        render_action_chart(df)

    st.divider()
    st.subheader(t["action_detail_header"])
    render_detail_table(df, t)

    st.divider()
    st.subheader(t["element_action_header"])
    render_element_action_table(element_df, t)

    st.markdown(
        f"<p class='caption-text'>{t['footer']}</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
