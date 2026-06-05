import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Presidential Elections · 1976–2024",
    layout="wide",
    page_icon="🗳️",
    initial_sidebar_state="expanded"
)

# ─── COLOR PALETTE ────────────────────────────────────────────────────────────
C = {
    "bg":       "#F1F5F9",
    "card":     "#FFFFFF",
    "dem":      "#1E3A8A",
    "rep":      "#881337",
    "other":    "#94A3B8",
    "text":     "#0F172A",
    "subtext":  "#475569",
    "border":   "#E2E8F0",
    "accent":   "#334155",
}

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] {{ background: {C['bg']}; }}
  [data-testid="stSidebar"] {{ background: {C['text']}; }}
  [data-testid="stSidebar"] * {{ color: #F1F5F9 !important; }}
  [data-testid="metric-container"] {{
      background: {C['card']}; border: 1px solid {C['border']};
      border-radius: 10px; padding: 16px 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  [data-testid="stMetricValue"] {{ color: {C['text']} !important; font-weight: 700; }}
  [data-testid="stMetricLabel"] {{ color: {C['subtext']} !important; font-size: 0.78rem !important; text-transform: uppercase; }}
  h1 {{ color: {C['text']}; font-weight: 800; }}
  h2 {{ color: {C['text']}; font-weight: 700; border-bottom: 2px solid {C['border']}; padding-bottom: 6px; }}
  h3 {{ color: {C['accent']}; font-weight: 600; }}
  .caption-text {{ color: {C['subtext']}; font-size: 0.78rem; margin-top: -10px; }}
</style>
""", unsafe_allow_html=True)

# ─── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    df = df[df["writein"] == False].copy()
    df["candidatevotes"] = pd.to_numeric(df["candidatevotes"], errors="coerce").fillna(0)
    df["totalvotes"]     = pd.to_numeric(df["totalvotes"],     errors="coerce").fillna(0)
    return df

df = load_data()

national      = df.groupby(["year","party_simplified"])["candidatevotes"].sum().reset_index()
state_totals  = df.drop_duplicates(["year","state"])[["year","state","state_po","totalvotes"]]
yearly_turnout = state_totals.groupby("year")["totalvotes"].sum().reset_index()
yearly_turnout.columns = ["year","nat_total"]

dr = national[national["party_simplified"].isin(["DEMOCRAT","REPUBLICAN"])]
dr_piv = dr.pivot_table(index="year", columns="party_simplified", values="candidatevotes", aggfunc="sum").fillna(0).reset_index()
dr_piv["total_dr"] = dr_piv["DEMOCRAT"] + dr_piv["REPUBLICAN"]
dr_piv["dem_share"] = dr_piv["DEMOCRAT"] / dr_piv["total_dr"] * 100
dr_piv["rep_share"] = dr_piv["REPUBLICAN"] / dr_piv["total_dr"] * 100
dr_piv["margin"]    = dr_piv["DEMOCRAT"] - dr_piv["REPUBLICAN"]
dr_piv = dr_piv.merge(yearly_turnout, on="year")

third   = national[~national["party_simplified"].isin(["DEMOCRAT","REPUBLICAN"])]
third_yr = third.groupby("year")["candidatevotes"].sum().reset_index()
third_yr = third_yr.merge(yearly_turnout, on="year")
third_yr["pct"] = third_yr["candidatevotes"] / third_yr["nat_total"] * 100

sm = df[df["party_simplified"].isin(["DEMOCRAT","REPUBLICAN"])].copy()
sm = sm.groupby(["year","state","state_po","party_simplified"])["candidatevotes"].sum().reset_index()
sm_piv = sm.pivot_table(index=["year","state","state_po"], columns="party_simplified", values="candidatevotes").fillna(0).reset_index()
sm_piv["total"]      = sm_piv["DEMOCRAT"] + sm_piv["REPUBLICAN"]
sm_piv["dem_share"]  = sm_piv["DEMOCRAT"] / sm_piv["total"] * 100
sm_piv["rep_share"]  = sm_piv["REPUBLICAN"] / sm_piv["total"] * 100
sm_piv["dem_margin"] = sm_piv["dem_share"] - 50
sm_piv["abs_margin"] = sm_piv["dem_margin"].abs()
sm_piv["winner"]     = sm_piv["dem_margin"].apply(lambda x: "Democrat" if x > 0 else "Republican")

state_win   = sm_piv.loc[sm_piv.groupby(["year","state"])["total"].idxmax()]
competitive = sm_piv.groupby("state")["abs_margin"].mean().reset_index().sort_values("abs_margin")

def flips(g):
    p = g.sort_values("year")["winner"].tolist()
    return sum(1 for i in range(1,len(p)) if p[i]!=p[i-1])
swing = state_win.groupby("state").apply(flips).reset_index()
swing.columns = ["state","flips"]
swing = swing.sort_values("flips", ascending=False)

lib = df[df["party_simplified"]=="LIBERTARIAN"].groupby("year")["candidatevotes"].sum().reset_index()
lib = lib.merge(yearly_turnout, on="year")
lib["pct"] = lib["candidatevotes"] / lib["nat_total"] * 100

latest_year = int(df["year"].max())
years_list  = sorted(df["year"].unique())

def styled(fig, height=420):
    fig.update_layout(
        height=height, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", color=C["text"], size=12),
        legend=dict(bgcolor="white", bordercolor=C["border"], borderwidth=1),
        margin=dict(l=10,r=10,t=50,b=10),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor=C["border"]),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor=C["border"]),
    )
    return fig

def card(title, caption=""):
    st.markdown(f"<h3 style='margin-bottom:2px'>{title}</h3>", unsafe_allow_html=True)
    if caption:
        st.markdown(f"<p class='caption-text'>{caption}</p>", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px 0;'>
      <div style='font-size:2rem;'>🗳️</div>
      <div style='font-size:1.05rem;font-weight:800;color:white;'>US Presidential<br>Elections</div>
      <div style='font-size:0.72rem;color:#94A3B8;margin-top:4px;'>1976 – 2024</div>
    </div>
    <hr style='border-color:#334155;margin:10px 0 20px 0;'>
    """, unsafe_allow_html=True)
    page = st.radio("NAVIGATION", [
        "📊  National Overview",
        "🗺️  State by State Breakdown",
        "📈  Explore Trends",
        "🔬  Advanced Analysis"
    ])
    st.markdown("<hr style='border-color:#334155;margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem;color:#64748B;text-align:center;'>MIT Election Data & Science Lab</div>", unsafe_allow_html=True)

page = page.split("  ",1)[1].strip()

# ══════════════ PAGE 1: NATIONAL OVERVIEW ════════════════════════════════════
if page == "National Overview":
    st.markdown("## 📊 National Overview")

    c1,c2,c3,c4,c5 = st.columns(5)
    dem_wins = sum(1 for y in years_list if len(dr_piv[(dr_piv["year"]==y)&(dr_piv["DEMOCRAT"]>dr_piv["REPUBLICAN"])]) > 0)
    max_turn = yearly_turnout.loc[yearly_turnout["nat_total"].idxmax()]
    third_max = third_yr.loc[third_yr["pct"].idxmax()]
    c1.metric("Elections", len(years_list))
    c2.metric("Democrat Popular Wins", dem_wins)
    c3.metric("Republican Popular Wins", len(years_list)-dem_wins)
    c4.metric("Peak Turnout Year", f"{int(max_turn['year'])}")
    c5.metric("Peak 3rd-Party Year", f"{int(third_max['year'])} ({third_max['pct']:.1f}%)")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        card("🥧 Overall Party Vote Share (All Elections)")
        total_by_party = df.groupby("party_simplified")["candidatevotes"].sum().reset_index()
        total_by_party["label"] = total_by_party["party_simplified"].map(
            {"DEMOCRAT":"Democrat","REPUBLICAN":"Republican","LIBERTARIAN":"Libertarian","OTHER":"Other"}
        ).fillna("Other")
        agg = total_by_party.groupby("label")["candidatevotes"].sum().reset_index()
        pie_colors = {"Democrat":C["dem"],"Republican":C["rep"],"Libertarian":"#7C3AED","Other":C["other"]}
        fig1 = go.Figure(go.Pie(
            labels=agg["label"], values=agg["candidatevotes"], hole=0.45,
            marker=dict(colors=[pie_colors.get(l,C["other"]) for l in agg["label"]],
                        line=dict(color="white",width=2)),
        ))
        fig1.update_layout(height=380, paper_bgcolor="white",
            font=dict(family="Inter, system-ui, sans-serif"),
            annotations=[dict(text="All<br>Votes",x=0.5,y=0.5,font_size=13,showarrow=False)])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        card("📊 National Popular Vote by Party & Year")
        dr_long = dr_piv.melt(id_vars="year",value_vars=["DEMOCRAT","REPUBLICAN"],var_name="Party",value_name="Votes")
        dr_long["Party"] = dr_long["Party"].map({"DEMOCRAT":"Democrat","REPUBLICAN":"Republican"})
        fig2 = px.bar(dr_long, x="year", y="Votes", color="Party", barmode="group",
                      color_discrete_map={"Democrat":C["dem"],"Republican":C["rep"]})
        fig2 = styled(fig2, 380)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        card("📈 National Voter Turnout (1976–2024)")
        fig3 = go.Figure(go.Scatter(
            x=yearly_turnout["year"], y=yearly_turnout["nat_total"],
            mode="lines+markers", line=dict(color=C["accent"],width=2.5),
            marker=dict(size=8,color=C["accent"],line=dict(color="white",width=2)),
            fill="tozeroy", fillcolor="rgba(51,65,85,0.08)"
        ))
        fig3 = styled(fig3, 360)
        fig3.update_layout(xaxis_title="Election Year",yaxis_title="Total Votes Cast",showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        card(f"🔽 Vote Funnel — {latest_year} Election")
        ly = df[df["year"]==latest_year]
        total_ly = int(state_totals[state_totals["year"]==latest_year]["totalvotes"].sum())
        dem_ly   = int(ly[ly["party_simplified"]=="DEMOCRAT"]["candidatevotes"].sum())
        rep_ly   = int(ly[ly["party_simplified"]=="REPUBLICAN"]["candidatevotes"].sum())
        lib_ly   = int(ly[ly["party_simplified"]=="LIBERTARIAN"]["candidatevotes"].sum())
        oth_ly   = total_ly - dem_ly - rep_ly - lib_ly
        fig4 = go.Figure(go.Funnel(
            y=["Total Votes","Democrat","Republican","Libertarian","Other"],
            x=[total_ly,dem_ly,rep_ly,lib_ly,oth_ly],
            textinfo="value+percent initial",
            marker=dict(color=[C["accent"],C["dem"],C["rep"],"#7C3AED",C["other"]])
        ))
        fig4.update_layout(height=360, paper_bgcolor="white", font=dict(family="Inter, system-ui, sans-serif"))
        st.plotly_chart(fig4, use_container_width=True)

    card("📉 Democrat vs Republican Two-Party Vote Share Over Time")
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=dr_piv["year"],y=dr_piv["dem_share"],name="Democrat",
        mode="lines+markers",line=dict(color=C["dem"],width=2.5),
        marker=dict(size=8,color=C["dem"],line=dict(color="white",width=2))))
    fig5.add_trace(go.Scatter(x=dr_piv["year"],y=dr_piv["rep_share"],name="Republican",
        mode="lines+markers",line=dict(color=C["rep"],width=2.5),
        marker=dict(size=8,color=C["rep"],line=dict(color="white",width=2))))
    fig5.add_hline(y=50,line_dash="dash",line_color=C["border"],line_width=1.5)
    fig5 = styled(fig5, 400)
    fig5.update_layout(xaxis_title="Election Year",yaxis_title="Vote Share (%)",yaxis_range=[44,58])
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════ PAGE 2: STATE BY STATE ═══════════════════════════════════════
elif page == "State by State Breakdown":
    st.markdown("## 🗺️ State by State Breakdown")
    col_pick,_ = st.columns([2,5])
    with col_pick:
        sel_state = st.selectbox("Select a State", sorted(sm_piv["state"].unique()),
                                  index=list(sorted(sm_piv["state"].unique())).index("CALIFORNIA"))
    st.markdown("---")
    state_df     = sm_piv[sm_piv["state"]==sel_state].sort_values("year")
    state_win_df = state_win[state_win["state"]==sel_state].sort_values("year")
    dem_wins_s = (state_win_df["winner"]=="Democrat").sum()
    rep_wins_s = (state_win_df["winner"]=="Republican").sum()
    flips_s    = swing[swing["state"]==sel_state]["flips"].values[0] if sel_state in swing["state"].values else 0
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Democrat Wins", int(dem_wins_s))
    k2.metric("Republican Wins", int(rep_wins_s))
    k3.metric("Avg. Margin", f"{state_df['abs_margin'].mean():.1f}%")
    k4.metric("Party Flips", int(flips_s))

    col1,col2 = st.columns(2)
    with col1:
        card(f"🥧 Election Wins by Party — {sel_state.title()}")
        fig6 = go.Figure(go.Pie(
            labels=["Democrat","Republican"], values=[int(dem_wins_s),int(rep_wins_s)], hole=0.45,
            marker=dict(colors=[C["dem"],C["rep"]],line=dict(color="white",width=2))
        ))
        fig6.update_layout(height=340, paper_bgcolor="white", font=dict(family="Inter, system-ui, sans-serif"))
        st.plotly_chart(fig6, use_container_width=True)

    with col2:
        card(f"📈 Democrat Vote Share Over Time — {sel_state.title()}")
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=state_df["year"],y=state_df["dem_share"],mode="lines+markers",
            line=dict(color=C["dem"],width=2.5),
            marker=dict(size=9,color=[C["dem"] if w=="Democrat" else C["rep"] for w in state_win_df["winner"]],
                        line=dict(color="white",width=2))))
        fig7.add_hline(y=50,line_dash="dash",line_color=C["border"],line_width=1.5)
        fig7 = styled(fig7, 340)
        fig7.update_layout(xaxis_title="Year",yaxis_title="Democrat Vote Share (%)")
        st.plotly_chart(fig7, use_container_width=True)

    card("📊 20 Most Competitive States (Avg Margin 1976–2024)")
    top20 = competitive.head(20)
    fig8 = px.bar(top20, x="abs_margin", y="state", orientation="h",
                  color="abs_margin",
                  color_continuous_scale=[[0,C["dem"]],[0.5,"#94A3B8"],[1,C["rep"]]],
                  labels={"abs_margin":"Avg Absolute Margin (%)","state":"State"})
    fig8.update_coloraxes(showscale=False)
    fig8 = styled(fig8, 500)
    fig8.update_layout(yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig8, use_container_width=True)

    card("🔄 Swing States — Party Flips (1976–2024)")
    fig9 = px.bar(swing[swing["flips"]>0], x="state", y="flips",
                  color="flips", color_continuous_scale=[[0,"#E2E8F0"],[1,C["text"]]],
                  labels={"flips":"Party Flips","state":"State"})
    fig9.update_coloraxes(showscale=False)
    fig9 = styled(fig9, 380)
    st.plotly_chart(fig9, use_container_width=True)

    card(f"🗺️ {latest_year} Election Results Map")
    ly_state = sm_piv[sm_piv["year"]==latest_year].copy()
    fig10 = px.choropleth(ly_state,locations="state_po",locationmode="USA-states",
        color="dem_margin", scope="usa",
        color_continuous_scale=[[0,C["rep"]],[0.5,"#F8FAFC"],[1,C["dem"]]],
        color_continuous_midpoint=0)
    fig10.update_layout(height=480, paper_bgcolor="white", margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig10, use_container_width=True)

# ══════════════ PAGE 3: EXPLORE TRENDS ═══════════════════════════════════════
elif page == "Explore Trends":
    st.markdown("## 📈 Explore Trends")
    col1,col2 = st.columns(2)
    with col1:
        card("📊 Histogram: Distribution of State Vote Margins")
        fig11 = px.histogram(sm_piv, x="dem_margin", nbins=40,
                             color_discrete_sequence=[C["dem"]],
                             labels={"dem_margin":"Democrat Margin (%)","count":"Frequency"})
        fig11.add_vline(x=0,line_dash="dash",line_color=C["rep"],line_width=2)
        fig11 = styled(fig11, 380)
        fig11.update_layout(xaxis_title="Democrat Margin (%)",yaxis_title="State-Elections",bargap=0.05)
        st.plotly_chart(fig11, use_container_width=True)

    with col2:
        card("🔵 Scatter: State Size vs Democrat Vote Share")
        scatter_yr = st.selectbox("Election Year", years_list[::-1], key="sc_yr")
        sc_df = sm_piv[sm_piv["year"]==scatter_yr].copy()
        fig12 = px.scatter(sc_df, x="total", y="dem_share", color="winner",
            size="total", size_max=50, hover_name="state",
            color_discrete_map={"Democrat":C["dem"],"Republican":C["rep"]},
            labels={"total":"Total Votes","dem_share":"Dem Share (%)","winner":"Winner"})
        fig12.add_hline(y=50,line_dash="dash",line_color=C["border"])
        fig12 = styled(fig12, 380)
        st.plotly_chart(fig12, use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        card("📈 Third-Party Vote Share Over Time")
        fig13 = go.Figure(go.Scatter(x=third_yr["year"],y=third_yr["pct"],mode="lines+markers",
            line=dict(color="#7C3AED",width=2.5),
            marker=dict(size=8,color="#7C3AED",line=dict(color="white",width=2)),
            fill="tozeroy",fillcolor="rgba(124,58,237,0.08)"))
        fig13 = styled(fig13, 360)
        fig13.update_layout(xaxis_title="Election Year",yaxis_title="Third-Party %",showlegend=False)
        st.plotly_chart(fig13, use_container_width=True)

    with col4:
        card("📈 Libertarian Party Vote Share (1976–2024)")
        fig14 = go.Figure(go.Scatter(x=lib["year"],y=lib["pct"],mode="lines+markers",
            line=dict(color="#F59E0B",width=2.5),
            marker=dict(size=8,color="#F59E0B",line=dict(color="white",width=2)),
            fill="tozeroy",fillcolor="rgba(245,158,11,0.08)"))
        fig14 = styled(fig14, 360)
        fig14.update_layout(xaxis_title="Election Year",yaxis_title="Libertarian %",showlegend=False)
        st.plotly_chart(fig14, use_container_width=True)

    card("🫧 Bubble Chart: State Turnout vs Competitiveness")
    bub_yr = st.selectbox("Filter by Decade",["All"]+[str(d)+"s" for d in [1970,1980,1990,2000,2010,2020]],key="bub_dec")
    bub_df = sm_piv.copy()
    if bub_yr != "All":
        dec = int(bub_yr[:4]); bub_df = bub_df[bub_df["year"].between(dec,dec+9)]
    fig15 = px.scatter(bub_df, x="year", y="abs_margin", size="total", color="winner",
        hover_name="state", size_max=55,
        color_discrete_map={"Democrat":C["dem"],"Republican":C["rep"]},
        labels={"abs_margin":"Absolute Margin (%)","year":"Year","total":"Votes","winner":"Winner"})
    fig15 = styled(fig15, 480)
    st.plotly_chart(fig15, use_container_width=True)

    card("📊 Histogram: Winning Margin Distribution by Party")
    fig16 = px.histogram(sm_piv, x="abs_margin", color="winner", barmode="overlay",
        nbins=35, opacity=0.78,
        color_discrete_map={"Democrat":C["dem"],"Republican":C["rep"]},
        labels={"abs_margin":"Absolute Margin (%)","winner":"Winner","count":"Frequency"})
    fig16 = styled(fig16, 380)
    fig16.update_layout(xaxis_title="Absolute Margin (%)",yaxis_title="State-Elections",bargap=0.03)
    st.plotly_chart(fig16, use_container_width=True)

# ══════════════ PAGE 4: ADVANCED ANALYSIS ════════════════════════════════════
elif page == "Advanced Analysis":
    st.markdown("## 🔬 Advanced Analysis")

    card("🔀 Pair Plot: Multi-Variable Scatter Matrix")
    pair_df = sm_piv[["dem_share","rep_share","abs_margin","total","year"]].dropna()
    pair_df_sample = pair_df.sample(min(800,len(pair_df)),random_state=42)
    pair_df_sample["Winner"] = (pair_df_sample["dem_share"]>50).map({True:"Democrat",False:"Republican"})
    dims = [
        dict(label="Dem Share (%)",values=pair_df_sample["dem_share"]),
        dict(label="Rep Share (%)",values=pair_df_sample["rep_share"]),
        dict(label="Abs Margin (%)",values=pair_df_sample["abs_margin"]),
        dict(label="Total Votes (M)",values=pair_df_sample["total"]/1e6),
    ]
    fig17 = go.Figure(go.Splom(dimensions=dims, showupperhalf=False,
        marker=dict(color=[0 if w=="Democrat" else 1 for w in pair_df_sample["Winner"]],
            colorscale=[[0,C["dem"]],[1,C["rep"]]],size=4,opacity=0.6,line=dict(width=0)),
        text=pair_df_sample["Winner"], hovertemplate="<b>%{text}</b><extra></extra>"))
    fig17.update_layout(height=560, paper_bgcolor="white", font=dict(family="Inter, system-ui, sans-serif"),
        margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig17, use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        card("🔵 Scatter: National Turnout vs Popular Vote Margin")
        scat_nat = dr_piv.copy()
        scat_nat["winner"] = scat_nat["margin"].apply(lambda x: "Democrat" if x>0 else "Republican")
        scat_nat["margin_abs"] = scat_nat["margin"].abs()
        fig18 = px.scatter(scat_nat, x="nat_total", y="margin_abs", color="winner",
            size="margin_abs", size_max=30, text="year",
            color_discrete_map={"Democrat":C["dem"],"Republican":C["rep"]},
            labels={"nat_total":"National Turnout","margin_abs":"Vote Margin","winner":"Winner"})
        fig18.update_traces(textposition="top center",textfont=dict(size=9,color=C["subtext"]))
        fig18 = styled(fig18, 400)
        st.plotly_chart(fig18, use_container_width=True)

    with col2:
        card("🔽 Funnel: Third-Party Share by Decade")
        third_yr["decade"] = (third_yr["year"]//10*10).astype(str)+"s"
        dec_avg = third_yr.groupby("decade")["pct"].mean().reset_index().sort_values("pct",ascending=False)
        fig19 = go.Figure(go.Funnel(y=dec_avg["decade"], x=dec_avg["pct"],
            textinfo="value+percent initial",
            marker=dict(color=[C["dem"],C["accent"],"#475569","#64748B","#94A3B8"][:len(dec_avg)])))
        fig19.update_layout(height=400, paper_bgcolor="white", font=dict(family="Inter, system-ui, sans-serif"))
        st.plotly_chart(fig19, use_container_width=True)

    card("📊 Top 20 Third-Party Candidates by Total Votes")
    tc = df[~df["party_simplified"].isin(["DEMOCRAT","REPUBLICAN"])].copy()
    tc = tc[tc["candidate"].notna()&(tc["candidate"].str.strip()!="")]
    top_tc = tc.groupby(["candidate","year","party_simplified"])["candidatevotes"].sum().reset_index()
    top_tc = top_tc.nlargest(20,"candidatevotes")
    top_tc["label"] = top_tc["candidate"].str.strip().str.title()+" ("+top_tc["year"].astype(str)+")"
    fig20 = px.bar(top_tc.sort_values("candidatevotes"), x="candidatevotes", y="label",
        orientation="h", color="party_simplified",
        color_discrete_map={"LIBERTARIAN":"#F59E0B","OTHER":"#94A3B8"},
        labels={"candidatevotes":"Total Votes","label":"Candidate","party_simplified":"Party"})
    fig20 = styled(fig20, 580)
    fig20.update_layout(yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig20, use_container_width=True)

    card("📈 Democrat Vote Share by US Region (1976–2024)")
    regions_map = {
        "AL":"South","AR":"South","DC":"South","FL":"South","GA":"South","KY":"South",
        "LA":"South","MS":"South","NC":"South","OK":"South","SC":"South","TN":"South",
        "TX":"South","VA":"South","WV":"South","MD":"South",
        "CT":"Northeast","DE":"Northeast","MA":"Northeast","ME":"Northeast","NH":"Northeast",
        "NJ":"Northeast","NY":"Northeast","PA":"Northeast","RI":"Northeast","VT":"Northeast",
        "IL":"Midwest","IN":"Midwest","IA":"Midwest","KS":"Midwest","MI":"Midwest",
        "MN":"Midwest","MO":"Midwest","NE":"Midwest","ND":"Midwest","OH":"Midwest",
        "SD":"Midwest","WI":"Midwest",
        "AK":"West","AZ":"West","CA":"West","CO":"West","HI":"West","ID":"West",
        "MT":"West","NM":"West","NV":"West","OR":"West","UT":"West","WA":"West","WY":"West",
    }
    sm_piv["region"] = sm_piv["state_po"].map(regions_map).fillna("South")
    reg = sm_piv.groupby(["year","region"])[["DEMOCRAT","REPUBLICAN"]].sum().reset_index()
    reg["dem_share"] = reg["DEMOCRAT"]/(reg["DEMOCRAT"]+reg["REPUBLICAN"])*100
    fig21 = px.line(reg, x="year", y="dem_share", color="region", markers=True,
        color_discrete_map={"South":C["rep"],"Northeast":C["dem"],"Midwest":"#7C3AED","West":"#0891B2"},
        labels={"dem_share":"Dem Share (%)","year":"Year","region":"Region"})
    fig21.add_hline(y=50,line_dash="dash",line_color=C["border"],line_width=1.5)
    fig21 = styled(fig21, 420)
    st.plotly_chart(fig21, use_container_width=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;color:{C["subtext"]};font-size:0.75rem;
            padding:30px 0 10px 0;border-top:1px solid {C["border"]};margin-top:40px;'>
  <b>US Presidential Elections Dashboard</b> · 1976–2024 &nbsp;|&nbsp;
  Data: MIT Election Data + Science Lab &nbsp;|&nbsp; Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)