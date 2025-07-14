import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

st.markdown(
    """
    <style>
    .stApp {
        background-color: #111 !important;
        color: ##A855F7 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #A855F7 !important;
    }
    .stMarkdown p, .stMarkdown {
        color: ##A855F7 !important;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111 !important;
    }
    /* Sidebar and widget label text, file name, selectbox, slider, etc. */
    label, .st-af, .st-ag, .st-cq, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz, .st-da, .st-db, .st-dc, .st-dd, .st-de, .st-df, .st-dg, .st-dh, .st-di, .st-dj, .st-dk, .st-dl, .st-dm, .st-dn, .st-do, .st-dp, .st-dq, .st-dr, .st-ds, .st-dt, .st-du, .st-dv, .st-dw, .st-dx, .st-dy, .st-dz, .st-e0, .st-e1, .st-e2, .st-e3, .st-e4, .st-e5, .st-e6, .st-e7, .st-e8, .st-e9, .st-ea, .st-eb, .st-ec, .st-ed, .st-ee, .st-ef, .st-eg, .st-eh, .st-ei, .st-ej, .st-ek, .st-el, .st-em, .st-en, .st-eo, .st-ep, .st-eq, .st-er, .st-es, .st-et, .st-eu, .st-ev, .st-ew, .st-ex, .st-ey, .st-ez, .st-fa, .st-fb, .st-fc, .st-fd, .st-fe, .st-ff, .st-fg, .st-fh, .st-fi, .st-fj, .st-fk, .st-fl, .st-fm, .st-fn, .st-fo, .st-fp, .st-fq, .st-fr, .st-fs, .st-ft, .st-fu, .st-fv, .st-fw, .st-fx, .st-fy, .st-fz, .st-ga, .st-gb, .st-gc, .st-gd, .st-ge, .st-gf, .st-gg, .st-gh, .st-gi, .st-gj, .st-gk, .st-gl, .st-gm, .st-gn, .st-go, .st-gp, .st-gq, .st-gr, .st-gs, .st-gt, .st-gu, .st-gv, .st-gw, .st-gx, .st-gy, .st-gz, .st-ha, .st-hb, .st-hc, .st-hd, .st-he, .st-hf, .st-hg, .st-hh, .st-hi, .st-hj, .st-hk, .st-hl, .st-hm, .st-hn, .st-ho, .st-hp, .st-hq, .st-hr, .st-hs, .st-ht, .st-hu, .st-hv, .st-hw, .st-hx, .st-hy, .st-hz, .st-ia, .st-ib, .st-ic, .st-id, .st-ie, .st-if, .st-ig, .st-ih, .st-ii, .st-ij, .st-ik, .st-il, .st-im, .st-in, .st-io, .st-ip, .st-iq, .st-ir, .st-is, .st-it, .st-iu, .st-iv, .st-iw, .st-ix, .st-iy, .st-iz, .st-ja, .st-jb, .st-jc, .st-jd, .st-je, .st-jf, .st-jg, .st-jh, .st-ji, .st-jj, .st-jk, .st-jl, .st-jm, .st-jn, .st-jo, .st-jp, .st-jq, .st-jr, .st-js, .st-jt, .st-ju, .st-jv, .st-jw, .st-jx, .st-jy, .st-jz, .st-ka, .st-kb, .st-kc, .st-kd, .st-ke, .st-kf, .st-kg, .st-kh, .st-ki, .st-kj, .st-kk, .st-kl, .st-km, .st-kn, .st-ko, .st-kp, .st-kq, .st-kr, .st-ks, .st-kt, .st-ku, .st-kv, .st-kw, .st-kx, .st-ky, .st-kz, .st-la, .st-lb, .st-lc, .st-ld, .st-le, .st-lf, .st-lg, .st-lh, .st-li, .st-lj, .st-lk, .st-ll, .st-lm, .st-ln, .st-lo, .st-lp, .st-lq, .st-lr, .st-ls, .st-lt, .st-lu, .st-lv, .st-lw, .st-lx, .st-ly, .st-lz, .st-ma, .st-mb, .st-mc, .st-md, .st-me, .st-mf, .st-mg, .st-mh, .st-mi, .st-mj, .st-mk, .st-ml, .st-mm, .st-mn, .st-mo, .st-mp, .st-mq, .st-mr, .st-ms, .st-mt, .st-mu, .st-mv, .st-mw, .st-mx, .st-my, .st-mz, .st-na, .st-nb, .st-nc, .st-nd, .st-ne, .st-nf, .st-ng, .st-nh, .st-ni, .st-nj, .st-nk, .st-nl, .st-nm, .st-nn, .st-no, .st-np, .st-nq, .st-nr, .st-ns, .st-nt, .st-nu, .st-nv, .st-nw, .st-nx, .st-ny, .st-nz, .st-oa, .st-ob, .st-oc, .st-od, .st-oe, .st-of, .st-og, .st-oh, .st-oi, .st-oj, .st-ok, .st-ol, .st-om, .st-on, .st-oo, .st-op, .st-oq, .st-or, .st-os, .st-ot, .st-ou, .st-ov, .st-ow, .st-ox, .st-oy, .st-oz, .st-pa, .st-pb, .st-pc, .st-pd, .st-pe, .st-pf, .st-pg, .st-ph, .st-pi, .st-pj, .st-pk, .st-pl, .st-pm, .st-pn, .st-po, .st-pp, .st-pq, .st-pr, .st-ps, .st-pt, .st-pu, .st-pv, .st-pw, .st-px, .st-py, .st-pz, .st-qa, .st-qb, .st-qc, .st-qd, .st-qe, .st-qf, .st-qg, .st-qh, .st-qi, .st-qj, .st-qk, .st-ql, .st-qm, .st-qn, .st-qo, .st-qp, .st-qq, .st-qr, .st-qs, .st-qt, .st-qu, .st-qv, .st-qw, .st-qx, .st-qy, .st-qz, .st-ra, .st-rb, .st-rc, .st-rd, .st-re, .st-rf, .st-rg, .st-rh, .st-ri, .st-rj, .st-rk, .st-rl, .st-rm, .st-rn, .st-ro, .st-rp, .st-rq, .st-rr, .st-rs, .st-rt, .st-ru, .st-rv, .st-rw, .st-rx, .st-ry, .st-rz, .st-sa, .st-sb, .st-sc, .st-sd, .st-se, .st-sf, .st-sg, .st-sh, .st-si, .st-sj, .st-sk, .st-sl, .st-sm, .st-sn, .st-so, .st-sp, .st-sq, .st-sr, .st-ss, .st-st, .st-su, .st-sv, .st-sw, .st-sx, .st-sy, .st-sz, .st-ta, .st-tb, .st-tc, .st-td, .st-te, .st-tf, .st-tg, .st-th, .st-ti, .st-tj, .st-tk, .st-tl, .st-tm, .st-tn, .st-to, .st-tp, .st-tq, .st-tr, .st-ts, .st-tt, .st-tu, .st-tv, .st-tw, .st-tx, .st-ty, .st-tz, .st-ua, .st-ub, .st-uc, .st-ud, .st-ue, .st-uf, .st-ug, .st-uh, .st-ui, .st-uj, .st-uk, .st-ul, .st-um, .st-un, .st-uo, .st-up, .st-uq, .st-ur, .st-us, .st-ut, .st-uu, .st-uv, .st-uw, .st-ux, .st-uy, .st-uz, .st-va, .st-vb, .st-vc, .st-vd, .st-ve, .st-vf, .st-vg, .st-vh, .st-vi, .st-vj, .st-vk, .st-vl, .st-vm, .st-vn, .st-vo, .st-vp, .st-vq, .st-vr, .st-vs, .st-vt, .st-vu, .st-vv, .st-vw, .st-vx, .st-vy, .st-vz, .st-wa, .st-wb, .st-wc, .st-wd, .st-we, .st-wf, .st-wg, .st-wh, .st-wi, .st-wj, .st-wk, .st-wl, .st-wm, .st-wn, .st-wo, .st-wp, .st-wq, .st-wr, .st-ws, .st-wt, .st-wu, .st-wv, .st-ww, .st-wx, .st-wy, .st-wz, .st-xa, .st-xb, .st-xc, .st-xd, .st-xe, .st-xf, .st-xg, .st-xh, .st-xi, .st-xj, .st-xk, .st-xl, .st-xm, .st-xn, .st-xo, .st-xp, .st-xq, .st-xr, .st-xs, .st-xt, .st-xu, .st-xv, .st-xw, .st-xx, .st-xy, .st-xz, .st-ya, .st-yb, .st-yc, .st-yd, .st-ye, .st-yf, .st-yg, .st-yh, .st-yi, .st-yj, .st-yk, .st-yl, .st-ym, .st-yn, .st-yo, .st-yp, .st-yq, .st-yr, .st-ys, .st-yt, .st-yu, .st-yv, .st-yw, .st-yx, .st-yy, .st-yz, .st-za, .st-zb, .st-zc, .st-zd, .st-ze, .st-zf, .st-zg, .st-zh, .st-zi, .st-zj, .st-zk, .st-zl, .st-zm, .st-zn, .st-zo, .st-zp, .st-zq, .st-zr, .st-zs, .st-zt, .st-zu, .st-zv, .st-zw, .st-zx, .st-zy, .st-zz {
        color: #A855F7 !important;
    }
    /* File name in uploader */
    .st-dq, .st-emotion-cache-1cypcdb {
        color: #A855F7 !important;
    }
    /* Expander/section frame border */
    .stExpander, .st-emotion-cache-1v0mbdj, .st-emotion-cache-1vzeuhh {
        border: 2px solid #A855F7 !important;
        box-shadow: 0 0 8px #7209b744 !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# main section
st.markdown('<h1 style="color:#A855F7;">ML Model App</h1>', unsafe_allow_html=True)

def compute_data_quality_metrics(df):

    total_cells = df.size
    non_null_cells = df.count().sum()
    completeness_pct = round((non_null_cells / total_cells) * 100, 2)
    missing_pct = round(100 - completeness_pct, 2)
    duplicate_pct = round((df.duplicated().sum() / len(df)) * 100, 2)
    constant_cols_pct = round((sum(df.nunique(dropna=False) == 1) / df.shape[1]) * 100, 2)
    high_cardinality_pct = round((sum(df.nunique(dropna=False) > 0.9 * len(df)) / df.shape[1]) * 100, 2)
    numeric_cols = df.select_dtypes(include='number').shape[1]
    total_cols = df.shape[1]
    valid_type_pct = round((numeric_cols / total_cols) * 100, 2)

    score = (
        0.30 * completeness_pct +
        0.15 * (100 - duplicate_pct) +
        0.15 * (100 - constant_cols_pct) +
        0.15 * (100 - high_cardinality_pct) +
        0.25 * valid_type_pct
    )
    score = round(score, 2)

    return {
        "Completeness %": completeness_pct,
        "Missing Value %": missing_pct,
        "Duplicate Rows %": duplicate_pct,
        "Constant Columns %": constant_cols_pct,
        "High Cardinality Columns %": high_cardinality_pct,
        "Data Quality Score %": score
    }

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.sidebar.markdown("### File Info")
        st.sidebar.write(f"**Name:** {uploaded_file.name}")
        st.sidebar.write(f"**Size:** {file_size:.2f} MB")
        
        if file_size > 200:
            st.error("File size exceeds 200MB limit. Please upload a smaller file.")
            st.stop()
        
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.sidebar.write(f"**Shape:** {df.shape}")
        st.sidebar.markdown(
            """
            <div style="
                background-color:#222; 
                border:2px solid #A855F7; 
                border-radius:8px; 
                padding:8px; 
                margin:8px 0; 
                color:#A855F7; 
                font-weight:bold; 
                display:flex; 
                align-items:center;
            ">
            ✅ <span style="margin-left:8px;">Successfully loaded!</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.sidebar.divider()
        st.sidebar.markdown("### Data Quality Metrics")
        metrics = compute_data_quality_metrics(df)
        for k, v in metrics.items():
            if k == "Data Quality Score %":
                if v > 90:
                    icon = "🟢"
                elif v >= 70:
                    icon = "🟡"
                else:
                    icon = "🔴"
                st.sidebar.write(f"{k}: {v} {icon}")
            else:
                st.sidebar.write(f"{k}: {v}")
        st.sidebar.divider()
        st.sidebar.markdown("### Data Preview")
        st.sidebar.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.write("This might be due to:")
        st.write("- File corruption")
        st.write("- Unsupported file format")
        st.write("- File size too large")
        st.write("- Permission issues")
        st.stop()
else:
    st.sidebar.markdown(
        """
        <div style="
            background-color:#222; 
            border:2px solid #A855F7; 
            border-radius:8px; 
            padding:8px; 
            margin:8px 0; 
            color:#A855F7; 
            font-weight:bold; 
            display:flex; 
            align-items:center;
        ">
        ⚠️ <span style="margin-left:8px;">Please upload a CSV or Excel file.</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# target column selection
with st.expander("Data"):

    st.write("**Select your target column:**")
    y_col = st.selectbox("Target column (y)", df.columns)

    if 'df_active' not in st.session_state:
        st.session_state['df_active'] = df
    if 'oversampled' not in st.session_state:
        st.session_state['oversampled'] = False

    df_active = st.session_state['df_active']

    if df_active[y_col].nunique() == 2:
        counts = df_active[y_col].value_counts()
        imbalance = counts.get(0, 0) / counts.get(1, 1) if counts.get(1, 1) != 0 else float('inf')
        st.write(f"**Target imbalance:** {imbalance:.2f}")

        if imbalance > 5.67:
            st.markdown(
                """
                <div style="
                    background-color: #2d0a4b;
                    border: 2px solid #A855F7;
                    border-radius: 10px;
                    color: #E9D5FF;
                    padding: 12px 18px;
                    margin: 10px 0;
                    font-weight: bold;
                    box-shadow: 0 0 8px #7209b744;
                    display: flex;
                    align-items: center;
                ">
                    ⚠️ Target class imbalance is high. Consider using techniques like oversampling.
                </div>
                """,
                unsafe_allow_html=True
            )
            desired_imbalance = st.number_input(
                "Choose your desired target imbalance (0/1 ratio):",
                min_value=1.86, max_value=5.67, value=2.0, step=0.1
            )
            if st.button("Oversample to desired imbalance"):
                class_0 = df_active[df_active[y_col] == 0]
                class_1 = df_active[df_active[y_col] == 1]
                n_class_1 = len(class_1)
                n_class_0_new = int(desired_imbalance * n_class_1)
                class_0_oversampled = class_0.sample(n=n_class_0_new, replace=True, random_state=42)
                df_oversampled = pd.concat([class_0_oversampled, class_1], ignore_index=True)
                st.session_state['df_active'] = df_oversampled
                st.session_state['oversampled'] = True
                st.markdown(
                    """
                    <div style="
                        background-color: #2d0a4b; 
                        border:2px solid #A855F7; 
                        border-radius:10px; 
                        color:#E9D5FF; 
                        padding:12px 18px; 
                        margin:10px 0; 
                        font-weight:bold; 
                        box-shadow:0 0 8px #7209b744; 
                        display:flex; 
                        align-items:center;
                    ">
                        ✅ Oversampling done!
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                df_active = df_oversampled
                
        elif imbalance < 1.86:
            st.markdown(
                """
                <div style="
                    background-color: #2d0a4b;
                    border: 2px solid #A855F7;
                    border-radius: 10px;
                    color: #E9D5FF;
                    padding: 12px 18px;
                    margin: 10px 0;
                    font-weight: bold;
                    box-shadow: 0 0 8px #7209b744;
                    display: flex;
                    align-items: center;
                ">
                    ⚠️ Target class imbalance is low. Consider using techniques like undersampling.
                </div>
                """,
                unsafe_allow_html=True
            )
            desired_imbalance = st.number_input(
                "Choose your desired target imbalance (0/1 ratio):",
                min_value=1.86, max_value=5.67, value=2.0, step=0.1
            )
            if st.button("Undersample to desired imbalance"):
                class_0 = df_active[df_active[y_col] == 0]
                class_1 = df_active[df_active[y_col] == 1]
                n_class_1_new = int(len(class_0) / desired_imbalance)
                class_1_undersampled = class_1.sample(n=n_class_1_new, replace=False, random_state=42)
                df_undersampled = pd.concat([class_0, class_1_undersampled], ignore_index=True)
                st.session_state['df_active'] = df_undersampled
                st.session_state['oversampled'] = False
                st.markdown(
                    """
                    <div style="
                        background-color: #2d0a4b; 
                        border:2px solid #A855F7; 
                        border-radius:10px; 
                        color:#E9D5FF; 
                        padding:12px 18px; 
                        margin:10px 0; 
                        font-weight:bold; 
                        box-shadow:0 0 8px #7209b744; 
                        display:flex; 
                        align-items:center;
                    ">
                        ✅ Undersampling done!
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                df_active = df_undersampled

    else:
        st.info("Selected target is not binary")

# data explore
with st.expander("Data Exploration"):
    if st.button("Explore Data"):
        st.session_state['show_explore'] = True
    if st.session_state.get('show_explore', False):
        st.write("**Descriptive statistics:**")
        st.dataframe(df_active.describe(include='all').T)

        st.write("**Feature histograms:**")
        feature = st.selectbox("Select feature for histogram", [col for col in df_active.columns if col != y_col], key="explore_hist_feature")
        if pd.api.types.is_numeric_dtype(df_active[feature]):
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df_active[feature], marker_color='#A855F7'))
            fig.update_layout(title=f"Histogram of {feature}", xaxis_title=feature, yaxis_title="Count", xaxis_showgrid=False, yaxis_showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = go.Figure()
            counts = df_active[feature].value_counts()
            fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts.values, marker_color='#A855F7'))
            fig.update_layout(title=f"Bar plot of {feature}", xaxis_title=feature, yaxis_title="Count", xaxis_showgrid=False, yaxis_showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

        st.write("**Correlation heatmap:**")
        corr = df_active.corr(numeric_only=True)
        if not corr.empty:
            import plotly.express as px
            fig = px.imshow(
                corr,
                color_continuous_scale=["#E9D5FF", "#C084FC", "#A855F7", "#6D28D9"],
                labels=dict(color="Correlation"),
                aspect="auto"
            )
            fig.update_layout(
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                font=dict(color="#A855F7"),
                plot_bgcolor="#111",
                paper_bgcolor="#111"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric features for correlation heatmap.")

# data cleaning
with st.expander("Data Cleaning"):
    df_active = st.session_state['df_active']
    st.write("**Drop columns by null threshold:**")
    null_thresh = st.slider("Set null threshold (fraction)", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    null_frac = df_active.isnull().mean()
    cols_to_drop = null_frac[null_frac > null_thresh].index.tolist()
    st.write(f"Columns above threshold: {cols_to_drop}")
    if st.button("Drop selected columns"):
        df_active = df_active.drop(columns=cols_to_drop)
        st.session_state['df_active'] = df_active
        st.markdown(
            """
            <div style="
                background-color: #2d0a4b; 
                border:2px solid #A855F7; 
                border-radius:10px; 
                color:#E9D5FF; 
                padding:12px 18px; 
                margin:10px 0; 
                font-weight:bold; 
                box-shadow:0 0 8px #7209b744; 
                display:flex; 
                align-items:center;
            ">
                ✅ Successfully dropped!
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("**Fill null values:**")
    num_fill = st.selectbox("Fill numeric nulls with:", ["0", "mean", "median", "drop rows", "keep nulls"])
    cat_fill = st.selectbox("Fill categorical nulls with:", ["unknown", "mode", "drop rows", "keep nulls"])
    if st.button("Apply null filling"):
        df_temp = df_active.copy()
        num_cols = df_temp.select_dtypes(include='number').columns
        cat_cols = df_temp.select_dtypes(exclude='number').columns
        if num_fill == "0":
            df_temp[num_cols] = df_temp[num_cols].fillna(0)
        elif num_fill == "mean":
            df_temp[num_cols] = df_temp[num_cols].fillna(df_temp[num_cols].mean())
        elif num_fill == "median":
            df_temp[num_cols] = df_temp[num_cols].fillna(df_temp[num_cols].median())
        elif num_fill == "drop rows":
            df_temp = df_temp.dropna(subset=num_cols)
        # else keep nulls
        if cat_fill == "unknown":
            df_temp[cat_cols] = df_temp[cat_cols].fillna("unknown")
        elif cat_fill == "mode":
            for col in cat_cols:
                mode = df_temp[col].mode()
                if not mode.empty:
                    df_temp[col] = df_temp[col].fillna(mode[0])
        elif cat_fill == "drop rows":
            df_temp = df_temp.dropna(subset=cat_cols)
        # else keep nulls
        st.session_state['df_active'] = df_temp
        st.markdown(
            """
            <div style="
                background-color: #2d0a4b; 
                border:2px solid #A855F7; 
                border-radius:10px; 
                color:#E9D5FF; 
                padding:12px 18px; 
                margin:10px 0; 
                font-weight:bold; 
                box-shadow:0 0 8px #7209b744; 
                display:flex; 
                align-items:center;
            ">
                ✅ Successfully filled!
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("**Outlier detection and handling:**")
    outlier_method = st.selectbox("Choose outlier detection method:", ["IQR", "Z-score", "Isolation Forest", "DBSCAN"])
    outlier_cols = st.multiselect("Select columns for outlier detection", df_active.select_dtypes(include='number').columns)
    if outlier_cols and st.button("Detect outliers"):
        df_temp = df_active.copy()
        outlier_mask = pd.Series([False]*len(df_temp))
        if outlier_method == "IQR":
            for col in outlier_cols:
                Q1 = df_temp[col].quantile(0.25)
                Q3 = df_temp[col].quantile(0.75)
                IQR = Q3 - Q1
                mask = (df_temp[col] < (Q1 - 1.5 * IQR)) | (df_temp[col] > (Q3 + 1.5 * IQR))
                outlier_mask = outlier_mask | mask
        elif outlier_method == "Z-score":
            from scipy.stats import zscore
            for col in outlier_cols:
                mask = np.abs(zscore(df_temp[col].dropna())) > 3
                mask_full = pd.Series(False, index=df_temp.index)
                mask_full[df_temp[col].dropna().index[mask]] = True
                outlier_mask = outlier_mask | mask_full
        elif outlier_method == "Isolation Forest":
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=0.05, random_state=42)
            mask = iso.fit_predict(df_temp[outlier_cols]) == -1
            outlier_mask = outlier_mask | pd.Series(mask, index=df_temp.index)
        elif outlier_method == "DBSCAN":
            from sklearn.cluster import DBSCAN
            db = DBSCAN(eps=3, min_samples=5)
            labels = db.fit_predict(df_temp[outlier_cols])
            mask = labels == -1
            outlier_mask = outlier_mask | pd.Series(mask, index=df_temp.index)
        st.session_state['outlier_mask'] = outlier_mask
        st.session_state['outlier_df_temp'] = df_temp
        st.write(f"Outliers detected: {outlier_mask.sum()} rows")
    if 'outlier_mask' in st.session_state and st.button("Drop detected outliers"):
        df_temp = st.session_state['outlier_df_temp']
        outlier_mask = st.session_state['outlier_mask']
        df_active = df_temp[~outlier_mask].reset_index(drop=True)
        st.session_state['df_active'] = df_active
        st.markdown(
            """
            <div style="
                background-color: #2d0a4b; 
                border:2px solid #A855F7; 
                border-radius:10px; 
                color:#E9D5FF; 
                padding:12px 18px; 
                margin:10px 0; 
                font-weight:bold; 
                box-shadow:0 0 8px #7209b744; 
                display:flex; 
                align-items:center;
            ">
                ✅ Successfully handle outlier!
            </div>
            """,
            unsafe_allow_html=True
        )
        del st.session_state['outlier_mask']
        del st.session_state['outlier_df_temp']

#feature selection
with st.expander("Feature Selection"):
    st.write("**Feature selection using feature importance and correlation filtering:**")

    id_col = st.selectbox("Select ID column", df_active.columns)
    target_col = st.selectbox("Select target column", [col for col in df_active.columns if col != id_col])

    df_work = df_active.copy()


    threshold = st.slider("Correlation threshold for dropping features", min_value=0.5, max_value=1.0, value=0.9, step=0.05)
    top_percent = st.slider("Top percent of features to keep", min_value=1, max_value=100, value=20, step=1)
    test_size = st.slider("Test size for train/test split", min_value=0.1, max_value=0.5, value=0.3, step=0.01, format="%.2f")

    if st.button("Select Features"):
        def clean_feature_names(df):
            df.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '')
                             .replace(' ', '_').replace('"', '').replace("'", '')
                             .replace('{', '').replace('}', '').replace(':', '') for col in df.columns]
            return df

        def remove_high_corr_features(X, importance_dict, threshold=0.9):
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = set()
            for col in upper.columns:
                for row in upper.index:
                    if upper.loc[row, col] > threshold:
                        if importance_dict.get(row, 0) > importance_dict.get(col, 0):
                            to_drop.add(col)
                        else:
                            to_drop.add(row)
            return X.drop(columns=to_drop)

        def get_feature_importance(model, feature_names, model_name):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            importance_df['Importance'] = importance_df['Importance'].apply(lambda x: f"{x:.4f}")
            st.write(f"🔹 Feature Importance for {model_name}:")
            st.dataframe(importance_df)
            return importance_df

        def select_top_features(model, X_train, y_train, model_name, threshold, top_percent):
            model.fit(X_train, y_train)
            importances = model.feature_importances_
            importance_dict = dict(zip(X_train.columns, importances))

            X_filtered = remove_high_corr_features(X_train, importance_dict, threshold=threshold)

            model.fit(X_filtered, y_train)
            importances = model.feature_importances_
            feature_df = pd.DataFrame({'Feature': X_filtered.columns, 'Importance': importances})

            quantile_threshold = feature_df['Importance'].quantile(1 - top_percent / 100)
            selected_features_df = feature_df[feature_df['Importance'] >= quantile_threshold].sort_values(by='Importance', ascending=False)

            return selected_features_df['Feature'].values

        df_raw = clean_feature_names(df_work)
        X = df_raw.drop(columns=[id_col, target_col])
        y = df_raw[target_col]

        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42),
            'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42), 
        }

        selected_features_dict = {}
        feature_importance_dict = {}

        for name, model in models.items():
            selected_features = select_top_features(model, X_train, y_train, name, threshold, top_percent)
            selected_features_dict[name] = selected_features
            fitted_model = model.fit(X_train[selected_features], y_train)
            feature_importance_dict[name] = get_feature_importance(fitted_model, selected_features, name)

        feature_stats = {}
        for name, importance_df in feature_importance_dict.items():
            imp = importance_df['Importance'].astype(float)
            norm_importance = (imp - imp.min()) / (imp.max() - imp.min() + 1e-9)
            for feature, norm_imp in zip(importance_df['Feature'], norm_importance):
                if feature not in feature_stats:
                    feature_stats[feature] = {'count': 0, 'total_importance': 0.0, 'models': []}
                feature_stats[feature]['count'] += 1
                feature_stats[feature]['total_importance'] += norm_imp
                feature_stats[feature]['models'].append(name)

        feature_ranking = pd.DataFrame([
            {
                'Feature': feat,
                'Models_Selected': stat['count'],
                'Avg_Normalized_Importance': stat['total_importance'] / stat['count']
            }
            for feat, stat in feature_stats.items()
        ])

        feature_ranking = feature_ranking.sort_values(
            by=['Models_Selected', 'Avg_Normalized_Importance'], ascending=[False, False]
        )

        st.write("Feature Ranking (Top 30)")
        st.dataframe(feature_ranking.head(30))

        suspicious_features = []
        for name, importance_df in feature_importance_dict.items():
            imp = importance_df['Importance'].astype(float)
            norm_importance = (imp - imp.min()) / (imp.max() - imp.min() + 1e-9)
            for feature, norm_imp in zip(importance_df['Feature'], norm_importance):
                if norm_imp > 0.95:
                    suspicious_features.append((feature, name, norm_imp))

        if suspicious_features:
            st.markdown(
                """
                <div style="
                    background-color: #2d0a4b; 
                    border:2px solid #A855F7; 
                    border-radius:10px; 
                    color:#E9D5FF; 
                    padding:12px 18px; 
                    margin:10px 0; 
                    font-weight:bold; 
                    box-shadow:0 0 8px #7209b744; 
                    display:flex; 
                    align-items:center;
                "">
                    ⚠️ Suspicious features with very high normalized importance (>0.95)!
                </div>
                """,
                unsafe_allow_html=True
            )

            for feat, model, norm_imp in suspicious_features:
                st.write(f"Feature: {feat} | Model: {model} | Normalized Importance: {norm_imp:.2f}")
        
        else:
            st.markdown(
                """
                <div style="
                    background-color: #2d0a4b; 
                    border:2px solid #A855F7; 
                    border-radius:10px; 
                    color:#E9D5FF; 
                    padding:12px 18px; 
                    margin:10px 0; 
                    font-weight:bold; 
                    box-shadow:0 0 8px #7209b744; 
                    display:flex; 
                    align-items:center;
                "">
                    ⚠️ No suspicious features detected with normalized importance > 0.95
                </div>
                """,
                unsafe_allow_html=True
            )

        st.session_state['selected_features_dict'] = selected_features_dict
        st.session_state['suspicious_features'] = suspicious_features
        st.session_state['models'] = models
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['feature_selected'] = True

#modeling
with st.expander("Modeling"):

    selected_features_dict = st.session_state.get('selected_features_dict', {})
    suspicious_features = st.session_state.get('suspicious_features', [])
    models = st.session_state.get('models', {})
    X_train = st.session_state.get('X_train', None)
    X_test = st.session_state.get('X_test', None)
    y_train = st.session_state.get('y_train', None)
    y_test = st.session_state.get('y_test', None)
    feature_selected = st.session_state.get('feature_selected', False)
    if not feature_selected or not selected_features_dict:
        st.markdown(
            """
            <div style="
                background-color: #2d0a4b; 
                border:2px solid #A855F7; 
                border-radius:10px; 
                color:#E9D5FF; 
                padding:12px 18px; 
                margin:10px 0; 
                font-weight:bold; 
                box-shadow:0 0 8px #7209b744; 
                display:flex; 
                align-items:center;
            "">
                ⚠️ Please run feature selection before modeling.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()

    def filter_suspicious_features(selected_features_dict, suspicious_features):
        sus_feature_names = set([feat for feat, _, _ in suspicious_features])
        filtered_dict = {}
        for name in selected_features_dict:
            filtered = [f for f in selected_features_dict[name] if f not in sus_feature_names]
            filtered_dict[name] = filtered
        return filtered_dict

    def evaluate_model(model, X_train, X_test, y_train, y_test, selected_features):
        model.fit(X_train[selected_features], y_train)
        y_train_pred = model.predict_proba(X_train[selected_features])[:, 1]
        y_test_pred = model.predict_proba(X_test[selected_features])[:, 1]
        y_test_pred_label = model.predict(X_test[selected_features])
        return {
            'Train ROC AUC': roc_auc_score(y_train, y_train_pred),
            'Test ROC AUC': roc_auc_score(y_test, y_test_pred),
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'y_test_pred_label': y_test_pred_label
        }

    filtered_features_dict = filter_suspicious_features(selected_features_dict, suspicious_features)
    results_dict = {}

    fig_train_roc = go.Figure()
    for name, model in models.items():
        features = filtered_features_dict.get(name, [])
        if not features or X_train is None or X_test is None or y_train is None or y_test is None:
            st.warning(f"No features left for {name} after removing suspicious features or missing training data.")
            continue
        results = evaluate_model(model, X_train, X_test, y_train, y_test, features)
        results_dict[name] = results
        fpr, tpr, _ = roc_curve(y_train, results['y_train_pred'])
        roc_auc = auc(fpr, tpr)
        fig_train_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={roc_auc:.3f})'))
    fig_train_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='navy'), showlegend=False
    ))
    fig_train_roc.update_layout(
        title="Train ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=800, height=500, legend=dict(x=0.7, y=0.05), template="plotly_white"
    )
    st.plotly_chart(fig_train_roc, use_container_width=True)

    fig_test_roc = go.Figure()
    for name, results in results_dict.items():
        fpr, tpr, _ = roc_curve(y_test, results['y_test_pred'])
        roc_auc = auc(fpr, tpr)
        fig_test_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={roc_auc:.3f})'))
    fig_test_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='navy'), showlegend=False
    ))
    fig_test_roc.update_layout(
        title="Test ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=800, height=500, legend=dict(x=0.7, y=0.05), template="plotly_white"
    )
    st.plotly_chart(fig_test_roc, use_container_width=True)

    if results_dict:
        model_names = list(results_dict.keys())
        train_roc = [results_dict[n]['Train ROC AUC'] for n in model_names]
        test_roc = [results_dict[n]['Test ROC AUC'] for n in model_names]
        power_loss = [tr - te for tr, te in zip(train_roc, test_roc)]

        fig_loss = go.Figure(data=[
            go.Bar(x=model_names, y=power_loss, marker_color='#A855F7')
        ])
        fig_loss.update_layout(
            title="Power Loss (Train ROC - Test ROC) for Each Model",
            yaxis_title="Power Loss",
            xaxis_title="Model",
            width=700,
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    st.subheader("Model Performance Metrics")
    for name, results in results_dict.items():
        y_test_pred_label = results['y_test_pred_label']
        acc = accuracy_score(y_test, y_test_pred_label)
        prec = precision_score(y_test, y_test_pred_label, zero_division=0)
        rec = recall_score(y_test, y_test_pred_label, zero_division=0)
        f1 = f1_score(y_test, y_test_pred_label, zero_division=0)
        cm = confusion_matrix(y_test, y_test_pred_label)
        st.markdown(f"**{name}**")
        st.write(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1-score: {f1:.3f}")
        st.write("Confusion Matrix:")
        import plotly.express as px
        cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"])
        fig_cm = px.imshow(
            cm_df,
            color_continuous_scale=["#E9D5FF", "#C084FC", "#A855F7", "#6D28D9"],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            text_auto=True,
            aspect="auto"
        )
        fig_cm.update_layout(
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            font=dict(color="#A855F7"),
            plot_bgcolor="#111",
            paper_bgcolor="#111",
            width=400,
            height=300
        )
        st.plotly_chart(fig_cm, use_container_width=False)
                
                









