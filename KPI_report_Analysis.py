import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.mixture import GaussianMixture
import io

st.set_page_config(
    page_title="Performance Bias Detection System - Banpu HR",
    layout="wide"
)

# =====================================================
# ฟังก์ชันวิเคราะห์
# =====================================================

def detect_outliers_with_gmm(df):
    """ตรวจจับ Outliers ด้วย Gaussian Mixture Model"""
    
    df_clustering = df.copy()
    outlier_records = []
    
    fig = go.Figure()
    period_list = list(df_clustering.groupby(["Assessment Year", "Period"]).groups.keys())
    
    for i, ((year, period), g) in enumerate(df_clustering.groupby(["Assessment Year", "Period"])):
        
        X = g[["Behavioral score", "Performance score"]].values
        
        gmm = GaussianMixture(n_components=3, random_state=42)
        g["cluster"] = gmm.fit_predict(X)
        g["log_likelihood"] = gmm.score_samples(X)
        
        threshold = np.percentile(g["log_likelihood"], 5)
        g["is_outlier"] = g["log_likelihood"] < threshold
        
        outlier_year = g[g["is_outlier"] == True][
            ["Appraisee name", "Appraiser Name", "Performance score", "Behavioral score"]
        ]
        
        for _, row in outlier_year.iterrows():
            outlier_records.append({
                "Appraisee name": row["Appraisee name"],
                "Appraiser Name": row["Appraiser Name"],
                "Assessment Year": year,
                "Period": period
            })
        
        fig.add_trace(
            go.Scatter(
                x=g["Behavioral score"],
                y=g["Performance score"],
                mode="markers",
                visible=(i == 0),
                marker=dict(
                    size=9,
                    symbol=np.where(g["is_outlier"], "x", "circle"),
                    color=np.where(g["is_outlier"], "red", "blue")
                ),
                text=(
                    "Appraisee: " + g["Appraisee name"].astype(str) +
                    "<br>Appraiser: " + g["Appraiser Name"].astype(str)
                ),
                hovertemplate=
                    "%{text}<br>" +
                    "Behavioral: %{x}<br>" +
                    "Performance: %{y}<extra></extra>"
            )
        )
    
    buttons = []
    for i, (year, period) in enumerate(period_list):
        vis = [False] * len(period_list)
        vis[i] = True
        buttons.append(
            dict(
                label=f"{year} - {period}",
                method="update",
                args=[{"visible": vis}]
            )
        )
    
    fig.update_layout(
        title="Behavioral vs Performance with GMM Outliers",
        xaxis_title="Behavioral Score",
        yaxis_title="Performance Score",
        height=600,
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=1.15,
                y=1
            )
        ]
    )
    
    outlier_df = pd.DataFrame(outlier_records)
    
    return fig, outlier_df


def analyze_temporal_trends(df):
    """วิเคราะห์แนวโน้มของ Appraiser ตามเวลา"""
    
    df = df.copy()
    df['year_period'] = df['Assessment Year'].astype(str) + '_' + df['Period']
    df['sort_key'] = df['Assessment Year'] * 10 + df['Period'].map({'1H': 1, '2H': 2})
    df = df.sort_values('sort_key')
    
    period_scores = (
        df
        .groupby(['Appraiser Name', 'year_period', 'sort_key'])
        .agg(
            score_mean=('Total', 'mean'),
            count=('Total', 'count'),
            unique_appraisees=('Appraisee name', 'nunique')
        )
        .round(2)
        .reset_index()
    )
    
    trend_results = []
    
    for appraiser in period_scores['Appraiser Name'].unique():
        
        appraiser_data = (
            period_scores[period_scores['Appraiser Name'] == appraiser]
            .sort_values('sort_key')
        )
        
        periods = appraiser_data['year_period'].values
        scores = appraiser_data['score_mean'].values
        counts = appraiser_data['count'].values
        
        if len(scores) < 2:
            continue
        
        x = np.arange(len(scores))
        slope, intercept = np.polyfit(x, scores, 1)
        
        y_pred = slope * x + intercept
        ss_res = np.sum((scores - y_pred) ** 2)
        ss_tot = np.sum((scores - np.mean(scores)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        score_std = np.std(scores)
        score_range = scores.max() - scores.min()
        
        trend_results.append({
            'Appraiser Name': appraiser,
            'periods_count': len(periods),
            'first_period': periods[0],
            'last_period': periods[-1],
            'first_score': round(scores[0], 2),
            'last_score': round(scores[-1], 2),
            'score_change': round(scores[-1] - scores[0], 2),
            'avg_score': round(np.mean(scores), 2),
            'score_std': round(score_std, 2),
            'score_range': round(score_range, 2),
            'trend_slope': round(slope, 3),
            'trend_intercept': round(intercept, 3),
            'trend_r_squared': round(r_squared, 3),
            # 'unique_appraisee_total': int(appraiser_data['unique_appraisees'].sum()),
            'evaluation_time_count': int(counts.sum()),
            'score_timeline': ' → '.join([f"{s:.1f}" for s in scores])
        })
    
    trend_df = pd.DataFrame(trend_results)
    
    def classify_trend(row):
        slope = row['trend_slope']
        r_sq = row['trend_r_squared']
        
        if r_sq < 0.3:
            consistency = "Low"
        elif r_sq < 0.7:
            consistency = "Medium"
        else:
            consistency = "High"
        
        if slope > 0.5:
            direction = "Increasing"
        elif slope < -0.5:
            direction = "Decreasing"
        else:
            direction = "Stable"
        
        return f"{direction} ({consistency} consistency)"
    
    trend_df['trend_pattern'] = trend_df.apply(classify_trend, axis=1)
    trend_df = trend_df.sort_values('trend_slope', ascending=False)
    
    return trend_df, period_scores


def plot_appraiser_trends(period_scores, trend_df):
    """สร้างกราฟแนวโน้มของ Appraiser"""
    
    fig = go.Figure()
    appraiser_list = period_scores['Appraiser Name'].unique()
    
    for i, appraiser in enumerate(appraiser_list):
        
        g = (
            period_scores[period_scores['Appraiser Name'] == appraiser]
            .sort_values('sort_key')
        )
        
        trend_row = trend_df[trend_df['Appraiser Name'] == appraiser]
        
        if len(trend_row) == 0:
            continue
        
        slope = trend_row['trend_slope'].values[0]
        intercept = trend_row['trend_intercept'].values[0]
        
        x_labels = g['year_period'].values
        y_scores = g['score_mean'].values
        x_num = np.arange(len(y_scores))
        y_reg = slope * x_num + intercept
        
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=y_scores,
                mode='lines+markers',
                name=f"{appraiser} (Actual)",
                visible=True if i == 0 else False,
                customdata=g['count'],
                hovertemplate=
                    "Appraiser: %{text}<br>" +
                    "Period: %{x}<br>" +
                    "Avg Score: %{y}<br>" +
                    "Count: %{customdata}<extra></extra>",
                text=[appraiser] * len(g)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=y_reg,
                mode='lines',
                name=f"{appraiser} (Trend)",
                visible=True if i == 0 else False,
                line=dict(dash='dash'),
                hovertemplate="Regression: %{y}<extra></extra>"
            )
        )
    
    buttons = []
    total_traces = len(appraiser_list) * 2
    
    for i, appraiser in enumerate(appraiser_list):
        visible = [False] * total_traces
        visible[i*2] = True
        visible[i*2 + 1] = True
        
        buttons.append(
            dict(
                label=str(appraiser),
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"Trend Analysis: {appraiser}"}
                ]
            )
        )
    
    fig.update_layout(
        title=f"Trend Analysis: {appraiser_list[0]}",
        xaxis_title="Period",
        yaxis_title="Average Total Score",
        height=600,
        showlegend=False,
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=1.15,
                y=1
            )
        ]
    )
    
    return fig


def calculate_bias_score(trend_df, outlier_df):
    """คำนวณคะแนน Bias และจัดระดับความเสี่ยง"""
    
    outlier_count = outlier_df.groupby("Appraiser Name").size().reset_index(name="outlier_freq")
    
    bias_df = trend_df.merge(outlier_count, on="Appraiser Name", how="left")
    bias_df["outlier_freq"] = bias_df["outlier_freq"].fillna(0)
        
    # ตรวจว่าเป็น Stable หรือไม่
    bias_df["is_stable"] = bias_df["trend_pattern"].str.contains("Stable")

    # -------------------------
    # จัดกลุ่ม 3 กรณี
    # -------------------------
    def classify_bias_case(row):
        stable = row["is_stable"]
        has_outlier = row["outlier_freq"] > 0

        if stable and has_outlier:
            return "case 1: ให้คะแนนสม่ำเสมอ | มี Outlier"
        elif not stable and has_outlier:
            return "case 2: ให้คะแนนไม่สม่ำเสมอ | มี Outlier"
        elif not stable and not has_outlier:
            return "case 3: ให้คะแนนไม่สม่ำเสมอ | ไม่มี Outlier"
        elif stable and not has_outlier:
            return "case 4: ให้คะแนนสม่ำเสมอ | ไม่มี Outlier"

    
    bias_df["BIAS_DETECTION"] = bias_df.apply(classify_bias_case, axis=1)
    
    return bias_df


def convert_df_to_excel(dataframes_dict):
    """แปลง DataFrames หลายตัวเป็นไฟล์ Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


# =====================================================
# Streamlit UI
# =====================================================

st.title("Performance Bias Detection System - Banpu HR")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "เลือกไฟล์ CSV หรือ Excel",
        type=['csv', 'xlsx', 'xls'],
        help="อัพโหลดไฟล์ข้อมูลการประเมิน"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ System Info")
    st.info("""
    **Features:**
    - Outlier Detection (GMM)
    - Trend Analysis
    - Bias Risk Scoring
    - Export Results
    """)

# Main Content
if uploaded_file is None:
    st.info("👈 กรุณาอัพโหลดไฟล์ข้อมูลเพื่อเริ่มการวิเคราะห์")
    
    st.markdown("### 📋 ข้อมูลที่ต้องการในไฟล์:")
    st.markdown("""
    - `Appraisee name` - ชื่อผู้ถูกประเมิน
    - `Appraiser Name` - ชื่อผู้ประเมิน
    - `Assessment Year` - ปีที่ประเมิน
    - `Period` - ครั้งที่ประเมิน (เช่น 1H, 2H)
    - `Behavioral score` - คะแนนพฤติกรรม
    - `Performance score` - คะแนนผลงาน
    - `Total` - คะแนนรวม
    """)

else:
    try:
        # โหลดข้อมูล
        if uploaded_file.name.endswith('.csv'):
            model_df = pd.read_csv(uploaded_file)
        else:
            model_df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ โหลดข้อมูลสำเร็จ: {len(model_df)} แถว")
        
        # แสดงข้อมูลตัวอย่าง
        with st.expander("ข้อมูลตัวอย่าง"):
            st.dataframe(model_df.head(10))
        
        st.markdown("---")
        
        # ปุ่มวิเคราะห์
        if st.button("🚀 เริ่มการวิเคราะห์", type="primary", use_container_width=True):
            
            with st.spinner("🔄 กำลังวิเคราะห์ข้อมูล..."):
                
                # 1. Outlier Detection
                st.markdown("## 1. Detect Outlier by Clustering model")
                fig_outlier, outlier_df = detect_outliers_with_gmm(model_df)
                st.plotly_chart(fig_outlier, use_container_width=True)
                
                st.markdown("### สรุปผล Outlier ที่ตรวจจับได้ทุกปี:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("จำนวน Outliers", len(outlier_df))
                with col2:
                    st.metric("Appraisers ที่มี Outliers", outlier_df['Appraiser Name'].nunique())
                
                with st.expander("📊 ดูรายละเอียด Outliers"):
                    st.dataframe(outlier_df, use_container_width=True)
                
                st.markdown("---")
                
                # 2. Trend Analysis
                st.markdown("## 2. Appraiser Trend Analysis")
                trend_df, period_scores = analyze_temporal_trends(model_df)
                fig_trend = plot_appraiser_trends(period_scores, trend_df)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                with st.expander("📊 ดูรายละเอียด Trend Analysis"):
                    st.dataframe(trend_df, use_container_width=True)
                
                st.markdown("---")
                
                # 3. Bias Detection
                st.markdown("## 3. Bias Detection")
                bias_df = calculate_bias_score(trend_df, outlier_df)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    case1 = len(bias_df[bias_df["BIAS_DETECTION"] == "case 1: ให้คะแนนสม่ำเสมอ | มี Outlier"])
                    st.metric("case 1", case1)

                with col2:
                    case2 = len(bias_df[bias_df["BIAS_DETECTION"] == "case 2: ให้คะแนนไม่สม่ำเสมอ | มี Outlier"])
                    st.metric("case 2", case2)

                with col3:
                    case3 = len(bias_df[bias_df["BIAS_DETECTION"] == "case 3: ให้คะแนนไม่สม่ำเสมอ | ไม่มี Outlier"])
                    st.metric("case 3", case3)

                with col4:
                    case4 = len(bias_df[bias_df["BIAS_DETECTION"] == "case 4: ให้คะแนนสม่ำเสมอ | ไม่มี Outlier"])
                    st.metric("case 4", case4)
                
                # แสดงตาราง Bias
                st.dataframe(
                    bias_df[[
                        'Appraiser Name', 'evaluation_time_count', 'score_timeline', 'outlier_freq', 'BIAS_DETECTION',
                    ]],
                    use_container_width=True
                )
                
                st.markdown("---")
                
                # 4. Download Section
                st.markdown("## 💾 4. Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download Excel (All Results)
                    excel_data = convert_df_to_excel({
                        'Bias Detection': bias_df,
                        'Trend Analysis': trend_df,
                        'Outliers': outlier_df
                    })
                    
                    st.download_button(
                        label="📥 ดาวน์โหลดผลลัพธ์ทั้งหมด (Excel)",
                        data=excel_data,
                        file_name="bias_detection_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                with col2:
                    # Download CSV (Bias Detection)
                    csv_data = bias_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 ดาวน์โหลด Bias Detection (CSV)",
                        data=csv_data,
                        file_name="bias_detection.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                st.success("✅ การวิเคราะห์เสร็จสมบูรณ์!")
                
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        st.info("กรุณาตรวจสอบว่าไฟล์มีคอลัมน์ที่จำเป็นครบถ้วน")