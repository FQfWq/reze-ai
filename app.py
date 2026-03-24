import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
from openai import OpenAI

st.set_page_config(page_title="Reze Ai", layout="centered", initial_sidebar_state="collapsed")

# ==================== 高級深色 + 鮮明純白文字 + Reze 背景 ====================
st.markdown("""
<style>
    .main, .stApp, .block-container {
        background-color: #05080f !important;
        color: #ffffff !important;
    }
    .stSidebar { background-color: #0a0f1c !important; }

    h1 { 
        font-weight: 500; 
        letter-spacing: -0.8px; 
        text-align: center; 
        color: #d4af37;           /* 標題保留金色 */
        font-size: 2.75rem;
        margin: 40px 0 15px 0;
    }
    .subtitle {
        text-align: center;
        color: #ffffff;
        font-size: 1.1rem;
        margin-bottom: 50px;
        letter-spacing: 3px;
    }

    /* 強制所有文字為鮮明純白色 */
    p, span, div, label, .stMarkdown, .stMetric label, .stMetric div[data-testid="stMetricValue"],
    .stCaption, .report-box *, .stExpander {
        color: #ffffff !important;
    }

    /* Reze 左右背景圖 */
    .reze-bg {
        position: fixed;
        top: 0; bottom: 0;
        width: 50%;
        background-size: cover;
        background-position: center;
        opacity: 0.28;
        z-index: 1;
        pointer-events: none;
    }
    .reze-left  { left: 0; background-image: url('https://i.pinimg.com/originals/2d/da/9c/2dda9c229714591bf636e718220f25b9.jpg'); }
    .reze-right { right: 0; background-image: url('https://i.pinimg.com/originals/74/ad/6e/74ad6e3c5c24906849beba055453bd4f.jpg'); }

    /* 黑色底板 */
    .content-base {
        position: relative;
        z-index: 5;
        background-color: #0a0f1c;
        border-radius: 20px;
        padding: 50px 40px 55px 40px;
        margin: 35px auto;
        max-width: 1150px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.75);
        border: 1px solid #1e2a44;
    }

    /* Plotly 圖表強制鮮明白色文字與深色背景 */
    .js-plotly-plot .plotly .bg, .js-plotly-plot .plotly .paper {
        background-color: #0a0f1c !important;
    }
    .js-plotly-plot text, .js-plotly-plot .axis-title, .js-plotly-plot .tick text {
        fill: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Reze 左右背景圖
st.markdown("""
    <div class="reze-bg reze-left"></div>
    <div class="reze-bg reze-right"></div>
""", unsafe_allow_html=True)

st.title("Reze Ai")
st.markdown('<p class="subtitle">專業・優雅・精準技術分析</p>', unsafe_allow_html=True)

# ====================== 設定 ======================
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
if not MINIMAX_API_KEY:
    st.error("未設定 API 金鑰")
    st.stop()

client = OpenAI(
    base_url="https://api.minimaxi.com/v1",
    api_key=MINIMAX_API_KEY
)

# ====================== 股票對照表 ======================
stock_map = {
    "台積電": "2330.TW", "2330": "2330.TW",
    "鴻海": "2317.TW", "2317": "2317.TW",
    "聯發科": "2454.TW", "2454": "2454.TW",
    "騰訊": "0700.HK", "0700": "0700.HK",
    "阿里": "9988.HK", "阿里巴巴": "9988.HK", "9988": "9988.HK",
    "快手": "1024.HK", "1024": "1024.HK",
    "美團": "3690.HK", "3690": "3690.HK",
    "小米": "1810.HK", "1810": "1810.HK",
    "比亞迪": "1211.HK", "1211": "1211.HK",
    "中芯國際": "0981.HK", "0981": "0981.HK",
    "蘋果": "AAPL", "AAPL": "AAPL",
    "輝達": "NVDA", "NVDA": "NVDA",
    "微軟": "MSFT", "MSFT": "MSFT",
    "特斯拉": "TSLA", "TSLA": "TSLA",
}

def normalize_symbol(input_str: str) -> str:
    s = input_str.strip().upper()
    if s.endswith(('.HK', '.TW', '.SH', '.SZ')):
        return s
    if s.isdigit():
        if len(s) in (4, 5):
            code = s[1:] if len(s) == 5 and s.startswith('0') else s
            return f"{code}.HK"
        elif len(s) == 6:
            return f"{s}.SH" if s.startswith(('600','601','603','688')) else f"{s}.SZ"
    return s

@st.cache_data(ttl=300)
def get_stock_data(raw_input: str):
    try:
        symbol = normalize_symbol(raw_input)
        symbol = stock_map.get(raw_input.strip(), symbol)

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="max")

        if len(hist) == 0:
            return None, None, None, "無法取得股票資料，請確認代碼。"

        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        hist['MA60'] = hist['Close'].rolling(60).mean()

        hist['BB_mid'] = hist['Close'].rolling(20).mean()
        bb_std = hist['Close'].rolling(20).std()
        hist['BB_upper'] = hist['BB_mid'] + 2 * bb_std
        hist['BB_lower'] = hist['BB_mid'] - 2 * bb_std

        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean().replace(0, 1)
        hist['RSI'] = 100 - (100 / (1 + avg_gain / avg_loss))

        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        dif = exp12 - exp26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_signal = "金叉（看漲）" if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2] else \
                      "死叉（看跌）" if dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2] else "持平"

        current_price = hist['Close'].iloc[-1]
        current_rsi = hist['RSI'].iloc[-1]
        bb_pos = (current_price - hist['BB_lower'].iloc[-1]) / (hist['BB_upper'].iloc[-1] - hist['BB_lower'].iloc[-1]) if not pd.isna(hist['BB_lower'].iloc[-1]) else 0.5
        bb_status = "接近上軌（可能超買）" if bb_pos > 0.8 else "接近下軌（可能超賣）" if bb_pos < 0.2 else "中軌附近"

        data = {
            "symbol": symbol,
            "name": ticker.info.get('longName') or ticker.info.get('shortName') or "未知公司",
            "price": round(current_price, 2),
            "change_pct": round((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100, 2) if len(hist) > 1 else 0,
            "rsi": round(current_rsi, 1),
            "bb_status": bb_status,
            "ma5": round(hist['MA5'].iloc[-1], 2),
            "ma20": round(hist['MA20'].iloc[-1], 2),
            "ma60": round(hist['MA60'].iloc[-1], 2) if not pd.isna(hist['MA60'].iloc[-1]) else None,
        }

        warning = f"歷史資料僅 {len(hist)} 天" if len(hist) < 60 else None
        return data, {"signal": macd_signal, "rsi": current_rsi, "bb": bb_status}, hist, warning

    except Exception:
        return None, None, None, "獲取數據失敗，請稍後再試。"

# ====================== 主介面 ======================
symbol_input = st.text_input("輸入股票名稱或代碼", value="0700", placeholder="例如：0700、9988、2330、NVDA")

if st.button("開始分析"):
    with st.spinner("正在獲取數據並生成報告..."):
        data, macd_info, hist, warning = get_stock_data(symbol_input)

        if not data:
            st.markdown(f'<div class="warning-box">{warning}</div>', unsafe_allow_html=True)
        else:
            if warning:
                st.markdown(f'<div class="warning-box">{warning}</div>', unsafe_allow_html=True)

            st.markdown('<div class="content-base">', unsafe_allow_html=True)

            st.markdown(f"### {data['name']}（{data['symbol']}）", unsafe_allow_html=True)
            
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.metric("目前股價", f"{data['price']} 元", f"{data['change_pct']}%")
            st.metric("MACD 訊號", macd_info['signal'])
            st.metric("RSI", f"{data['rsi']}")
            st.metric("布林帶位置", macd_info['bb'])
            ma60_str = f"{data['ma60']}" if data['ma60'] is not None else "資料不足"
            st.metric("均線", f"{data['ma5']} / {data['ma20']} / {ma60_str}")
            st.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown('</div>', unsafe_allow_html=True)

            # 技術圖表（確保文字鮮明白色）
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                row_heights=[0.55, 0.225, 0.225],
                                subplot_titles=("價格走勢", "成交量", "RSI 指標"))

            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                         low=hist['Low'], close=hist['Close']), row=1, col=1)

            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA5'], line=dict(color='#f59e0b')), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='#22d3ee')), row=1, col=1)
            if not hist['MA60'].isna().all():
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], line=dict(color='#a855f7')), row=1, col=1)

            fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], line=dict(color='#64748b', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_lower'], line=dict(color='#64748b', dash='dash')), row=1, col=1)

            colors = ['#ef4444' if o > c else '#22c55e' for o, c in zip(hist['Open'], hist['Close'])]
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors), row=2, col=1)

            fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='#c084fc')), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=3, col=1, annotation_text="超買")
            fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", row=3, col=1, annotation_text="超賣")

            fig.update_layout(height=780, template="plotly_dark",
                              paper_bgcolor="#0a0f1c", plot_bgcolor="#0a0f1c",
                              xaxis_rangeslider_visible=False,
                              font=dict(color="#ffffff"))

            st.plotly_chart(fig, use_container_width=True)

            # AI 報告 Prompt
            prompt = f"""
請用優雅簡潔的繁體中文撰寫一份完整的股票分析報告。

股票：{data['name']}（{data['symbol']}）
目前股價：{data['price']} 元
漲跌幅：{data['change_pct']}%
MACD 訊號：{macd_info['signal']}
RSI：{data['rsi']}
布林帶位置：{macd_info['bb']}
均線：5日 {data['ma5']}，20日 {data['ma20']}，60日 {data.get('ma60', '資料不足')}

請按照以下順序完整輸出：
1. 目前技術趨勢
2. 關鍵指標觀察
3. 操作建議（短期與中長期）
4. 主要風險
5. 綜合評分（滿分10分）

最後必須加上獨立的「優雅總結」一段文字，明確說明是否適合長期持有並說明理由。
"""

            with st.spinner("正在生成專業報告..."):
                try:
                    response = client.chat.completions.create(
                        model="MiniMax-M2.5",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.65,
                        max_tokens=2000
                    )
                    report = response.choices[0].message.content.strip()
                    st.markdown("### 專業分析報告")
                    st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"報告生成失敗: {str(e)}")

            st.markdown('</div>', unsafe_allow_html=True)  # 結束黑色底板

# ====================== 側邊欄 ======================
with st.sidebar:
    with st.expander("使用說明"):
        st.write("直接輸入股票代碼或名稱即可查詢")
        st.write("支援港股、台股、美股熱門標的")
    st.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")