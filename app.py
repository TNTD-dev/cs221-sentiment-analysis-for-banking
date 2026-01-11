"""
Sentiment Analysis for Banking - Streamlit Demo App
Phân tích cảm xúc cho ngành ngân hàng với PhoBERT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import json
from pathlib import Path

# Import utilities
from utils.model_loader import (
    load_model_and_tokenizer, 
    get_label_name, 
    get_label_emoji, 
    get_label_color
)
from utils.predictor import predict_sentiment, format_confidence_dict
from utils.analyzer import (
    load_test_data,
    generate_predictions,
    get_error_samples,
    get_correct_samples,
    get_confusion_pairs,
    explain_prediction,
    filter_by_label,
    get_per_class_metrics,
    get_error_statistics
)

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Demo",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_model():
    """Load model and tokenizer"""
    if 'model' not in st.session_state:
        model, tokenizer, device = load_model_and_tokenizer('models/phobert')
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.device = device
    
    return st.session_state.model, st.session_state.tokenizer, st.session_state.device


def load_predictions_cache():
    """Load or generate predictions on test set"""
    if 'predictions_df' not in st.session_state:
        model, tokenizer, device = load_model()
        test_df = load_test_data()
        
        if not test_df.empty:
            predictions_df = generate_predictions(test_df, model, tokenizer, device)
            st.session_state.predictions_df = predictions_df
        else:
            st.session_state.predictions_df = pd.DataFrame()
    
    return st.session_state.predictions_df


# ========================================
# PAGE 1: DEMO PREDICTION
# ========================================
def page_demo():
    st.markdown('<div class="main-header">🏦 Dự Đoán Sentiment Banking </div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Nhập câu tiếng Việt để phân tích cảm xúc với PhoBERT</div>', unsafe_allow_html=True)
    
    # Load model
    model, tokenizer, device = load_model()
    
    if model is None:
        st.error("❌ Không thể load model. Vui lòng kiểm tra folder models/phobert/")
        return
    
    # Input area
    st.markdown("### 📝 Nhập văn bản")
    
    # Example texts
    examples = {
        "Negative": "Gọi không được mà tốn tiền như gì ấy",
        "Neutral": "Tôi muốn biết thông tin về sản phẩm này",
        "Positive": "Vietcombank ngân hàng tốt, dịch vụ tuyệt vời"
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝 Ví dụ Negative", width='stretch'):
            st.session_state.demo_text = examples["Negative"]
    with col2:
        if st.button("📝 Ví dụ Neutral", width='stretch'):
            st.session_state.demo_text = examples["Neutral"]
    with col3:
        if st.button("📝 Ví dụ Positive", width='stretch'):
            st.session_state.demo_text = examples["Positive"]
    
    # Text input
    user_text = st.text_area(
        "Nhập câu của bạn:",
        value=st.session_state.get('demo_text', ''),
        height=100,
        placeholder="Ví dụ: Ngân hàng này dịch vụ rất tốt..."
    )
    
    # Predict button
    if st.button("Dự đoán", type="primary", width='stretch'):
        if user_text.strip():
            with st.spinner('🔄 Đang phân tích...'):
                # Predict
                predicted_label, confidence_scores = predict_sentiment(
                    user_text, model, tokenizer, device
                )
                
                # Display results
                st.markdown("---")
                st.markdown("### 🎯 Kết quả dự đoán")
                
                # Main prediction
                emoji = get_label_emoji(predicted_label)
                label_name = get_label_name(predicted_label)
                confidence = confidence_scores[predicted_label] * 100
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"<h1 style='text-align: center; font-size: 4rem;'>{emoji}</h1>", unsafe_allow_html=True)
                    
                    if predicted_label == 0:
                        st.markdown(f"<h2 style='text-align: center;' class='negative'>{label_name}</h2>", unsafe_allow_html=True)
                    elif predicted_label == 1:
                        st.markdown(f"<h2 style='text-align: center;' class='neutral'>{label_name}</h2>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='text-align: center;' class='positive'>{label_name}</h2>", unsafe_allow_html=True)
                    
                    st.markdown(f"<p style='text-align: center; font-size: 1.2rem;'>Độ tin cậy: <b>{confidence:.1f}%</b></p>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Confidence scores
                st.markdown("### 📊 Chi tiết độ tin cậy")
                
                confidence_dict = format_confidence_dict(confidence_scores)
                
                # Create bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(confidence_dict.values()),
                        y=list(confidence_dict.keys()),
                        orientation='h',
                        marker=dict(
                            color=['#dc3545', '#6c757d', '#28a745'],
                        ),
                        text=[f"{v:.1f}%" for v in confidence_dict.values()],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Phân bố xác suất cho từng nhãn",
                    xaxis_title="Confidence (%)",
                    yaxis_title="Sentiment",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, width='stretch')
                
        else:
            st.warning("⚠️ Vui lòng nhập văn bản!")


# ========================================
# PAGE 2: ERROR ANALYSIS
# ========================================
def page_error_analysis():
    st.markdown('<div class="main-header">🔍 Phân Tích Lỗi</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Phân tích mẫu dự đoán đúng và sai trên test set</div>', unsafe_allow_html=True)
    
    # Load predictions
    predictions_df = load_predictions_cache()
    
    if predictions_df.empty:
        st.error("❌ Không thể load test data!")
        return
    
    # Overall statistics
    stats = get_error_statistics(predictions_df)
    
    st.markdown("### 📈 Tổng quan")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tổng số mẫu", stats['total_samples'])
    with col2:
        st.metric("Dự đoán đúng", stats['correct_predictions'])
    with col3:
        st.metric("Dự đoán sai", stats['incorrect_predictions'])
    with col4:
        st.metric("Accuracy", f"{stats['accuracy']*100:.2f}%")
    
    st.markdown("---")
    
    # Tabs for correct and incorrect
    tab1, tab2 = st.tabs(["❌ Mẫu Sai", "✅ Mẫu Đúng"])
    
    # Tab 1: Incorrect predictions
    with tab1:
        error_df = get_error_samples(predictions_df)
        
        st.markdown(f"### Tổng số mẫu sai: {len(error_df)}")
        
        # Confusion pairs
        confusion_pairs = get_confusion_pairs(predictions_df)
        
        if confusion_pairs:
            st.markdown("#### 📊 Các loại lỗi phổ biến")
            
            pairs_df = pd.DataFrame([
                {'Loại lỗi': k, 'Số lượng': v}
                for k, v in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
            ])
            
            fig = px.bar(
                pairs_df,
                x='Số lượng',
                y='Loại lỗi',
                orientation='h',
                title='Phân bố các loại lỗi'
            )
            st.plotly_chart(fig, width='stretch')
        
        # Filter by error type
        st.markdown("#### 🔎 Xem chi tiết mẫu sai")
        
        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        col1, col2 = st.columns(2)
        with col1:
            filter_true = st.selectbox(
                "Lọc theo nhãn thực tế:",
                options=['Tất cả'] + list(label_map.values()),
                key='error_true_label'
            )
        with col2:
            filter_pred = st.selectbox(
                "Lọc theo nhãn dự đoán:",
                options=['Tất cả'] + list(label_map.values()),
                key='error_pred_label'
            )
        
        # Apply filters
        filtered_errors = error_df.copy()
        if filter_true != 'Tất cả':
            true_label_id = [k for k, v in label_map.items() if v == filter_true][0]
            filtered_errors = filtered_errors[filtered_errors['label'] == true_label_id]
        
        if filter_pred != 'Tất cả':
            pred_label_id = [k for k, v in label_map.items() if v == filter_pred][0]
            filtered_errors = filtered_errors[filtered_errors['predicted_label'] == pred_label_id]
        
        st.markdown(f"**Hiển thị {len(filtered_errors)} mẫu**")
        
        # Display samples
        for idx, row in filtered_errors.head(10).iterrows():
            with st.expander(f"📄 Mẫu {idx}: {row['text'][:80]}..."):
                st.markdown(f"**Text gốc:** {row['text']}")
                st.markdown(f"**Text đã xử lý:** {row['text_clean']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    true_name = label_map[row['label']]
                    st.markdown(f"**Nhãn thực tế:**  \n{get_label_emoji(row['label'])} {true_name}")
                with col2:
                    pred_name = label_map[row['predicted_label']]
                    st.markdown(f"**Dự đoán:**  \n{get_label_emoji(row['predicted_label'])} {pred_name}")
                with col3:
                    st.markdown(f"**Confidence:**  \n{row['max_confidence']*100:.1f}%")
                
                # Explanation
                explanation = explain_prediction(
                    row['text_clean'],
                    row['label'],
                    row['predicted_label'],
                    row['max_confidence'],
                    row['is_correct']
                )
                st.info(explanation)
    
    # Tab 2: Correct predictions
    with tab2:
        correct_df = get_correct_samples(predictions_df)
        
        st.markdown(f"### Tổng số mẫu đúng: {len(correct_df)}")
        
        # Sort by confidence
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Top 5 mẫu có confidence cao nhất")
            top_confident = correct_df.nlargest(5, 'max_confidence')
            
            for idx, row in top_confident.iterrows():
                with st.expander(f"Confidence: {row['max_confidence']*100:.1f}% - {row['text'][:60]}..."):
                    st.markdown(f"**Text:** {row['text_clean']}")
                    label_name = label_map[row['label']]
                    st.markdown(f"**Nhãn:** {get_label_emoji(row['label'])} {label_name}")
                    st.success(f"✅ Model rất tự tin với {row['max_confidence']*100:.1f}%")
        
        with col2:
            st.markdown("#### 🤔 Top 5 mẫu có confidence thấp nhất (nhưng vẫn đúng)")
            low_confident = correct_df.nsmallest(5, 'max_confidence')
            
            for idx, row in low_confident.iterrows():
                with st.expander(f"Confidence: {row['max_confidence']*100:.1f}% - {row['text'][:60]}..."):
                    st.markdown(f"**Text:** {row['text_clean']}")
                    label_name = label_map[row['label']]
                    st.markdown(f"**Nhãn:** {get_label_emoji(row['label'])} {label_name}")
                    st.warning(f"⚠️ Model phân vân (confidence thấp: {row['max_confidence']*100:.1f}%)")


# ========================================
# PAGE 3: METRICS DASHBOARD
# ========================================
def page_metrics():
    st.markdown('<div class="main-header">📊 Metrics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Chi tiết hiệu năng của PhoBERT model</div>', unsafe_allow_html=True)
    
    # Load metrics from JSON
    metrics_path = Path('results/metrics.json')
    if not metrics_path.exists():
        st.error("❌ Không tìm thấy file results/metrics.json")
        return
    
    with open(metrics_path, 'r', encoding='utf-8') as f:
        all_metrics = json.load(f)
    
    # Get PhoBERT metrics
    phobert_metrics = None
    for m in all_metrics:
        if m['Model'] == 'PhoBERT':
            phobert_metrics = m
            break
    
    if phobert_metrics is None:
        st.error("❌ Không tìm thấy metrics cho PhoBERT")
        return
    
    # Overall metrics
    st.markdown("### 🎯 Overall Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{phobert_metrics['Accuracy']*100:.2f}%")
    with col2:
        st.metric("F1 Score (Macro)", f"{phobert_metrics['F1 (Macro)']*100:.2f}%")
    with col3:
        st.metric("Precision (Weighted)", f"{phobert_metrics['Precision (Weighted)']*100:.2f}%")
    with col4:
        st.metric("Recall (Weighted)", f"{phobert_metrics['Recall (Weighted)']*100:.2f}%")
    
    st.markdown("---")
    
    # Confusion Matrix
    st.markdown("### 🔲 Confusion Matrix")
    
    cm_path = Path('results/confusion_matrix_phobert.png')
    if cm_path.exists():
        cm_image = Image.open(cm_path)
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(cm_image, width='stretch')
    else:
        st.warning("⚠️ Không tìm thấy confusion matrix image")
    
    st.markdown("---")
    
    # Per-class metrics
    st.markdown("### 📋 Per-Class Performance")
    
    predictions_df = load_predictions_cache()
    if not predictions_df.empty:
        per_class = get_per_class_metrics(predictions_df)
        st.dataframe(per_class, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # Model comparison
    st.markdown("### 🏆 So sánh với các models khác")
    
    # Create comparison dataframe
    comparison_data = []
    for m in all_metrics:
        comparison_data.append({
            'Model': m['Model'],
            'Accuracy': m['Accuracy'] * 100,
            'F1 (Macro)': m['F1 (Macro)'] * 100,
            'Training Time (s)': m['Training Time (s)']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Bar chart for accuracy
    fig1 = px.bar(
        comparison_df,
        x='Model',
        y='Accuracy',
        title='Accuracy Comparison',
        color='Accuracy',
        color_continuous_scale='Blues'
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, width='stretch')
    
    # Table with all metrics
    st.markdown("#### 📊 Bảng so sánh chi tiết")
    st.dataframe(comparison_df, width='stretch', hide_index=True)


# ========================================
# PAGE 4: SAMPLE EXPLORER
# ========================================
def page_sample_explorer():
    st.markdown('<div class="main-header">🎯 Sample Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Khám phá và phân tích các mẫu cụ thể</div>', unsafe_allow_html=True)
    
    # Load predictions
    predictions_df = load_predictions_cache()
    
    if predictions_df.empty:
        st.error("❌ Không thể load test data!")
        return
    
    # Filters
    st.markdown("### 🔍 Bộ lọc")
    
    col1, col2 = st.columns(2)
    
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    with col1:
        selected_true_label = st.selectbox(
            "Chọn nhãn thực tế:",
            options=['Tất cả'] + list(label_map.values()),
            key='explorer_true_label'
        )
    
    with col2:
        prediction_status = st.selectbox(
            "Trạng thái dự đoán:",
            options=['Tất cả', 'Đúng', 'Sai'],
            key='explorer_status'
        )
    
    # Apply filters
    filtered_df = predictions_df.copy()
    
    if selected_true_label != 'Tất cả':
        true_label_id = [k for k, v in label_map.items() if v == selected_true_label][0]
        filtered_df = filtered_df[filtered_df['label'] == true_label_id]
    
    if prediction_status == 'Đúng':
        filtered_df = filtered_df[filtered_df['is_correct'] == True]
    elif prediction_status == 'Sai':
        filtered_df = filtered_df[filtered_df['is_correct'] == False]
    
    st.markdown(f"**Tìm thấy {len(filtered_df)} mẫu phù hợp**")
    
    # Random sample button
    if st.button("🎲 Lấy 5 mẫu ngẫu nhiên", type="primary"):
        st.session_state.random_samples = filtered_df.sample(min(5, len(filtered_df)))
    
    # Display samples
    if 'random_samples' in st.session_state and not st.session_state.random_samples.empty:
        st.markdown("---")
        st.markdown("### 📄 Các mẫu được chọn")
        
        for idx, row in st.session_state.random_samples.iterrows():
            with st.container():
                st.markdown(f"#### Mẫu {idx}")
                
                # Text
                st.markdown(f"**Text gốc:** {row['text']}")
                st.markdown(f"**Text đã xử lý:** {row['text_clean']}")
                
                # Labels and predictions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    true_label_name = label_map[row['label']]
                    st.markdown(f"**Nhãn thực tế:**")
                    st.markdown(f"<h2 style='text-align: center;'>{get_label_emoji(row['label'])} {true_label_name}</h2>", unsafe_allow_html=True)
                
                with col2:
                    pred_label_name = label_map[row['predicted_label']]
                    st.markdown(f"**Dự đoán:**")
                    st.markdown(f"<h2 style='text-align: center;'>{get_label_emoji(row['predicted_label'])} {pred_label_name}</h2>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"**Độ tin cậy:**")
                    st.markdown(f"<h2 style='text-align: center;'>{row['max_confidence']*100:.1f}%</h2>", unsafe_allow_html=True)
                
                # Confidence breakdown
                confidence_data = {
                    'Negative': row['confidence_negative'] * 100,
                    'Neutral': row['confidence_neutral'] * 100,
                    'Positive': row['confidence_positive'] * 100
                }
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(confidence_data.values()),
                        y=list(confidence_data.keys()),
                        orientation='h',
                        marker=dict(color=['#dc3545', '#6c757d', '#28a745']),
                        text=[f"{v:.1f}%" for v in confidence_data.values()],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Chi tiết confidence",
                    xaxis_title="Confidence (%)",
                    height=250,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Explanation
                explanation = explain_prediction(
                    row['text_clean'],
                    row['label'],
                    row['predicted_label'],
                    row['max_confidence'],
                    row['is_correct']
                )
                
                if row['is_correct']:
                    st.success(explanation)
                else:
                    st.error(explanation)
                
                st.markdown("---")


# ========================================
# MAIN APP
# ========================================
def main():
    # Sidebar
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Chọn trang:",
        ["🏦 Demo Prediction", "🔍 Error Analysis", "📊 Metrics Dashboard", "🎯 Sample Explorer"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Thông tin")
    st.sidebar.info("""
    **Sentiment Analysis for Banking**
    
    - Model: PhoBERT
    - Classes: Negative, Neutral, Positive 
    - Accuracy: ~94.6%
    
    Dự án phân tích cảm xúc cho ngành ngân hàng sử dụng PhoBERT.
    """)
    
    # Route to pages
    if page == "🏦 Demo Prediction":
        page_demo()
    elif page == "🔍 Error Analysis":
        page_error_analysis()
    elif page == "📊 Metrics Dashboard":
        page_metrics()
    elif page == "🎯 Sample Explorer":
        page_sample_explorer()


if __name__ == "__main__":
    main()

