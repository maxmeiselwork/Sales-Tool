"""
AI-Powered Buyer Scoring Tool - Updated for Snowflake Integration
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime

from src.model import BuyerScoringModel
from src.database import SnowflakeManager
from src.utils import validate_csv_data

# Page configuration
st.set_page_config(
    page_title="AI Buyer Scoring Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function"""
    
    # Header
    st.title("🎯 AI-Powered Buyer Scoring Tool")
    st.markdown("""
    **Select companies from your Snowflake database and get AI-powered lead scoring!**
    
    This tool uses advanced AI to rank your potential buyers by likelihood to purchase your product.
    """)
    
    # Initialize components
    if 'model' not in st.session_state:
        with st.spinner("🤖 Loading AI model..."):
            try:
                st.session_state.model = BuyerScoringModel()
            except Exception as e:
                st.error(f"❌ Failed to load AI model: {str(e)}")
                st.info("💡 Check your .env file and ensure Hugging Face token is set")
                return
    
    if 'db' not in st.session_state:
        try:
            st.session_state.db = SnowflakeManager()
        except Exception as e:
            st.error(f"❌ Failed to connect to Snowflake: {str(e)}")
            st.info("💡 Check your .env file for Snowflake credentials")
            return
    
    # Sidebar for model management and settings
    with st.sidebar:
        st.header("🧠 Model Management")
        
        # Model status
        if st.session_state.model and st.session_state.model.is_ready():
            st.success("✅ AI Model Ready")
            if st.session_state.model.is_trained():
                st.info("🎓 Using Fine-tuned Model")
            else:
                st.info("📚 Using Base Model")
        else:
            st.error("❌ Model Not Loaded")
        
        # Database status
        if st.session_state.db and st.session_state.db.is_connected():
            st.success("✅ Snowflake Connected")
            
            # Show database stats
            stats = st.session_state.db.get_company_stats()
            if stats:
                st.metric("Total Companies", f"{stats.get('total_companies', 0):,}")
        else:
            st.error("❌ Snowflake Not Connected")
            return
        
         # Training section
        st.subheader("🏢 Company Knowledge Training")
        if st.button("🧠 Train AI on Company Database"):
            # Get all company data for training
            with st.spinner("📊 Loading company data from Snowflake..."):
                company_data = st.session_state.db.get_all_companies_for_training()
            
            if len(company_data) > 1000:
                with st.spinner("🚀 Training AI model on company knowledge..."):
                    success = st.session_state.model.train_on_company_data(company_data)
                if success:
                    st.success("✅ AI now knows your company database!")
                    st.balloons()
                    st.experimental_rerun()
            else:
                st.warning("⚠️ Need at least 1,000 companies to train effectively. Upload your data first!")
        
        # Historical scoring training
        st.subheader("🎯 Scoring Improvement")
        if st.button("📊 Train on Scoring History"):
            historical_data = st.session_state.db.get_historical_data()
            
            if len(historical_data) > 5:
                with st.spinner("🚀 Training model on scoring patterns..."):
                    success = st.session_state.model.train_model(historical_data)
                if success:
                    st.success("✅ Improved scoring accuracy!")
                    st.experimental_rerun()
            else:
                st.warning("⚠️ Need at least 5 historical scores to improve. Score some buyers first!")
        
        # Data statistics
        st.subheader("📈 Scoring Statistics")
        total_records = st.session_state.db.get_record_count()
        st.metric("Total Scored Buyers", total_records)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Step 1: Product Description
        st.header("1️⃣ Describe Your Product")
        product_description = st.text_area(
            "What are you selling?",
            placeholder="e.g., 'Salesforce CRM alternative for growing companies that need better customer relationship management'",
            height=120,
            help="Be specific about your target market, key features, and ideal customer size"
        )
        
        # Step 2: Select Companies from Snowflake
        st.header("2️⃣ Select Companies to Score")
        
        # Company selection filters
        with st.expander("🔍 Filter Companies", expanded=True):
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                industry_filter = st.text_input(
                    "Industry (optional)",
                    placeholder="e.g., Technology, Healthcare",
                    help="Filter by industry - leave blank for all"
                )
                
                location_filter = st.text_input(
                    "Location (optional)",
                    placeholder="e.g., San Francisco, New York",
                    help="Filter by location - leave blank for all"
                )
            
            with col_filter2:
                min_employees = st.number_input(
                    "Min Employees",
                    min_value=0,
                    value=0,
                    help="Minimum number of employees"
                )
                
                max_employees = st.number_input(
                    "Max Employees",
                    min_value=0,
                    value=0,
                    help="Maximum employees (0 = no limit)"
                )
        
        # Number of companies to score
        num_companies = st.number_input(
            "Number of companies to score",
            min_value=10,
            max_value=1000,
            value=100,
            help="More companies = more comprehensive results but slower processing"
        )
        
        # Load companies button
        if st.button("📊 Load Companies from Snowflake", type="primary"):
            with st.spinner("🔍 Searching companies in Snowflake..."):
                
                # Build filters
                filters = {}
                if industry_filter:
                    filters['industry'] = industry_filter
                if location_filter:
                    filters['location'] = location_filter
                if min_employees > 0:
                    filters['min_employees'] = min_employees
                if max_employees > 0:
                    filters['max_employees'] = max_employees
                
                # Get companies
                companies_df = st.session_state.db.get_companies_for_scoring(
                    limit=num_companies,
                    filters=filters
                )
                
                if len(companies_df) > 0:
                    st.session_state.selected_companies = companies_df
                    st.success(f"✅ Loaded {len(companies_df)} companies!")
                else:
                    st.warning("⚠️ No companies found matching your criteria")
    
    with col2:
        # Instructions and tips
        st.header("💡 Tips for Better Scoring")
        st.markdown("""
        **Product Description:**
        - Be specific about your target market
        - Mention company size preferences
        - Include key use cases
        
        **Company Selection:**
        - Use filters to target specific segments
        - Start with smaller batches (100-200 companies)
        - Consider your ideal customer profile
        
        **Scoring Strategy:**
        - Score similar companies together
        - Review and correct scores to improve AI
        - Use historical data for model training
        """)
        
        # Show database statistics
        if st.session_state.db and st.session_state.db.is_connected():
            stats = st.session_state.db.get_company_stats()
            if stats and stats.get('top_industries'):
                st.subheader("📊 Database Overview")
                st.write("**Top Industries:**")
                for industry in stats['top_industries'][:5]:
                    st.write(f"• {industry['industry']}: {industry['count']:,} companies")
    
    # Display selected companies and scoring
    if 'selected_companies' in st.session_state and len(st.session_state.selected_companies) > 0:
        
        companies_df = st.session_state.selected_companies
        
        # Data preview
        with st.expander("👀 Preview Selected Companies", expanded=True):
            st.dataframe(companies_df.head(10), use_container_width=True)
            
            # Data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Companies", len(companies_df))
            with col2:
                if 'industry' in companies_df.columns:
                    st.metric("Industries", companies_df['industry'].nunique())
            with col3:
                if 'employee_count' in companies_df.columns:
                    avg_size = companies_df['employee_count'].mean()
                    st.metric("Avg Company Size", f"{avg_size:.0f}" if not pd.isna(avg_size) else "N/A")
        
        # Scoring section
        st.header("3️⃣ AI Scoring")
        
        if st.button("🚀 Score Selected Companies", type="primary", disabled=not product_description):
            if not product_description:
                st.error("❌ Please enter a product description first")
                return
            
            if not st.session_state.model or not st.session_state.model.is_ready():
                st.error("❌ AI model not ready. Please check your configuration.")
                return
            
            # Score the companies
            with st.spinner("🤖 AI is analyzing your companies..."):
                scored_results = st.session_state.model.score_buyers(
                    companies_df, 
                    product_description
                )
            
            # Save to database
            if st.session_state.db and st.session_state.db.is_connected():
                st.session_state.db.save_scoring_results(scored_results, product_description)
            
            # Display results
            display_scoring_results(scored_results)

def display_scoring_results(scored_results):
    """Display the scoring results with analytics"""
    
    st.success(f"✅ Successfully scored {len(scored_results)} companies!")
    
    # Summary metrics
    st.subheader("📊 Scoring Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_score = len(scored_results[scored_results['score'] >= 8])
        st.metric("🔥 Hot Leads (8-10)", high_score, delta=f"{high_score/len(scored_results)*100:.1f}%")
    
    with col2:
        warm_score = len(scored_results[(scored_results['score'] >= 6) & (scored_results['score'] < 8)])
        st.metric("🌟 Warm Leads (6-7)", warm_score, delta=f"{warm_score/len(scored_results)*100:.1f}%")
    
    with col3:
        cold_score = len(scored_results[(scored_results['score'] >= 4) & (scored_results['score'] < 6)])
        st.metric("❄️ Cold Leads (4-5)", cold_score, delta=f"{cold_score/len(scored_results)*100:.1f}%")
    
    with col4:
        poor_score = len(scored_results[scored_results['score'] < 4])
        st.metric("❌ Poor Fit (<4)", poor_score, delta=f"{poor_score/len(scored_results)*100:.1f}%")
    
    # Results table
    st.subheader("🎯 Scored Results")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Minimum Score Filter", 1, 10, 1)
    with col2:
        show_top_n = st.selectbox("Show Top N Results", [10, 25, 50, 100, "All"], index=1)
    
    # Filter data
    filtered_results = scored_results[scored_results['score'] >= min_score]
    
    if show_top_n != "All":
        filtered_results = filtered_results.head(show_top_n)
    
    # Display table with styling
    styled_results = filtered_results.copy()
    
    # Color code scores
    def color_score(val):
        if val >= 8:
            return 'background-color: #d4edda; color: #155724'  # Green
        elif val >= 6:
            return 'background-color: #fff3cd; color: #856404'  # Yellow
        elif val >= 4:
            return 'background-color: #f8d7da; color: #721c24'  # Light red
        else:
            return 'background-color: #f5c6cb; color: #721c24'  # Red
    
    # Apply styling
    display_columns = ['company_name', 'industry', 'employee_count', 'location', 'score', 'reason']
    display_columns = [col for col in display_columns if col in styled_results.columns]
    
    styled_df = styled_results[display_columns].style.applymap(
        color_score, subset=['score']
    ).format({'score': '{:.1f}'})
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Download options
    st.subheader("📥 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv_data = scored_results.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"buyer_scores_{timestamp}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel download (if openpyxl is installed)
        try:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                scored_results.to_excel(writer, sheet_name='Scored Buyers', index=False)
            
            st.download_button(
                label="📊 Download Excel",
                data=buffer.getvalue(),
                file_name=f"buyer_scores_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.info("Install openpyxl for Excel export: `pip install openpyxl`")

# Run the app
if __name__ == "__main__":
    main()