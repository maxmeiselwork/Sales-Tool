"""
AI-Powered Buyer Scoring Tool - Dynamic Table Support
Main Streamlit Application with Dynamic Table Selection
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
    st.title("AI-Powered Buyer Scoring Tool")
    st.markdown("""
    **Select any table from your Snowflake database and get AI-powered lead scoring!**
    
    This tool automatically adapts to your table structure and uses advanced AI to rank potential buyers.
    """)
    
    # Initialize components
    if 'model' not in st.session_state:
        with st.spinner("Loading AI model..."):
            try:
                st.session_state.model = BuyerScoringModel()
            except Exception as e:
                st.error(f"Failed to load AI model: {str(e)}")
                st.info("Check your .env file and ensure Hugging Face token is set")
                return
    
    if 'db' not in st.session_state:
        try:
            st.session_state.db = SnowflakeManager()
        except Exception as e:
            st.error(f"Failed to connect to Snowflake: {str(e)}")
            st.info("Check your .env file for Snowflake credentials")
            return
    
    # Sidebar for model management and settings
    with st.sidebar:
        st.header("Model Management")
        
        # Model status
        if st.session_state.model and st.session_state.model.is_ready():
            st.success("AI Model Ready")
            if st.session_state.model.is_trained():
                st.info("Using Fine-tuned Model")
            else:
                st.info("Using Base Model")
        else:
            st.error("Model Not Loaded")
        
        # Database status
        if st.session_state.db and st.session_state.db.is_connected():
            st.success("Snowflake Connected")
            
            # Table selection
            st.subheader("Select Data Table")
            available_tables = st.session_state.db.get_available_tables()
            
            if available_tables:
                # Create options with table info
                table_options = []
                for table in available_tables:
                    row_count = f"{table['row_count']:,}" if table['row_count'] else "Unknown"
                    table_options.append(f"{table['name']} ({row_count} rows)")
                
                selected_option = st.selectbox(
                    "Choose table to analyze:",
                    table_options,
                    help="Select which table contains your company data"
                )
                
                # Extract table name
                selected_table = selected_option.split(' (')[0] if selected_option else None
                st.session_state.selected_table = selected_table
                
                # Show table info
                if selected_table:
                    table_info = next((t for t in available_tables if t['name'] == selected_table), None)
                    if table_info:
                        st.metric("Rows", f"{table_info['row_count']:,}" if table_info['row_count'] else "Unknown")
                        st.metric("Columns", len(table_info['columns']))
                        
                        # Show column mapping info
                        mapping_info = st.session_state.db.get_column_mapping_info(selected_table)
                        st.metric("Mapped Fields", f"{mapping_info['mapped_fields']}/{mapping_info['total_columns']}")
                        
                        # Show table preview
                        with st.expander("Table Preview"):
                            preview_df = st.session_state.db.get_table_preview(selected_table, 5)
                            if not preview_df.empty:
                                st.dataframe(preview_df, use_container_width=True)
                        
                        # Show column mapping
                        with st.expander("Column Mapping"):
                            mapping = mapping_info['mapping']
                            if mapping:
                                for standard_field, actual_column in mapping.items():
                                    if actual_column:
                                        st.write(f"**{standard_field}** → `{actual_column}`")
                            else:
                                st.write("No automatic mapping available - using raw columns")
            else:
                st.warning("No tables found in the current schema")
                return
        else:
            st.error("Snowflake Not Connected")
            return
        
        # Training section (only show if table is selected)
        if hasattr(st.session_state, 'selected_table') and st.session_state.selected_table:
            st.subheader("Company Knowledge Training")
            if st.button("Train AI on Selected Table"):
                # Get all company data for training from selected table
                with st.spinner(f"Loading company data from {st.session_state.selected_table}..."):
                    company_data = st.session_state.db.get_all_companies_for_training(st.session_state.selected_table)
                
                if len(company_data) > 1000:
                    with st.spinner("Training AI model on company knowledge..."):
                        success = st.session_state.model.train_on_company_data(company_data)
                    if success:
                        st.success("AI now knows your company database!")
                        st.balloons()
                        st.rerun()
                else:
                    st.warning(f"Need at least 1,000 companies to train effectively. Found {len(company_data)} companies in selected table.")
            
            # Historical scoring training
            st.subheader("Scoring Improvement")
            if st.button("Train on Scoring History"):
                historical_data = st.session_state.db.get_historical_data()
                
                if len(historical_data) > 5:
                    with st.spinner("Training model on scoring patterns..."):
                        success = st.session_state.model.train_model(historical_data)
                    if success:
                        st.success("Improved scoring accuracy!")
                        st.rerun()
                else:
                    st.warning("Need at least 5 historical scores to improve. Score some buyers first!")
        
        # Data statistics
        st.subheader("Scoring Statistics")
        total_records = st.session_state.db.get_record_count()
        st.metric("Total Scored Buyers", total_records)
    
    # Main content area
    if not hasattr(st.session_state, 'selected_table') or not st.session_state.selected_table:
        st.info("Please select a table from the sidebar to begin")
        return
    
    selected_table = st.session_state.selected_table
    
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
        
        # Filter & Score Companies
        st.subheader(f"Filter & Score Companies from {selected_table}")
        
        # Company selection filters
        with st.expander("🔍 Filter Companies", expanded=True):
            col_filter1, col_filter2 = st.columns(2)
            
            # Get mapping info to show relevant filters
            mapping_info = st.session_state.db.get_column_mapping_info(selected_table)
            mapping = mapping_info['mapping']
            
            with col_filter1:
                # Only show industry filter if industry column is mapped
                if mapping.get('industry'):
                    industry_filter = st.text_input(
                        "Industry (optional)",
                        placeholder="e.g., Technology, Healthcare",
                        help="Filter by industry - leave blank for all"
                    )
                else:
                    industry_filter = None
                
                # Only show location filter if location/city column is mapped
                if mapping.get('location') or mapping.get('country'):
                    location_filter = st.text_input(
                        "Location (optional)",
                        placeholder="e.g., San Francisco, New York",
                        help="Filter by location - leave blank for all"
                    )
                else:
                    location_filter = None
            
            with col_filter2:
                # Only show employee filters if employee count is mapped
                if mapping.get('employee_count'):
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
                else:
                    min_employees = 0
                    max_employees = 0
                    st.info("Employee count filtering not available for this table")
        
        # Number of companies to score
        num_companies = st.number_input(
            "Number of companies to score",
            min_value=10,
            max_value=1000,
            value=100,
            help="More companies = more comprehensive results but slower processing"
        )
        
        # Score companies button - combined action
        if st.button("🎯 Score Best Companies", type="primary", disabled=not product_description):
            if not product_description:
                st.error("❌ Please enter a product description first")
                return
            
            if not st.session_state.model or not st.session_state.model.is_ready():
                st.error("❌ AI model not ready. Please check your configuration.")
                return
            
            with st.spinner(f"🔍 Finding and scoring top companies from {selected_table}..."):
                
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
                
                # Get companies from selected table
                companies_df = st.session_state.db.get_companies_for_scoring(
                    table_name=selected_table,
                    limit=num_companies,
                    filters=filters
                )
                
                if len(companies_df) > 0:
                    # Score the companies
                    scored_results = st.session_state.model.score_buyers(
                        companies_df, 
                        product_description
                    )
                    
                    # Save to database with source table info
                    if st.session_state.db and st.session_state.db.is_connected():
                        st.session_state.db.save_scoring_results(
                            scored_results, 
                            product_description, 
                            selected_table
                        )
                    
                    # Display results
                    display_scoring_results(scored_results)
                else:
                    st.warning("No companies found matching your criteria")
    
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
        
        # Show table statistics
        if st.session_state.db and st.session_state.db.is_connected():
            stats = st.session_state.db.get_company_stats(selected_table)
            if stats and stats.get('top_industries'):
                st.subheader(f"{selected_table} Overview")
                st.write("**Top Industries:**")
                for industry in stats['top_industries'][:5]:
                    st.write(f"• {industry['industry']}: {industry['count']:,} companies")

def display_scoring_results(scored_results):
    """Display the scoring results with analytics"""
    
    st.success(f"Successfully scored {len(scored_results)} companies!")
    
    # Summary metrics - update ranges for decimals
    st.subheader("Scoring Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_score = len(scored_results[scored_results['score'] >= 8.0])
        st.metric("Hot Leads (8.0+)", high_score, delta=f"{high_score/len(scored_results)*100:.1f}%")
    
    with col2:
        warm_score = len(scored_results[(scored_results['score'] >= 6.0) & (scored_results['score'] < 8.0)])
        st.metric("Warm Leads (6.0-7.9)", warm_score, delta=f"{warm_score/len(scored_results)*100:.1f}%")
    
    with col3:
        cold_score = len(scored_results[(scored_results['score'] >= 4.0) & (scored_results['score'] < 6.0)])
        st.metric("Cold Leads (4.0-5.9)", cold_score, delta=f"{cold_score/len(scored_results)*100:.1f}%")
    
    with col4:
        poor_score = len(scored_results[scored_results['score'] < 4.0])
        st.metric("Poor Fit (<4.0)", poor_score, delta=f"{poor_score/len(scored_results)*100:.1f}%")
    
    # Results display - simplified view
    st.subheader("Top Scored Companies")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Minimum Score Filter", 1.0, 10.0, 1.0, step=0.1)
    with col2:
        show_top_n = st.selectbox("Show Top N Results", [10, 25, 50, 100, "All"], index=1)
    
    # Filter data
    filtered_results = scored_results[scored_results['score'] >= min_score]
    
    if show_top_n != "All":
        filtered_results = filtered_results.head(show_top_n)
    
    # Display simplified table
    for _, row in filtered_results.iterrows():
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"{row.get('company_name', 'Unknown Company')}")
                
                # Company details in one line
                details = []
                if row.get('industry'):
                    details.append(f"Industry: {row['industry']}")
                if row.get('employee_count'):
                    details.append(f"Size: {row['employee_count']} employees")
                if row.get('location'):
                    details.append(f"Location: {row['location']}")
                
                if details:
                    st.caption(" | ".join(details))
                
                # Reasoning in bullet points
                reason = row.get('reason', 'No analysis available')
                st.write("**Why this company needs your product:**")
                st.markdown(reason)
            
            with col2:
                # Score with color coding
                score = float(row.get('score', 5))
                if score >= 8.0:
                    st.success(f"**Score: {score:.1f}/10**")
                elif score >= 6.0:
                    st.warning(f"**Score: {score:.1f}/10**")
                else:
                    st.info(f"**Score: {score:.1f}/10**")
        
        st.divider()
    
    # Download section
    st.subheader("Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Format the dataframe for download
        download_df = scored_results.copy()
        download_df['score'] = download_df['score'].round(1)
        
        csv_data = download_df.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📄 Download Full Results (CSV)",
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
                download_df.to_excel(writer, sheet_name='Scored Buyers', index=False)
            
            st.download_button(
                label="📊 Download Full Results (Excel)",
                data=buffer.getvalue(),
                file_name=f"buyer_scores_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.info("Install openpyxl for Excel export: `pip install openpyxl`")

def show_table_management():
    """Show table management interface"""
    st.header("Table Management")
    
    if st.session_state.db and st.session_state.db.is_connected():
        available_tables = st.session_state.db.get_available_tables()
        
        if available_tables:
            st.subheader("Available Tables")
            
            for table in available_tables:
                with st.expander(f"{table['name']} ({table['row_count']:,} rows)"):
                    
                    # Show column information
                    st.write("**Columns:**")
                    for col in table['columns'][:10]:  # Show first 10 columns
                        nullable = "NULL" if col['nullable'] else "NOT NULL"
                        st.write(f"• `{col['name']}` ({col['type']}) - {nullable}")
                    
                    if len(table['columns']) > 10:
                        st.write(f"... and {len(table['columns']) - 10} more columns")
                    
                    # Show mapping
                    mapping_info = st.session_state.db.get_column_mapping_info(table['name'])
                    if mapping_info['mapping']:
                        st.write("**Field Mapping:**")
                        for standard_field, actual_column in mapping_info['mapping'].items():
                            if actual_column:
                                st.write(f"• {standard_field} → `{actual_column}`")
                    
                    # Show preview
                    if st.button(f"Preview {table['name']}", key=f"preview_{table['name']}"):
                        preview_df = st.session_state.db.get_table_preview(table['name'], 10)
                        if not preview_df.empty:
                            st.dataframe(preview_df, use_container_width=True)

# Add table management to sidebar
def add_table_management_to_sidebar():
    """Add table management option to sidebar"""
    if st.sidebar.button("Manage Tables"):
        st.session_state.show_table_management = True

# Run the app
if __name__ == "__main__":
    # Add table management option
    add_table_management_to_sidebar()
    
    # Show table management if requested
    if st.session_state.get('show_table_management', False):
        show_table_management()
        if st.button("← Back to Scoring"):
            st.session_state.show_table_management = False
            st.rerun()
    else:
        main()