"""
Utility functions for the AI Buyer Scoring Tool
Helper functions for data processing, validation, and common operations
"""

import pandas as pd
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the uploaded CSV has the required columns for buyer scoring.
    
    Args:
        df (pd.DataFrame): The uploaded dataframe
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    required_columns = ['company_name']  # Minimum required
    recommended_columns = ['company_name', 'industry', 'employee_count', 'location', 'domain']
    
    # Check if dataframe is empty
    if df.empty:
        return False, "CSV file is empty"
    
    # Check for required columns
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        return False, f"Missing required columns: {', '.join(missing_required)}"
    
    # Check for recommended columns and warn
    missing_recommended = [col for col in recommended_columns if col not in df.columns]
    if missing_recommended:
        logger.warning(f"Missing recommended columns: {', '.join(missing_recommended)}")
        st.warning(f"⚠️ Missing recommended columns: {', '.join(missing_recommended)}. Scoring may be less accurate.")
    
    return True, "CSV structure is valid"

def clean_company_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize company data for better AI processing.
    
    Args:
        df (pd.DataFrame): Raw company dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Clean company names
    if 'company_name' in df_clean.columns:
        df_clean['company_name'] = df_clean['company_name'].astype(str)
        df_clean['company_name'] = df_clean['company_name'].str.strip()
        df_clean['company_name'] = df_clean['company_name'].replace('nan', '')
    
    # Clean industry data
    if 'industry' in df_clean.columns:
        df_clean['industry'] = df_clean['industry'].astype(str)
        df_clean['industry'] = df_clean['industry'].str.strip()
        df_clean['industry'] = df_clean['industry'].replace('nan', 'Unknown')
    
    # Clean employee count
    if 'employee_count' in df_clean.columns:
        df_clean['employee_count'] = pd.to_numeric(df_clean['employee_count'], errors='coerce')
    
    # Clean location data
    if 'location' in df_clean.columns:
        df_clean['location'] = df_clean['location'].astype(str)
        df_clean['location'] = df_clean['location'].str.strip()
        df_clean['location'] = df_clean['location'].replace('nan', 'Unknown')
    
    # Remove rows with empty company names
    df_clean = df_clean[df_clean['company_name'] != '']
    df_clean = df_clean.dropna(subset=['company_name'])
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    logger.info(f"Cleaned data: {len(df_clean)} companies remaining")
    return df_clean

def format_employee_count(count: Any) -> str:
    """
    Format employee count for display and AI processing.
    
    Args:
        count: Employee count (could be number, string, or NaN)
        
    Returns:
        str: Formatted employee count description
    """
    if pd.isna(count) or count == '' or count == 'Unknown':
        return "Unknown size"
    
    try:
        num = int(float(count))
        if num < 10:
            return "Startup (1-9 employees)"
        elif num < 50:
            return "Small company (10-49 employees)"
        elif num < 250:
            return "Medium company (50-249 employees)"
        elif num < 1000:
            return "Large company (250-999 employees)"
        else:
            return "Enterprise (1000+ employees)"
    except (ValueError, TypeError):
        return "Unknown size"

def create_buyer_profile_text(row: pd.Series) -> str:
    """
    Create a text description of a buyer/company for AI processing.
    
    Args:
        row (pd.Series): Company data row
        
    Returns:
        str: Formatted text profile
    """
    profile_parts = []
    
    # Company name (required)
    if 'company_name' in row and pd.notna(row['company_name']):
        profile_parts.append(f"Company: {row['company_name']}")
    
    # Industry
    if 'industry' in row and pd.notna(row['industry']) and row['industry'] != 'Unknown':
        profile_parts.append(f"Industry: {row['industry']}")
    
    # Employee count
    if 'employee_count' in row:
        size_desc = format_employee_count(row['employee_count'])
        profile_parts.append(f"Size: {size_desc}")
    
    # Location
    if 'location' in row and pd.notna(row['location']) and row['location'] != 'Unknown':
        profile_parts.append(f"Location: {row['location']}")
    
    # Domain/Website
    if 'domain' in row and pd.notna(row['domain']):
        profile_parts.append(f"Website: {row['domain']}")
    
    return " | ".join(profile_parts)

def parse_ai_response(response: str) -> Tuple[int, str]:
    """
    Parse AI response to extract score and reasoning.
    
    Args:
        response (str): Raw AI response
        
    Returns:
        Tuple[int, str]: (score, reasoning)
    """
    try:
        # Try to find score pattern (number out of 10)
        score_patterns = [
            r'Score:\s*(\d+)',
            r'score:\s*(\d+)',
            r'(\d+)/10',
            r'(\d+)\s*out\s*of\s*10',
            r'Rating:\s*(\d+)'
        ]
        
        score = 5  # Default score
        for pattern in score_patterns:
            match = re.search(pattern, response)
            if match:
                score = int(match.group(1))
                break
        
        # Ensure score is between 1-10
        score = max(1, min(10, score))
        
        # Extract reasoning (everything after score or full response)
        reasoning = response.strip()
        
        # Try to find reasoning after common patterns
        reasoning_patterns = [
            r'Reason[ing]*:\s*(.*)',
            r'Because:\s*(.*)',
            r'Explanation:\s*(.*)'
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                break
        
        return score, reasoning
        
    except Exception as e:
        logger.error(f"Error parsing AI response: {e}")
        return 5, "Unable to parse AI response"

def save_results_to_csv(results: List[Dict], filename: Optional[str] = None) -> str:
    """
    Save scoring results to CSV file.
    
    Args:
        results (List[Dict]): List of scoring results
        filename (str, optional): Custom filename
        
    Returns:
        str: Filename of saved file
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"buyer_scores_{timestamp}.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    logger.info(f"Results saved to {filename}")
    return filename

def calculate_score_distribution(scores: List[int]) -> Dict[str, int]:
    """
    Calculate distribution of scores for analytics.
    
    Args:
        scores (List[int]): List of scores
        
    Returns:
        Dict[str, int]: Score distribution
    """
    distribution = {
        'High Priority (8-10)': len([s for s in scores if s >= 8]),
        'Medium Priority (5-7)': len([s for s in scores if 5 <= s <= 7]),
        'Low Priority (1-4)': len([s for s in scores if s <= 4])
    }
    return distribution

def format_score_for_display(score: int) -> str:
    """
    Format score with color coding for Streamlit display.
    
    Args:
        score (int): Score from 1-10
        
    Returns:
        str: Formatted score with emoji
    """
    if score >= 8:
        return f"🔥 {score}/10"
    elif score >= 6:
        return f"⚡ {score}/10"
    elif score >= 4:
        return f"⚠️ {score}/10"
    else:
        return f"❄️ {score}/10"

def validate_product_description(description: str) -> Tuple[bool, str]:
    """
    Validate product description input.
    
    Args:
        description (str): Product description
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    if not description or description.strip() == "":
        return False, "Product description cannot be empty"
    
    if len(description.strip()) < 10:
        return False, "Product description should be at least 10 characters long"
    
    if len(description) > 1000:
        return False, "Product description should be less than 1000 characters"
    
    return True, "Product description is valid"

def get_sample_data() -> pd.DataFrame:
    """
    Generate sample buyer data for testing.
    
    Returns:
        pd.DataFrame: Sample buyer data
    """
    sample_data = {
        'company_name': [
            'TechCorp Solutions', 'GreenEnergy Inc', 'RetailMax Group',
            'HealthcarePlus', 'FinanceSecure LLC', 'EduTech Academy',
            'ManufacturingPro', 'StartupInnovate', 'Enterprise Global',
            'LocalServices Co'
        ],
        'industry': [
            'Technology', 'Energy', 'Retail',
            'Healthcare', 'Financial Services', 'Education',
            'Manufacturing', 'Technology', 'Consulting',
            'Professional Services'
        ],
        'employee_count': [150, 2500, 800, 45, 1200, 75, 350, 12, 5000, 25],
        'location': [
            'San Francisco, CA', 'Austin, TX', 'New York, NY',
            'Boston, MA', 'Chicago, IL', 'Seattle, WA',
            'Detroit, MI', 'Palo Alto, CA', 'Atlanta, GA',
            'Portland, OR'
        ],
        'domain': [
            'techcorp.com', 'greenenergy.com', 'retailmax.com',
            'healthcareplus.com', 'financesecure.com', 'edutech.com',
            'manufacturingpro.com', 'startupinnovate.com', 'enterpriseglobal.com',
            'localservices.com'
        ]
    }
    
    return pd.DataFrame(sample_data)

def validate_csv_data(df: pd.DataFrame) -> list:
    """
    Wrapper for validate_csv_structure to return a list of error messages.
    """
    is_valid, message = validate_csv_structure(df)
    return [] if is_valid else [message]