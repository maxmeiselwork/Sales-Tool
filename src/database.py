"""
Snowflake Database Manager
Handles all database operations for buyer scoring data
"""

import os
import pandas as pd
import snowflake.connector
from datetime import datetime
from typing import Optional, List, Dict
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

class SnowflakeManager:
    """Manages Snowflake database connections and operations"""
    
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """Establish connection to Snowflake"""
        try:
            self.connection = snowflake.connector.connect(
                account=os.getenv('SNOWFLAKE_ACCOUNT'),
                user=os.getenv('SNOWFLAKE_USER'),
                password=os.getenv('SNOWFLAKE_PASSWORD'),
                warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
                database=os.getenv('SNOWFLAKE_DATABASE', 'BUYER_SCORING'),
                schema=os.getenv('SNOWFLAKE_SCHEMA', 'DATA')
            )
            
            # Test the connection
            cursor = self.connection.cursor()
            cursor.execute("SELECT CURRENT_VERSION()")
            version = cursor.fetchone()
            cursor.close()
            
            st.success(f"✅ Connected to Snowflake (Version: {version[0]})")
            
        except Exception as e:
            st.error(f"❌ Snowflake connection failed: {str(e)}")
            st.info("💡 Make sure your .env file has correct Snowflake credentials")
            self.connection = None
    
    def is_connected(self) -> bool:
        """Check if connected to Snowflake"""
        return self.connection is not None
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.is_connected():
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Create companies table (for raw company data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    id NUMBER AUTOINCREMENT PRIMARY KEY,
                    company_name VARCHAR(255) NOT NULL,
                    industry VARCHAR(100),
                    company_size VARCHAR(50),
                    employee_count NUMBER,
                    revenue VARCHAR(50),
                    location VARCHAR(255),
                    website VARCHAR(255),
                    domain VARCHAR(255),
                    contact_name VARCHAR(255),
                    contact_title VARCHAR(255),
                    contact_email VARCHAR(255),
                    phone VARCHAR(50),
                    linkedin_url VARCHAR(500),
                    funding_stage VARCHAR(50),
                    founded_year NUMBER,
                    description TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                    batch_id VARCHAR(100)
                )
            """)
            
            # Create buyers table (for scored results)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS buyers (
                    id NUMBER AUTOINCREMENT PRIMARY KEY,
                    company_name VARCHAR(255) NOT NULL,
                    industry VARCHAR(100),
                    company_size VARCHAR(50),
                    revenue VARCHAR(50),
                    location VARCHAR(255),
                    contact_name VARCHAR(255),
                    contact_title VARCHAR(255),
                    contact_email VARCHAR(255),
                    linkedin_url VARCHAR(500),
                    website VARCHAR(255),
                    employee_count NUMBER,
                    funding_stage VARCHAR(50),
                    current_tools VARCHAR(255),
                    score NUMBER(3,1),
                    reason TEXT,
                    product_description TEXT,
                    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                    upload_session VARCHAR(100)
                )
            """)
            
            # Create training_data table for model improvement
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id NUMBER AUTOINCREMENT PRIMARY KEY,
                    company_name VARCHAR(255) NOT NULL,
                    industry VARCHAR(100),
                    company_size VARCHAR(50),
                    revenue VARCHAR(50),
                    location VARCHAR(255),
                    product_description TEXT,
                    original_score NUMBER(3,1),
                    corrected_score NUMBER(3,1),
                    original_reason TEXT,
                    corrected_reason TEXT,
                    feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                    user_id VARCHAR(100)
                )
            """)
            
            cursor.close()
            st.success("✅ Database tables ready")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to create tables: {str(e)}")
            return False
    
    def get_companies_for_scoring(self, limit: int = 100, filters: Dict = None) -> pd.DataFrame:
        """Get companies from the database for scoring"""
        if not self.is_connected():
            return pd.DataFrame()
        
        try:
            cursor = self.connection.cursor()
            
            # Build WHERE clause based on filters
            where_conditions = []
            params = []
            
            if filters:
                if filters.get('industry'):
                    where_conditions.append("LOWER(industry) LIKE LOWER(%s)")
                    params.append(f"%{filters['industry']}%")
                
                if filters.get('min_employees'):
                    where_conditions.append("employee_count >= %s")
                    params.append(filters['min_employees'])
                
                if filters.get('max_employees'):
                    where_conditions.append("employee_count <= %s")
                    params.append(filters['max_employees'])
                
                if filters.get('location'):
                    where_conditions.append("LOWER(location) LIKE LOWER(%s)")
                    params.append(f"%{filters['location']}%")
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Get companies
            query = f"""
                SELECT 
                    company_name, industry, company_size, employee_count,
                    revenue, location, website, contact_name, contact_title,
                    contact_email, description
                FROM companies 
                WHERE {where_clause}
                ORDER BY RANDOM()  -- Get random sample
                LIMIT {limit}
            """
            
            cursor.execute(query, params)
            
            # Fetch data and column names
            columns = [desc[0].lower() for desc in cursor.description]
            data = cursor.fetchall()
            cursor.close()
            
            if data:
                df = pd.DataFrame(data, columns=columns)
                st.info(f"📊 Retrieved {len(df)} companies from Snowflake")
                return df
            else:
                st.info("📭 No companies found matching your criteria")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"❌ Failed to retrieve companies: {str(e)}")
            return pd.DataFrame()
    
    def get_company_stats(self) -> Dict:
        """Get statistics about companies in the database"""
        if not self.is_connected():
            return {}
        
        try:
            cursor = self.connection.cursor()
            stats = {}
            
            # Total companies
            cursor.execute("SELECT COUNT(*) FROM companies")
            stats['total_companies'] = cursor.fetchone()[0]
            
            # Companies by industry (top 10)
            cursor.execute("""
                SELECT industry, COUNT(*) as count 
                FROM companies 
                WHERE industry IS NOT NULL AND industry != ''
                GROUP BY industry 
                ORDER BY count DESC 
                LIMIT 10
            """)
            
            industries = []
            for row in cursor.fetchall():
                industries.append({'industry': row[0], 'count': row[1]})
            stats['top_industries'] = industries
            
            # Companies by size
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN employee_count < 10 THEN 'Startup (1-9)'
                        WHEN employee_count < 50 THEN 'Small (10-49)'
                        WHEN employee_count < 250 THEN 'Medium (50-249)'
                        WHEN employee_count < 1000 THEN 'Large (250-999)'
                        ELSE 'Enterprise (1000+)'
                    END as size_category,
                    COUNT(*) as count
                FROM companies 
                WHERE employee_count IS NOT NULL
                GROUP BY size_category
                ORDER BY count DESC
            """)
            
            sizes = []
            for row in cursor.fetchall():
                sizes.append({'size': row[0], 'count': row[1]})
            stats['company_sizes'] = sizes
            
            cursor.close()
            return stats
            
        except Exception as e:
            st.error(f"❌ Failed to get company stats: {str(e)}")
            return {}
    
    def save_scoring_results(self, scored_df: pd.DataFrame, product_description: str) -> bool:
        """Save scored results to Snowflake"""
        if not self.is_connected():
            st.warning("⚠️ No Snowflake connection - results not saved to database")
            return False
        
        try:
            # Ensure tables exist
            self.create_tables()
            
            cursor = self.connection.cursor()
            
            # Generate session ID
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Insert each row
            for _, row in scored_df.iterrows():
                cursor.execute("""
                    INSERT INTO buyers (
                        company_name, industry, company_size, revenue, location,
                        contact_name, contact_title, contact_email, linkedin_url, website,
                        employee_count, funding_stage, current_tools,
                        score, reason, product_description, upload_session
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    row.get('company_name', ''),
                    row.get('industry', ''),
                    row.get('company_size', ''),
                    row.get('revenue', ''),
                    row.get('location', ''),
                    row.get('contact_name', ''),
                    row.get('contact_title', ''),
                    row.get('contact_email', ''),
                    row.get('linkedin_url', ''),
                    row.get('website', ''),
                    row.get('employee_count', None),
                    row.get('funding_stage', ''),
                    row.get('current_tools', ''),
                    row.get('score', 5),
                    row.get('reason', ''),
                    product_description,
                    session_id
                ))
            
            self.connection.commit()
            cursor.close()
            
            st.success(f"✅ Saved {len(scored_df)} records to Snowflake (Session: {session_id})")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to save to Snowflake: {str(e)}")
            return False
    
    def get_historical_data(self, limit: int = 1000) -> pd.DataFrame:
        """Retrieve historical scoring data for model training"""
        if not self.is_connected():
            return pd.DataFrame()
        
        try:
            cursor = self.connection.cursor()
            
            # Get historical data with all necessary columns
            cursor.execute(f"""
                SELECT 
                    company_name, industry, company_size, revenue, location,
                    contact_name, contact_title, score, reason, product_description,
                    scored_at
                FROM buyers 
                WHERE score IS NOT NULL 
                ORDER BY scored_at DESC 
                LIMIT {limit}
            """)
            
            # Fetch data and column names
            columns = [desc[0].lower() for desc in cursor.description]
            data = cursor.fetchall()
            cursor.close()
            
            if data:
                df = pd.DataFrame(data, columns=columns)
                st.info(f"📚 Retrieved {len(df)} historical records for training")
                return df
            else:
                st.info("📭 No historical data found")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"❌ Failed to retrieve historical data: {str(e)}")
            return pd.DataFrame()
    
    def get_record_count(self) -> int:
        """Get total number of scored buyers"""
        if not self.is_connected():
            return 0
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM buyers")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
            
        except Exception as e:
            st.error(f"❌ Failed to get record count: {str(e)}")
            return 0
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """Get recent scoring sessions"""
        if not self.is_connected():
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT 
                    upload_session,
                    COUNT(*) as buyer_count,
                    AVG(score) as avg_score,
                    MIN(scored_at) as session_date,
                    product_description
                FROM buyers 
                WHERE upload_session IS NOT NULL
                GROUP BY upload_session, product_description
                ORDER BY session_date DESC
                LIMIT %s
            """, (limit,))
            
            columns = [desc[0].lower() for desc in cursor.description]
            data = cursor.fetchall()
            cursor.close()
            
            sessions = []
            for row in data:
                sessions.append(dict(zip(columns, row)))
            
            return sessions
            
        except Exception as e:
            st.error(f"❌ Failed to retrieve sessions: {str(e)}")
            return []
    
    def save_training_feedback(self, feedback_data: List[Dict]) -> bool:
        """Save user feedback for model improvement"""
        if not self.is_connected():
            return False
        
        try:
            cursor = self.connection.cursor()
            
            for feedback in feedback_data:
                cursor.execute("""
                    INSERT INTO training_data (
                        company_name, industry, company_size, revenue, location,
                        product_description, original_score, corrected_score,
                        original_reason, corrected_reason, user_id
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    feedback.get('company_name', ''),
                    feedback.get('industry', ''),
                    feedback.get('company_size', ''),
                    feedback.get('revenue', ''),
                    feedback.get('location', ''),
                    feedback.get('product_description', ''),
                    feedback.get('original_score', 0),
                    feedback.get('corrected_score', 0),
                    feedback.get('original_reason', ''),
                    feedback.get('corrected_reason', ''),
                    feedback.get('user_id', 'anonymous')
                ))
            
            self.connection.commit()
            cursor.close()
            
            st.success(f"✅ Saved {len(feedback_data)} feedback records")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to save feedback: {str(e)}")
            return False
    
    def get_analytics_data(self) -> Dict:
        """Get analytics data for dashboard"""
        if not self.is_connected():
            return {}
        
        try:
            cursor = self.connection.cursor()
            
            analytics = {}
            
            # Total buyers scored
            cursor.execute("SELECT COUNT(*) FROM buyers")
            analytics['total_buyers'] = cursor.fetchone()[0]
            
            # Average score
            cursor.execute("SELECT AVG(score) FROM buyers WHERE score IS NOT NULL")
            result = cursor.fetchone()
            analytics['avg_score'] = round(result[0], 2) if result[0] else 0
            
            # Score distribution
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN score >= 8 THEN 'Hot (8-10)'
                        WHEN score >= 6 THEN 'Warm (6-7)'
                        WHEN score >= 4 THEN 'Cold (4-5)'
                        ELSE 'Poor (<4)'
                    END as score_category,
                    COUNT(*) as count
                FROM buyers 
                WHERE score IS NOT NULL
                GROUP BY score_category
            """)
            
            score_dist = {}
            for row in cursor.fetchall():
                score_dist[row[0]] = row[1]
            analytics['score_distribution'] = score_dist
            
            # Top industries
            cursor.execute("""
                SELECT industry, COUNT(*) as count, AVG(score) as avg_score
                FROM buyers 
                WHERE industry IS NOT NULL AND industry != ''
                GROUP BY industry
                ORDER BY count DESC
                LIMIT 10
            """)
            
            industries = []
            for row in cursor.fetchall():
                industries.append({
                    'industry': row[0],
                    'count': row[1],
                    'avg_score': round(row[2], 2) if row[2] else 0
                })
            analytics['top_industries'] = industries
            
            cursor.close()
            return analytics
            
        except Exception as e:
            st.error(f"❌ Failed to get analytics: {str(e)}")
            return {}
    
    def close_connection(self):
        """Close the Snowflake connection"""
        if self.connection:
            self.connection.close()
            self.connection = None