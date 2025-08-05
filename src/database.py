"""
Dynamic Snowflake Database Manager
Automatically discovers tables and adapts to any schema structure
"""

import os
import pandas as pd
import snowflake.connector
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import streamlit as st
from dotenv import load_dotenv
import json

load_dotenv()

class SnowflakeManager:
    """Dynamic Snowflake manager that adapts to any table structure"""
    
    def __init__(self):
        self.connection = None
        self.available_tables = []
        self.table_schemas = {}
        self.column_mappings = {}
        self.connect()
        if self.is_connected():
            self.discover_tables()
    
    def connect(self):
        """Establish connection to Snowflake"""
        try:
            self.connection = snowflake.connector.connect(
                account=os.getenv('SNOWFLAKE_ACCOUNT'),
                user=os.getenv('SNOWFLAKE_USER'),
                password=os.getenv('SNOWFLAKE_PASSWORD'),
                warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
                database=os.getenv('SNOWFLAKE_DATABASE', 'BUYER_SCORING'),
                schema=os.getenv('SNOWFLAKE_SCHEMA', 'DATA'),
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
    
    def discover_tables(self):
        """Discover all available tables and their schemas"""
        if not self.is_connected():
            return
        
        try:
            cursor = self.connection.cursor()
            
            # Get all tables in the current schema
            cursor.execute("""
                SELECT TABLE_NAME, ROW_COUNT 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
                AND TABLE_TYPE = 'BASE TABLE'
                ORDER BY ROW_COUNT DESC NULLS LAST
            """)
            
            tables_info = cursor.fetchall()
            self.available_tables = []
            
            for table_name, row_count in tables_info:
                # Get column information for each table
                cursor.execute(f"""
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = '{table_name}'
                    AND TABLE_SCHEMA = CURRENT_SCHEMA()
                    ORDER BY ORDINAL_POSITION
                """)
                
                columns_info = cursor.fetchall()
                columns = []
                for col_name, data_type, is_nullable in columns_info:
                    columns.append({
                        'name': col_name,
                        'type': data_type,
                        'nullable': is_nullable == 'YES'
                    })
                
                table_info = {
                    'name': table_name,
                    'row_count': row_count or 0,
                    'columns': columns
                }
                
                self.available_tables.append(table_info)
                self.table_schemas[table_name] = columns
                
                # Create intelligent column mapping
                self.column_mappings[table_name] = self._create_column_mapping(columns)
            
            cursor.close()
            st.info(f"📊 Discovered {len(self.available_tables)} tables")
            
        except Exception as e:
            st.error(f"❌ Failed to discover tables: {str(e)}")
    
    def _create_column_mapping(self, columns: List[Dict]) -> Dict[str, str]:
        """Create intelligent mapping from standard fields to actual column names"""
        column_names = [col['name'].lower() for col in columns]
        column_name_dict = {col['name'].lower(): col['name'] for col in columns}
        mapping = {}
        
        # Define mapping patterns for common fields - Updated with your actual columns
        field_patterns = {
            'company_name': ['company_name', 'name', 'company', 'organization', 'org_name', 'business_name'],
            'industry': ['industry', 'sector', 'vertical', 'business_type', 'category'],
            'employee_count': ['current_employee_estimate', 'total_employee_estimate', 'employee_count', 'employees', 'staff_count', 'headcount', 'size'],
            'location': ['locality', 'location', 'city', 'address', 'headquarters', 'hq_location'],
            'country': ['country', 'nation', 'region'],
            'website': ['domain', 'website', 'url', 'web_address', 'site'],
            'revenue': ['revenue', 'sales', 'turnover', 'annual_revenue'],
            'founded_year': ['year_founded', 'founded_year', 'established', 'founded', 'incorporation_year'],
            'description': ['description', 'about', 'overview', 'summary', 'profile'],
            'linkedin_url': ['linkedin_url', 'linkedin', 'linkedin_profile'],
            'phone': ['phone', 'telephone', 'phone_number', 'contact_phone', 'phone_normalized'],
            'email': ['email', 'contact_email', 'email_address', 'email_normalized'],
            'company_size': ['size_range', 'company_size', 'size_category', 'business_size'],
            'id': ['id', 'company_id', 'record_id'],
            'batch_id': ['batch_id', 'batch', 'upload_batch'],
            'uploaded_at': ['uploaded_at', 'created_at', 'date_added', 'timestamp']
        }
        
        # Find best matches with improved logic
        for standard_field, patterns in field_patterns.items():
            best_match = None
            best_score = 0
            
            for pattern in patterns:
                for col_name in column_names:
                    # Exact match gets highest score
                    if pattern.lower() == col_name:
                        best_match = column_name_dict[col_name]
                        best_score = 100
                        break
                    # Partial match gets lower score
                    elif pattern.lower() in col_name:
                        score = len(pattern) / len(col_name) * 50
                        if score > best_score:
                            best_match = column_name_dict[col_name]
                            best_score = score
                    # Reverse partial match
                    elif col_name in pattern.lower():
                        score = len(col_name) / len(pattern) * 30
                        if score > best_score:
                            best_match = column_name_dict[col_name]
                            best_score = score
                
                if best_score == 100:  # Found exact match
                    break
            
            if best_match:
                mapping[standard_field] = best_match
        
        return mapping
    
    def get_available_tables(self) -> List[Dict]:
        """Get list of available tables with metadata"""
        return self.available_tables
    
    def get_table_schema(self, table_name: str) -> List[Dict]:
        """Get schema for a specific table"""
        return self.table_schemas.get(table_name, [])
    
    def get_table_preview(self, table_name: str, limit: int = 10) -> pd.DataFrame:
        """Get a preview of data from any table with type-safe casting"""
        if not self.is_connected():
            return pd.DataFrame()

        try:
            cursor = self.connection.cursor()

            # Build query with safe casting for all problematic column types
            schema = self.table_schemas.get(table_name, [])
            select_clauses = []
            for col in schema:
                col_name = col['name']
                col_type = col['type'].upper()
                # Convert all numeric and timestamp types to string to avoid overflow
                if col_type in ['NUMBER', 'BIGINT', 'DECIMAL', 'INT', 'INTEGER', 'FLOAT', 'DOUBLE']:
                    select_clauses.append(f"TO_VARCHAR({col_name}) AS {col_name}")
                elif 'TIMESTAMP' in col_type or col_type in ['TIMESTAMP_NTZ', 'TIMESTAMP_LTZ', 'TIMESTAMP_TZ']:
                    select_clauses.append(f"TO_VARCHAR({col_name}) AS {col_name}")
                else:
                    select_clauses.append(f"{col_name}")

            query = f"""
                SELECT {', '.join(select_clauses)} FROM {table_name}
                LIMIT {limit}
            """
            cursor.execute(query)

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            cursor.close()

            return pd.DataFrame(rows, columns=columns)

        except Exception as e:
            st.error(f"❌ Failed to preview table {table_name}: {str(e)}")
            return pd.DataFrame()
        
    def get_companies_for_scoring(self, table_name: str, limit: int = 100, filters: Dict = None) -> pd.DataFrame:
        """Get companies from any selected table for scoring"""
        if not self.is_connected():
            return pd.DataFrame()
        
        try:
            cursor = self.connection.cursor()
            mapping = self.column_mappings.get(table_name, {})
            
            # Build SELECT clause using available columns with safer data handling
            select_columns = []
            schema = self.table_schemas.get(table_name, [])
            for standard_field, actual_column in mapping.items():
                if actual_column:
                    # Cast ALL numeric and timestamp columns to string to avoid overflow
                    col_info = next((col for col in schema if col['name'] == actual_column), None)
                    if col_info:
                        col_type = col_info['type'].upper()
                        if col_type in ['NUMBER', 'BIGINT', 'DECIMAL', 'INT', 'INTEGER', 'FLOAT', 'DOUBLE']:
                            select_columns.append(f"TO_VARCHAR({actual_column}) as {standard_field}")
                        elif 'TIMESTAMP' in col_type or col_type in ['TIMESTAMP_NTZ', 'TIMESTAMP_LTZ', 'TIMESTAMP_TZ']:
                            select_columns.append(f"TO_VARCHAR({actual_column}) as {standard_field}")
                        else:
                            select_columns.append(f"{actual_column} as {standard_field}")
                    else:
                        select_columns.append(f"{actual_column} as {standard_field}")
            
            # If no mapping found, get all columns but be safe with large numbers
            if not select_columns:
                # Get column info to identify numeric columns
                schema = self.table_schemas.get(table_name, [])
                for col in schema:
                    if col['type'] in ['NUMBER', 'BIGINT'] and ('employee' in col['name'].lower() or 'estimate' in col['name'].lower()):
                        select_columns.append(f"CAST({col['name']} AS VARCHAR) as {col['name'].lower()}")
                    else:
                        select_columns.append(f"{col['name']} as {col['name'].lower()}")
            
            # Build WHERE clause based on filters
            where_conditions = ["1=1"]  # Always true condition
            params = []
            
            if filters and mapping:
                if filters.get('industry') and mapping.get('industry'):
                    where_conditions.append(f"LOWER({mapping['industry']}) LIKE LOWER(%s)")
                    params.append(f"%{filters['industry']}%")
                
                if filters.get('min_employees') and mapping.get('employee_count'):
                    where_conditions.append(f"CAST({mapping['employee_count']} AS NUMBER) >= %s")
                    params.append(filters['min_employees'])
                
                if filters.get('max_employees') and mapping.get('employee_count'):
                    where_conditions.append(f"CAST({mapping['employee_count']} AS NUMBER) <= %s")
                    params.append(filters['max_employees'])
                
                if filters.get('location'):
                    location_conditions = []
                    if mapping.get('location'):
                        location_conditions.append(f"LOWER({mapping['location']}) LIKE LOWER(%s)")
                        params.append(f"%{filters['location']}%")
                    if mapping.get('country'):
                        location_conditions.append(f"LOWER({mapping['country']}) LIKE LOWER(%s)")
                        params.append(f"%{filters['location']}%")
                    
                    if location_conditions:
                        where_conditions.append(f"({' OR '.join(location_conditions)})")
            
            where_clause = " AND ".join(where_conditions)
            
            # Build the query
            query = f"""
                SELECT {', '.join(select_columns)}
                FROM {table_name} 
                WHERE {where_clause}
                ORDER BY RANDOM()
                LIMIT {limit}
            """

            # Set session to handle large numbers safely
            cursor.execute("ALTER SESSION SET PYTHON_CONNECTOR_QUERY_RESULT_FORMAT = 'json'")
            cursor.execute(query, params)

            # Fetch data and column names
            columns = [desc[0].lower() for desc in cursor.description]
            data = cursor.fetchall()
            cursor.close()
            
            if data:
                df = pd.DataFrame(data, columns=columns)
                
                # Convert string employee counts back to numeric where possible
                if 'employee_count' in df.columns:
                    df['employee_count'] = pd.to_numeric(df['employee_count'], errors='coerce')
                
                # Post-process location if we have both city and country
                if 'location' in df.columns and 'country' in df.columns:
                    df['location'] = df.apply(lambda row: 
                        f"{row['location']}, {row['country']}" if pd.notna(row['location']) and pd.notna(row['country'])
                        else row['location'] if pd.notna(row['location'])
                        else row['country'] if pd.notna(row['country'])
                        else 'Unknown', axis=1)
                
                st.info(f"📊 Retrieved {len(df)} companies from table '{table_name}'")
                return df
            else:
                st.info("📭 No companies found matching your criteria")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"❌ Failed to retrieve companies from {table_name}: {str(e)}")
            return pd.DataFrame()
    
    def get_company_stats(self, table_name: str) -> Dict:
        """Get statistics about companies in the selected table"""
        if not self.is_connected():
            return {}
        
        try:
            cursor = self.connection.cursor()
            mapping = self.column_mappings.get(table_name, {})
            stats = {}
            
            # Total companies
            company_name_col = mapping.get('company_name', 'COMPANY_NAME')
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {company_name_col} IS NOT NULL")
            stats['total_companies'] = cursor.fetchone()[0]
            
            # Companies by industry (top 10) if industry column exists
            if mapping.get('industry'):
                cursor.execute(f"""
                    SELECT {mapping['industry']}, COUNT(*) as count 
                    FROM {table_name} 
                    WHERE {mapping['industry']} IS NOT NULL AND {mapping['industry']} != ''
                    GROUP BY {mapping['industry']} 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                
                industries = []
                for row in cursor.fetchall():
                    industries.append({'industry': row[0], 'count': row[1]})
                stats['top_industries'] = industries
            
            # Companies by size if employee count column exists
            if mapping.get('employee_count'):
                cursor.execute(f"""
                    SELECT 
                        CASE 
                            WHEN CAST({mapping['employee_count']} AS NUMBER) < 10 THEN 'Startup (1-9)'
                            WHEN CAST({mapping['employee_count']} AS NUMBER) < 50 THEN 'Small (10-49)'
                            WHEN CAST({mapping['employee_count']} AS NUMBER) < 250 THEN 'Medium (50-249)'
                            WHEN CAST({mapping['employee_count']} AS NUMBER) < 1000 THEN 'Large (250-999)'
                            ELSE 'Enterprise (1000+)'
                        END as size_category,
                        COUNT(*) as count
                    FROM {table_name} 
                    WHERE {mapping['employee_count']} IS NOT NULL
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
            st.error(f"❌ Failed to get company stats for {table_name}: {str(e)}")
            return {}
    
    def create_tables(self):
        """Create necessary application tables if they don't exist"""
        if not self.is_connected():
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Create buyers table (for scored results)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS buyers (
                    id NUMBER AUTOINCREMENT PRIMARY KEY,
                    source_table VARCHAR(100),
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
                    employee_count VARCHAR(50), -- Changed to VARCHAR to handle large numbers
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
                    source_table VARCHAR(100),
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
            st.success("✅ Application tables ready")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to create tables: {str(e)}")
            return False
    
    def save_scoring_results(self, scored_df: pd.DataFrame, product_description: str, source_table: str) -> bool:
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
                # Convert employee_count to string if it's numeric
                emp_count = row.get('employee_count', '')
                if pd.notna(emp_count) and isinstance(emp_count, (int, float)):
                    emp_count = str(int(emp_count))
                
                cursor.execute("""
                    INSERT INTO buyers (
                        source_table, company_name, industry, company_size, revenue, location,
                        contact_name, contact_title, contact_email, linkedin_url, website,
                        employee_count, funding_stage, current_tools,
                        score, reason, product_description, upload_session
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    source_table,
                    str(row.get('company_name', '')),
                    str(row.get('industry', '')),
                    str(row.get('company_size', '')),
                    str(row.get('revenue', '')),
                    str(row.get('location', '')),
                    str(row.get('contact_name', '')),
                    str(row.get('contact_title', '')),
                    str(row.get('contact_email', '')),
                    str(row.get('linkedin_url', '')),
                    str(row.get('website', '')),
                    str(emp_count),
                    str(row.get('funding_stage', '')),
                    str(row.get('current_tools', '')),
                    float(row.get('score', 5)),
                    str(row.get('reason', '')),
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
                    scored_at, source_table
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
            # If buyers table doesn't exist yet, return 0
            return 0
    
    def get_all_companies_for_training(self, table_name: str) -> pd.DataFrame:
        """Get all company data from selected table for AI training"""
        if not self.is_connected():
            return pd.DataFrame()

        try:
            cursor = self.connection.cursor()
            mapping = self.column_mappings.get(table_name, {})
            schema = self.table_schemas.get(table_name, [])

            # Always cast all fields to safe types
            select_columns = []
            schema = self.table_schemas.get(table_name, [])
            for field, actual_column in mapping.items():
                schema_entry = next((col for col in schema if col['name'] == actual_column), None)
                if not schema_entry:
                    continue
                col_type = schema_entry['type'].upper()
                # Convert ALL numeric and timestamp types to string
                if col_type in ['NUMBER', 'BIGINT', 'DECIMAL', 'INT', 'INTEGER', 'FLOAT', 'DOUBLE']:
                    select_columns.append(f"TO_VARCHAR({actual_column}) AS {field}")
                elif 'TIMESTAMP' in col_type or col_type in ['TIMESTAMP_NTZ', 'TIMESTAMP_LTZ', 'TIMESTAMP_TZ']:
                    select_columns.append(f"TO_VARCHAR({actual_column}) AS {field}")
                else:
                    select_columns.append(f"{actual_column} AS {field}")

            query = f"""
                SELECT {', '.join(select_columns)}
                FROM {table_name}
                WHERE {mapping.get('company_name', 'COMPANY_NAME')} IS NOT NULL 
                AND {mapping.get('company_name', 'COMPANY_NAME')} != ''
                ORDER BY RANDOM()
                LIMIT 50000
            """

            # Set session to handle large numbers safely
            cursor.execute("ALTER SESSION SET PYTHON_CONNECTOR_QUERY_RESULT_FORMAT = 'json'")
            cursor.execute(query)
            columns = [desc[0].lower() for desc in cursor.description]
            data = cursor.fetchall()
            cursor.close()

            if data:
                df = pd.DataFrame(data, columns=columns)
                if 'employee_count' in df.columns:
                    df['employee_count'] = pd.to_numeric(df['employee_count'], errors='coerce')
                return df
            else:
                st.info(f"📭 No company data found in table '{table_name}' for training")
                return pd.DataFrame()

        except Exception as e:
            st.error(f"❌ Failed to retrieve training data from {table_name}: {str(e)}")
            return pd.DataFrame()

    
    def get_column_mapping_info(self, table_name: str) -> Dict:
        """Get information about how columns are mapped for a table"""
        mapping = self.column_mappings.get(table_name, {})
        schema = self.table_schemas.get(table_name, [])
        
        return {
            'mapping': mapping,
            'schema': schema,
            'mapped_fields': len([v for v in mapping.values() if v]),
            'total_columns': len(schema)
        }
    
    def close_connection(self):
        """Close the Snowflake connection"""
        if self.connection:
            self.connection.close()
            self.connection = None