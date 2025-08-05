"""
Upload your companies_sorted.csv data to Snowflake BUYER_SCORING database
This script will efficiently upload your 7M company records
"""

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import math

load_dotenv()

class SnowflakeUploader:
    """Handle large CSV uploads to Snowflake"""
    
    def __init__(self):
        self.connection = None
        self.csv_encoding = 'utf-8'  # Default encoding
        self.connect()
    
    def connect(self):
        """Connect to Snowflake"""
        try:
            self.connection = snowflake.connector.connect(
                account=os.getenv('SNOWFLAKE_ACCOUNT'),
                user=os.getenv('SNOWFLAKE_USER'),
                password=os.getenv('SNOWFLAKE_PASSWORD'),
                warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
                database=os.getenv('SNOWFLAKE_DATABASE', 'BUYER_SCORING'),
                schema=os.getenv('SNOWFLAKE_SCHEMA', 'DATA')
            )
            print("✅ Connected to Snowflake")
            
            # Test connection
            cursor = self.connection.cursor()
            cursor.execute("SELECT CURRENT_VERSION()")
            version = cursor.fetchone()
            print(f"📊 Snowflake Version: {version[0]}")
            cursor.close()
            
        except Exception as e:
            print(f"❌ Snowflake connection failed: {str(e)}")
            print("💡 Check your .env file credentials")
            sys.exit(1)
    
    def create_company_table(self):
        """Create the companies table for your raw data with UPPERCASE column names"""
        try:
            cursor = self.connection.cursor()
            
            # Drop table if it exists to ensure clean structure
            cursor.execute("DROP TABLE IF EXISTS companies")
            
            # Create companies table with UPPERCASE column names (Snowflake standard)
            cursor.execute("""
                CREATE TABLE companies (
                    ID NUMBER AUTOINCREMENT PRIMARY KEY,
                    COMPANY_NAME VARCHAR(255) NOT NULL,
                    DOMAIN VARCHAR(255),
                    YEAR_FOUNDED NUMBER,
                    INDUSTRY VARCHAR(100),
                    SIZE_RANGE VARCHAR(50),
                    LOCALITY VARCHAR(255),
                    COUNTRY VARCHAR(100),
                    LINKEDIN_URL VARCHAR(500),
                    CURRENT_EMPLOYEE_ESTIMATE NUMBER,
                    TOTAL_EMPLOYEE_ESTIMATE NUMBER,
                    UPLOADED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                    BATCH_ID VARCHAR(100)
                )
            """)
            
            print("✅ Companies table created with proper structure")
            cursor.close()
            
        except Exception as e:
            print(f"❌ Failed to create table: {str(e)}")
            sys.exit(1)
    
    def validate_csv(self, csv_path):
        """Validate CSV structure and show info - flexible validation"""
        try:
            print(f"📂 Validating {csv_path}...")
            
            # Try different encodings to handle various file formats
            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
            sample_df = None
            working_encoding = None
            
            for encoding in encodings_to_try:
                try:
                    sample_df = pd.read_csv(csv_path, nrows=5, encoding=encoding)
                    working_encoding = encoding
                    print(f"✅ Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    if "codec" not in str(e).lower():
                        # If it's not an encoding error, re-raise it
                        raise e
                    continue
            
            if sample_df is None:
                print(f"❌ Could not read CSV with any of these encodings: {encodings_to_try}")
                return False
            
            # Store the working encoding for later use
            self.csv_encoding = working_encoding
            
            print(f"📋 CSV Columns: {list(sample_df.columns)}")
            print(f"📊 Sample data:")
            print(sample_df.head())
            
            # Check for at least one identifiable company column
            possible_name_columns = ['name', 'company_name', 'Company', 'company', 'Company Name']
            has_company_column = any(col in sample_df.columns for col in possible_name_columns)
            
            if not has_company_column:
                print(f"❌ No company name column found. Looking for one of: {possible_name_columns}")
                print(f"Available columns: {list(sample_df.columns)}")
                return False
            
            # Get total row count using the working encoding
            try:
                with open(csv_path, 'r', encoding=working_encoding) as f:
                    total_rows = sum(1 for line in f) - 1  # Subtract header
            except:
                # Fallback method
                total_rows = len(pd.read_csv(csv_path, encoding=working_encoding))
            
            print(f"📈 Total companies to upload: {total_rows:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ CSV validation failed: {str(e)}")
            return False
    
    def clean_dataframe(self, df):
        """Clean and standardize the dataframe for your specific CSV format"""
        print("🧹 Cleaning data...")
        
        # Reset index to avoid pandas warning
        df = df.reset_index(drop=True)
        
        # Remove unnamed index columns first
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Map your CSV columns to database columns (UPPERCASE to match Snowflake)
        column_mapping = {
            'name': 'COMPANY_NAME',
            'Company': 'COMPANY_NAME',
            'company': 'COMPANY_NAME',
            'Company Name': 'COMPANY_NAME',
            'domain': 'DOMAIN',
            'year founded': 'YEAR_FOUNDED',
            'industry': 'INDUSTRY',
            'size range': 'SIZE_RANGE',
            'locality': 'LOCALITY',
            'country': 'COUNTRY',
            'linkedin url': 'LINKEDIN_URL',
            'current employee estimate': 'CURRENT_EMPLOYEE_ESTIMATE',
            'total employee estimate': 'TOTAL_EMPLOYEE_ESTIMATE'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Ensure we have a company_name column
        if 'COMPANY_NAME' not in df.columns:
            print("❌ No company name column found after mapping")
            print(f"Available columns after mapping: {list(df.columns)}")
            return pd.DataFrame()  # Return empty dataframe
        
        # Remove rows with empty company names
        df = df.dropna(subset=['COMPANY_NAME'])
        df = df[df['COMPANY_NAME'].astype(str).str.strip() != '']
        df = df[df['COMPANY_NAME'].astype(str) != 'nan']
        
        # Clean company names
        df['COMPANY_NAME'] = df['COMPANY_NAME'].astype(str).str.strip()
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['YEAR_FOUNDED', 'CURRENT_EMPLOYEE_ESTIMATE', 'TOTAL_EMPLOYEE_ESTIMATE']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values appropriately for string columns
        string_columns = ['DOMAIN', 'INDUSTRY', 'SIZE_RANGE', 'LOCALITY', 'COUNTRY', 'LINKEDIN_URL']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        # Truncate long strings to fit database constraints
        df['COMPANY_NAME'] = df['COMPANY_NAME'].str[:255]
        if 'INDUSTRY' in df.columns:
            df['INDUSTRY'] = df['INDUSTRY'].str[:100]
        if 'LOCALITY' in df.columns:
            df['LOCALITY'] = df['LOCALITY'].str[:255]
        if 'COUNTRY' in df.columns:
            df['COUNTRY'] = df['COUNTRY'].str[:100]
        if 'DOMAIN' in df.columns:
            df['DOMAIN'] = df['DOMAIN'].str[:255]
        if 'LINKEDIN_URL' in df.columns:
            df['LINKEDIN_URL'] = df['LINKEDIN_URL'].str[:500]
        if 'SIZE_RANGE' in df.columns:
            df['SIZE_RANGE'] = df['SIZE_RANGE'].str[:50]
        
        # Only keep columns that exist in the database table
        expected_columns = ['COMPANY_NAME', 'DOMAIN', 'YEAR_FOUNDED', 'INDUSTRY', 'SIZE_RANGE', 
                           'LOCALITY', 'COUNTRY', 'LINKEDIN_URL', 'CURRENT_EMPLOYEE_ESTIMATE', 
                           'TOTAL_EMPLOYEE_ESTIMATE']
        
        # Keep only columns that exist in both the dataframe and expected columns
        final_columns = [col for col in expected_columns if col in df.columns]
        df = df[final_columns]
        
        print(f"✅ Cleaned data: {len(df)} companies remaining")
        print(f"📋 Final columns: {list(df.columns)}")
        return df
    
    def upload_in_batches(self, csv_path, batch_size=10000):
        """Upload CSV data in batches for efficiency"""
        try:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"🚀 Starting batch upload (Batch ID: {batch_id})")
            
            # Get total number of rows using the correct encoding
            try:
                with open(csv_path, 'r', encoding=self.csv_encoding) as f:
                    total_rows = sum(1 for line in f) - 1  # Subtract header
            except:
                # Fallback method
                total_rows = len(pd.read_csv(csv_path, encoding=self.csv_encoding))
            
            total_batches = math.ceil(total_rows / batch_size)
            
            print(f"📊 Total rows: {total_rows:,}")
            print(f"📦 Batch size: {batch_size:,}")
            print(f"🔄 Total batches: {total_batches}")
            
            uploaded_count = 0
            
            # Process in chunks using the correct encoding
            for batch_num, chunk_df in enumerate(pd.read_csv(csv_path, chunksize=batch_size, encoding=self.csv_encoding), 1):
                print(f"\n📦 Processing batch {batch_num}/{total_batches}")
                
                # Clean the chunk
                clean_chunk = self.clean_dataframe(chunk_df)
                
                if len(clean_chunk) == 0:
                    print("⚠️ No valid data in this batch, skipping...")
                    continue
                
                # Add batch metadata (UPPERCASE column names)
                clean_chunk['BATCH_ID'] = batch_id
                clean_chunk['UPLOADED_AT'] = datetime.now()
                
                # Debug: Print column names before upload
                print(f"🔍 DataFrame columns: {list(clean_chunk.columns)}")
                
                # Upload to Snowflake
                try:
                    success, nchunks, nrows, _ = write_pandas(
                        self.connection,
                        clean_chunk,
                        'COMPANIES',
                        auto_create_table=False,
                        overwrite=False
                    )
                    
                    if success:
                        uploaded_count += len(clean_chunk)
                        print(f"✅ Uploaded {len(clean_chunk)} companies (Total: {uploaded_count:,})")
                    else:
                        print(f"❌ Failed to upload batch {batch_num}")
                        
                except Exception as batch_error:
                    print(f"❌ Batch {batch_num} upload error: {str(batch_error)}")
                    continue
            
            print(f"\n🎉 Upload complete!")
            print(f"📊 Total companies uploaded: {uploaded_count:,}")
            print(f"🆔 Batch ID: {batch_id}")
            
            return True, uploaded_count, batch_id
            
        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")
            return False, 0, None
    
    def verify_upload(self, batch_id):
        """Verify the upload was successful"""
        try:
            cursor = self.connection.cursor()
            
            # Count total records
            cursor.execute("SELECT COUNT(*) FROM companies")
            total_count = cursor.fetchone()[0]
            
            # Count records from this batch (UPPERCASE column names)
            cursor.execute("SELECT COUNT(*) FROM companies WHERE BATCH_ID = %s", (batch_id,))
            batch_count = cursor.fetchone()[0]
            
            # Sample some records (UPPERCASE column names)
            cursor.execute("""
                SELECT COMPANY_NAME, INDUSTRY, LOCALITY, COUNTRY
                FROM companies 
                WHERE BATCH_ID = %s 
                LIMIT 5
            """, (batch_id,))
            
            samples = cursor.fetchall()
            
            print(f"\n📊 Upload Verification:")
            print(f"Total companies in database: {total_count:,}")
            print(f"Companies from this batch: {batch_count:,}")
            print(f"\n📋 Sample records:")
            for i, (name, industry, locality, country) in enumerate(samples, 1):
                print(f"{i}. {name} | {industry} | {locality}, {country}")
            
            cursor.close()
            
        except Exception as e:
            print(f"❌ Verification failed: {str(e)}")

def main():
    """Main upload process"""
    print("🏔️ Snowflake Company Data Uploader")
    print("=" * 50)
    
    # Configuration
    CSV_PATH = "data/companies_sorted.csv"
    BATCH_SIZE = 10000  # Adjust based on your system memory
    
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV file not found: {CSV_PATH}")
        print("💡 Make sure your companies_sorted.csv is in the data/ folder")
        return
    
    # Initialize uploader
    uploader = SnowflakeUploader()
    
    # Create tables
    uploader.create_company_table()
    
    # Validate CSV
    if not uploader.validate_csv(CSV_PATH):
        print("❌ CSV validation failed")
        return
    
    # Confirm upload
    print(f"\n⚠️  You're about to upload data from: {CSV_PATH}")
    print(f"📊 Batch size: {BATCH_SIZE:,} records per batch")
    
    confirm = input("\n🤔 Continue with upload? (y/n): ").lower().strip()
    if confirm != 'y':
        print("❌ Upload cancelled")
        return
    
    # Start upload
    success, count, batch_id = uploader.upload_in_batches(CSV_PATH, BATCH_SIZE)
    
    if success:
        print("\n🎉 SUCCESS! Your data is now in Snowflake")
        
        # Verify upload
        uploader.verify_upload(batch_id)
        
        print(f"\n🎯 Next steps:")
        print(f"1. Your {count:,} companies are now in BUYER_SCORING.DATA.COMPANIES")
        print(f"2. Run your Streamlit app: streamlit run app.py")
        print(f"3. The app will now pull data from Snowflake!")
        print(f"4. Use 'Train on Historical Data' to train your AI model")
        
    else:
        print("❌ Upload failed. Check error messages above.")

if __name__ == "__main__":
    main()