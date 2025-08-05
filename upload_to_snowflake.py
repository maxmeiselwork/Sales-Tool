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
        """Create the companies table for your raw data"""
        try:
            cursor = self.connection.cursor()
            
            # Create companies table (separate from scored buyers)
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
            
            print("✅ Companies table ready")
            cursor.close()
            
        except Exception as e:
            print(f"❌ Failed to create table: {str(e)}")
            sys.exit(1)
    
    def validate_csv(self, csv_path):
        """Validate CSV structure and show info"""
        try:
            print(f"📂 Validating {csv_path}...")
            
            # Read first few rows to check structure
            sample_df = pd.read_csv(csv_path, nrows=5)
            
            print(f"📋 CSV Columns: {list(sample_df.columns)}")
            print(f"📊 Sample data:")
            print(sample_df.head())
            
            # Check for required columns
            required_cols = ['company_name']
            missing_cols = [col for col in required_cols if col not in sample_df.columns]
            
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                return False
            
            # Get total row count
            total_rows = sum(1 for line in open(csv_path)) - 1  # Subtract header
            print(f"📈 Total companies to upload: {total_rows:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ CSV validation failed: {str(e)}")
            return False
    
    def clean_dataframe(self, df):
        """Clean and standardize the dataframe"""
        print("🧹 Cleaning data...")
        
        # Remove rows with empty company names
        df = df.dropna(subset=['company_name'])
        df = df[df['company_name'].str.strip() != '']
        
        # Clean company names
        df['company_name'] = df['company_name'].str.strip()
        
        # Standardize column names and handle missing columns
        column_mapping = {
            'Company': 'company_name',
            'Company Name': 'company_name',
            'Industry': 'industry',
            'Size': 'company_size',
            'Employee Count': 'employee_count',
            'Employees': 'employee_count',
            'Location': 'location',
            'City': 'location',
            'Revenue': 'revenue',
            'Website': 'website',
            'Domain': 'domain',
            'Contact Name': 'contact_name',
            'Contact': 'contact_name',
            'Title': 'contact_title',
            'Email': 'contact_email'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Ensure numeric columns are properly typed
        if 'employee_count' in df.columns:
            df['employee_count'] = pd.to_numeric(df['employee_count'], errors='coerce')
        
        if 'founded_year' in df.columns:
            df['founded_year'] = pd.to_numeric(df['founded_year'], errors='coerce')
        
        # Fill NaN values appropriately
        string_columns = ['industry', 'company_size', 'location', 'website', 'domain', 
                         'contact_name', 'contact_title', 'contact_email', 'revenue']
        
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # Truncate long strings to fit database constraints
        df['company_name'] = df['company_name'].str[:255]
        if 'industry' in df.columns:
            df['industry'] = df['industry'].str[:100]
        if 'location' in df.columns:
            df['location'] = df['location'].str[:255]
        
        print(f"✅ Cleaned data: {len(df)} companies remaining")
        return df
    
    def upload_in_batches(self, csv_path, batch_size=10000):
        """Upload CSV data in batches for efficiency"""
        try:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"🚀 Starting batch upload (Batch ID: {batch_id})")
            
            # Get total number of rows
            total_rows = sum(1 for line in open(csv_path)) - 1
            total_batches = math.ceil(total_rows / batch_size)
            
            print(f"📊 Total rows: {total_rows:,}")
            print(f"📦 Batch size: {batch_size:,}")
            print(f"🔄 Total batches: {total_batches}")
            
            uploaded_count = 0
            
            # Process in chunks
            for batch_num, chunk_df in enumerate(pd.read_csv(csv_path, chunksize=batch_size), 1):
                print(f"\n📦 Processing batch {batch_num}/{total_batches}")
                
                # Clean the chunk
                clean_chunk = self.clean_dataframe(chunk_df)
                
                if len(clean_chunk) == 0:
                    print("⚠️ No valid data in this batch, skipping...")
                    continue
                
                # Add batch metadata
                clean_chunk['batch_id'] = batch_id
                clean_chunk['uploaded_at'] = datetime.now()
                
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
            
            # Count records from this batch
            cursor.execute("SELECT COUNT(*) FROM companies WHERE batch_id = %s", (batch_id,))
            batch_count = cursor.fetchone()[0]
            
            # Sample some records
            cursor.execute("""
                SELECT company_name, industry, location 
                FROM companies 
                WHERE batch_id = %s 
                LIMIT 5
            """, (batch_id,))
            
            samples = cursor.fetchall()
            
            print(f"\n📊 Upload Verification:")
            print(f"Total companies in database: {total_count:,}")
            print(f"Companies from this batch: {batch_count:,}")
            print(f"\n📋 Sample records:")
            for i, (name, industry, location) in enumerate(samples, 1):
                print(f"{i}. {name} | {industry} | {location}")
            
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