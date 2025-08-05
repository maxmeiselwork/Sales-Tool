"""
Unit tests for the AI Buyer Scoring Tool
Tests for model.py functions and scoring logic
"""

import unittest
import pandas as pd
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.model import create_scoring_prompt, BuyerScorer
from src.utils import (
    validate_csv_structure, 
    clean_company_data, 
    format_employee_count,
    create_buyer_profile_text,
    parse_ai_response,
    validate_product_description
)

class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_df = pd.DataFrame({
            'company_name': ['TechCorp', 'HealthCare Inc', ''],
            'industry': ['Technology', 'Healthcare', 'Unknown'],
            'employee_count': [100, 'nan', 50],
            'location': ['SF', 'Boston', 'NY']
        })
    
    def test_validate_csv_structure_valid(self):
        """Test CSV validation with valid data"""
        valid_df = pd.DataFrame({'company_name': ['Test Corp']})
        is_valid, message = validate_csv_structure(valid_df)
        self.assertTrue(is_valid)
        self.assertEqual(message, "CSV structure is valid")
    
    def test_validate_csv_structure_missing_required(self):
        """Test CSV validation with missing required columns"""
        invalid_df = pd.DataFrame({'random_column': ['data']})
        is_valid, message = validate_csv_structure(invalid_df)
        self.assertFalse(is_valid)
        self.assertIn("Missing required columns", message)
    
    def test_validate_csv_structure_empty(self):
        """Test CSV validation with empty dataframe"""
        empty_df = pd.DataFrame()
        is_valid, message = validate_csv_structure(empty_df)
        self.assertFalse(is_valid)
        self.assertEqual(message, "CSV file is empty")
    
    def test_clean_company_data(self):
        """Test data cleaning function"""
        cleaned_df = clean_company_data(self.sample_df)
        
        # Should remove rows with empty company names
        self.assertEqual(len(cleaned_df), 2)
        
        # Should clean company names
        self.assertNotIn('', cleaned_df['company_name'].values)
    
    def test_format_employee_count(self):
        """Test employee count formatting"""
        self.assertEqual(format_employee_count(5), "Startup (1-9 employees)")
        self.assertEqual(format_employee_count(25), "Small company (10-49 employees)")
        self.assertEqual(format_employee_count(100), "Medium company (50-249 employees)")
        self.assertEqual(format_employee_count(500), "Large company (250-999 employees)")
        self.assertEqual(format_employee_count(2000), "Enterprise (1000+ employees)")
        self.assertEqual(format_employee_count('nan'), "Unknown size")
        self.assertEqual(format_employee_count(None), "Unknown size")
    
    def test_create_buyer_profile_text(self):
        """Test buyer profile text creation"""
        row = pd.Series({
            'company_name': 'TechCorp',
            'industry': 'Technology',
            'employee_count': 100,
            'location': 'San Francisco'
        })
        
        profile = create_buyer_profile_text(row)
        self.assertIn('TechCorp', profile)
        self.assertIn('Technology', profile)
        self.assertIn('Medium company', profile)
        self.assertIn('San Francisco', profile)
    
    def test_parse_ai_response(self):
        """Test AI response parsing"""
        # Test score extraction
        response1 = "Score: 8\nReason: Good fit because tech company"
        score1, reason1 = parse_ai_response(response1)
        self.assertEqual(score1, 8)
        self.assertIn("Good fit", reason1)
        
        # Test different format
        response2 = "This company gets a 9/10 rating due to size"
        score2, reason2 = parse_ai_response(response2)
        self.assertEqual(score2, 9)
        
        # Test edge cases
        response3 = "No clear score here"
        score3, reason3 = parse_ai_response(response3)
        self.assertEqual(score3, 5)  # Default score
    
    def test_validate_product_description(self):
        """Test product description validation"""
        # Valid description
        valid_desc = "CRM software for growing companies"
        is_valid, message = validate_product_description(valid_desc)
        self.assertTrue(is_valid)
        
        # Empty description
        is_valid, message = validate_product_description("")
        self.assertFalse(is_valid)
        self.assertIn("cannot be empty", message)
        
        # Too short
        is_valid, message = validate_product_description("CRM")
        self.assertFalse(is_valid)
        self.assertIn("at least 10 characters", message)
        
        # Too long
        long_desc = "x" * 1001
        is_valid, message = validate_product_description(long_desc)
        self.assertFalse(is_valid)
        self.assertIn("less than 1000 characters", message)

class TestModel(unittest.TestCase):
    """Test model functions"""
    
    def test_create_scoring_prompt(self):
        """Test scoring prompt creation"""
        product_desc = "CRM software for small businesses"
        buyer_profile = "Company: TechCorp | Industry: Technology | Size: Medium"
        
        prompt = create_scoring_prompt(product_desc, buyer_profile)
        
        self.assertIn(product_desc, prompt)
        self.assertIn(buyer_profile, prompt)
        self.assertIn("score from 1 to 10", prompt.lower())
        self.assertIn("reason", prompt.lower())
    
    def test_buyer_scorer_initialization(self):
        """Test BuyerScorer class initialization"""
        # Test with mock API key
        os.environ['OPENAI_API_KEY'] = 'test-key'
        scorer = BuyerScorer(model_type='openai')
        self.assertEqual(scorer.model_type, 'openai')
        
        # Test HuggingFace initialization
        scorer_hf = BuyerScorer(model_type='huggingface')
        self.assertEqual(scorer_hf.model_type, 'huggingface')

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_flow(self):
        """Test complete scoring flow with sample data"""
        # Create sample data
        sample_data = pd.DataFrame({
            'company_name': ['TechCorp', 'HealthCare Inc'],
            'industry': ['Technology', 'Healthcare'],
            'employee_count': [100, 200],
            'location': ['SF', 'Boston']
        })
        
        # Validate structure
        is_valid, message = validate_csv_structure(sample_data)
        self.assertTrue(is_valid)
        
        # Clean data
        cleaned_data = clean_company_data(sample_data)
        self.assertEqual(len(cleaned_data), 2)
        
        # Create profiles
        for idx, row in cleaned_data.iterrows():
            profile = create_buyer_profile_text(row)
            self.assertIsInstance(profile, str)
            self.assertTrue(len(profile) > 0)
        
        # Validate product description
        product_desc = "CRM software for growing companies"
        is_valid, message = validate_product_description(product_desc)
        self.assertTrue(is_valid)
        
        # Create prompt
        prompt = create_scoring_prompt(product_desc, profile)
        self.assertIsInstance(prompt, str)
        self.assertTrue(len(prompt) > 0)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)