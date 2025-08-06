"""
AI Model for Buyer Scoring - Company Knowledge Based - OPTIMIZED VERSION
Trains on your company database to understand company characteristics
OPTIMIZED for 7M+ records with intelligent sampling and efficient training
"""

import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import streamlit as st
from typing import Tuple, List, Dict
import json
import re
from dotenv import load_dotenv
import random
from collections import Counter

load_dotenv()

class BuyerScoringModel:
    """AI model trained on company data for intelligent buyer scoring - OPTIMIZED VERSION"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trained_model_path = "models/company_trained_model"
        self.model_name = os.getenv('MODEL_NAME', 'microsoft/DialoGPT-medium')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Optimization settings
        self.max_training_samples = 150000  # Increased but manageable
        self.industry_sample_limit = 3000   # Max per industry
        self.company_size_sample_limit = 2500  # Max per size category
        
        # Initialize model
        self.load_model()
    
    def load_model(self):
        """Load the AI model (base or company-trained)"""
        try:
            # Check if we have a company-trained model
            if os.path.exists(self.trained_model_path):
                st.info("Loading company-trained model...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.trained_model_path)
                self.model = AutoModelForCausalLM.from_pretrained(self.trained_model_path)
            else:
                st.info("Loading base model from Hugging Face...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            # Configure tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            st.success("Model loaded successfully")
            
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference"""
        return self.model is not None and self.tokenizer is not None
    
    def is_trained(self) -> bool:
        """Check if using a company-trained model"""
        return os.path.exists(self.trained_model_path)
    
    def create_company_knowledge_text(self, company_row) -> str:
        """Create CONCISE knowledge text about a company for training - OPTIMIZED"""
        parts = []
        
        company_name = company_row.get('company_name', 'Unknown Company')
        parts.append(f"Company: {company_name}")
        
        if company_row.get('industry'):
            parts.append(f"Industry: {company_row['industry']}")
        
        # Handle employee count efficiently
        employee_count = company_row.get('employee_count')
        if employee_count and pd.notna(employee_count):
            try:
                if isinstance(employee_count, str):
                    emp_count = int(employee_count)
                else:
                    emp_count = int(employee_count)
                
                parts.append(f"Size: {emp_count:,} employees")
                
                # Add concise size category
                if emp_count < 50:
                    parts.append("Type: Small")
                elif emp_count < 500:
                    parts.append("Type: Medium")
                elif emp_count < 5000:
                    parts.append("Type: Large")
                else:
                    parts.append("Type: Enterprise")
            except (ValueError, TypeError):
                parts.append(f"Size: {employee_count}")
        
        if company_row.get('location'):
            parts.append(f"Location: {company_row['location']}")
        
        if company_row.get('founded_year') and pd.notna(company_row['founded_year']):
            try:
                year = int(float(company_row['founded_year']))
                age = 2024 - year
                if age < 5:
                    parts.append("Stage: Startup")
                elif age < 15:
                    parts.append("Stage: Growth")
                else:
                    parts.append("Stage: Established")
            except (ValueError, TypeError):
                pass
        
        return " | ".join(parts)
    
    def intelligent_sampling(self, company_df: pd.DataFrame) -> pd.DataFrame:
        """Intelligently sample companies for diverse, representative training data"""
        st.info(f"Intelligent sampling from {len(company_df):,} companies...")
        
        sampled_companies = []
        
        # STRATEGY 1: Industry-based sampling
        if 'industry' in company_df.columns:
            industry_counts = company_df['industry'].value_counts()
            st.info(f"Found {len(industry_counts)} unique industries")
            
            for industry in industry_counts.head(25).index:  # Top 25 industries
                industry_df = company_df[company_df['industry'] == industry]
                
                # Sample proportionally but cap at limit
                industry_sample_size = min(
                    self.industry_sample_limit, 
                    max(500, int(len(industry_df) * 0.02))  # At least 500 or 2% of industry
                )
                
                if len(industry_df) > industry_sample_size:
                    sample = industry_df.sample(n=industry_sample_size, random_state=42)
                else:
                    sample = industry_df
                
                sampled_companies.append(sample)
                
            st.info(f"Sampled from {len(industry_counts.head(25))} top industries")
        
        # STRATEGY 2: Company size sampling
        if 'employee_count' in company_df.columns:
            # Define size buckets
            def get_size_bucket(emp_count):
                try:
                    count = int(emp_count) if pd.notna(emp_count) else 0
                    if count < 50: return 'small'
                    elif count < 500: return 'medium'
                    elif count < 5000: return 'large'
                    else: return 'enterprise'
                except: return 'unknown'
            
            company_df['size_bucket'] = company_df['employee_count'].apply(get_size_bucket)
            
            for size_bucket in ['small', 'medium', 'large', 'enterprise']:
                size_df = company_df[company_df['size_bucket'] == size_bucket]
                if len(size_df) > 0:
                    sample_size = min(self.company_size_sample_limit, len(size_df))
                    if len(size_df) > sample_size:
                        sample = size_df.sample(n=sample_size, random_state=42)
                    else:
                        sample = size_df
                    sampled_companies.append(sample)
            
            # Clean up temporary column
            company_df = company_df.drop('size_bucket', axis=1)
        
        # STRATEGY 3: Geographic diversity
        if 'country' in company_df.columns:
            top_countries = company_df['country'].value_counts().head(15).index
            for country in top_countries:
                country_df = company_df[company_df['country'] == country]
                sample_size = min(2000, max(200, len(country_df) // 10))
                if len(country_df) > sample_size:
                    sample = country_df.sample(n=sample_size, random_state=42)
                else:
                    sample = country_df
                sampled_companies.append(sample)
        
        # Combine all samples and remove duplicates
        if sampled_companies:
            combined_df = pd.concat(sampled_companies, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['company_name'], keep='first')
        else:
            # Fallback: random sampling
            sample_size = min(self.max_training_samples, len(company_df))
            combined_df = company_df.sample(n=sample_size, random_state=42)
        
        # Final cap to ensure manageable training size
        if len(combined_df) > self.max_training_samples:
            combined_df = combined_df.sample(n=self.max_training_samples, random_state=42)
        
        st.success(f"Intelligently selected {len(combined_df):,} diverse companies for training")
        return combined_df.reset_index(drop=True)
    
    def create_efficient_training_examples(self, sample_df: pd.DataFrame) -> List[str]:
        """Create efficient, high-quality training examples - OPTIMIZED"""
        training_texts = []
        
        progress_bar = st.progress(0)
        batch_size = 1000
        
        for i in range(0, len(sample_df), batch_size):
            batch_end = min(i + batch_size, len(sample_df))
            progress = batch_end / len(sample_df)
            progress_bar.progress(progress)
            
            batch_df = sample_df.iloc[i:batch_end]
            
            for _, row in batch_df.iterrows():
                knowledge_text = self.create_company_knowledge_text(row)
                
                # Create focused training example
                training_example = f"""Company Profile: {knowledge_text}

Business Context: This company in the {row.get('industry', 'unknown')} sector """
                
                # Add targeted business insights
                employee_count = row.get('employee_count')
                if employee_count and pd.notna(employee_count):
                    try:
                        emp_count = int(employee_count) if isinstance(employee_count, str) else int(employee_count)
                        
                        if emp_count < 50:
                            training_example += "needs affordable, simple solutions for small teams. Priorities: cost-effectiveness, ease of use, quick setup."
                        elif emp_count < 500:
                            training_example += "requires scalable solutions for growing operations. Priorities: growth support, integration capabilities, team collaboration."
                        elif emp_count < 5000:
                            training_example += "needs enterprise-grade solutions for complex operations. Priorities: reliability, security, advanced features."
                        else:
                            training_example += "requires sophisticated enterprise solutions. Priorities: compliance, security, scalability, custom integration."
                    except (ValueError, TypeError):
                        training_example += "has specific operational needs requiring tailored solutions."
                
                training_texts.append(training_example)
        
        progress_bar.progress(1.0)
        progress_bar.empty()
        
        return training_texts
    
    def train_on_company_data(self, company_df: pd.DataFrame) -> bool:
        """OPTIMIZED training on company data for effectiveness AND speed"""
        if not self.is_ready():
            st.error("❌ Base model not loaded. Cannot train.")
            return False
        
        try:
            st.info("Starting OPTIMIZED company knowledge training...")
            
            # STEP 1: Intelligent sampling for diverse, representative data
            sample_df = self.intelligent_sampling(company_df)
            
            # STEP 2: Create efficient training examples
            st.info("Creating high-quality training examples...")
            training_texts = self.create_efficient_training_examples(sample_df)
            
            st.info(f"Created {len(training_texts):,} optimized training examples")
            
            # STEP 3: Efficient tokenization
            def tokenize_function_optimized(examples):
                texts = examples['text'] if isinstance(examples['text'], list) else [examples['text']]
                return self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=320,  # Optimized length - not too short, not too long
                    return_tensors=None
                )

            dataset = Dataset.from_dict({'text': training_texts})
            tokenized_dataset = dataset.map(
                tokenize_function_optimized, 
                batched=True,
                batch_size=1000,  # Larger batches for efficiency
                remove_columns=dataset.column_names,
                num_proc=2 if os.cpu_count() > 2 else 1  # Parallel processing
            )

            # STEP 4: Optimized LoRA configuration
            for param in self.model.parameters():
                param.requires_grad = True

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,  # Good balance of quality and speed
                lora_alpha=32,
                lora_dropout=0.05,  # Lower dropout for faster convergence
                target_modules=["c_attn"]
            )

            model = get_peft_model(self.model, peft_config)
            model.train()
            model.enable_input_require_grads()

            # Verify trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            st.info(f"Training {trainable_params:,} parameters ({100 * trainable_params / total_params:.1f}% of model)")

            # STEP 5: Optimized training configuration
            # Detect hardware capabilities
            has_gpu = torch.cuda.is_available()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if has_gpu else 0
            
            # Adaptive batch size based on hardware
            if has_gpu and gpu_memory > 8:
                batch_size = 8
                gradient_steps = 2
            elif has_gpu:
                batch_size = 4
                gradient_steps = 4
            else:
                batch_size = 2
                gradient_steps = 8
            
            # Calculate optimal max_steps for faster training
            total_samples = len(tokenized_dataset)
            effective_batch_size = batch_size * gradient_steps
            max_steps = min(8000, total_samples // effective_batch_size)  # Cap at 8k steps
            
            training_args = TrainingArguments(
                output_dir="./company_training_results",
                num_train_epochs=1,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_steps,
                learning_rate=3e-4,  # Slightly higher for faster convergence
                logging_steps=100,
                save_steps=max(500, max_steps // 4),  # Save 4 times during training
                eval_strategy="no",
                save_total_limit=2,
                remove_unused_columns=False,
                report_to="none",
                gradient_checkpointing=False,
                fp16=has_gpu,  # Use mixed precision if GPU available
                dataloader_drop_last=True,
                warmup_steps=max(50, max_steps // 20),  # 5% warmup
                dataloader_num_workers=2 if os.cpu_count() > 2 else 0,
                optim="adamw_torch",
                max_steps=max_steps,  # Limit steps for faster training
                lr_scheduler_type="cosine",  # Better convergence
                weight_decay=0.01,
                adam_epsilon=1e-6,
                max_grad_norm=1.0,
            )
            
            st.info(f"   Training configuration:")
            st.info(f"   • Batch size: {batch_size} × {gradient_steps} = {effective_batch_size}")
            st.info(f"   • Max steps: {max_steps:,}")
            st.info(f"   • Hardware: {'GPU' if has_gpu else 'CPU'}")
            st.info(f"   • Mixed precision: {has_gpu}")

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
            
            st.info("Starting optimized training... (Estimated time: 30-90 minutes)")
            
            # Training with progress updates
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("AI learning your company database..."):
                trainer.train()
            
            progress_bar.empty()
            status_text.empty()
            
            # Save the trained model
            os.makedirs(self.trained_model_path, exist_ok=True)
            model.save_pretrained(self.trained_model_path)
            self.tokenizer.save_pretrained(self.trained_model_path)
            
            # Update current model
            self.model = model
            
            st.success("Optimized company knowledge training completed!")
            st.success(f"Model trained on {len(sample_df):,} diverse companies")
            st.info("Your AI now understands company characteristics and business needs!")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            return False
    
    def create_scoring_prompt(self, buyer_data: Dict, product_description: str) -> str:
        """Create intelligent scoring prompt using company knowledge"""
        
        company_knowledge = self.create_company_knowledge_text(buyer_data)
        
        prompt = f"""You are an expert B2B sales analyst with deep knowledge of companies and their needs.

PRODUCT/SERVICE TO EVALUATE: {product_description}

COMPANY TO SCORE: {company_knowledge}

TASK: Analyze if this company would be interested in buying this product/service and provide a score from 1-10.

ANALYSIS FRAMEWORK:
1. Industry Fit: Does this company's industry typically need this type of product?
2. Company Size Match: Is the company the right size for this product?
3. Business Stage: Is the company at a stage where they'd invest in this?
4. Geographic Fit: Are there location/market considerations?
5. Business Problems: What problems does this company likely face that this product could solve?

SCORING GUIDE:
- 9-10: Perfect fit - Company definitely needs this and can afford it
- 7-8: Strong fit - High likelihood of interest and purchase
- 5-6: Moderate fit - Some potential but needs more qualification
- 3-4: Poor fit - Low likelihood but possible niche use case
- 1-2: No fit - Company would not benefit from this product

Provide your analysis in this format:
Score: [1-10]
Reason: [Detailed explanation of why this company would or wouldn't buy this product, mentioning specific business needs and challenges they likely face]

Score:"""
        
        return prompt
    
    def extract_score_and_reason(self, buyer_data, response: str) -> Tuple[float, str]:
        """Extract score and reason from model response"""
        try:
            response = response.strip()
            score = 5.0  # default
            reason = "Analysis in progress"
            
            # Look for score patterns
            score_patterns = [
                r'Score:\s*(\d+\.?\d*)',
                r'score:\s*(\d+\.?\d*)',
                r'(\d+\.?\d*)/10',
                r'Rating:\s*(\d+\.?\d*)'
            ]
            
            for pattern in score_patterns:
                matches = re.findall(pattern, response)
                if matches:
                    score = float(matches[-1])
                    break
            
            # Look for detailed analysis
            analysis_patterns = [
                r'Detailed Analysis:\s*(.*?)(?=\n\n|\Z)',
                r'Analysis:\s*(.*?)(?=\n\n|\Z)',
                r'Reason:\s*(.*?)(?=\n\n|\Z)'
            ]
            
            for pattern in analysis_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    reason = match.group(1).strip()
                    break
            
            if reason == "Analysis in progress" and len(response) > 50:
                parts = re.split(r'Score:\s*\d+\.?\d*', response, flags=re.IGNORECASE)
                if len(parts) > 1:
                    reason = parts[-1].strip()
            
            if len(reason) < 20:
                reason = f"Company analysis: This {buyer_data.get('industry', 'company')} with {buyer_data.get('employee_count', 'unknown')} employees may have specific needs that align with the product offering."
            
            score = max(1.0, min(10.0, score))
            
            return score, reason
            
        except Exception as e:
            st.warning(f"⚠️ Error parsing response: {str(e)}")
            return 5.0, f"Analysis of {buyer_data.get('company_name', 'this company')} requires further review to determine product fit and potential challenges they face."
    
    def score_single_buyer(self, buyer_data: Dict, product_description: str) -> Tuple[float, str]:
        """Score a single buyer using company-trained AI"""
        if not self.is_ready():
            return 5.0, "Model not ready"
        
        try:
            # Use enhanced rule-based scoring with AI insights
            score = self._calculate_enhanced_score(buyer_data, product_description)
            reason = self._generate_detailed_analysis(buyer_data, product_description, score)
            
            return score, reason
            
        except Exception as e:
            st.error(f"❌ Scoring failed for {buyer_data.get('company_name', 'Unknown')}: {str(e)}")
            return 5.0, f"Analysis pending for {buyer_data.get('company_name', 'this company')}"
    
    def score_buyers(self, buyer_df: pd.DataFrame, product_description: str) -> pd.DataFrame:
        """Score all buyers using company-trained AI"""
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_buyers = len(buyer_df)
        
        for index, row in buyer_df.iterrows():
            progress = (index + 1) / total_buyers
            progress_bar.progress(progress)
            status_text.text(f"AI analyzing buyer {index + 1} of {total_buyers}: {row.get('company_name', 'Unknown')}")
            
            buyer_dict = row.to_dict()
            score, reason = self.score_single_buyer(buyer_dict, product_description)
            
            result_row = buyer_dict.copy()
            result_row['score'] = score
            result_row['reason'] = reason
            result_row['scored_at'] = pd.Timestamp.now()
            
            results.append(result_row)
        
        progress_bar.empty()
        status_text.empty()
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False).reset_index(drop=True)
        
        return results_df
    
    def train_model(self, historical_data: pd.DataFrame) -> bool:
        """Train model on historical scoring data for improved accuracy"""
        if not self.is_ready():
            st.error("❌ Base model not loaded. Cannot train.")
            return False
        
        try:
            st.info("Building historical scoring dataset...")
            
            training_texts = []

            for _, row in historical_data.iterrows():
                company_profile = self.create_company_knowledge_text(row)
                
                training_example = f"""Product: {row.get('product_description', 'Unknown product')}
Company: {company_profile}
Score: {row.get('score', 5)}
Reason: {row.get('reason', 'No reason provided')}"""
                
                training_texts.append(training_example)

            st.info(f"Created {len(training_texts)} historical training examples")

            def tokenize_function(examples):
                texts = examples['text'] if isinstance(examples['text'], list) else [examples['text']]
                return self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=320,
                    return_tensors=None
                )

            dataset = Dataset.from_dict({'text': training_texts})
            tokenized_dataset = dataset.map(
                tokenize_function, 
                batched=True,
                remove_columns=dataset.column_names
            )
            
            for param in self.model.parameters():
                param.requires_grad = True

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["c_attn"]
            )

            model = get_peft_model(self.model, peft_config)
            model.train()
            model.enable_input_require_grads()

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            st.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            training_args = TrainingArguments(
                output_dir="./historical_training_results",
                num_train_epochs=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                logging_steps=50,
                save_steps=500,
                eval_strategy="no",
                save_total_limit=1,
                remove_unused_columns=False,
                report_to="none",
                gradient_checkpointing=False,
                fp16=torch.cuda.is_available(),
                dataloader_drop_last=True,
                warmup_steps=50,
                dataloader_num_workers=0,
                optim="adamw_torch"
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
            
            st.info("Training on historical scoring data...")
            with st.spinner("Improving scoring accuracy..."):
                trainer.train()
            
            os.makedirs(self.trained_model_path, exist_ok=True)
            model.save_pretrained(self.trained_model_path)
            self.tokenizer.save_pretrained(self.trained_model_path)
            
            self.model = model
            
            st.success("Historical scoring training completed!")
            return True
            
        except Exception as e:
            st.error(f"❌ Historical training failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'is_company_trained': self.is_trained(),
            'is_ready': self.is_ready(),
            'device': self.device,
            'trained_model_exists': os.path.exists(self.trained_model_path),
            'model_type': 'Company-trained AI' if self.is_trained() else 'Base Model'
        }
    
    def _calculate_enhanced_score(self, buyer_data: Dict, product_description: str) -> float:
        """Enhanced rule-based scoring using company knowledge"""
        score = 5.0  # Base score as float
        
        # Industry analysis
        industry = str(buyer_data.get('industry', '')).lower()
        product_lower = product_description.lower()
        
        # Enhanced industry matching
        industry_matches = {
            'technology': ['software', 'saas', 'platform', 'api', 'cloud', 'data', 'analytics', 'ai', 'automation'],
            'healthcare': ['health', 'medical', 'patient', 'clinical', 'telemedicine', 'emr', 'hipaa'],
            'financial': ['finance', 'payment', 'banking', 'compliance', 'risk', 'fintech', 'trading'],
            'retail': ['ecommerce', 'inventory', 'pos', 'customer', 'shopping', 'omnichannel'],
            'manufacturing': ['automation', 'supply', 'quality', 'erp', 'logistics', 'iot', 'maintenance'],
            'education': ['learning', 'student', 'campus', 'lms', 'education', 'academic'],
            'real estate': ['property', 'lease', 'tenant', 'building', 'facility'],
            'legal': ['legal', 'compliance', 'contract', 'litigation', 'law', 'attorney'],
            'consulting': ['consulting', 'advisory', 'strategy', 'management', 'professional'],
            'marketing': ['marketing', 'advertising', 'campaign', 'social', 'brand', 'crm'],
            'insurance': ['insurance', 'claims', 'underwriting', 'actuarial', 'risk'],
            'construction': ['construction', 'project', 'building', 'contractor', 'architecture']
        }
        
        # Score based on industry-product fit
        for ind, keywords in industry_matches.items():
            if ind in industry:
                matches = sum(1 for keyword in keywords if keyword in product_lower)
                if matches > 0:
                    score += min(3.0, matches * 0.5)  # Cap industry bonus at +3.0
                    break
        
        # Employee count analysis with enhanced logic
        try:
            emp_count = int(str(buyer_data.get('employee_count', '0')).replace(',', ''))
            
            # Product complexity indicators
            is_enterprise_product = any(word in product_lower for word in 
                ['enterprise', 'corporate', 'advanced', 'professional', 'premium'])
            is_simple_product = any(word in product_lower for word in 
                ['simple', 'basic', 'starter', 'small', 'affordable'])
            is_scalable_product = any(word in product_lower for word in 
                ['scalable', 'growth', 'flexible', 'modular'])
            
            if emp_count < 50:  # Small companies
                if is_simple_product or 'startup' in product_lower:
                    score += 2.0
                elif is_enterprise_product:
                    score -= 1.0
                else:
                    score += 1.0
                    
            elif 50 <= emp_count < 500:  # Medium companies
                if is_scalable_product or 'growth' in product_lower:
                    score += 2.0
                elif is_simple_product:
                    score += 1.0
                else:
                    score += 1.0
                    
            elif 500 <= emp_count < 5000:  # Large companies
                if is_enterprise_product:
                    score += 2.0
                elif is_scalable_product:
                    score += 1.0
                elif is_simple_product:
                    score -= 1.0
                    
            else:  # Enterprise (5000+)
                if is_enterprise_product:
                    score += 3.0
                elif is_scalable_product:
                    score += 1.0
                elif is_simple_product:
                    score -= 2.0
                    
        except (ValueError, TypeError):
            pass
        
        # Location-based adjustments
        location = str(buyer_data.get('location', '')).lower()
        country = str(buyer_data.get('country', '')).lower()
        
        # Tech hubs get bonus for tech products
        tech_hubs = ['san francisco', 'silicon valley', 'new york', 'boston', 'seattle', 'austin', 'london', 'toronto']
        if any(hub in location for hub in tech_hubs) and 'technology' in industry:
            score += 1.0
        
        # Developed markets for premium products
        developed_markets = ['united states', 'canada', 'uk', 'germany', 'france', 'australia', 'japan']
        if any(market in country for market in developed_markets) and is_enterprise_product:
            score += 1.0
        
        # Company maturity indicators
        founded_year = buyer_data.get('founded_year')
        if founded_year and pd.notna(founded_year):
            try:
                year = int(float(founded_year))
                company_age = 2025 - year
                
                if company_age < 3:  # Very young startup
                    if 'startup' in product_lower or is_simple_product:
                        score += 1.0
                elif company_age < 10:  # Growth stage
                    if is_scalable_product:
                        score += 1.0
                else:  # Established company
                    if is_enterprise_product:
                        score += 1.0
            except (ValueError, TypeError):
                pass
        
        # Website and digital presence (if available)
        website = str(buyer_data.get('website', '')).lower()
        if website and website != 'nan':
            # Companies with websites are more likely to adopt digital solutions
            if any(word in product_lower for word in ['digital', 'online', 'cloud', 'saas']):
                score += 1.0
        
        # Final adjustments and bounds
        score = max(1.0, min(10.0, score))
        
        return round(score, 1)

    def _generate_detailed_analysis(self, buyer_data: Dict, product_description: str, score: float) -> str:
        """Generate product-focused analysis with bullet points"""
        company_name = buyer_data.get('company_name', 'This company')
        industry = buyer_data.get('industry', 'Unknown industry')  
        employee_count = buyer_data.get('employee_count', 'Unknown size')
        
        # Extract key product benefits from description
        product_lower = product_description.lower()
        
        # Build bullet-point analysis focused on product fit
        bullets = []
        
        # Industry-specific benefits
        if 'financial' in industry.lower():
            if any(word in product_lower for word in ['database', 'real-time', 'analytics', 'performance']):
                bullets.append("• **Regulatory Compliance**: Fast data retrieval supports real-time risk monitoring and audit trail requirements")
                bullets.append("• **Transaction Processing**: Low-latency database critical for high-frequency trading and payment processing")
                bullets.append("• **Customer Analytics**: Real-time personalization enables better client recommendations and fraud detection")
        
        elif 'technology' in industry.lower():
            if any(word in product_lower for word in ['database', 'api', 'cloud', 'platform']):
                bullets.append("• **Scalable Infrastructure**: Tech companies need high-performance systems that grow with user base")
                bullets.append("• **Developer Productivity**: Fast queries and reliable APIs reduce development bottlenecks")
                bullets.append("• **Customer Experience**: Sub-millisecond response times directly impact user satisfaction")
        
        elif 'healthcare' in industry.lower():
            if any(word in product_lower for word in ['database', 'real-time', 'analytics']):
                bullets.append("• **Patient Data Management**: Real-time access to medical records improves care delivery")
                bullets.append("• **HIPAA Compliance**: Secure, fast data systems meet healthcare regulatory requirements")
                bullets.append("• **Clinical Decision Support**: Instant data retrieval enables better diagnostic decisions")
        
        elif 'retail' in industry.lower() or 'ecommerce' in industry.lower():
            if any(word in product_lower for word in ['database', 'real-time', 'personalization', 'analytics']):
                bullets.append("• **Inventory Management**: Real-time data prevents stockouts and overstock situations")
                bullets.append("• **Customer Personalization**: Fast recommendation engines increase conversion rates")
                bullets.append("• **Fraud Prevention**: Real-time transaction monitoring reduces payment fraud")
        
        # Size-specific benefits  
        try:
            emp_count = int(str(employee_count).replace(',', ''))
            if 50 <= emp_count <= 500:
                bullets.append("• **Scale Readiness**: Mid-size team needs infrastructure that grows with expanding data requirements")
                bullets.append("• **Cost Efficiency**: Right-sized solution avoids enterprise overhead while providing performance")
            elif emp_count > 500:
                bullets.append("• **Enterprise Performance**: Large organization requires sub-millisecond response times for critical operations")
                bullets.append("• **Infrastructure Optimization**: Existing systems likely struggling with query volume and speed requirements")
            elif emp_count < 50:
                bullets.append("• **Startup Agility**: Small team needs simple, powerful tools that don't require dedicated DevOps")
                bullets.append("• **Competitive Edge**: Fast data access helps startups compete with larger companies")
        except:
            pass
        
        # Product-specific pain points this solves
        if 'database' in product_lower and 'real-time' in product_lower:
            bullets.append("• **Query Performance Issues**: Current database likely causing delays in data-driven decision making")
            bullets.append("• **Scalability Bottlenecks**: Growing data volume overwhelming existing infrastructure capacity")
        
        if 'ai' in product_lower or 'ml' in product_lower:
            bullets.append("• **AI/ML Infrastructure**: Feature stores and model inference require ultra-fast data access")
            bullets.append("• **Competitive Advantage**: Real-time personalization directly impacts revenue and customer satisfaction")
        
        if 'analytics' in product_lower:
            bullets.append("• **Business Intelligence**: Fast analytics enable real-time business decisions and competitive responses")
            bullets.append("• **Data-Driven Culture**: Quick insights help teams make better decisions faster")
        
        # Add general business impact based on score
        if score >= 8.0:
            bullets.append("• **Immediate ROI**: Strong technical fit suggests quick implementation and measurable performance gains")
            bullets.append("• **Strategic Priority**: Core business functions directly benefit from this technology upgrade")
        elif score >= 6.0:
            bullets.append("• **Growth Enabler**: Solution addresses scaling challenges as business expands")
            bullets.append("• **Operational Efficiency**: Performance improvements reduce manual workarounds and system delays")
        elif score >= 4.0:
            bullets.append("• **Future Consideration**: May become relevant as company grows or technology needs evolve")
            bullets.append("• **Niche Applications**: Specific use cases could benefit from this technology")
        
        # If no specific bullets, add generic ones
        if not bullets:
            bullets.append("• **Performance Improvement**: Database optimization typically yields measurable efficiency gains")
            bullets.append("• **Future-Proofing**: Scalable infrastructure supports business growth and data expansion")
        
        return "\n".join(bullets)