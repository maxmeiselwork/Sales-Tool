"""
AI Model for Buyer Scoring - Company Knowledge Based
Trains on your company database to understand company characteristics
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

load_dotenv()

class BuyerScoringModel:
    """AI model trained on company data for intelligent buyer scoring"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trained_model_path = "models/company_trained_model"
        self.model_name = os.getenv('MODEL_NAME', 'microsoft/DialoGPT-medium')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model
        self.load_model()
    
    def load_model(self):
        """Load the AI model (base or company-trained)"""
        try:
            # Check if we have a company-trained model
            if os.path.exists(self.trained_model_path):
                st.info("🏢 Loading company-trained model...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.trained_model_path)
                self.model = AutoModelForCausalLM.from_pretrained(self.trained_model_path)
            else:
                st.info("📚 Loading base model from Hugging Face...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            # Configure tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            st.success("✅ Model loaded successfully")
            
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
        """Create knowledge text about a company for training"""
        parts = []
        
        company_name = company_row.get('company_name', 'Unknown Company')
        parts.append(f"Company: {company_name}")
        
        if company_row.get('industry'):
            parts.append(f"Industry: {company_row['industry']}")
        
        if company_row.get('employee_count') and pd.notna(company_row['employee_count']):
            emp_count = int(company_row['employee_count'])
            parts.append(f"Employees: {emp_count:,}")
            
            # Add size category
            if emp_count < 50:
                parts.append("Size: Small startup/company")
            elif emp_count < 500:
                parts.append("Size: Medium company")
            elif emp_count < 5000:
                parts.append("Size: Large company")
            else:
                parts.append("Size: Enterprise corporation")
        
        if company_row.get('location'):
            parts.append(f"Location: {company_row['location']}")
        
        if company_row.get('country'):
            parts.append(f"Country: {company_row['country']}")
        
        if company_row.get('founded_year') and pd.notna(company_row['founded_year']):
            year = int(company_row['founded_year'])
            age = 2024 - year
            parts.append(f"Founded: {year} ({age} years old)")
            
            # Add maturity insight
            if age < 5:
                parts.append("Maturity: Young startup")
            elif age < 15:
                parts.append("Maturity: Growing company")
            else:
                parts.append("Maturity: Established company")
        
        if company_row.get('website'):
            parts.append(f"Website: {company_row['website']}")
        
        return " | ".join(parts)
    
    def train_on_company_data(self, company_df: pd.DataFrame) -> bool:
        """Train the model to understand company characteristics"""
        if not self.is_ready():
            st.error("❌ Base model not loaded. Cannot train.")
            return False
        
        try:
            st.info("🏗️ Building company knowledge dataset...")
            
            # Create training texts from company data
            training_texts = []
            
            # Sample companies if dataset is too large
            if len(company_df) > 50000:
                st.info(f"📊 Sampling 50,000 companies from {len(company_df):,} total companies...")
                sample_df = company_df.sample(n=50000, random_state=42)
            else:
                sample_df = company_df
            
            # Create knowledge texts
            progress_bar = st.progress(0)
            for idx, (_, row) in enumerate(sample_df.iterrows()):
                if idx % 1000 == 0:
                    progress_bar.progress(idx / len(sample_df))
                
                knowledge_text = self.create_company_knowledge_text(row)
                
                # Create training example with business context
                training_example = f"""Company Profile: {knowledge_text}

Business Analysis: This company operates in the {row.get('industry', 'unknown')} sector. """
                
                # Add business insights based on company characteristics
                if row.get('employee_count') and pd.notna(row['employee_count']):
                    emp_count = int(row['employee_count'])
                    if emp_count < 50:
                        training_example += "As a small company, they likely need cost-effective solutions, have limited IT resources, and value simple, easy-to-implement tools. "
                    elif emp_count < 500:
                        training_example += "As a medium-sized company, they likely have dedicated departments, need scalable solutions, and are growing their operations. "
                    elif emp_count < 5000:
                        training_example += "As a large company, they likely have complex needs, established processes, and require enterprise-grade solutions. "
                    else:
                        training_example += "As an enterprise corporation, they likely have sophisticated requirements, compliance needs, and substantial budgets for technology. "
                
                # Add industry-specific insights
                industry = str(row.get('industry', '')).lower()
                if 'technology' in industry or 'software' in industry:
                    training_example += "Technology companies typically need development tools, cloud services, and productivity software. "
                elif 'healthcare' in industry or 'medical' in industry:
                    training_example += "Healthcare companies typically need compliance solutions, patient management systems, and secure data handling. "
                elif 'financial' in industry or 'banking' in industry:
                    training_example += "Financial companies typically need security solutions, compliance tools, and data analytics platforms. "
                elif 'retail' in industry:
                    training_example += "Retail companies typically need e-commerce solutions, inventory management, and customer analytics. "
                elif 'manufacturing' in industry:
                    training_example += "Manufacturing companies typically need supply chain solutions, quality control systems, and operational efficiency tools. "
                
                training_texts.append(training_example)
            
            progress_bar.progress(1.0)
            st.info(f"📚 Created {len(training_texts):,} training examples")
            
            # Create dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding=True,
                    max_length=512  # Shorter for efficiency
                )
            
            dataset = Dataset.from_dict({'text': training_texts})
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Configure LoRA for efficient fine-tuning
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,  # Increased for better learning
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] if "llama" in self.model_name.lower() else ["c_attn"]
            )
            
            # Apply LoRA to model
            model = get_peft_model(self.model, peft_config)
            
            # Data collator for language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # Causal LM, not masked LM
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir="./company_training_results",
                num_train_epochs=2,  # Fewer epochs for large dataset
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                learning_rate=1e-4,
                logging_steps=50,
                save_steps=500,
                evaluation_strategy="no",
                save_total_limit=2,
                remove_unused_columns=False,
                report_to="none",
                gradient_checkpointing=True,
                fp16=torch.cuda.is_available(),
                dataloader_drop_last=True,
                warmup_steps=100
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
            
            st.info("🚀 Starting company knowledge training...")
            with st.spinner("Training model on company data..."):
                trainer.train()
            
            # Save the company-trained model
            os.makedirs(self.trained_model_path, exist_ok=True)
            model.save_pretrained(self.trained_model_path)
            self.tokenizer.save_pretrained(self.trained_model_path)
            
            # Update the current model instance
            self.model = model
            
            st.success("✅ Company knowledge training completed!")
            return True
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
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
    
    def extract_score_and_reason(self, response: str) -> Tuple[int, str]:
        """Extract score and reason from model response"""
        try:
            lines = response.strip().split('\n')
            score = 5  # default
            reason = "AI analysis completed"
            
            for line in lines:
                line = line.strip()
                
                # Look for score
                if line.startswith('Score:') or 'score' in line.lower():
                    numbers = re.findall(r'\b([1-9]|10)\b', line)
                    if numbers:
                        score = int(numbers[0])
                
                # Look for reason
                elif line.startswith('Reason:'):
                    reason = line.replace('Reason:', '').strip()
                elif len(line) > 20 and not line.startswith('Score:') and not line.startswith('TASK:'):
                    reason = line
            
            # Validate score range
            score = max(1, min(10, score))
            
            return score, reason
            
        except Exception as e:
            st.warning(f"⚠️ Error parsing response: {str(e)}")
            return 5, "Error in AI analysis - using default score"
    
    def score_single_buyer(self, buyer_data: Dict, product_description: str) -> Tuple[int, str]:
        """Score a single buyer using company-trained AI"""
        if not self.is_ready():
            return 5, "Model not ready"
        
        try:
            # Create intelligent prompt
            prompt = self.create_scoring_prompt(buyer_data, product_description)
            
            # Tokenize
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=200,  # More tokens for detailed reasoning
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            # Extract score and reason
            score, reason = self.extract_score_and_reason(response)
            
            return score, reason
            
        except Exception as e:
            st.error(f"❌ Scoring failed for {buyer_data.get('company_name', 'Unknown')}: {str(e)}")
            return 5, f"Scoring error: {str(e)}"
    
    def score_buyers(self, buyer_df: pd.DataFrame, product_description: str) -> pd.DataFrame:
        """Score all buyers using company-trained AI"""
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_buyers = len(buyer_df)
        
        for index, row in buyer_df.iterrows():
            # Update progress
            progress = (index + 1) / total_buyers
            progress_bar.progress(progress)
            status_text.text(f"🤖 AI analyzing buyer {index + 1} of {total_buyers}: {row.get('company_name', 'Unknown')}")
            
            # Score this buyer
            buyer_dict = row.to_dict()
            score, reason = self.score_single_buyer(buyer_dict, product_description)
            
            # Add score and reason to the row data
            result_row = buyer_dict.copy()
            result_row['score'] = score
            result_row['reason'] = reason
            result_row['scored_at'] = pd.Timestamp.now()
            
            results.append(result_row)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Convert to DataFrame and sort by score
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False).reset_index(drop=True)
        
        return results_df
    
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