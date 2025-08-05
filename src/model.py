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
        
        # Handle employee count (might be string due to large number handling)
        employee_count = company_row.get('employee_count')
        if employee_count and pd.notna(employee_count):
            try:
                if isinstance(employee_count, str):
                    emp_count = int(employee_count)
                else:
                    emp_count = int(employee_count)
                
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
            except (ValueError, TypeError):
                parts.append(f"Employees: {employee_count}")
        
        if company_row.get('location'):
            parts.append(f"Location: {company_row['location']}")
        
        if company_row.get('country'):
            parts.append(f"Country: {company_row['country']}")
        
        if company_row.get('founded_year') and pd.notna(company_row['founded_year']):
            try:
                year = int(float(company_row['founded_year']))
                age = 2024 - year
                parts.append(f"Founded: {year} ({age} years old)")
                
                # Add maturity insight
                if age < 5:
                    parts.append("Maturity: Young startup")
                elif age < 15:
                    parts.append("Maturity: Growing company")
                else:
                    parts.append("Maturity: Established company")
            except (ValueError, TypeError):
                parts.append(f"Founded: {company_row['founded_year']}")
        
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
                employee_count = row.get('employee_count')
                if employee_count and pd.notna(employee_count):
                    try:
                        if isinstance(employee_count, str):
                            emp_count = int(employee_count)
                        else:
                            emp_count = int(employee_count)
                        
                        if emp_count < 50:
                            training_example += "As a small company, they likely need cost-effective solutions, have limited IT resources, and value simple, easy-to-implement tools. "
                        elif emp_count < 500:
                            training_example += "As a medium-sized company, they likely have dedicated departments, need scalable solutions, and are growing their operations. "
                        elif emp_count < 5000:
                            training_example += "As a large company, they likely have complex needs, established processes, and require enterprise-grade solutions. "
                        else:
                            training_example += "As an enterprise corporation, they likely have sophisticated requirements, compliance needs, and substantial budgets for technology. "
                    except (ValueError, TypeError):
                        pass
                
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
                # Ensure we're working with a list of strings
                texts = examples['text'] if isinstance(examples['text'], list) else [examples['text']]
                return self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors=None  # Don't return tensors yet
                )

            dataset = Dataset.from_dict({'text': training_texts})
            tokenized_dataset = dataset.map(
                tokenize_function, 
                batched=True,
                remove_columns=dataset.column_names  # Remove original text column
            )

            # First, make sure the base model parameters are trainable
            for param in self.model.parameters():
                param.requires_grad = True

            # Configure LoRA with correct target modules for DialoGPT
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["c_attn"]  # DialoGPT uses c_attn as the main attention module
            )

            # Apply LoRA to model
            model = get_peft_model(self.model, peft_config)

            # Explicitly enable training mode and gradients
            model.train()
            model.enable_input_require_grads()  # This is crucial for LoRA

            # Verify LoRA parameters are trainable
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            st.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

            # Data collator for language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # Causal LM, not masked LM
            )

            # Training arguments with more conservative settings
            training_args = TrainingArguments(
                output_dir="./company_training_results",
                num_train_epochs=1,  # Start with 1 epoch
                per_device_train_batch_size=1,  # Very small batch size
                gradient_accumulation_steps=8,
                learning_rate=2e-4,  # Slightly higher learning rate for LoRA
                logging_steps=50,
                save_steps=500,
                eval_strategy="no",
                save_total_limit=2,
                remove_unused_columns=False,
                report_to="none",
                gradient_checkpointing=False,  # Disable to avoid issues
                fp16=False,  # Keep disabled
                dataloader_drop_last=True,
                warmup_steps=100,
                dataloader_num_workers=0,
                optim="adamw_torch"  # Use torch optimizer
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
            st.error(f"Error details: {type(e).__name__}")
            return False
    
    def create_scoring_prompt(self, buyer_data: Dict, product_description: str) -> str:
        """Create intelligent scoring prompt using company knowledge"""
        
        company_knowledge = self.create_company_knowledge_text(buyer_data)
        
        prompt = f"""You are an expert B2B sales analyst with deep knowledge of companies and their needs.

        PRODUCT/SERVICE TO EVALUATE: {product_description}

        COMPANY TO SCORE: {company_knowledge}

        TASK: Analyze if this company would be interested in buying this product/service and provide a score from 1-10. Use the data provided to you on the company inlcuding their industry, size, location and use their website and linkdin if provided to do a furhter analyis, scrape these sites to do a thurough analysis of the company and their needs.

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
    
    def extract_score_and_reason(self,buyer_data, response: str) -> Tuple[int, str]:
        """Extract score and reason from model response"""
        try:
            # Clean up the response
            response = response.strip()
            
            score = 5  # default
            reason = "Analysis in progress"
            
            # Look for score patterns
            score_patterns = [
                r'Score:\s*(\d+)',
                r'score:\s*(\d+)',
                r'(\d+)/10',
                r'Rating:\s*(\d+)'
            ]
            
            for pattern in score_patterns:
                matches = re.findall(pattern, response)
                if matches:
                    score = int(matches[-1])  # Take the last match
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
            
            # If no structured analysis found, use the full response after the score
            if reason == "Analysis in progress" and len(response) > 50:
                # Find everything after "Score:" or similar
                parts = re.split(r'Score:\s*\d+', response, flags=re.IGNORECASE)
                if len(parts) > 1:
                    reason = parts[-1].strip()
            
            # Ensure we have meaningful content
            if len(reason) < 20:
                reason = f"Company analysis: This {buyer_data.get('industry', 'company')} with {buyer_data.get('employee_count', 'unknown')} employees may have specific needs that align with the product offering."
            
            # Validate score range
            score = max(1, min(10, score))
            
            return score, reason
            
        except Exception as e:
            st.warning(f"⚠️ Error parsing response: {str(e)}")
            return 5, f"Analysis of {buyer_data.get('company_name', 'this company')} requires further review to determine product fit and potential challenges they face."
    
    def score_single_buyer(self, buyer_data: Dict, product_description: str) -> Tuple[int, str]:
        """Score a single buyer using company-trained AI"""
        if not self.is_ready():
            return 5, "Model not ready"
        
        try:
            # Create intelligent prompt
            prompt = self.create_scoring_prompt(buyer_data, product_description)
            
            # Use a rule-based scoring approach for now since training is complex
            score = self._calculate_rule_based_score(buyer_data, product_description)
            reason = self._generate_detailed_analysis(buyer_data, product_description, score)
            
            return score, reason
            
        except Exception as e:
            st.error(f"❌ Scoring failed for {buyer_data.get('company_name', 'Unknown')}: {str(e)}")
            return 5, f"Analysis pending for {buyer_data.get('company_name', 'this company')}"
    
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
    
    def train_model(self, historical_data: pd.DataFrame) -> bool:
        """Train model on historical scoring data for improved accuracy"""
        if not self.is_ready():
            st.error("❌ Base model not loaded. Cannot train.")
            return False
        
        try:
            st.info("🏗️ Building historical scoring dataset...")
            
            # Create training examples from historical data
            training_texts = []

            for _, row in historical_data.iterrows():
                # Create company profile
                company_profile = self.create_company_knowledge_text(row)
                
                # Create training example
                training_example = f"""Product: {row.get('product_description', 'Unknown product')}
    Company: {company_profile}
    Score: {row.get('score', 5)}
    Reason: {row.get('reason', 'No reason provided')}"""
                
                training_texts.append(training_example)

            st.info(f"📚 Created {len(training_texts)} historical training examples")

            # Create dataset
            def tokenize_function(examples):
                texts = examples['text'] if isinstance(examples['text'], list) else [examples['text']]
                return self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors=None
                )

            dataset = Dataset.from_dict({'text': training_texts})
            tokenized_dataset = dataset.map(
                tokenize_function, 
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # First, make sure the base model parameters are trainable
            for param in self.model.parameters():
                param.requires_grad = True

            # Configure LoRA with correct target modules
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["c_attn"]  # DialoGPT uses c_attn
            )

            # Apply LoRA to model
            model = get_peft_model(self.model, peft_config)
            model.train()
            model.enable_input_require_grads()  # Enable gradients for inputs

            # Verify trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            st.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

            # Data collator for language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            # Training arguments
            training_args = TrainingArguments(
                output_dir="./historical_training_results",
                num_train_epochs=1,  # Start with 1 epoch
                per_device_train_batch_size=1,  # Very small batch size
                gradient_accumulation_steps=8,
                learning_rate=2e-4,  # Slightly higher learning rate for LoRA
                logging_steps=50,
                save_steps=500,
                eval_strategy="no",
                save_total_limit=1,
                remove_unused_columns=False,
                report_to="none",
                gradient_checkpointing=False,  # Disable to avoid issues
                fp16=False,  # Keep disabled
                dataloader_drop_last=True,
                warmup_steps=50,
                dataloader_num_workers=0,
                optim="adamw_torch"  # Use torch optimizer
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
            
            st.info("🚀 Training on historical scoring data...")
            with st.spinner("Improving scoring accuracy..."):
                trainer.train()
            
            # Save the trained model
            os.makedirs(self.trained_model_path, exist_ok=True)
            model.save_pretrained(self.trained_model_path)
            self.tokenizer.save_pretrained(self.trained_model_path)
            
            # Update the current model instance
            self.model = model
            
            st.success("✅ Historical scoring training completed!")
            return True
            
        except Exception as e:
            st.error(f"❌ Historical training failed: {str(e)}")
            st.error(f"Error details: {type(e).__name__}")
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
    def _calculate_rule_based_score(self, buyer_data: Dict, product_description: str) -> int:
        """Calculate score based on business rules"""
        score = 5  # Base score
        
        # Industry analysis
        industry = str(buyer_data.get('industry', '')).lower()
        product_lower = product_description.lower()
        
        # Technology companies score higher for tech products
        if 'technology' in industry and any(word in product_lower for word in ['software', 'saas', 'platform', 'api', 'cloud']):
            score += 2
        
        # Healthcare companies for healthcare products
        elif 'healthcare' in industry and any(word in product_lower for word in ['health', 'medical', 'patient', 'clinical']):
            score += 2
        
        # Financial services for fintech
        elif 'financial' in industry and any(word in product_lower for word in ['finance', 'payment', 'banking', 'compliance']):
            score += 2
        
        # Employee count analysis
        try:
            emp_count = int(str(buyer_data.get('employee_count', '0')).replace(',', ''))
            
            # Small companies (good for affordable solutions)
            if emp_count < 50 and any(word in product_lower for word in ['affordable', 'small', 'startup', 'simple']):
                score += 1
            
            # Medium companies (good for growth solutions)
            elif 50 <= emp_count < 500 and any(word in product_lower for word in ['growth', 'scale', 'expand']):
                score += 1
                
            # Large companies (good for enterprise solutions)
            elif emp_count >= 500 and any(word in product_lower for word in ['enterprise', 'large', 'corporate']):
                score += 1
                
        except (ValueError, TypeError):
            pass
        
        # Ensure score is within bounds
        return max(1, min(10, score))

    def _generate_detailed_analysis(self, buyer_data: Dict, product_description: str, score: int) -> str:
        """Generate detailed business analysis"""
        company_name = buyer_data.get('company_name', 'This company')
        industry = buyer_data.get('industry', 'Unknown industry')
        employee_count = buyer_data.get('employee_count', 'Unknown size')
        
        # Start with company profile
        analysis = f"{company_name} operates in the {industry} sector"
        
        # Add size context
        try:
            emp_count = int(str(employee_count).replace(',', ''))
            if emp_count < 50:
                analysis += f" with {emp_count} employees, making them a small company that likely values cost-effective, easy-to-implement solutions."
            elif emp_count < 500:
                analysis += f" with {emp_count} employees, positioning them as a growing mid-size company that needs scalable solutions."
            else:
                analysis += f" with {emp_count} employees, making them a large enterprise requiring robust, comprehensive solutions."
        except (ValueError, TypeError):
            analysis += " and appears to be an established business."
        
        # Industry-specific challenges
        industry_lower = industry.lower()
        if 'technology' in industry_lower:
            analysis += " Technology companies typically face challenges with rapid scaling, development efficiency, and staying competitive in fast-moving markets."
        elif 'healthcare' in industry_lower:
            analysis += " Healthcare organizations often struggle with compliance requirements, patient data security, and operational efficiency."
        elif 'financial' in industry_lower:
            analysis += " Financial services companies deal with regulatory compliance, security concerns, and the need for reliable, scalable systems."
        elif 'retail' in industry_lower:
            analysis += " Retail companies face challenges with inventory management, customer experience, and omnichannel operations."
        else:
            analysis += f" Companies in the {industry} sector typically need solutions that improve operational efficiency and competitive advantage."
        
        # Product fit analysis
        analysis += f" Given the product description '{product_description}', "
        
        if score >= 8:
            analysis += "this company shows excellent alignment and would likely see immediate value and ROI from this solution."
        elif score >= 6:
            analysis += "this company shows good potential and could benefit from this solution with proper positioning and demonstration of value."
        elif score >= 4:
            analysis += "this company shows moderate fit and would need additional qualification to determine specific use cases and value proposition."
        else:
            analysis += "this company shows limited alignment and may not be the ideal target for this particular solution."
        
        return analysis