"""
AI Model for Buyer Scoring
Handles LLaMA/GPT integration and fine-tuning
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
    pipeline
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
    """AI model for scoring buyer likelihood"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trained_model_path = "models/trained_buyer_model"
        self.model_name = os.getenv('MODEL_NAME', 'microsoft/DialoGPT-medium')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model
        self.load_model()
    
    def load_model(self):
        """Load the AI model (base or fine-tuned)"""
        try:
            # Check if we have a fine-tuned model
            if os.path.exists(self.trained_model_path):
                st.info("🎓 Loading fine-tuned model...")
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
        """Check if using a fine-tuned model"""
        return os.path.exists(self.trained_model_path)
    
    def create_scoring_prompt(self, buyer_data: Dict, product_description: str) -> str:
        """Create a structured prompt for the AI model"""
        
        prompt = f"""You are an expert B2B sales analyst. Score this potential buyer for the given product.

PRODUCT: {product_description}

BUYER INFORMATION:
Company: {buyer_data.get('company_name', 'Unknown')}
Industry: {buyer_data.get('industry', 'Unknown')}
Size: {buyer_data.get('company_size', 'Unknown')}
Revenue: {buyer_data.get('revenue', 'Unknown')}
Location: {buyer_data.get('location', 'Unknown')}
Contact: {buyer_data.get('contact_name', 'N/A')} - {buyer_data.get('contact_title', 'N/A')}

SCORING CRITERIA:
- Industry fit (does this industry typically need this product?)
- Company size match (is the company the right size for this product?)
- Revenue capacity (can they afford this product?)
- Growth stage (are they in a stage where they'd buy this?)
- Geographic considerations

INSTRUCTIONS:
Provide a score from 1-10 where:
- 9-10: Perfect fit, high likelihood to buy
- 7-8: Good fit, strong potential
- 5-6: Moderate fit, some potential
- 3-4: Poor fit, low potential
- 1-2: Very poor fit, unlikely to buy

Respond in this exact format:
Score: [number 1-10]
Reason: [brief explanation of why this score makes sense]

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
                    # Extract number from line
                    numbers = re.findall(r'\b([1-9]|10)\b', line)
                    if numbers:
                        score = int(numbers[0])
                
                # Look for reason
                elif line.startswith('Reason:'):
                    reason = line.replace('Reason:', '').strip()
                elif len(line) > 10 and not line.startswith('Score:'):
                    # If it's a substantial line and not the score line, treat as reason
                    reason = line
            
            # Validate score range
            score = max(1, min(10, score))
            
            return score, reason
            
        except Exception as e:
            st.warning(f"⚠️ Error parsing response: {str(e)}")
            return 5, "Error in AI analysis - using default score"
    
    def score_single_buyer(self, buyer_data: Dict, product_description: str) -> Tuple[int, str]:
        """Score a single buyer using the AI model"""
        if not self.is_ready():
            return 5, "Model not ready"
        
        try:
            # Create prompt
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
                    max_new_tokens=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
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
        """Score all buyers in the DataFrame"""
        results = []
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_buyers = len(buyer_df)
        
        for index, row in buyer_df.iterrows():
            # Update progress
            progress = (index + 1) / total_buyers
            progress_bar.progress(progress)
            status_text.text(f"Scoring buyer {index + 1} of {total_buyers}: {row.get('company_name', 'Unknown')}")
            
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
    
    def prepare_training_data(self, historical_df: pd.DataFrame) -> Tuple[List[str], List[int]]:
        """Prepare historical data for model training"""
        training_texts = []
        labels = []
        
        for _, row in historical_df.iterrows():
            # Create training example
            buyer_info = {
                'company_name': row.get('company_name', ''),
                'industry': row.get('industry', ''),
                'company_size': row.get('company_size', ''),
                'revenue': row.get('revenue', ''),
                'location': row.get('location', ''),
                'contact_name': row.get('contact_name', ''),
                'contact_title': row.get('contact_title', '')
            }
            
            prompt = self.create_scoring_prompt(buyer_info, row.get('product_description', ''))
            
            # Add the correct answer
            training_text = f"{prompt} {row.get('score', 5)}\nReason: {row.get('reason', 'Training example')}"
            
            training_texts.append(training_text)
            labels.append(int(row.get('score', 5)))
        
        return training_texts, labels
    
    def train_model(self, historical_data: pd.DataFrame) -> bool:
        """Fine-tune the model on historical scoring data"""
        if not self.is_ready():
            st.error("❌ Base model not loaded. Cannot train.")
            return False
        
        try:
            st.info("🔄 Preparing training data...")
            
            # Prepare training data
            texts, scores = self.prepare_training_data(historical_data)
            
            if len(texts) < 5:
                st.warning("⚠️ Need at least 5 training examples. Current count: {len(texts)}")
                return False
            
            st.info(f"📚 Training on {len(texts)} examples...")
            
            # Create dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding=True,
                    max_length=1024
                )
            
            dataset = Dataset.from_dict({
                'text': texts,
                'labels': scores
            })
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Configure LoRA for efficient fine-tuning
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"] if "llama" in self.model_name.lower() else ["c_attn"]
            )
            
            # Apply LoRA to model
            model = get_peft_model(self.model, peft_config)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=5e-5,
                logging_steps=10,
                save_steps=100,
                evaluation_strategy="no",
                save_total_limit=2,
                remove_unused_columns=False,
                report_to="none",  # Disable wandb logging
                gradient_checkpointing=True,
                fp16=torch.cuda.is_available()
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=self.tokenizer,
            )
            
            st.info("🚀 Starting fine-tuning process...")
            
            # Train the model
            trainer.train()
            
            # Save the fine-tuned model
            os.makedirs(self.trained_model_path, exist_ok=True)
            model.save_pretrained(self.trained_model_path)
            self.tokenizer.save_pretrained(self.trained_model_path)
            
            # Update the current model instance
            self.model = model
            
            st.success("✅ Model fine-tuning completed!")
            return True
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained(),
            'is_ready': self.is_ready(),
            'device': self.device,
            'trained_model_exists': os.path.exists(self.trained_model_path)
        }