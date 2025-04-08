"""
Translation module for AutoWealthTranslate.

This module is responsible for translating document components using LLMs.
"""

import os
import time
import logging
from typing import List, Dict, Any, Union
import openai
import tiktoken
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from auto_wealth_translate.utils.logger import get_logger
from auto_wealth_translate.core.document_processor import (
    DocumentComponent, TextComponent, TableComponent, 
    ImageComponent, ChartComponent
)

logger = get_logger(__name__)

class TranslationService:
    """
    Service for translating document components.
    """
    
    def __init__(self, source_lang: str = "en", target_lang: str = "zh", model: str = "gpt-4"):
        """
        Initialize the translation service.
        
        Args:
            source_lang: Source language code (e.g., 'en', 'fr')
            target_lang: Target language code (e.g., 'zh', 'fr')
            model: Model to use for translation (e.g., 'gpt-4', 'grok-2')
            
        Note:
            To use the OpenAI API for translation, you need to set the OPENAI_API_KEY
            environment variable. 
            
            To use the xAI Grok API, you need to set XAI_API_KEY environment variable.
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model = model
        
        # Max tokens for context length (model dependent)
        if "gpt-3.5" in model:
            self.max_tokens = 4000 
        elif "grok" in model:
            self.max_tokens = 8000  # Assuming Grok has similar context length to GPT-4
        else:
            self.max_tokens = 8000  # Default for GPT-4 and others
        
        # Language names for reference
        self.language_names = {
            "en": "English",
            "zh": "Chinese",
            "fr": "French",
            "es": "Spanish",
            "de": "German",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
            "ar": "Arabic",
            "it": "Italian",
            "pt": "Portuguese"
        }
        
        logger.info(f"Setting up translation from {self.language_names.get(source_lang, source_lang)} to {self.language_names.get(target_lang, target_lang)}")
        
        # Initialize OpenAI API if using GPT models
        if model.startswith("gpt"):
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning("OpenAI API key not found in environment variables. Please set the OPENAI_API_KEY environment variable to use translation functionality.")
            else:
                logger.info(f"Initialized OpenAI {model} model")
        
        # Initialize xAI API if using Grok models
        elif model.startswith("grok"):
            self.api_key = os.environ.get("XAI_API_KEY")
            if not self.api_key:
                logger.warning("xAI API key not found in environment variables. Please set the XAI_API_KEY environment variable to use Grok translation functionality.")
            else:
                logger.info(f"Initialized xAI {model} model")
                
        # Create tokenizer for token counting
        try:
            if model.startswith("gpt"):
                self.tokenizer = tiktoken.encoding_for_model(model)
            elif model.startswith("grok"):
                # Use a similar tokenizer to GPT models since we don't have a specific one for Grok
                self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            else:
                self.tokenizer = None
        except:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            
    def _count_tokens(self, text):
        """Count the number of tokens in a text string."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Rough estimate if tokenizer not available
        return len(text.split()) * 1.5
    
    def translate(self, components: List[DocumentComponent]) -> List[DocumentComponent]:
        """
        Translate all components of a document.
        
        Args:
            components: List of document components
            
        Returns:
            List of translated document components
        """
        source_lang_name = self.language_names.get(self.source_lang, self.source_lang)
        target_lang_name = self.language_names.get(self.target_lang, self.target_lang)
        
        logger.info(f"Translating document from {source_lang_name} to {target_lang_name} using {self.model}")
        
        # Check if API key is available when using GPT models
        if self.model.startswith("gpt") and not self.api_key:
            logger.error("OpenAI API key not provided. Translation will return original text.")
            logger.error("Please set OPENAI_API_KEY environment variable or provide it in the application.")
            
            # Return original components with a warning
            for comp in components:
                if isinstance(comp, TextComponent):
                    comp.text = f"[API KEY MISSING] {comp.text}"
            return components
            
        # Extract financial terms for consistent translation
        financial_terms = self._extract_financial_terms(components)
        if financial_terms:
            logger.info(f"Extracted {len(financial_terms)} financial terms for consistent translation: {', '.join(financial_terms[:5])}{'...' if len(financial_terms) > 5 else ''}")
        
        # Initialize translated components
        translated_components = []
        
        # Use ThreadPoolExecutor for parallel translation of text components
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit text components for translation
            future_to_component = {}
            
            # Count of components by type for logging
            component_types = {'text': 0, 'table': 0, 'image': 0, 'chart': 0, 'other': 0}
            
            # Process different component types
            for component in components:
                if isinstance(component, TextComponent):
                    component_types['text'] += 1
                    # Submit text components for translation
                    future = executor.submit(
                        self._translate_text_component, 
                        component, 
                        financial_terms
                    )
                    future_to_component[future] = component
                    
                elif isinstance(component, TableComponent):
                    component_types['table'] += 1
                    # Submit table components for translation
                    future = executor.submit(
                        self._translate_table_component, 
                        component,
                        financial_terms
                    )
                    future_to_component[future] = component
                    
                elif isinstance(component, ImageComponent):
                    component_types['image'] += 1
                    translated_components.append(component)
                elif isinstance(component, ChartComponent):
                    component_types['chart'] += 1
                    translated_components.append(component)
                else:
                    component_types['other'] += 1
                    # For non-translatable components, just copy them
                    translated_components.append(component)
            
            # Log component counts
            logger.info(f"Document contains: {component_types['text']} text components, {component_types['table']} tables, " +
                        f"{component_types['image']} images, {component_types['chart']} charts, {component_types['other']} other components")
            
            # Process completed translations
            total_futures = len(future_to_component)
            
            if total_futures > 0:
                logger.info(f"Translating {total_futures} components...")
                
                completed = 0
                successful = 0
                failed = 0
                
                for future in as_completed(future_to_component):
                    try:
                        translated_component = future.result()
                        translated_components.append(translated_component)
                        successful += 1
                        
                    except Exception as exc:
                        original_component = future_to_component[future]
                        logger.error(f"Error translating component {original_component.component_id}: {str(exc)}")
                        # Fall back to the original component in case of error
                        translated_components.append(original_component)
                        failed += 1
                    
                    # Log progress
                    completed += 1
                    if completed % 10 == 0 or completed == total_futures:
                        logger.info(f"Translation progress: {completed}/{total_futures} components " +
                                   f"({successful} successful, {failed} failed)")
                
                logger.info(f"Translation complete: {successful} components translated successfully, {failed} components failed")
            else:
                logger.info("No translatable components found in document")
        
        return translated_components
    
    def _extract_financial_terms(self, components: List[DocumentComponent]) -> List[str]:
        """Extract common financial terms for consistent translation."""
        # Common financial terms to look for
        common_terms = [
            "Portfolio", "Asset Allocation", "Diversification",
            "Investment", "Returns", "Risk Management",
            "Equity", "Fixed Income", "Cash Equivalent",
            "Mutual Fund", "ETF", "Stocks", "Bonds",
            "Retirement", "IRA", "401(k)", "Tax",
            "Estate Planning", "Insurance", "Annuity",
            "Financial Plan", "Wealth Management", "Net Worth",
            "Income", "Expenses", "Budget", "Savings"
        ]
        
        # Extract terms from components
        terms = set()
        for component in components:
            if isinstance(component, TextComponent):
                for term in common_terms:
                    if term.lower() in component.text.lower():
                        terms.add(term)
            elif isinstance(component, TableComponent):
                for row in component.rows:
                    for cell in row:
                        for term in common_terms:
                            if term.lower() in cell.lower():
                                terms.add(term)
                                
        return list(terms)
    
    def _translate_text_component(self, component: TextComponent, financial_terms: List[str]) -> TextComponent:
        """Translate a text component."""
        if not component.text.strip():
            return component
            
        try:
            # Translate the text content
            translated_text = self._translate_text(component.text, financial_terms)
            
            # Create a new component with translated text
            return TextComponent(
                component_id=component.component_id,
                component_type=component.component_type,
                page_number=component.page_number,
                text=translated_text,
                font_info=component.font_info,
                position=component.position,
                is_header=component.is_header,
                is_footer=component.is_footer
            )
        except Exception as e:
            logger.error(f"Error translating text component: {str(e)}")
            return component  # Return original on error
    
    def _translate_table_component(self, component: TableComponent, financial_terms: List[str]) -> TableComponent:
        """Translate a table component."""
        try:
            # Translate each cell
            translated_rows = []
            for row in component.rows:
                translated_row = [self._translate_text(cell, financial_terms) for cell in row]
                translated_rows.append(translated_row)
            
            # Create a new component with translated text
            return TableComponent(
                component_id=component.component_id,
                component_type=component.component_type,
                page_number=component.page_number,
                rows=translated_rows,
                position=component.position
            )
        except Exception as e:
            logger.error(f"Error translating table component: {str(e)}")
            return component  # Return original on error
    
    def _translate_text(self, text: str, financial_terms: List[str] = None) -> str:
        """
        Translate a text string.
        
        Args:
            text: Text to translate
            financial_terms: List of financial terms for consistent translation
            
        Returns:
            Translated text
        """
        if not text.strip():
            return text
        
        # Handle numbers, dates, email addresses and URLs
        # This ensures they remain unchanged during translation
        text_with_placeholders, placeholders = self._prepare_text_for_translation(text)
        
        try:
            if self.model.startswith("gpt"):
                translated_text = self._translate_with_openai(text_with_placeholders, financial_terms)
            else:
                # Fall back to dummy translation for non-OpenAI models
                source_lang_name = self.language_names.get(self.source_lang, self.source_lang)
                target_lang_name = self.language_names.get(self.target_lang, self.target_lang)
                translated_text = f"[{source_lang_name} → {target_lang_name}] {text}"
            
            # Restore placeholders
            restored_text = self._restore_placeholders(translated_text, placeholders)
            return restored_text
                
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text  # Return original text on error
    
    def _prepare_text_for_translation(self, text):
        """Prepare text for translation by replacing special items with placeholders."""
        import re
        
        # Patterns to match
        patterns = {
            'number': r'\b\d+(\.\d+)?\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://[^\s]+'
        }
        
        placeholders = {}
        text_with_placeholders = text
        
        # Replace patterns with placeholders
        for pattern_type, pattern in patterns.items():
            matches = re.finditer(pattern, text_with_placeholders)
            for i, match in enumerate(matches):
                placeholder = f"__{pattern_type}_{i}__"
                placeholders[placeholder] = match.group(0)
                text_with_placeholders = text_with_placeholders.replace(match.group(0), placeholder)
        
        return text_with_placeholders, placeholders
    
    def _restore_placeholders(self, translated_text, placeholders):
        """Restore placeholders in translated text with original values."""
        restored_text = translated_text
        for placeholder, original in placeholders.items():
            restored_text = restored_text.replace(placeholder, original)
        return restored_text
    
    def _translate_with_openai(self, text: str, target_lang: str = None, financial_terms: List[str] = None, temperature: float = 0.3) -> str:
        """
        Translate text using OpenAI or xAI API.
        
        Args:
            text: Text to translate
            target_lang: Target language code (overrides self.target_lang if provided)
            financial_terms: List of financial terms for consistent translation
            temperature: Temperature for OpenAI generation (lower for more consistency)
            
        Returns:
            Translated text
        """
        if not self.api_key:
            logger.warning("No API key provided, returning original text")
            return f"[NO API KEY] {text}"
        
        # Use provided target_lang if available, otherwise use instance target_lang
        actual_target_lang = target_lang if target_lang is not None else self.target_lang
        
        # Check if the text is too long and needs to be chunked
        if self._count_tokens(text) > self.max_tokens // 2:
            return self._translate_long_text(text, financial_terms, actual_target_lang)
        
        # Prepare system message with instructions
        source_lang_name = self.language_names.get(self.source_lang, self.source_lang)
        target_lang_name = self.language_names.get(actual_target_lang, actual_target_lang)
        
        system_message = f"You are a professional translator specializing in financial documents. Translate from {source_lang_name} to {target_lang_name}."
        
        # Add financial term glossary if available
        if financial_terms and len(financial_terms) > 0:
            terms_text = ", ".join(financial_terms)
            system_message += f" Ensure consistent translation of the following financial terms: {terms_text}."
            
        system_message += " Preserve formatting, numbers, and special characters. Maintain the professional tone of financial documents."
        
        # Log important info
        model_provider = "OpenAI" if self.model.startswith("gpt") else "xAI Grok" if self.model.startswith("grok") else "Custom"
        logger.info(f"Translating text with {model_provider} ({len(text)} chars) from {source_lang_name} to {target_lang_name}")
        if actual_target_lang == "zh":
            logger.info("Chinese translation requested - ensuring proper character encoding")
        
        try:
            # Configure client based on model type
            if self.model.startswith("gpt"):
                # Call OpenAI API for translation using the latest client format
                client = openai.OpenAI(api_key=self.api_key)
            elif self.model.startswith("grok"):
                # Call xAI API for translation
                client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.x.ai/v1"
                )
            else:
                # Default to OpenAI
                client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": text}
                ],
                temperature=temperature,  # Use provided temperature
                max_tokens=self.max_tokens // 2
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Verify Chinese translation when appropriate
            if actual_target_lang == "zh":
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in translated_text)
                if not has_chinese:
                    logger.warning(f"{model_provider} translation did not return Chinese characters. Result: {translated_text[:100]}...")
                else:
                    logger.info(f"Chinese characters verified in {model_provider} translation output")
                    
            return translated_text
            
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            
            # Simple retry with backoff in case of rate limiting
            if "rate limit" in str(e).lower():
                logger.info("Rate limit hit, retrying after delay...")
                time.sleep(2)
                try:
                    # Configure client based on model type (same as above)
                    if self.model.startswith("gpt"):
                        client = openai.OpenAI(api_key=self.api_key)
                    elif self.model.startswith("grok"):
                        client = openai.OpenAI(
                            api_key=self.api_key,
                            base_url="https://api.x.ai/v1"
                        )
                    else:
                        client = openai.OpenAI(api_key=self.api_key)
                    
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": text}
                        ],
                        temperature=temperature,  # Use provided temperature
                        max_tokens=self.max_tokens // 2
                    )
                    
                    translated_text = response.choices[0].message.content.strip()
                    
                    # Verify Chinese translation when appropriate
                    if actual_target_lang == "zh":
                        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in translated_text)
                        if not has_chinese:
                            logger.warning(f"Retry translation did not return Chinese characters.")
                        else:
                            logger.info("Chinese characters verified in retry translation")
                            
                    return translated_text
                except Exception as e2:
                    logger.error(f"API retry failed: {str(e2)}")
            
            return text  # Return original text on error
    
    def _translate_long_text(self, text: str, financial_terms: List[str] = None, target_lang: str = None) -> str:
        """
        Handle translation of long text by splitting it into chunks.
        
        Args:
            text: Long text to translate
            financial_terms: List of financial terms for consistent translation
            target_lang: Target language code (overrides self.target_lang if provided)
            
        Returns:
            Combined translated text
        """
        # Use provided target_lang if available, otherwise use instance target_lang
        actual_target_lang = target_lang if target_lang is not None else self.target_lang
        
        # Simple text chunking by sentences
        import re
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = self._count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, start a new chunk
            if current_length + sentence_length > (self.max_tokens // 2):
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add any remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        logger.info(f"Split long text ({len(text)} chars) into {len(chunks)} chunks for translation")
        
        # Translate each chunk
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Translating chunk {i+1}/{len(chunks)} of long text")
            translated_chunk = self._translate_with_openai(
                text=chunk,
                target_lang=actual_target_lang,
                financial_terms=financial_terms
            )
            translated_chunks.append(translated_chunk)
            
            # Add a small delay between API calls to avoid rate limiting
            if i < len(chunks) - 1:
                time.sleep(1)  # Increased delay to better handle rate limits
        
        # Combine translated chunks
        result = " ".join(translated_chunks)
        
        # Verify Chinese translation when appropriate
        if actual_target_lang == "zh":
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in result)
            if not has_chinese:
                logger.warning("Long text translation did not produce Chinese characters")
            else:
                logger.info("Chinese characters verified in long text translation")
                
        return result
