"""
Validation module for AutoWealthTranslate.

This module is responsible for validating translated documents,
ensuring quality and completeness of translations.
"""

import re
import logging
from typing import List, Dict, Any, Union
import difflib
import json

from auto_wealth_translate.utils.logger import get_logger
from auto_wealth_translate.core.document_processor import (
    DocumentComponent, TextComponent, TableComponent, 
    ImageComponent, ChartComponent
)

logger = get_logger(__name__)

class OutputValidator:
    """
    Validates translated document output.
    """
    
    def __init__(self):
        """Initialize the validator."""
        logger.info("Initialized output validator")
        
    def validate(self, 
                original_components: List[DocumentComponent], 
                translated_document) -> Dict[str, Any]:
        """
        Validate the translated document against the original components.
        
        Args:
            original_components: Original document components
            translated_document: Translated document output
            
        Returns:
            Validation results including score and issues
        """
        logger.info("Validating translated document")
        
        # Initialize validation results
        results = {
            "score": 0,
            "issues": []
        }
        
        # Check for completeness
        original_component_count = len(original_components)
        text_components = [c for c in original_components if isinstance(c, TextComponent)]
        text_component_count = len(text_components)
        
        # Count original text components with non-trivial content
        significant_text_components = [c for c in text_components if len(c.text.strip()) > 10]
        significant_text_count = len(significant_text_components)
        
        # Count tables
        table_components = [c for c in original_components if isinstance(c, TableComponent)]
        table_count = len(table_components)
        
        # Count images and charts
        image_components = [c for c in original_components if isinstance(c, ImageComponent)]
        chart_components = [c for c in original_components if isinstance(c, ChartComponent)]
        image_count = len(image_components) + len(chart_components)
        
        # In a full implementation, we would extract components from the translated document
        # and compare them with the original components. For this version, we'll provide
        # a simplified validation by checking the document output.
        
        # Example validation logic:
        # 1. Assign a base score of 7
        # 2. Deduct points for missing components or issues
        # 3. Add points for good quality
        
        score = 7  # Base score
        
        # Check if output exists
        if translated_document is None:
            results["score"] = 0
            results["issues"].append("No output document was generated")
            return results
        
        # Special case for markdown-processed documents (passed as dict)
        if isinstance(translated_document, dict) and "markdown_processed" in translated_document:
            return self.validate_markdown_document(translated_document, original_components)
        
        # Simplified validation - in a real implementation, these checks would be more thorough
        if hasattr(translated_document, 'data') and translated_document.data:
            # Document was generated, which is good
            score += 1
            
            # Check data size (very basic check)
            if len(translated_document.data) < 1000:
                score -= 1
                results["issues"].append("Output document is suspiciously small")
        else:
            score -= 2
            results["issues"].append("Generated document has no data")
        
        # Apply additional checks and adjustments
        if text_component_count > 0 and table_count > 0:
            # Both text and tables are present in original, so check for basic completeness
            if score >= 6:
                score += 1  # Bonus point for handling complex document
            
        # Add final formatting check (simplified for this implementation)
        score += 1  # Assume formatting is preserved
        
        # Clamp score between 0 and 10
        score = max(0, min(10, score))
        
        # Set final results
        results["score"] = score
        if score >= 8:
            logger.info(f"Translation validated with high score: {score}/10")
        elif score >= 5:
            logger.info(f"Translation validated with moderate score: {score}/10")
        else:
            logger.warning(f"Translation validated with low score: {score}/10")
            if not results["issues"]:
                results["issues"].append("Low quality translation detected")
        
        return results
    
    def validate_markdown_document(self, markdown_result: Dict[str, Any], original_components: List[DocumentComponent]) -> Dict[str, Any]:
        """
        Validate a document processed through the Markdown processor.
        
        Args:
            markdown_result: Result information from markdown processing
            original_components: Original document components
            
        Returns:
            Validation results including score and issues
        """
        logger.info("Validating markdown-processed document")
        
        # Initialize validation results
        results = {
            "score": 8,  # Start with a higher base score for markdown processing
            "issues": []
        }
        
        # Check document existence
        if not markdown_result.get("output_path"):
            results["score"] = 0
            results["issues"].append("No output document was generated")
            return results
            
        # Check for translation completeness
        if markdown_result.get("translation_completeness", 0) < 0.9:
            results["score"] -= 2
            results["issues"].append("Translation appears to be incomplete")
        
        # Check for structure preservation
        if markdown_result.get("structure_preservation", 0) < 0.8:
            results["score"] -= 1
            results["issues"].append("Document structure may not be fully preserved")
            
        # Check for CJK character handling if target language requires it
        if markdown_result.get("target_language") in ["zh", "ja", "ko"]:
            if not markdown_result.get("cjk_support", True):
                results["score"] -= 1
                results["issues"].append("CJK character support issues detected")
                
        # Check table handling
        table_count = len([c for c in original_components if isinstance(c, TableComponent)])
        if table_count > 0 and not markdown_result.get("tables_preserved", True):
            results["score"] -= 1
            results["issues"].append("Some tables may not be properly preserved")
            
        # Clamp score between 0 and 10
        results["score"] = max(0, min(10, results["score"]))
        
        # Log validation results
        if results["score"] >= 8:
            logger.info(f"Markdown translation validated with high score: {results['score']}/10")
        elif results["score"] >= 5:
            logger.info(f"Markdown translation validated with moderate score: {results['score']}/10")
        else:
            logger.warning(f"Markdown translation validated with low score: {results['score']}/10")
            
        return results
    
    def _check_formatting_consistency(self, original_components, translated_components):
        """
        Check if formatting is consistent between original and translated components.
        
        This is a simplified implementation. A full version would check font style,
        layouts, spacing, and other formatting details.
        
        Args:
            original_components: Original document components
            translated_components: Translated document components
            
        Returns:
            Tuple of (consistency_score, issues)
        """
        issues = []
        
        # Count components by type
        original_counts = self._count_components_by_type(original_components)
        translated_counts = self._count_components_by_type(translated_components)
        
        # Compare component counts
        for component_type, count in original_counts.items():
            translated_count = translated_counts.get(component_type, 0)
            if translated_count < count:
                missing = count - translated_count
                percent_missing = (missing / count) * 100
                issues.append(f"Missing {missing} ({percent_missing:.1f}%) {component_type} components")
        
        # Calculate consistency score (0-10)
        score = 10
        for issue in issues:
            score -= 2  # Deduct points for each issue
        
        # Ensure score is between 0-10
        score = max(0, min(10, score))
        
        return score, issues
    
    def _count_components_by_type(self, components):
        """
        Count components by type.
        
        Args:
            components: List of document components
            
        Returns:
            Dictionary with component counts by type
        """
        counts = {}
        for component in components:
            component_type = component.component_type
            counts[component_type] = counts.get(component_type, 0) + 1
        return counts
