"""
Tests for the translator module.
"""

import unittest
from unittest.mock import MagicMock, patch
import os

from auto_wealth_translate.core.translator import TranslationService


class TestTranslationService(unittest.TestCase):
    """Tests for the TranslationService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock environment variable for OpenAI API key
        os.environ["OPENAI_API_KEY"] = "test-key"
    
    @patch('auto_wealth_translate.core.translator.openai')
    def test_initialization(self, mock_openai):
        """Test that the service initializes correctly."""
        service = TranslationService(target_lang="fr")
        
        self.assertEqual(service.target_lang, "fr")
        self.assertEqual(service.model, "gpt-4")
        self.assertEqual(service.api_key, "test-key")
        
        # Check that OpenAI was initialized with the API key
        mock_openai.api_key = service.api_key
    
    @patch('auto_wealth_translate.core.translator.TranslationService._call_llm')
    def test_extract_financial_terms(self, mock_call_llm):
        """Test the extraction of financial terms."""
        service = TranslationService(target_lang="fr")
        
        # Create test components with financial terms
        components = {
            'sections': [
                {
                    'type': 'text',
                    'content': [
                        {
                            'text': 'The portfolio has a 3.5% dividend yield and 15% ROI.'
                        },
                        {
                            'text': 'Consider investing in the EmergingMarkets fund.'
                        }
                    ]
                }
            ]
        }
        
        terms = service._extract_financial_terms(components)
        
        # Check that the terms are extracted
        self.assertIn('dividend yield', terms)
        self.assertIn('ROI', terms)
        self.assertIn('EmergingMarkets fund', terms)
    
    @patch('auto_wealth_translate.core.translator.TranslationService._call_llm')
    def test_translate_financial_terms(self, mock_call_llm):
        """Test the translation of financial terms."""
        service = TranslationService(target_lang="fr")
        
        # Mock the LLM response for financial terms
        mock_call_llm.return_value = """
ROI: ROI
dividend yield: rendement en dividendes
EmergingMarkets fund: fonds des marchés émergents
"""
        
        terms = ["ROI", "dividend yield", "EmergingMarkets fund"]
        translations = service._translate_financial_terms(terms)
        
        # Check the translations
        self.assertEqual(translations["ROI"], "ROI")
        self.assertEqual(translations["dividend yield"], "rendement en dividendes")
        self.assertEqual(translations["EmergingMarkets fund"], "fonds des marchés émergents")
    
    @patch('auto_wealth_translate.core.translator.TranslationService._call_llm')
    def test_translate_section(self, mock_call_llm):
        """Test the translation of text sections."""
        service = TranslationService(target_lang="fr")
        
        # Mock the LLM response for text translation
        mock_call_llm.return_value = "Ceci est un texte traduit."
        
        sections = [
            {
                'type': 'text',
                'content': [
                    {
                        'text': 'This is some text.'
                    }
                ]
            }
        ]
        
        financial_terms = {}
        translated_sections = service._translate_sections(sections, financial_terms)
        
        # Check the translation
        self.assertEqual(translated_sections[0]['content'][0]['text'], "Ceci est un texte traduit.")
    
    @patch('auto_wealth_translate.core.translator.TranslationService._call_llm')
    def test_translate_chart(self, mock_call_llm):
        """Test the translation of chart elements."""
        service = TranslationService(target_lang="fr")
        
        # Mock the LLM response for chart element translation
        mock_call_llm.return_value = """
Chart title: Rendement des investissements
X-axis label: Trimestre
Y-axis label: Rendement (%)
Legend items: Actions, Obligations, Liquidités
"""
        
        charts = [
            {
                'chart_type': 'line',
                'title': 'Investment Returns',
                'x_label': 'Quarter',
                'y_label': 'Return (%)',
                'legend_items': ['Stocks', 'Bonds', 'Cash'],
            }
        ]
        
        translated_charts = service._translate_charts(charts)
        
        # Check the translations
        self.assertEqual(translated_charts[0]['title'], "Rendement des investissements")
        self.assertEqual(translated_charts[0]['x_label'], "Trimestre")
        self.assertEqual(translated_charts[0]['y_label'], "Rendement (%)")
        self.assertEqual(translated_charts[0]['legend_items'][0], "Actions")
        self.assertEqual(translated_charts[0]['legend_items'][1], "Obligations")
        self.assertEqual(translated_charts[0]['legend_items'][2], "Liquidités")


if __name__ == '__main__':
    unittest.main()
