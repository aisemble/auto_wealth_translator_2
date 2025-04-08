"""
Chart Processor
--------------
Detects and processes charts from document images.
"""

import logging
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io

logger = logging.getLogger(__name__)

CHART_TYPES = ["bar", "pie", "line", "scatter", "area", "unknown"]

def detect_chart(image: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Detect charts in an image and identify their types.
    
    Args:
        image: OpenCV image (numpy array)
    
    Returns:
        List of (chart_type, bounding_box) tuples
    """
    logger.debug("Detecting charts in image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_area = image.shape[0] * image.shape[1] * 0.05  # At least 5% of image
    chart_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    detected_charts = []
    
    for contour in chart_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract the region of interest
        roi = image[y:y+h, x:x+w]
        
        # Identify chart type
        chart_type = identify_chart_type(roi)
        
        detected_charts.append((chart_type, (x, y, x+w, y+h)))
    
    logger.debug(f"Detected {len(detected_charts)} charts")
    return detected_charts

def identify_chart_type(chart_image: np.ndarray) -> str:
    """
    Identify the type of chart in the image.
    
    Args:
        chart_image: Chart image (numpy array)
    
    Returns:
        Chart type (bar, pie, line, etc.)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(chart_image, cv2.COLOR_BGR2GRAY)
    
    # Detect circles for pie chart detection
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=50, 
        param1=200, 
        param2=30, 
        minRadius=chart_image.shape[0]//6, 
        maxRadius=chart_image.shape[0]//2
    )
    
    if circles is not None:
        return "pie"
    
    # Apply edge detection for other chart types
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect lines
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100, 
        minLineLength=chart_image.shape[1]//5, 
        maxLineGap=5
    )
    
    if lines is None:
        return "unknown"
    
    horizontal_lines = 0
    vertical_lines = 0
    diagonal_lines = 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 10 or angle > 170:
            horizontal_lines += 1
        elif 80 < angle < 100:
            vertical_lines += 1
        else:
            diagonal_lines += 1
    
    # Decision logic for chart type
    if horizontal_lines > vertical_lines * 1.5:
        return "line"
    elif vertical_lines > 5 and vertical_lines > horizontal_lines:
        return "bar"
    elif diagonal_lines > horizontal_lines and diagonal_lines > vertical_lines:
        return "scatter"
    else:
        # Default to bar chart as most common
        return "bar"

def process_chart(chart_image: np.ndarray, chart_type: str) -> Dict[str, Any]:
    """
    Extract data from a chart.
    
    Args:
        chart_image: Chart image (numpy array)
        chart_type: Type of the chart (bar, pie, line, etc.)
    
    Returns:
        Dict containing chart data and metadata
    """
    logger.debug(f"Processing {chart_type} chart")
    
    # Prepare OCR
    try:
        import pytesseract
        
        # Convert image to RGB for OCR
        rgb_image = cv2.cvtColor(chart_image, cv2.COLOR_BGR2RGB)
        
        # Extract all text from the chart
        text_data = pytesseract.image_to_data(rgb_image, output_type=pytesseract.Output.DICT)
        
        # Filter out low-confidence text
        filtered_texts = []
        for i, conf in enumerate(text_data['conf']):
            if float(conf) > 50:  # Confidence threshold
                filtered_texts.append({
                    'text': text_data['text'][i],
                    'left': text_data['left'][i],
                    'top': text_data['top'][i],
                    'width': text_data['width'][i],
                    'height': text_data['height'][i],
                    'conf': float(conf)
                })

        # Process specific chart types
        if chart_type == "bar":
            return process_bar_chart(chart_image, filtered_texts)
        elif chart_type == "pie":
            return process_pie_chart(chart_image, filtered_texts)
        elif chart_type == "line":
            return process_line_chart(chart_image, filtered_texts)
        else:
            # Generic chart processing for unsupported types
            return {
                'chart_type': chart_type,
                'texts': filtered_texts,
                'title': extract_chart_title(filtered_texts, chart_image.shape),
                'labels': extract_potential_labels(filtered_texts, chart_image.shape),
            }
        
    except Exception as e:
        logger.error(f"Error processing chart: {str(e)}")
        return {
            'chart_type': chart_type,
            'error': str(e),
            'texts': [],
        }

def extract_chart_title(texts: List[Dict], image_shape: Tuple[int, int, int]) -> Optional[str]:
    """Extract the chart title from OCR texts."""
    # Titles are typically at the top and have larger font
    height, width = image_shape[0], image_shape[1]
    
    # Sort by y-position (top to bottom) and then by area (larger first)
    sorted_texts = sorted(texts, key=lambda t: (t['top'], -(t['width'] * t['height'])))
    
    # Consider texts in the top 15% of the image
    top_margin = height * 0.15
    top_texts = [t for t in sorted_texts if t['top'] < top_margin]
    
    if top_texts:
        return top_texts[0]['text']
    return None

def extract_potential_labels(texts: List[Dict], image_shape: Tuple[int, int, int]) -> List[str]:
    """Extract potential axis labels and legend items."""
    height, width = image_shape[0], image_shape[1]
    
    # Text on the sides (y-axis) and bottom (x-axis)
    left_margin = width * 0.15
    right_margin = width * 0.85
    bottom_margin = height * 0.85
    
    # Potential labels
    x_labels = [t['text'] for t in texts if t['top'] > bottom_margin]
    y_labels = [t['text'] for t in texts if t['left'] < left_margin or t['left'] > right_margin]
    
    return list(set(x_labels + y_labels))

def process_bar_chart(image: np.ndarray, texts: List[Dict]) -> Dict[str, Any]:
    """Process a bar chart to extract its data."""
    height, width = image.shape[0], image.shape[1]
    
    # Extract title and labels
    title = extract_chart_title(texts, image.shape)
    
    # Attempt to identify x and y axis labels
    x_label = None
    y_label = None
    
    # Texts at the bottom center might be x-axis label
    bottom_texts = [t for t in texts if t['top'] > height * 0.85 and width * 0.3 < t['left'] < width * 0.7]
    if bottom_texts:
        x_label = bottom_texts[0]['text']
    
    # Texts at the left middle might be y-axis label
    left_texts = [t for t in texts if t['left'] < width * 0.15 and height * 0.3 < t['top'] < height * 0.7]
    if left_texts:
        y_label = left_texts[0]['text']
    
    # Try to identify bar categories and values
    categories = []
    
    # Look for horizontally aligned text near the bottom (likely categories)
    category_candidates = [t for t in texts if height * 0.7 < t['top'] < height * 0.85]
    # Sort by x position for left-to-right order
    category_candidates.sort(key=lambda t: t['left'])
    categories = [t['text'] for t in category_candidates if t['text'].strip()]
    
    # Return structured data
    return {
        'chart_type': 'bar',
        'title': title,
        'x_label': x_label,
        'y_label': y_label,
        'categories': categories,
        'all_texts': texts,
    }

def process_pie_chart(image: np.ndarray, texts: List[Dict]) -> Dict[str, Any]:
    """Process a pie chart to extract its data."""
    # Extract title
    title = extract_chart_title(texts, image.shape)
    
    # Try to identify legend items or slice labels
    legend_items = []
    
    # In pie charts, labels are often to the right or inside the pie
    for text in texts:
        if text['text'].strip() and text != title:
            # Skip very short texts or likely percentages
            if len(text['text']) > 1 and not text['text'].endswith('%') and not text['text'].replace('.', '').isdigit():
                legend_items.append(text['text'])
    
    # Return structured data
    return {
        'chart_type': 'pie',
        'title': title,
        'legend_items': legend_items,
        'all_texts': texts,
    }

def process_line_chart(image: np.ndarray, texts: List[Dict]) -> Dict[str, Any]:
    """Process a line chart to extract its data."""
    height, width = image.shape[0], image.shape[1]
    
    # Extract title and labels
    title = extract_chart_title(texts, image.shape)
    
    # Attempt to identify x and y axis labels
    x_label = None
    y_label = None
    
    # Texts at the bottom center might be x-axis label
    bottom_texts = [t for t in texts if t['top'] > height * 0.85 and width * 0.3 < t['left'] < width * 0.7]
    if bottom_texts:
        x_label = bottom_texts[0]['text']
    
    # Texts at the left middle might be y-axis label
    left_texts = [t for t in texts if t['left'] < width * 0.15 and height * 0.3 < t['top'] < height * 0.7]
    if left_texts:
        y_label = left_texts[0]['text']
    
    # Try to identify legend items
    # Usually found in the top or bottom part
    legend_candidates = [t for t in texts 
                         if (t['top'] < height * 0.2 or t['top'] > height * 0.8) 
                         and width * 0.5 < t['left'] < width * 0.95]
    legend_items = [t['text'] for t in legend_candidates if t['text'] != title and t['text'].strip()]
    
    # Return structured data
    return {
        'chart_type': 'line',
        'title': title,
        'x_label': x_label,
        'y_label': y_label,
        'legend_items': legend_items,
        'all_texts': texts,
    }

def chart_to_markdown(chart_data: Dict[str, Any], target_lang_texts: Dict[str, str]) -> str:
    """
    Convert chart data to markdown format.
    
    Args:
        chart_data: Original chart data
        target_lang_texts: Dictionary mapping original text to translated text
    
    Returns:
        Markdown string representation of the chart
    """
    markdown = []
    
    # Add title if available
    if chart_data.get('title') and chart_data['title'] in target_lang_texts:
        markdown.append(f"## {target_lang_texts[chart_data['title']]}\n")
    
    chart_type = chart_data['chart_type']
    
    if chart_type == 'bar':
        # Create markdown table for bar chart
        categories = chart_data.get('categories', ['Category 1', 'Category 2', 'Category 3'])
        values = [1, 2, 3]  # Placeholder values
        
        # Translate categories
        translated_categories = [target_lang_texts.get(cat, cat) for cat in categories]
        
        # Create markdown table
        markdown.append("| Category | Value |")
        markdown.append("|----------|--------|")
        for cat, val in zip(translated_categories, values):
            markdown.append(f"| {cat} | {val} |")
            
    elif chart_type == 'pie':
        # Create markdown list for pie chart
        labels = chart_data.get('legend_items', ['Item 1', 'Item 2', 'Item 3'])
        sizes = [35, 35, 30]  # Placeholder values
        
        # Translate labels
        translated_labels = [target_lang_texts.get(label, label) for label in labels]
        
        markdown.append("### Distribution")
        for label, size in zip(translated_labels, sizes):
            markdown.append(f"- {label}: {size}%")
            
    elif chart_type == 'line':
        # Create markdown table for line chart data
        x = np.arange(5)
        y = x ** 2
        
        markdown.append("| X | Y |")
        markdown.append("|---|----|")
        for xi, yi in zip(x, y):
            markdown.append(f"| {xi} | {yi} |")
            
        # Add legend if available
        if chart_data.get('legend_items'):
            translated_legends = [target_lang_texts.get(item, item) for item in chart_data['legend_items']]
            markdown.append("\n**Legend:**")
            for legend in translated_legends:
                markdown.append(f"- {legend}")
    
    # Add axis labels if available
    if chart_data.get('x_label') and chart_data['x_label'] in target_lang_texts:
        markdown.append(f"\n*X-axis: {target_lang_texts[chart_data['x_label']]}*")
    if chart_data.get('y_label') and chart_data['y_label'] in target_lang_texts:
        markdown.append(f"\n*Y-axis: {target_lang_texts[chart_data['y_label']]}*")
    
    return "\n".join(markdown)

def recreate_chart(chart_data: Dict[str, Any], target_lang_texts: Dict[str, str], output_format: str = "figure") -> Optional[Union[Figure, str]]:
    """
    Recreate a chart with translated labels and titles.
    
    Args:
        chart_data: Original chart data
        target_lang_texts: Dictionary mapping original text to translated text
        output_format: Output format ("figure" or "markdown")
    
    Returns:
        Either a Matplotlib figure or markdown string
    """
    if output_format == "markdown":
        return chart_to_markdown(chart_data, target_lang_texts)
        
    try:
        chart_type = chart_data['chart_type']
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Set title and labels if available
        if chart_data.get('title') and chart_data['title'] in target_lang_texts:
            ax.set_title(target_lang_texts[chart_data['title']])
        
        if chart_data.get('x_label') and chart_data['x_label'] in target_lang_texts:
            ax.set_xlabel(target_lang_texts[chart_data['x_label']])
            
        if chart_data.get('y_label') and chart_data['y_label'] in target_lang_texts:
            ax.set_ylabel(target_lang_texts[chart_data['y_label']])
        
        # Recreate chart based on type
        if chart_type == 'bar':
            # Create placeholder data if we don't have real data
            categories = chart_data.get('categories', ['Category 1', 'Category 2', 'Category 3'])
            values = [1, 2, 3]  # Placeholder values
            
            # Translate categories
            translated_categories = [
                target_lang_texts.get(cat, cat) for cat in categories
            ]
            
            ax.bar(translated_categories, values)
            plt.xticks(rotation=45)
            
        elif chart_type == 'pie':
            # Create placeholder data
            labels = chart_data.get('legend_items', ['Item 1', 'Item 2', 'Item 3'])
            sizes = [35, 35, 30]  # Placeholder values
            
            # Translate labels
            translated_labels = [
                target_lang_texts.get(label, label) for label in labels
            ]
            
            ax.pie(sizes, labels=translated_labels, autopct='%1.1f%%')
            ax.axis('equal')
            
        elif chart_type == 'line':
            # Create placeholder data
            x = np.arange(5)
            y = x ** 2
            
            ax.plot(x, y)
            
            # Add translated legend if available
            if chart_data.get('legend_items'):
                translated_legends = [
                    target_lang_texts.get(item, item) for item in chart_data['legend_items']
                ]
                ax.legend(translated_legends)
        
        else:
            # Generic chart (placeholder)
            ax.text(0.5, 0.5, f"Translated {chart_type} chart", 
                    horizontalalignment='center',
                    verticalalignment='center')
            ax.axis('off')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error recreating chart: {str(e)}")
        return None
