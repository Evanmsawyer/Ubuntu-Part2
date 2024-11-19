import json
import argparse
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import re
from html import unescape
import warnings

# Suppress specific BeautifulSoup warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def clean_text(text):
    # Remove all non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Limit word length to 30 characters
    words = text.split()
    words = [word[:30] for word in words]
    return ' '.join(words)

def clean_html(html_content):
    try:
        html_content = unescape(html_content)
        
        # Use 'lxml' parser if available, fall back to 'html.parser'
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except:
            soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator=' ')
        
        return clean_text(text)
    except Exception as e:
        print(f"Error cleaning HTML: {e}")
        return ""  # Return empty string if cleaning fails

def clean_json_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        if 'Title' in item:
            item['Title'] = clean_html(item['Title'])
        if 'Body' in item:
            item['Body'] = clean_html(item['Body'])
        if 'Tags' in item:
            # Clean and parse the Tags field
            cleaned_tags = clean_html(item['Tags'])
            item['Tags'] = cleaned_tags.split()  # Split by spaces to create a list of tags
        if 'Text' in item:
            item['Text'] = clean_html(item['Text'])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Clean HTML content in multiple JSON files')
    parser.add_argument('--topics_files', nargs='+', default=['topics_1.json', 'topics_2.json'], help='List of topics files')
    parser.add_argument('--output_files', nargs='+', default=['cleaned_topics1.json', 'cleaned_topics2.json'], help='Output files')
    args = parser.parse_args()

    if len(args.topics_files) != len(args.output_files):
        print("Error: The number of input files must match the number of output files.")
        return

    for input_file, output_file in zip(args.topics_files, args.output_files):
        clean_json_file(input_file, output_file)
        print(f"Cleaned content from {input_file} has been written to {output_file}")
        
if __name__ == "__main__":
    main()