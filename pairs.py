import json
import os
import pandas as pd
from collections import defaultdict

def load_json_file(file_path):
    """Load and return JSON file content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Convert list to dictionary with Id as key
        if isinstance(data, list):
            return {item['Id']: item for item in data}
        return data

def read_tsv_file(file_path):
    """Read TSV qrels file into a dictionary."""
    qrels = defaultdict(set)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                topic_id, _, doc_id, relevance = parts
                if int(relevance) > 0:
                    qrels[topic_id].add(doc_id)
    return qrels

def read_results_file(file_path):
    """Read TSV results file into a nested dictionary."""
    results = defaultdict(dict)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                topic_id, _, doc_id, rank, score, _ = parts
                if int(rank) <= 5:  # Only consider top 5 results
                    results[topic_id][doc_id] = {
                        'rank': int(rank),
                        'score': float(score)
                    }
    return results

def calculate_precision_at_5(retrieved_docs, relevant_docs):
    """Calculate precision@5 for a single topic."""
    relevant_retrieved = sum(1 for doc_id in list(retrieved_docs.keys())[:5] 
                           if doc_id in relevant_docs)
    return relevant_retrieved / 5

def analyze_topic_pairs(results_dir, topics_file, qrels_file):
    """Analyze results files to find contrasting topic pairs."""
    # Load topics and qrels
    topics = load_json_file(topics_file)
    qrels = read_tsv_file(qrels_file)
    
    pairs_analysis = {}
    
    # Process each results file
    for filename in os.listdir(results_dir):
        if not filename.endswith('.tsv'):
            continue
            
        print(f"\nAnalyzing {filename}...")
        file_path = os.path.join(results_dir, filename)
        results = read_results_file(file_path)
        
        # Calculate precision for each topic
        topic_performance = {}
        for topic_id in results:
            # Get relevant documents for this topic from qrels
            relevant_docs = qrels.get(topic_id, set())
            
            # Calculate precision@5
            precision = calculate_precision_at_5(results[topic_id], relevant_docs)
            
            # Get topic text if available
            topic_info = topics.get(topic_id, {})
            topic_text = f"{topic_info.get('Title', '')} - {topic_info.get('Body', '')}"
            if not topic_text.strip():
                topic_text = 'Topic text not found'
            
            topic_performance[topic_id] = {
                'precision': precision,
                'topic_text': topic_text,
                'retrieved_docs': list(results[topic_id].keys())[:5],
                'relevant_docs': list(relevant_docs)
            }
        
        # Sort topics by precision
        sorted_topics = sorted(topic_performance.items(), 
                             key=lambda x: x[1]['precision'])
        
        # Find worst and best performing topics
        worst_topics = [(tid, data) for tid, data in sorted_topics 
                       if data['precision'] == 0][:3]
        best_topics = [(tid, data) for tid, data in sorted_topics[::-1] 
                      if data['precision'] > 0][:3]
        
        if not worst_topics or not best_topics:
            print(f"Insufficient contrasting pairs found for {filename}")
            continue
        
        # Select one clear contrasting pair
        pairs = [{
            'good_topic': {
                'id': best_topics[0][0],
                'text': best_topics[0][1]['topic_text'],
                'precision': best_topics[0][1]['precision'],
                'retrieved': best_topics[0][1]['retrieved_docs'],
                'relevant': best_topics[0][1]['relevant_docs']
            },
            'bad_topic': {
                'id': worst_topics[0][0],
                'text': worst_topics[0][1]['topic_text'],
                'precision': worst_topics[0][1]['precision'],
                'retrieved': worst_topics[0][1]['retrieved_docs'],
                'relevant': worst_topics[0][1]['relevant_docs']
            }
        }]
        
        pairs_analysis[filename] = pairs
    
    return pairs_analysis

def save_analysis_report(pairs_analysis, output_file):
    """Save the analysis results to a formatted text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Topic Pair Analysis Report\n")
        f.write("=========================\n\n")
        
        for filename, pairs in pairs_analysis.items():
            f.write(f"\nResults File: {filename}\n")
            f.write("=" * (len(filename) + 14) + "\n\n")
            
            for pair in pairs:
                # Good topic details
                f.write("Good Performing Topic:\n")
                f.write(f"  ID: {pair['good_topic']['id']}\n")
                f.write(f"  Text: {pair['good_topic']['text'][:200]}...\n")
                f.write(f"  Precision@5: {pair['good_topic']['precision']:.2f}\n")
                f.write("  Retrieved docs: " + 
                       ", ".join(pair['good_topic']['retrieved'][:5]) + "\n")
                f.write("  Relevant docs: " + 
                       ", ".join(pair['good_topic']['relevant']) + "\n\n")
                
                # Bad topic details
                f.write("Poor Performing Topic:\n")
                f.write(f"  ID: {pair['bad_topic']['id']}\n")
                f.write(f"  Text: {pair['bad_topic']['text'][:200]}...\n")
                f.write(f"  Precision@5: {pair['bad_topic']['precision']:.2f}\n")
                f.write("  Retrieved docs: " + 
                       ", ".join(pair['bad_topic']['retrieved'][:5]) + "\n")
                f.write("  Relevant docs: " + 
                       ", ".join(pair['bad_topic']['relevant']) + "\n\n")
                
                f.write("-" * 50 + "\n\n")

def main():
    # Set your paths here
    results_dir = r"F:\Project-p2\results"
    topics_file = r"F:\Project-p2\topics_1.json"
    qrels_file = r"F:\Project-p2\qrel_1.tsv"
    output_file = "topic_pairs_analysis.txt"
    
    print("Starting analysis...")
    pairs_analysis = analyze_topic_pairs(results_dir, topics_file, qrels_file)
    
    print("Saving analysis report...")
    save_analysis_report(pairs_analysis, output_file)
    
    print(f"Analysis complete! Results saved to {output_file}")

if __name__ == "__main__":
    main()