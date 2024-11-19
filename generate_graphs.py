import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_results_file(file_path):
    """Read TSV results file into a nested dictionary."""
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            topic_id, _, doc_id, rank, score, _ = line.strip().split('\t')
            if topic_id not in results:
                results[topic_id] = {}
            results[topic_id][doc_id] = float(score)
    return results

def read_qrels_file(qrels_path):
    """Read QRELS file into a nested dictionary."""
    qrels = {}
    with open(qrels_path, 'r') as f:
        for line in f:
            topic_id, _, doc_id, relevance = line.strip().split()
            if topic_id not in qrels:
                qrels[topic_id] = {}
            qrels[topic_id][doc_id] = int(relevance)
    return qrels

def plot_precision_at_5_bar_plot(qrels, run_results, result_file: str):
    """Generate precision@5 bar plot for a single result file."""
    topic_precision = {}
    for topic_id in run_results:
        if topic_id in qrels:
            relevant_docs = set(doc_id for doc_id, rel in qrels[topic_id].items() if rel > 0)
            retrieved_docs = list(run_results[topic_id].keys())[:5]
            relevant_retrieved = len([doc for doc in retrieved_docs if doc in relevant_docs])
            topic_precision[topic_id] = relevant_retrieved / 5

    sorted_precision = sorted(topic_precision.items(), key=lambda x: x[1], reverse=True)
    
    # Handle case with few topics
    if len(sorted_precision) < 10:
        selected_topics = sorted_precision  # Use all topics if we have fewer than 10
    else:
        # Select 50 topics evenly across the range
        indices = np.linspace(0, len(sorted_precision) - 1, min(50, len(sorted_precision)), dtype=int)
        selected_topics = [sorted_precision[i] for i in indices]
    
    if not selected_topics:
        print(f"Warning: No topics found for {result_file}")
        return topic_precision
    
    # Create the plot
    topic_ids, precisions = zip(*selected_topics)
    plt.figure(figsize=(20, 10))
    bars = plt.bar(range(1, len(topic_ids) + 1), precisions, width=0.8)
    
    # Add topic IDs as labels
    for i, (topic_id, precision) in enumerate(selected_topics):
        plt.text(i + 1, precision, topic_id, ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Customize the plot
    plt.xlabel('Topics (Ranked by Precision@5)')
    plt.ylabel('Precision@5')
    plt.title(f'Precision@5 Bar Plot for {os.path.basename(result_file)}')
    plt.ylim(0, 1)
    plt.xlim(0, len(topic_ids) + 1)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    # Save the plot
    output_filename = f"precision_at_5_{os.path.splitext(os.path.basename(result_file))[0]}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    return topic_precision

def process_all_results(results_dir, qrels_path):
    """Process all result files in the directory and generate summary statistics."""
    qrels = read_qrels_file(qrels_path)
    summary_stats = {}
    
    # Print total number of topics in qrels for debugging
    print(f"Total number of topics in qrels: {len(qrels)}")
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.tsv'):
            file_path = os.path.join(results_dir, filename)
            print(f"\nProcessing {filename}")
            
            run_results = read_results_file(file_path)
            print(f"Number of topics in {filename}: {len(run_results)}")
            
            # Generate plot
            topic_precision = plot_precision_at_5_bar_plot(qrels, run_results, filename)
            
            if topic_precision:
                # Calculate summary statistics
                precisions = list(topic_precision.values())
                avg_precision = np.mean(precisions)
                median_precision = np.median(precisions)
                std_precision = np.std(precisions)
                
                summary_stats[filename] = {
                    'average_precision@5': avg_precision,
                    'median_precision@5': median_precision,
                    'std_precision@5': std_precision,
                    'num_topics': len(topic_precision)
                }
                
                print(f"Processed {filename} - {len(topic_precision)} topics")
            else:
                print(f"Warning: No valid precision values for {filename}")
    
    return summary_stats

def main():
    # Set your paths here using raw strings
    results_dir = r"C:\Users\evanm\OneDrive\Documents\IR\A3\results"
    qrels_path = r"C:\Users\evanm\OneDrive\Documents\IR\A3\qrel_1.tsv"
    
    # Process all results and get summary statistics
    print("Starting processing...")
    summary_stats = process_all_results(results_dir, qrels_path)
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')
    summary_df.to_csv('precision_summary_stats.csv')
    
    print("\nSummary statistics:")
    print(summary_df)

if __name__ == "__main__":
    main()