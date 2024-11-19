import os
import json
import logging
import argparse
import numpy as np
from typing import Dict, List, Tuple, Set
from scipy import stats
import matplotlib.pyplot as plt
import ranx
from dataclasses import dataclass
import pandas as pd
import seaborn as sns
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SignificanceResult:
    metric: str
    model1_mean: float
    model2_mean: float
    difference: float
    improvement_percentage: float
    t_statistic: float
    t_p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    significant_t_test: bool
    significant_wilcoxon: bool
    topic_scores_model1: List[float]
    topic_scores_model2: List[float]

class ModelComparator:
    def __init__(self, qrels_file: str, results_dir: str, output_dir: str):
        self.qrels = self._load_qrels(qrels_file)
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define metrics
        self.metrics = {
            "ndcg@5": ("ndcg", 5),
            "precision@5": ("precision", 5),
            "map": ("map", None),
            "mrr": ("mrr", None)
        }

    @staticmethod
    def _load_qrels(filepath: str) -> Dict[str, Dict[str, int]]:
        """Load QREL file."""
        qrels = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                query_id, _, doc_id, score = line.strip().split('\t')
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = int(score)
        return qrels

    def _load_run_file(self, filepath: str) -> Dict[str, Dict[str, float]]:
        """Load run file in TREC format with better error handling."""
        results = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 6:
                            query_id, _, doc_id, _, score, _ = parts
                            if query_id not in results:
                                results[query_id] = {}
                            results[query_id][doc_id] = float(score)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping malformed line in {filepath}: {line.strip()}")
                        continue
        except Exception as e:
            logger.error(f"Error reading run file {filepath}: {str(e)}")
            raise
        
        if not results:
            logger.warning(f"No valid results found in {filepath}")
        
        return results

    def evaluate_single_topic(self, qrels: ranx.Qrels, run: ranx.Run, 
                         metric: str, cutoff: int = None) -> float:
        """Evaluate a single topic with a given metric."""
        try:
            # Construct the metric string with cutoff if needed
            metric_str = metric
            if cutoff is not None:
                if metric in ["ndcg", "precision"]:
                    metric_str = f"{metric}@{cutoff}"
            
            # Use the general evaluate function
            result = ranx.evaluate(qrels, run, metrics=[metric_str])
            return float(result[metric_str])
        except Exception as e:
            logger.error(f"Error evaluating metric {metric}: {str(e)}")
            return 0.0

    def perform_significance_test(self, 
                            results1: Dict[str, Dict[str, float]], 
                            results2: Dict[str, Dict[str, float]], 
                            metric_name: str) -> SignificanceResult:
        """Perform statistical significance tests for a specific metric."""
        scores1 = []
        scores2 = []
        
        # Get the proper metric name
        ranx_metric_name, _ = self.metrics[metric_name]
        
        # Calculate per-topic scores
        for topic_id in self.qrels.keys():
            if topic_id not in results1 or topic_id not in results2:
                continue
            
            try:
                # Get relevant documents for this topic
                rel_docs = set(doc_id for doc_id, score in self.qrels[topic_id].items() if score > 0)
                
                # Get ranked lists
                ranked_list1 = list(results1[topic_id].keys())
                ranked_list2 = list(results2[topic_id].keys())
                
                # Calculate scores based on metric
                if metric_name.startswith("ndcg@"):
                    k = int(metric_name.split("@")[1])
                    # Calculate DCG and IDCG
                    score1 = self._calculate_ndcg(ranked_list1, rel_docs, k)
                    score2 = self._calculate_ndcg(ranked_list2, rel_docs, k)
                elif metric_name.startswith("precision@"):
                    k = int(metric_name.split("@")[1])
                    score1 = self._calculate_precision(ranked_list1, rel_docs, k)
                    score2 = self._calculate_precision(ranked_list2, rel_docs, k)
                elif metric_name == "map":
                    score1 = self._calculate_map(ranked_list1, rel_docs)
                    score2 = self._calculate_map(ranked_list2, rel_docs)
                elif metric_name == "mrr":
                    score1 = self._calculate_mrr(ranked_list1, rel_docs)
                    score2 = self._calculate_mrr(ranked_list2, rel_docs)
                else:
                    logger.error(f"Unsupported metric: {metric_name}")
                    continue
                
                scores1.append(score1)
                scores2.append(score2)
            except Exception as e:
                logger.error(f"Error evaluating topic {topic_id}: {str(e)}")
                continue
        
        # Convert to numpy arrays
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        
        if len(scores1) < 2:
            logger.warning(f"Not enough samples for statistical tests for {metric_name}")
            return SignificanceResult(
                metric=metric_name,
                model1_mean=np.mean(scores1) if len(scores1) > 0 else 0,
                model2_mean=np.mean(scores2) if len(scores2) > 0 else 0,
                difference=0,
                improvement_percentage=0,
                t_statistic=0,
                t_p_value=1,
                wilcoxon_statistic=0,
                wilcoxon_p_value=1,
                significant_t_test=False,
                significant_wilcoxon=False,
                topic_scores_model1=scores1.tolist(),
                topic_scores_model2=scores2.tolist()
            )
        
        try:
            t_stat, t_p_value = stats.ttest_rel(scores1, scores2)
            w_stat, w_p_value = stats.wilcoxon(scores1, scores2)
        except Exception as e:
            logger.warning(f"Statistical test failed for {metric_name}: {str(e)}")
            t_stat, t_p_value = 0, 1
            w_stat, w_p_value = 0, 1
        
        mean1 = np.mean(scores1)
        mean2 = np.mean(scores2)
        
        # Calculate improvement percentage from model1 (baseline) to model2
        if mean1 > 0:
            improvement_percentage = ((mean2 - mean1) / mean1) * 100
        else:
            improvement_percentage = 0.0 if mean2 == 0 else float('inf')
        
        return SignificanceResult(
            metric=metric_name,
            model1_mean=mean1,
            model2_mean=mean2,
            difference=mean2 - mean1,
            improvement_percentage=improvement_percentage,
            t_statistic=t_stat,
            t_p_value=t_p_value,
            wilcoxon_statistic=w_stat,
            wilcoxon_p_value=w_p_value,
            significant_t_test=t_p_value < 0.05,
            significant_wilcoxon=w_p_value < 0.05,
            topic_scores_model1=scores1.tolist(),
            topic_scores_model2=scores2.tolist()
        )

    def _calculate_precision(self, ranked_list: List[str], rel_docs: Set[str], k: int) -> float:
        """Calculate precision@k."""
        if not ranked_list:
            return 0.0
        
        top_k = ranked_list[:k]
        relevant_retrieved = sum(1 for doc in top_k if doc in rel_docs)
        return relevant_retrieved / k

    def _calculate_mrr(self, ranked_list: List[str], rel_docs: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not ranked_list or not rel_docs:
            return 0.0
        
        for rank, doc in enumerate(ranked_list, 1):
            if doc in rel_docs:
                return 1.0 / rank
        return 0.0

    def _calculate_map(self, ranked_list: List[str], rel_docs: Set[str]) -> float:
        """Calculate Mean Average Precision."""
        if not ranked_list or not rel_docs:
            return 0.0
        
        relevant_retrieved = 0
        sum_precision = 0.0
        
        for rank, doc in enumerate(ranked_list, 1):
            if doc in rel_docs:
                relevant_retrieved += 1
                sum_precision += relevant_retrieved / rank
        
        if len(rel_docs) == 0:
            return 0.0
        return sum_precision / len(rel_docs)

    def _calculate_ndcg(self, ranked_list: List[str], rel_docs: Set[str], k: int) -> float:
        """Calculate NDCG@k."""
        if not ranked_list:
            return 0.0
        
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG
        for i, doc in enumerate(ranked_list[:k], 1):
            if doc in rel_docs:
                dcg += 1.0 / np.log2(i + 1)
        
        # Calculate IDCG
        for i in range(min(len(rel_docs), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        return dcg / idcg

    def plot_boxplot_comparison(self, results: List[SignificanceResult], 
                              model1_name: str, model2_name: str):
        """Create boxplot comparison for all metrics."""
        plt.figure(figsize=(15, 8))
        
        data = []
        for result in results:
            # Add model 1 scores
            data.extend([{
                'Metric': result.metric,
                'Score': score,
                'Model': model1_name
            } for score in result.topic_scores_model1])
            
            # Add model 2 scores
            data.extend([{
                'Metric': result.metric,
                'Score': score,
                'Model': model2_name
            } for score in result.topic_scores_model2])
        
        df = pd.DataFrame(data)
        
        sns.boxplot(x='Metric', y='Score', hue='Model', data=df)
        plt.title('Score Distribution by Metric and Model')
        plt.xticks(rotation=45)
        
        # Add significance markers
        for i, result in enumerate(results):
            max_val = max(max(result.topic_scores_model1), max(result.topic_scores_model2))
            if result.significant_t_test:
                plt.text(i, max_val + 0.02, '*', ha='center', fontsize=12)
            if result.significant_wilcoxon:
                plt.text(i, max_val + 0.04, '†', ha='center', fontsize=12)
        
        plt.figtext(0.02, 0.02, '* p < 0.05 (t-test)\n† p < 0.05 (Wilcoxon)', fontsize=8)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison_boxplot.png')
        plt.close()

    def create_summary_table(self, results: List[SignificanceResult], 
                           model1_name: str, model2_name: str):
        """Create and save summary table."""
        rows = []
        for result in results:
            row = {
                'Metric': result.metric,
                f'{model1_name} Mean': f"{result.model1_mean:.4f}",
                f'{model2_name} Mean': f"{result.model2_mean:.4f}",
                'Difference': f"{result.difference:.4f}",
                'Improvement %': f"{result.improvement_percentage:.2f}%",
                't-test p-value': f"{result.t_p_value:.4f}",
                'Wilcoxon p-value': f"{result.wilcoxon_p_value:.4f}",
                'Significant': '✓' if result.significant_t_test or result.significant_wilcoxon else '✗'
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'significance_test_results.csv', index=False)
        
        html = df.style\
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#f0f0f0')]},
                {'selector': 'td', 'props': [('padding', '8px')]}
            ])\
            .highlight_max(subset=[f'{model1_name} Mean', f'{model2_name} Mean'], color='lightgreen')\
            .to_html()
        
        with open(self.output_dir / 'significance_test_results.html', 'w') as f:
            f.write(html)

def main():
    parser = argparse.ArgumentParser(description='Perform significance testing between two IR models')
    parser.add_argument('--qrels', 
                       default='qrel_1.tsv',
                       help='Path to QREL file')
    parser.add_argument('--results-dir', 
                       default='results',
                       help='Directory containing result files')
    parser.add_argument('--model2', 
                       default='result_bi_1.tsv',
                       help='First model result file')
    parser.add_argument('--model1', 
                       default='result_advanced_bm25_1.tsv',
                       help='Second model result file')
    parser.add_argument('--model2-name', 
                       default='Bi-encoder',
                       help='Name of first model')
    parser.add_argument('--model1-name', 
                       default='BM25',
                       help='Name of second model')
    parser.add_argument('--output-dir', 
                       default='significance_results', 
                       help='Directory for output files')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ModelComparator(args.qrels, args.results_dir, args.output_dir)
    
    # Load results
    results1 = comparator._load_run_file(Path(args.results_dir) / args.model1)
    results2 = comparator._load_run_file(Path(args.results_dir) / args.model2)
    
    # Perform significance tests for all metrics
    test_results = []
    for metric in comparator.metrics.keys():
        result = comparator.perform_significance_test(results1, results2, metric)
        test_results.append(result)
        
        # Log results
        logger.info(f"\nResults for {metric}:")
        logger.info(f"{args.model1_name} mean: {result.model1_mean:.4f}")
        logger.info(f"{args.model2_name} mean: {result.model2_mean:.4f}")
        logger.info(f"Improvement: {result.improvement_percentage:.2f}%")
        logger.info(f"t-test p-value: {result.t_p_value:.4f}")
        logger.info(f"Wilcoxon p-value: {result.wilcoxon_p_value:.4f}")
    
    # Create visualizations and reports
    comparator.plot_boxplot_comparison(test_results, args.model1_name, args.model2_name)
    comparator.create_summary_table(test_results, args.model1_name, args.model2_name)
    
    logger.info(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()