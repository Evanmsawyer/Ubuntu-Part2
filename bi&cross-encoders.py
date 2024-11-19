import os
import json
import csv
import torch
import math
import time
import string
import random
import logging
import argparse
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import (
    SentenceTransformer, 
    CrossEncoder, 
    InputExample, 
    losses,
    util,
    evaluation,
    SentencesDataset
)
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import ranx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader as TorchDataLoader
import re
import pickle
import multiprocessing
from functools import partial
from typing import Dict, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Topic:
    id: str
    title: str
    body: str
    tags: List[str]

    def get_query_text(self) -> str:
        """Get formatted query text for the topic."""
        return f"[TITLE]{self.title}[BODY]{self.body}"

@dataclass
class Document:
    id: str
    text: str
    score: int

def clean_text(text: str) -> str:
    """Clean text by removing special characters, extra whitespace, and normalizing."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    return text.strip()

class DataManager:
    @staticmethod
    def read_collection(filepath: str) -> Dict[str, Document]:
        """Read collection from JSON file with robust encoding handling."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_docs = json.load(f)
            
            documents = {}
            for doc in raw_docs:
                documents[doc['Id']] = Document(
                    id=doc['Id'],
                    text=doc['Text'],
                    score=int(doc['Score'])
                )
            return documents
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                raw_docs = json.load(f)
            
            documents = {}
            for doc in raw_docs:
                documents[doc['Id']] = Document(
                    id=doc['Id'],
                    text=doc['Text'],
                    score=int(doc['Score'])
                )
            return documents

    @staticmethod
    def load_topics(filepath: str) -> Dict[str, Topic]:
        """Load and preprocess topics from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_topics = json.load(f)
        
        topics = {}
        for item in raw_topics:
            title = item['Title'].translate(str.maketrans('', '', string.punctuation))
            body = item['Body'].translate(str.maketrans('', '', string.punctuation))
            tags = eval(item['Tags'])
            
            topics[item['Id']] = Topic(
                id=item['Id'],
                title=title,
                body=body,
                tags=tags
            )
        return topics

    @staticmethod
    def read_qrels(filepath: str) -> Dict[str, Dict[str, int]]:
        """Read and parse QREL file."""
        qrels = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t', lineterminator='\n')
            for line in reader:
                query_id, _, doc_id, score = line
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = int(score)
        return qrels

class BiEncoder:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 batch_size: int = 32, 
                 token: str = "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"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, token=token)
        self.model.to(self.device)
        self.batch_size = batch_size

    def encode_corpus(self, collection: Dict[str, Document]) -> Dict[str, torch.Tensor]:
        """Encode entire document collection with optimized batching."""
        logger.info("Encoding document collection...")
        doc_texts = []
        doc_ids = []
        
        for doc_id, doc in collection.items():
            doc_texts.append(doc.text)
            doc_ids.append(doc_id)
        
        embeddings = self.model.encode(
            doc_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True  
        )
        
        return {doc_id: embedding for doc_id, embedding in zip(doc_ids, embeddings)}

    def search(self, query: str, doc_embeddings: Dict[str, torch.Tensor], top_k: int = 100) -> Dict[str, float]:
        """Optimized search using pre-normalized embeddings."""
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True  
        )
        
        doc_ids = list(doc_embeddings.keys())
        doc_embeddings_tensor = torch.stack([doc_embeddings[doc_id] for doc_id in doc_ids])
        
        similarities = torch.mm(query_embedding.unsqueeze(0), doc_embeddings_tensor.t()).squeeze(0)
        
        top_k_scores, top_k_indices = torch.topk(similarities, min(top_k, len(doc_ids)))
        
        results = {
            doc_ids[idx]: score.item() 
            for idx, score in zip(top_k_indices.tolist(), top_k_scores.tolist())
        }
        
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    def prepare_training_data(self, topics: Dict[str, Topic], 
                            qrels: Dict[str, Dict[str, int]],
                            collection: Dict[str, Document]) -> Tuple[List[InputExample], List[InputExample]]:
        """Prepare data for model fine-tuning with correct InputExample format."""
        train_samples = []
        eval_samples = []
        
        for topic_id, topic_qrels in qrels.items():
            if topic_id not in topics:
                continue
                
            topic = topics[topic_id]
            query = topic.get_query_text()
            
            # Create positive and negative pairs
            for doc_id, relevance in topic_qrels.items():
                if doc_id not in collection:
                    continue
                    
                doc = collection[doc_id]
                if relevance >= 2:  # Highly relevant
                    sample = InputExample(texts=[query, doc.text], label=1.0)
                    train_samples.append(sample)
                    eval_samples.append(sample)
                elif relevance == 0:  # Non-relevant
                    sample = InputExample(texts=[query, doc.text], label=0.0)
                    train_samples.append(sample)
                    eval_samples.append(sample)
        
        return train_samples, eval_samples

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for InputExample objects."""
        texts1 = []
        texts2 = []
        labels = []
        
        for example in batch:
            texts1.append(example.texts[0])
            texts2.append(example.texts[1])
            labels.append(example.label)
            
        return [texts1, texts2], torch.tensor(labels, dtype=torch.float)

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts with gradient tracking."""
        features = self.model.tokenize(texts)
        features = {k: v.to(self.device) for k, v in features.items()}
        
        # Get embeddings from the model with gradient tracking
        with torch.set_grad_enabled(True):
            embeddings = self.model(features)['sentence_embedding']
        return embeddings

    def fine_tune(self, train_samples: List[InputExample], eval_samples: List[InputExample],
                 num_epochs: int = 1) -> None:
        """Fine-tune the bi-encoder model with proper gradient handling."""
        logger.info(f"Fine-tuning bi-encoder for {num_epochs} epochs...")
        
        os.makedirs("models/bi_encoder_ft", exist_ok=True)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        train_dataloader = DataLoader(
            train_samples, 
            shuffle=True, 
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )
        
        warmup_steps = int(len(train_dataloader) * 0.1)
        
        self.model.train()
        for epoch in range(num_epochs):
            training_steps = 0
            total_loss = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch_texts, labels in progress_bar:
                optimizer.zero_grad()
                
                texts1, texts2 = batch_texts
                embeddings1 = self.encode_batch(texts1)
                embeddings2 = self.encode_batch(texts2)
                
                scores = torch.sum(embeddings1 * embeddings2, dim=1)
                
                labels = labels.to(self.device)
                loss = torch.nn.MSELoss()(scores, labels)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                training_steps += 1
                if training_steps < warmup_steps:
                    lr_scale = min(1.0, float(training_steps) / float(warmup_steps))
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr_scale * 2e-5
                
                total_loss += loss.item()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")
        
        # Save the model
        self.model.save("models/bi_encoder_ft")
        logger.info("Model saved successfully")

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
                 batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CrossEncoder(model_name, device=self.device)
        
        special_tokens = ["[TITLE]", "[BODY]"]
        self.model.tokenizer.add_tokens(special_tokens, special_tokens=True)
        self.model.model.resize_token_embeddings(len(self.model.tokenizer))


    def rerank(self, query: str, doc_scores: Dict[str, float], 
               collection: Dict[str, Document]) -> Dict[str, float]:
        """Rerank documents using cross-encoder."""
        pairs = [[query, collection[doc_id].text] for doc_id in doc_scores.keys()]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        reranked = {
            doc_id: float(score) 
            for doc_id, score in zip(doc_scores.keys(), scores)
        }
        return dict(sorted(reranked.items(), key=lambda x: x[1], reverse=True))

    def prepare_training_data(self, topics: Dict[str, Topic],
                            qrels: Dict[str, Dict[str, int]],
                            collection: Dict[str, Document]) -> Tuple[List[InputExample], Dict]:
        """Prepare training data for cross-encoder with proper InputExample format."""
        train_samples = []
        validation_samples = {}
        
        for topic_id, topic_qrels in qrels.items():
            if topic_id not in topics:
                continue
                
            topic = topics[topic_id]
            query = topic.get_query_text()
            
            if topic_id not in validation_samples:
                validation_samples[topic_id] = {
                    'query': query,
                    'positive': [], 
                    'negative': []   
                }
            
            for doc_id, relevance in topic_qrels.items():
                if doc_id not in collection:
                    continue
                    
                doc = collection[doc_id]
                label = 1.0 if relevance >= 1 else 0.0
                
                train_samples.append(InputExample(texts=[query, doc.text], label=label))
                
                if label == 1.0:
                    validation_samples[topic_id]['positive'].append(doc.text)
                else:
                    validation_samples[topic_id]['negative'].append(doc.text)
        
        return train_samples, validation_samples


    def fine_tune(self, train_samples: List[InputExample], validation_samples: Dict,
                 num_epochs: int = 1) -> None:
        """Fine-tune the cross-encoder model."""
        logger.info(f"Fine-tuning cross-encoder for {num_epochs} epochs...")
        
        os.makedirs("models/cross_encoder_ft", exist_ok=True)

        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=self.batch_size)
        
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
        
        #Used built in loss function for cross-encoder, bi-encoder had to use custom loss function
        self.model.fit(
            train_dataloader=train_dataloader,
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            output_path="models/cross_encoder_ft",
            save_best_model=True,
            show_progress_bar=True
        )
        
        self.model.save("models/cross_encoder_ft")
        logger.info("Model saved successfully")

class CachedBiEncoder(BiEncoder):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 batch_size: int = 32, 
                 token: str = "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",
                 cache_dir: Optional[str] = None):
        super().__init__(model_name, batch_size, token)
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def encode_corpus(self, collection: Dict[str, Document]) -> Dict[str, torch.Tensor]:
        """Encode corpus with parallel text cleaning."""
        if not self.cache_dir:
            return super().encode_corpus(collection)
            
        cache_file = os.path.join(self.cache_dir, "doc_embeddings.pt")
        
        try:
            if os.path.exists(cache_file):
                logger.info("Loading cached document embeddings...")
                return torch.load(cache_file, map_location=self.device)
            
            logger.info("Computing document embeddings...")
            doc_texts = []
            doc_ids = []
            
            # Clean texts in parallel
            with multiprocessing.Pool() as pool:
                texts_to_clean = [doc.text for doc in collection.values()]
                cleaned_texts = pool.map(clean_text, texts_to_clean)
            
            for doc_id, cleaned_text in zip(collection.keys(), cleaned_texts):
                doc_texts.append(cleaned_text)
                doc_ids.append(doc_id)
            
            # Use automatic mixed precision for encoding
            with torch.cuda.amp.autocast():
                embeddings = self.model.encode(
                    doc_texts,
                    batch_size=256,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                    device=self.device,
                    normalize_embeddings=True
                )
            
            result = {doc_id: embedding for doc_id, embedding in zip(doc_ids, embeddings)}
            
            try:
                torch.save(result, cache_file)
                logger.info(f"Saved embeddings cache to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during document encoding: {str(e)}")
            return super().encode_corpus(collection)
    
def process_cross_encoder_batch(args):
    """Process a batch of documents with cross-encoder (for multiprocessing)."""
    query, doc_pairs, model, device = args
    model.to(device)
    with torch.no_grad():
        scores = model.predict(doc_pairs, batch_size=32)
    return scores.tolist()

class ParallelCrossEncoder(CrossEncoderReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
                 batch_size: int = 16):
        super().__init__(model_name, batch_size)

    def process_batch(self, args):
        """Process a batch of documents with cross-encoder."""
        query, pairs, device = args
        with torch.no_grad():
            scores = self.model.predict(pairs, batch_size=32)
        return scores.tolist()

    def rerank(self, query: str, doc_scores: Dict[str, float], 
               collection: Dict[str, Document]) -> Dict[str, float]:
        """Rerank documents using parallel processing."""
        query = clean_text(query)
        
        pairs = [[query, clean_text(collection[doc_id].text)] 
                for doc_id in doc_scores.keys()]
        
        if len(pairs) <= 100:
            with torch.no_grad():
                scores = self.model.predict(pairs, batch_size=32)
            reranked = {
                doc_id: float(score)
                for doc_id, score in zip(doc_scores.keys(), scores)
            }
            return dict(sorted(reranked.items(), key=lambda x: x[1], reverse=True))
        
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        batch_size = math.ceil(len(pairs) / num_cores)
        batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
        
        try:
            process_args = [
                (query, batch, self.device)
                for batch in batches
            ]
            
            # Process in parallel
            with multiprocessing.Pool(num_cores) as pool:
                results = pool.map(self.process_batch, process_args)
            
            all_scores = []
            for batch_scores in results:
                all_scores.extend(batch_scores)
            
            reranked = {
                doc_id: float(score)
                for doc_id, score in zip(doc_scores.keys(), all_scores)
            }
            return dict(sorted(reranked.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.warning(f"Parallel processing failed: {str(e)}. Falling back to sequential processing.")
            # Fallback to sequential processing
            with torch.no_grad():
                scores = self.model.predict(pairs, batch_size=32)
            reranked = {
                doc_id: float(score)
                for doc_id, score in zip(doc_scores.keys(), scores)
            }
            return dict(sorted(reranked.items(), key=lambda x: x[1], reverse=True))


class Evaluator:
    @staticmethod
    def evaluate_results(qrels: Dict[str, Dict[str, int]], 
                        results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Evaluate retrieval results."""
        metrics = [
            "map",
            "mrr",
            "ndcg@5",
            "precision@1",
            "precision@5"
        ]
        
        return ranx.evaluate(
            ranx.Qrels.from_dict(qrels),
            ranx.Run.from_dict(results),
            metrics
        )

    @staticmethod
    def plot_metrics(results: Dict[str, float], title: str, output_file: str):
        """Create visualization of evaluation metrics."""
        plt.figure(figsize=(10, 6))
        metrics = list(results.keys())
        scores = list(results.values())
        
        plt.bar(metrics, scores, color='skyblue')
        plt.title(title)
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        for i, score in enumerate(scores):
            plt.text(i, score + 0.02, f'{score:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    @staticmethod
    def plot_precision_at_k(qrels: Dict[str, Dict[str, int]], 
                           results: Dict[str, Dict[str, float]], 
                           k: int, title: str, output_file: str,
                           max_topics: int = 30):
        """Create precision@k plot for individual topics."""
        topic_precision = {}
        
        for topic_id in results:
            if topic_id not in qrels:
                continue
                
            relevant = set(doc_id for doc_id, rel in qrels[topic_id].items() if rel > 0)
            retrieved = list(results[topic_id].keys())[:k]
            relevant_retrieved = len([doc for doc in retrieved if doc in relevant])
            
            topic_precision[topic_id] = relevant_retrieved / k
        
        sorted_topics = sorted(
            topic_precision.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_topics]
        
        plt.figure(figsize=(15, 6))
        topics, precisions = zip(*sorted_topics)
        
        plt.bar(range(len(topics)), precisions, color='skyblue')
        plt.title(title)
        plt.xlabel('Topic ID')
        plt.ylabel(f'Precision@{k}')
        plt.xticks(range(len(topics)), topics, rotation=90)
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

class ResultWriter:
    @staticmethod
    def write_results(results: Dict[str, Dict[str, float]], output_file: str):
        """Write results in TREC format."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for topic_id, doc_scores in results.items():
                for rank, (doc_id, score) in enumerate(doc_scores.items(), 1):
                    f.write(f"{topic_id}\tQ0\t{doc_id}\t{rank}\t{score}\tBERT\n")

def process_single_run(topics_file: str, collection: Dict[str, Document],
                      bi_encoder: CachedBiEncoder, cross_encoder: ParallelCrossEncoder,
                      qrels: Optional[Dict[str, Dict[str, int]]] = None,
                      is_finetuned: bool = False,
                      output_dir: str = "results") -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Optimized process_single_run with progress tracking."""
    
    logger.info(f"Starting processing of {topics_file}")
    topics = DataManager.load_topics(topics_file)
    topic_num = "1" if "topics_1" in topics_file else "2"
    model_suffix = "ft" if is_finetuned else ""
    
    logger.info("Loading document embeddings...")
    doc_embeddings = bi_encoder.encode_corpus(collection)
    logger.info("Converting embeddings to tensor...")
    
    # Convert doc embeddings to tensor once and move to GPU
    doc_ids = list(doc_embeddings.keys())
    doc_embeddings_tensor = torch.stack([doc_embeddings[doc_id] for doc_id in doc_ids]).to(bi_encoder.device)
    
    logger.info(f"Processing {len(topics)} queries...")
    all_queries = []
    topic_ids = []
    
    with multiprocessing.Pool() as pool:
        cleaned_queries = pool.map(clean_text, [topic.get_query_text() for topic in topics.values()])
    
    for topic_id, cleaned_query in zip(topics.keys(), cleaned_queries):
        all_queries.append(cleaned_query)
        topic_ids.append(topic_id)
    
    # Encode all queries in one batch
    logger.info("Encoding queries...")
    with torch.no_grad(), torch.cuda.amp.autocast():  
        query_embeddings = bi_encoder.model.encode(
            all_queries,
            batch_size=len(all_queries),
            show_progress_bar=True,
            convert_to_tensor=True,
            device=bi_encoder.device,
            normalize_embeddings=True
        )
        
        logger.info("Computing similarities...")
        similarities = torch.mm(query_embeddings, doc_embeddings_tensor.t())
        top_k_values, top_k_indices = torch.topk(similarities, k=100)
    
    logger.info("Processing bi-encoder results...")
    bi_results = {}
    ce_results = {}
    
    all_pairs = []
    pair_mappings = []
    
    for i, topic_id in enumerate(topic_ids):
        topic_scores = {
            doc_ids[idx.item()]: score.item()
            for idx, score in zip(top_k_indices[i], top_k_values[i])
        }
        bi_results[topic_id] = dict(sorted(topic_scores.items(), key=lambda x: x[1], reverse=True))
        
        pairs = [[all_queries[i], clean_text(collection[doc_id].text)] 
                for doc_id in bi_results[topic_id].keys()]
        all_pairs.extend(pairs)
        pair_mappings.extend([(topic_id, doc_id) for doc_id in bi_results[topic_id].keys()])
    
    if cross_encoder is not None:
        logger.info("Processing cross-encoder reranking...")
        CROSS_ENCODER_BATCH_SIZE = 1024 
        all_scores = []
        
        total_batches = math.ceil(len(all_pairs) / CROSS_ENCODER_BATCH_SIZE)
        
        with tqdm(total=total_batches, desc="Cross-encoder batches") as pbar:
            for i in range(0, len(all_pairs), CROSS_ENCODER_BATCH_SIZE):
                batch_pairs = all_pairs[i:i + CROSS_ENCODER_BATCH_SIZE]
                with torch.no_grad(), torch.cuda.amp.autocast():
                    batch_scores = cross_encoder.model.predict(
                        batch_pairs,
                        batch_size=CROSS_ENCODER_BATCH_SIZE,
                        show_progress_bar=False
                    )
                    all_scores.extend(batch_scores)
                pbar.update(1)
        
        logger.info("Organizing cross-encoder results...")
        ce_results = {}
        for (topic_id, doc_id), score in zip(pair_mappings, all_scores):
            if topic_id not in ce_results:
                ce_results[topic_id] = {}
            ce_results[topic_id][doc_id] = float(score)
        
        for topic_id in ce_results:
            ce_results[topic_id] = dict(sorted(ce_results[topic_id].items(), key=lambda x: x[1], reverse=True))
    
    logger.info("Writing results...")
    ResultWriter.write_results(bi_results, os.path.join(output_dir, f"result_bi_{model_suffix}_{topic_num}.tsv"))
    if ce_results:
        ResultWriter.write_results(ce_results, os.path.join(output_dir, f"result_ce_{model_suffix}_{topic_num}.tsv"))
    
    if qrels:
        logger.info("Evaluating results...")
        evaluate_and_plot_results(bi_results, ce_results, qrels, topics_file, is_finetuned)
    
    logger.info(f"Completed processing of {topics_file}")
    return bi_results, ce_results

def evaluate_and_plot_results(bi_results: Dict[str, Dict[str, float]],
                            ce_results: Dict[str, Dict[str, float]],
                            qrels: Dict[str, Dict[str, int]],
                            topics_file: str,
                            is_finetuned: bool):
    """Evaluate and create visualizations for results."""
    
    model_type = "Fine-tuned" if is_finetuned else "Base"
    
    bi_eval = Evaluator.evaluate_results(qrels, bi_results)
    logger.info(f"\nBi-encoder ({model_type}) evaluation results:")
    for metric, score in bi_eval.items():
        logger.info(f"{metric}: {score:.4f}")
    
    ce_eval = Evaluator.evaluate_results(qrels, ce_results)
    logger.info(f"\nCross-encoder ({model_type}) evaluation results:")
    for metric, score in ce_eval.items():
        logger.info(f"{metric}: {score:.4f}")
    
    base_name = os.path.splitext(os.path.basename(topics_file))[0]
    os.makedirs("plots", exist_ok=True)
    
    Evaluator.plot_metrics(
        bi_eval,
        f"Bi-encoder {model_type} Evaluation",
        f"plots/bi_encoder_{model_type.lower()}_{base_name}_metrics.png"
    )
    
    Evaluator.plot_metrics(
        ce_eval,
        f"Cross-encoder {model_type} Evaluation",
        f"plots/cross_encoder_{model_type.lower()}_{base_name}_metrics.png"
    )
    
    max_topics = 30 if "topics_2" in topics_file else None
    
    Evaluator.plot_precision_at_k(
        qrels, bi_results, k=5,
        title=f"Bi-encoder {model_type} Precision@5 by Topic",
        output_file=f"plots/bi_encoder_{model_type.lower()}_{base_name}_precision.png",
        max_topics=max_topics
    )
    
    Evaluator.plot_precision_at_k(
        qrels, ce_results, k=5,
        title=f"Cross-encoder {model_type} Precision@5 by Topic",
        output_file=f"plots/cross_encoder_{model_type.lower()}_{base_name}_precision.png",
        max_topics=max_topics
    )

def fine_tune_models(bi_encoder: BiEncoder, cross_encoder: CrossEncoderReranker,
                    topics: Dict[str, Topic], qrels: Dict[str, Dict[str, int]],
                    collection: Dict[str, Document], num_epochs: int = 5):
    """Fine-tune both bi-encoder and cross-encoder models with improved error handling."""
    
    logger.info("Preparing training data...")
    
    try:
        bi_train_samples, bi_eval_samples = bi_encoder.prepare_training_data(
            topics, qrels, collection
        )
        
        ce_train_samples, ce_validation_samples = cross_encoder.prepare_training_data(
            topics, qrels, collection
        )
        
        logger.info(f"Prepared {len(bi_train_samples)} bi-encoder training samples")
        logger.info(f"Prepared {len(ce_train_samples)} cross-encoder training samples")
        
        os.makedirs("models/bi_encoder_ft/checkpoints", exist_ok=True)
        os.makedirs("models/cross_encoder_ft/checkpoints", exist_ok=True)
        
        try:
            bi_encoder.fine_tune(
                bi_train_samples,
                bi_eval_samples,
                num_epochs=num_epochs  
            )
            logger.info("Bi-encoder fine-tuning completed successfully")
            
            cross_encoder.fine_tune(
               ce_train_samples,
               ce_validation_samples,
               num_epochs=num_epochs  
            )
            logger.info("Cross-encoder fine-tuning completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model fine-tuning: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")
        raise

def cleanup_model_directories():
    """Clean up model directories before training."""
    import shutil
    
    dirs = [
        "models/bi_encoder_ft",
        "models/cross_encoder_ft",
        "models/bi_encoder_ft/checkpoints",
        "models/cross_encoder_ft/checkpoints"
    ]
    
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def optimize_training_config(bi_encoder, cross_encoder):
    """Optimize training configuration for both encoders."""
    from torch.cuda.amp import autocast
    num_epochs = 10
    
    bi_encoder.batch_size = 128
    cross_encoder.batch_size = 64

    bi_encoder.model.gradient_checkpointing_enable()
    
    if hasattr(cross_encoder.model, 'model'):
        cross_encoder.model.model.gradient_checkpointing_enable()
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if torch.cuda.get_device_capability()[0] >= 7:
            torch.backends.cuda.matmul.allow_tf32 = True
    
    return num_epochs

def main():
    parser = argparse.ArgumentParser(description='Advanced Information Retrieval System')
    parser.add_argument('--answers_file', default='Answers.json')
    parser.add_argument('--topics_files', nargs='+', default=['topics_1.json', 'topics_2.json'])
    parser.add_argument('--qrels_file', default='qrel_1.tsv')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--no-cache', action='store_true', help='Disable embedding caching')
    parser.add_argument('--cache-dir', type=str, default='cache', help='Directory for cached embeddings')
    args = parser.parse_args()

    start_time = time.time()
    
    # Configure GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if torch.cuda.get_device_capability()[0] >= 7:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
 
    # Create necessary directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    if not args.no_cache:
        os.makedirs(args.cache_dir, exist_ok=True)
    cleanup_model_directories()
    
    # Configure logging
    logger.info("Loading data...")
    try:
        collection = DataManager.read_collection(args.answers_file)
        qrels = DataManager.read_qrels(args.qrels_file) if args.qrels_file else None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
    
    # Initialize models with optimized settings
    logger.info("Initializing models...")
    try:
        bi_encoder = CachedBiEncoder(
            batch_size=args.batch_size,
            cache_dir=None if args.no_cache else args.cache_dir
        )
        cross_encoder = ParallelCrossEncoder(batch_size=args.batch_size)
        
        # Optimize training configuration
        num_epochs = optimize_training_config(bi_encoder, cross_encoder)
        
        # Process with base models
        logger.info("Processing with base models...")
        for topics_file in args.topics_files:
            logger.info(f"\nProcessing {topics_file}")
            try:
                process_single_run(
                    topics_file,
                    collection,
                    bi_encoder,
                    cross_encoder,
                    qrels if "topics_1" in topics_file else None,
                    is_finetuned=False,
                    output_dir=args.output_dir
                )
            except Exception as e:
                logger.error(f"Error processing {topics_file}: {str(e)}")
                continue

        # Fine-tune models if applicable
        if "topics_1.json" in args.topics_files and qrels:
            logger.info("\nFine-tuning models...")
            try:
                topics = DataManager.load_topics("topics_1.json")
                fine_tune_models(
                    bi_encoder,
                    cross_encoder,
                    topics,
                    qrels,
                    collection,
                    num_epochs=num_epochs
                )
                
                # Process with fine-tuned models
                logger.info("\nProcessing with fine-tuned models...")
                for topics_file in args.topics_files:
                    logger.info(f"\nProcessing {topics_file}")
                    try:
                        process_single_run(
                            topics_file,
                            collection,
                            bi_encoder,
                            cross_encoder,
                            qrels if "topics_1" in topics_file else None,
                            is_finetuned=True,
                            output_dir=args.output_dir
                        )
                    except Exception as e:
                        logger.error(f"Error processing fine-tuned {topics_file}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error during fine-tuning: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
    
    finally:
        # Clean up and report
        end_time = time.time()
        logger.info(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
        
        # Clean up cache if requested
        if args.no_cache and os.path.exists(args.cache_dir):
            import shutil
            shutil.rmtree(args.cache_dir)

if __name__ == "__main__":
    main()