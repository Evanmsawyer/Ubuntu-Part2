# Advanced Information Retrieval System

A Python-based information retrieval system that implements a bi-encoder/cross-encoder architecture for efficient document retrieval and reranking. The system utilizes BERT-based models from the Sentence Transformers library and supports model fine-tuning for improved performance.

## Features

- **Dual Encoder Architecture**\
  - Bi-encoder for efficient initial retrieval\
  - Cross-encoder for high-quality reranking\
  - Support for model fine-tuning

- **Performance Optimizations**\
  - GPU acceleration with CUDA support\
  - Parallel processing for large document collections\
  - Embedding caching system\
  - Automatic mixed precision (AMP) training\
  - Memory-efficient batch processing

- **Evaluation & Visualization**\
  - Comprehensive evaluation metrics (MAP, MRR, NDCG@5, P@1, P@5)\
  - Automated performance visualization\
  - Per-topic precision analysis\
  - TREC-format result output

## Prerequisites

```bash\
pip install torch sentence-transformers ranx matplotlib tqdm sklearn\
```

## Usage

### Basic Usage

```bash\
python bi&cross-encoders.py --answers_file Answers.json --topics_files topics_1.json topics_2.json --qrels_file qrel_1.tsv\
```

### Command Line Arguments

- `--answers_file`: Path to the document collection JSON file (default: 'Answers.json')\
- `--topics_files`: List of topic files to process (default: ['topics_1.json', 'topics_2.json'])\
- `--qrels_file`: Path to relevance judgments file (default: 'qrel_1.tsv')\
- `--output_dir`: Directory for output files (default: 'results')\
- `--batch_size`: Batch size for model inference (default: 64)\
- `--no-cache`: Disable embedding caching\
- `--cache-dir`: Directory for cached embeddings (default: 'cache')

## Input File Formats

### Document Collection (Answers.json)\
```json\
[\
    {\
        "Id": "doc_id",\
        "Text": "document text",\
        "Score": 0\
    }\
]\
```

### Topics File (topics_*.json)\
```json\
[\
    {\
        "Id": "topic_id",\
        "Title": "topic title",\
        "Body": "topic description",\
        "Tags": "['tag1', 'tag2']"\
    }\
]\
```

### Relevance Judgments (qrel_*.tsv)\
```\
topic_id    0    doc_id    relevance_score\
```

## Output

The system generates several types of output:

1\. **Results Files**\
   - `results/result_bi_*.tsv`: Bi-encoder retrieval results\
   - `results/result_ce_*.tsv`: Cross-encoder reranking results

2\. **Evaluation Plots**\
   - `plots/bi_encoder_*.png`: Bi-encoder performance metrics\
   - `plots/cross_encoder_*.png`: Cross-encoder performance metrics

3\. **Model Checkpoints**\
   - `models/bi_encoder_ft/`: Fine-tuned bi-encoder model\
   - `models/cross_encoder_ft/`: Fine-tuned cross-encoder model

## Key Components

### DataManager\
- Handles data loading and preprocessing\
- Supports robust encoding handling\
- Implements efficient data cleaning

### BiEncoder\
- Initial retrieval model\
- Supports embedding caching\
- Implements efficient batch processing

### CrossEncoderReranker\
- High-precision reranking\
- Parallel processing support\
- Optimized for large document collections

### Evaluator\
- Comprehensive metric calculation\
- Automated visualization generation\
- Per-topic performance analysis

## Performance Optimization Tips

1\. Enable GPU acceleration when available\
2\. Use embedding caching for large collections\
3\. Adjust batch sizes based on available memory\
4\. Enable parallel processing for large datasets

## Error Handling

The system implements robust error handling:\
- Graceful fallback for parallel processing\
- Automatic cache recovery\
- Comprehensive logging system

## Logging

The system uses Python's logging module with INFO level by default. Logs include:\
- Data loading progress\
- Model initialization status\
- Training progress\
- Error messages\
- Performance metrics

## License

This project is available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.