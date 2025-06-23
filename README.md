# Amharic E-commerce Data Extractor

Transform messy Telegram posts into a smart FinTech engine that reveals which vendors are the best candidates for a loan.

## Business Need

EthioMart has a vision to become the primary hub for all Telegram-based e-commerce activities in Ethiopia. This project focuses on fine-tuning LLM's for Amharic Named Entity Recognition (NER) system that extracts key business entities such as product names, prices, and locations from text, images, and documents shared across Telegram channels.

## Key Objectives

1. Develop a repeatable workflow for data ingestion from Telegram channels
2. Fine-tune a transformer-based model for high accuracy Amharic NER
3. Extract Product, Price, and Location entities from unstructured Amharic text
4. Compare multiple approaches and deliver model recommendations

## Project Structure

```
├── data/
│   ├── raw/           # Raw Telegram data
│   ├── processed/     # Preprocessed data
│   └── labeled/       # CoNLL format labeled data
├── src/
│   ├── data_ingestion/    # Telegram scraping
│   ├── preprocessing/     # Text processing
│   ├── labeling/         # CoNLL labeling tools
│   └── utils/           # Utility functions
├── config/              # Configuration files
├── logs/               # Application logs
├── notebook/           # Jupyter notebooks
└── tests/             # Unit tests
```

## Entity Types

- **Product Names or Types**: Items being sold
- **Location Mentions**: Geographic locations
- **Monetary Values or Prices**: Pricing information
- **Optional**: Delivery fees, Contact information

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Task 1: Data Ingestion and Preprocessing
```bash
python -m src.data_ingestion.telegram_scraper
```

### Task 2: Data Labeling
```bash
python -m src.labeling.conll_labeler
```

## Branches

- `main`: Main development branch
- `task-1`: Data ingestion and preprocessing
- `task-2`: Data labeling in CoNLL format