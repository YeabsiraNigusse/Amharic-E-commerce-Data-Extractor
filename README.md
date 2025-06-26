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

## Task Status

- ✅ **Task 1**: Data ingestion and preprocessing (COMPLETED)
- ✅ **Task 2**: Data labeling in CoNLL format (COMPLETED)
- ✅ **Task 3**: Fine Tune NER Model (COMPLETED)
- ✅ **Task 4**: Model Comparison & Selection (COMPLETED)
- ✅ **Task 5**: Model interpretability with SHAP and LIME (COMPLETED)
- ✅ **Task 6**: Vendor scorecard for micro-lending (COMPLETED)

## Branches

- `main`: Main development branch
- `task-1`: Data ingestion and preprocessing ✅
- `task-2`: Data labeling in CoNLL format ✅
- `task-3`: Model training and evaluation ✅
- `task-4`: Model deployment and API ✅
- `feature/tasks-5-6-implementation`: Model interpretability and vendor scorecard ✅

## Quick Start

### Task 1: Data Ingestion and Preprocessing
```bash
# Setup environment
cp .env.example .env
# Add your Telegram API credentials to .env

# Install dependencies
pip install -r requirements.txt

# Run Task 1
python run_task1.py
```

### Task 2: CoNLL Format Labeling
```bash
# Run Task 2
python run_task2.py
# Select option 1 to create sample labeled dataset
```

### Task 3: Fine Tune NER Model
```bash
# Run Task 3 - Fine-tune NER model
python src/models/ner_trainer.py
```

### Task 4: Model Comparison & Selection
```bash
# Run Task 4 - Compare and select best model
python src/models/model_evaluator.py
```

### Task 5: Model Interpretability
```bash
# Run Task 5 - Model Interpretability with SHAP and LIME
python run_task5_interpretability.py
```

### Task 6: Vendor Scorecard for Micro-Lending
```bash
# Run Task 6 - Vendor Analytics and Scorecard Generation
python run_task6_vendor_scorecard.py
```

## Sample Output

The project generates a labeled dataset with 50 Amharic messages in CoNLL format:

```
# Message ID: sample_001
# Text: የሕፃናት ልብስ ዋጋ 500 ብር ነው። በቦሌ አካባቢ ይገኛል።

የሕፃናት	O
ልብስ	O
ዋጋ	B-PRICE
500	B-PRICE
ብር	I-PRICE
ነው	I-PRICE
።	O
በቦሌ	O
አካባቢ	B-LOCATION
ይገኛል	O
።	O
```

## Results

### Task 1 & 2: Data Processing and Labeling
- **50 labeled messages** in CoNLL format
- **445 tokens** with **109 entities** (24.5% coverage)
- **Entity distribution**: PRICE (133), LOCATION (43), CONTACT_INFO (5)
- **Output files**:
  - `data/labeled/amharic_ner_sample_50_messages.txt`
  - `data/labeled/amharic_ner_sample_50_messages.json`

### Task 3 & 4: Fine-Tuning and Model Selection
- **Fine-Tuned NER Model**: XLM-RoBERTa-based transformer optimized for Amharic
- **Model Comparison**: Systematic evaluation of multiple model architectures
- **Best Model Selection**: Data-driven selection based on performance metrics
- **Optimized Performance**: Fine-tuned for Ethiopian e-commerce entity detection

### Task 5: Model Interpretability
- **Trained NER Model**: XLM-RoBERTa-based transformer for Amharic
- **SHAP Analysis**: Global feature importance for entity detection
- **LIME Explanations**: Local interpretability for individual predictions
- **Difficult Cases Analysis**: Identification and explanation of model struggles
- **Output files**:
  - `reports/model_evaluation_report.txt`
  - `reports/interpretability_report.txt`
  - `reports/lime_*_importance.png`
  - `reports/confusion_matrix.png`

### Task 6: Vendor Scorecard for Micro-Lending
- **Vendor Analytics Engine**: Comprehensive business performance analysis
- **Lending Scores**: 0-100 scale based on activity, engagement, and transparency
- **Risk Assessment**: Multi-dimensional vendor evaluation
- **Interactive Dashboards**: HTML visualizations for vendor comparison
- **Output files**:
  - `reports/vendor_scorecards/vendor_scorecard.xlsx`
  - `reports/vendor_scorecards/comprehensive_scorecard_report.txt`
  - `reports/vendor_scorecards/*.html` (visualizations)

## Project Completion

🎉 **ALL 6 TASKS COMPLETED** 🎉

This project represents a complete end-to-end solution for Amharic e-commerce data extraction, from raw data processing to business intelligence and micro-lending support.

For detailed implementation reports, see:
- [FINAL_IMPLEMENTATION_REPORT.md](FINAL_IMPLEMENTATION_REPORT.md)
- [PROJECT_COMPLETION_SUMMARY.md](PROJECT_COMPLETION_SUMMARY.md)
