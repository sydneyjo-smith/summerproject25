# Summer Project 25 – Ocular Image Preprocessing Pipelines

This repository contains a modular framework for preprocessing ocular images to support AI-based classification of infectious eye diseases. Pipelines are designed to improve image quality while tracking runtime performance, enabling both reproducibility and deployment assessment in low-resource settings.

---

## Project Structure

- **`pipelines/`** – Complete preprocessing pipelines (baseline `p0` and experimental variants `p2–p13`).  
  - Each pipeline includes an embedded timer to log runtime performance.  

- **`src/`** – Individual preprocessing modules (contrast enhancement, filtering, cropping, denoising, etc.).  
  - Each module can be unit tested by appending the `individual_tests.py` code and running it in the terminal.  

- **`analysis/`** – Evaluation and statistical analysis scripts.  
  - `image_tests/` – Metric tests applied to processed outputs.  
  - `friedman_all.py` – Friedman test across pipelines.  
  - `friedman_test_relative_sharpness.py` – Specialized test for relative sharpness.  
  - `boxplot.py` – Generates dissertation-ready boxplots.  

- **`data/`**  
  - `raw_images/` – Place unprocessed image datasets here before running pipelines.  
  - `processed_images/` – Pipeline outputs are stored here for downstream testing and analysis.  

- **`Pipeline_Images/`**  
  - `metric_tests/` – Intermediate test results (including Excel summaries).  
  - `figures_dissertation/` – Final boxplots and figures for reporting.  

---

## Usage

1. **Prepare your environment**
   ```bash
   pip install -r requirements.txt
2. **Run a Pipeline**
   place raw image into data/raw_images
3. **Run evaluations**
   with boxplot.py, image_tests.ipynb, etc.
