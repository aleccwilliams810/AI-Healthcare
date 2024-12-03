# Compassionate Cancer Care Assistant

This project explores the application of large language models to create a compassionate virtual assistant for cancer care. The assistant is designed to provide emotional and informational support to families and patients, with a specific focus on interactions involving children. The model integrates reasoning methods like Tree of Thought (ToT) and Chain of Thought (CoT) and leverages Synthea-generated synthetic patient data.

Only a few example functions are implemented.

## Features
- **Emotionally Supportive Responses**: Tailored for cancer care scenarios involving children as patients or as supporters of loved ones.
- **Data Integration**: Utilizes Synthea datasets for enriched responses, including:
  - Common cancer types treated
  - Top medications prescribed
  - Healthcare expenses and coverage data


## Requirements
- **Dependencies**:
  - [Ollama](https://ollama.com/): Used to run local LLMs (e.g., MedLlama2 and TinyLlama).
  - Python libraries: `pandas`, `ollama.Client`

## Datasets
- **Synthea Synthetic Patient Data**: Downloaded from [Synthea Downloads](https://synthea.mitre.org/downloads).
  - Includes cancer care-related data such as medications, procedures, and expenses.

## Models
- **MedLlama2**: Provides nuanced responses and excels in reasoning tasks.
- **TinyLlama**: Faster responses for lightweight interactions but less robust in reasoning.

## How to Run
1. **Download Ollama**:
   - Visit [Ollama](https://ollama.com/) and install the Ollama client.
2. **Pull Models**:
   - Use Ollama to download and serve the **MedLlama2** and **TinyLlama** models.
   - ollama pull medllama2 (/tinyllama)
   - ollama serve
3. **Set Up Environment**:
   - Download the required Synthea datasets and place them in the `csv` directory.
        - Careplans, Medications, Patients 
   - Install the necessary Python libraries: `pandas`, `ollama.Client`.
4. **Interact**:
   - Open the provided Jupyter Notebook file.
   - Interact with functions using string input questions that align with the designed use cases.
5. **Explore**:
   - Test scenarios like querying cancer descriptions, navigating side effects, or understanding diagnoses with Tree of Thought and Chain of Thought reasoning.

## Notes
- This project is for educational purposes and does not provide medical advice.
- Ensure models are served locally via Ollama for optimal performance.