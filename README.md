# 🏛️ Indian Constitution LLM (Fine-tuned on Legal Data) - In Progress

This Google Colab notebook provides a complete script to fine-tune a Large Language Model (Mistral-7B) on the Indian Constitution and related legal documents. It includes data preparation from various sources (JSON, Kaggle datasets, user-uploaded PDFs), LoRA fine-tuning using Unsloth, model saving, Hugging Face Hub upload, and a Gradio interface for interactive Q&A.

## 🚀 Features

*   **Data Ingestion**: Loads Indian Constitution articles from a JSON file, Supreme Court judgments from Kaggle, and can process user-uploaded PDF documents (e.g., full Constitution text).
*   **Efficient Fine-tuning**: Utilizes `unsloth` for fast and memory-efficient LoRA fine-tuning of `Mistral-7B-Instruct-v0.3-bnb-4bit`.
*   **Hugging Face Hub Integration**: Seamlessly uploads the fine-tuned model and tokenizer to your specified Hugging Face repository.
*   **Model Evaluation**: Includes a basic evaluation script with predefined test questions.
*   **Interactive Interface**: Deploys a Gradio web interface for real-time question-answering with the fine-tuned model.

## 🚀 Screenshot
![law-screenshot](law-screenshot.png)
![law-screenshot](law-screenshot-progree-2.png)
![law-screenshot](law-screenshot-progress-3.png)
![gradio-app](gradio-app.png)

## ⚙️ Setup and Prerequisites

1.  **Google Colab**: This notebook is designed to run on Google Colab, leveraging its free GPU resources (preferably T4 or A100 for faster training).
2.  **Hugging Face Token**:
    *   Create an account on [Hugging Face](https://huggingface.co/join).
    *   Generate a new access token with "write" permissions: [Hugging Face Settings -> Access Tokens](https://huggingface.co/settings/tokens).
    *   Replace `"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"` in the notebook with your actual token.
3.  **Kaggle API Token (Optional but Recommended)**:
    *   Create an account on [Kaggle](https://www.kaggle.com/).
    *   Go to your account settings (`kaggle.com/<your-username>/account`).
    *   Under the "API" section, click "Create New API Token". This will download a `kaggle.json` file.
    *   You will be prompted to upload this file during notebook execution.
4.  **Data Files (Optional but Recommended)**:
    *   **`INDIA.json`**: A JSON file containing structured data (e.g., articles of the Indian Constitution with `article` and `description` fields). You will be prompted to upload this.
    *   **PDF Constitution Files**: Any PDF documents containing parts of the Indian Constitution or other legal texts you wish to include in training. You will be prompted to upload these.
    *   **Kaggle Dataset**: The notebook is pre-configured to download `adarshsingh0903/legal-dataset-sc-judgments-india-19502024`. This requires your `kaggle.json`.

## 📖 How to Use

1.  **Open in Google Colab**: Upload the notebook (`.ipynb` file) to Google Colab(recommended) or on own system.
2.  **Run All Cells**: Execute the notebook cells sequentially.
3.  **Upload Files**:
    *   When prompted, upload your `INDIA.json` file.
    *   *(Note: If you don't upload `INDIA.json` or PDFs, a default, limited set of Constitution articles will be used for training.)*
4.  **Monitor Training**: The training process will show progress updates. Note that `max_steps` is set to `50` for demonstration purposes. For a more robust model, you would increase `max_steps` and potentially `epochs`.
5.  **Check Hugging Face Upload**: After training, the model will attempt to upload to your Hugging Face repository(I already created). Check the console output for success/failure messages and the link to your model.
6.  **Interact with Gradio**: An interactive Gradio interface will launch, where you can ask questions to your fine-tuned LLM, basic things only sharing, this is in development.

## 📊 Dataset Details

The training data is aggregated from:
*   **`INDIA.json`**: Structured JSON data with "instruction" and "response" fields, primarily for constitutional articles.
*   **Kaggle Legal Judgments**: Top 100 entries from `adarshsingh0903/legal-dataset-sc-judgments-india-19502024`, formatted as summarization tasks. (Note: Only first 100 entries are used to limit memory/training time for this Colab example).
*   **User-uploaded PDFs**: Text extracted from the first 10 pages of each uploaded PDF, formatted as general content descriptions.

The total number of training examples depends on the uploaded data. The final dataset size will be printed before training starts.

## 🧠 Model Details

*   **Base Model**: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` (Mistral-7B, 4-bit quantized)
*   **Fine-tuning Method**: LoRA (Low-Rank Adaptation) using `unsloth` for efficiency.
*   **LoRA Parameters**: `r=16`, `lora_alpha=16`, `lora_dropout=0`, `bias="none"`.
*   **Training Arguments**:
    *   `per_device_train_batch_size=2`
    *   `gradient_accumulation_steps=4`
    *   `warmup_steps=5`
    *   `max_steps=50` (adjust for longer training)
    *   `learning_rate=2e-4`
    *   `optim="adamw_8bit"`

## 💡 Evaluation

A basic evaluation is performed using a small set of hardcoded constitutional questions. The model's response is checked for the presence of specific keywords to determine correctness. This gives a quick indication of how well the model has learned from the provided data.

## ⚠️ Important Notes

*   **Training Steps**: The `max_steps=50` in the `TrainingArguments` is very low and is primarily for demonstrating the pipeline quickly. For a production-ready or higher-performing model, you would need significantly more training data and steps (e.g., several hundreds or thousands).
*   **PDF Processing**: The PDF processing extracts raw text and might not perfectly handle complex layouts or tables.
*   **Kaggle Data Limit**: Only the first 100 rows of the Kaggle dataset are processed to manage Colab's memory limits and speed up the example.
*   **Hugging Face Token Security**: Be cautious with your Hugging Face token. Ensure it's not exposed accidentally. The `login()` function handles it securely within the Colab environment.
*   **Model Size**: The fine-tuned LoRA adapter is small, but the base model is still large. Inference might be slow on CPU or less powerful GPUs.

Feel free to experiment with different datasets, training parameters, and models to achieve better results!

---
