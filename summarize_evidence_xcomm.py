import os
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from tqdm import tqdm

summarize_model_name = "google/pegasus-cnn_dailymail"  # Default PEGASUS model for summarization
summarize_tokenizer = PegasusTokenizer.from_pretrained(summarize_model_name)
summarize_model = PegasusForConditionalGeneration.from_pretrained(summarize_model_name)

def summarize_text_with_pegasus(text, tokenizer, model, target_token_count=2048):
    """
    Summarizes a long text using the PEGASUS model from Hugging Face.

    Parameters:
    - text (str): The input text to summarize.
    - model_name (str): The name of the PEGASUS model to use.
    - target_token_count (int): Approximate number of tokens desired in the summary.

    Returns:
    - str: The summarized text.
    """

    # Define maximum input length for the model
    max_input_length = tokenizer.model_max_length  # Typically 1024 for PEGASUS

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # Split the input into chunks if it exceeds the model's maximum input length
    input_ids = inputs["input_ids"][0]
    total_tokens = input_ids.size(0)
    if total_tokens <= max_input_length:
        chunks = [input_ids]
    else:
        # Split into overlapping chunks to maintain context
        stride = max_input_length // 2  
        chunks = []
        for i in range(0, total_tokens, stride):
            chunk = input_ids[i:i + max_input_length]
            chunks.append(chunk)
            if i + max_input_length >= total_tokens:
                break
    # Generate summaries for each chunk
    summaries = []
    for chunk in chunks:
        input_chunk = chunk.unsqueeze(0)  # Add batch dimension
        summary_ids = model.generate(input_chunk, max_length=256, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Combine summaries
    combined_summary = ' '.join(summaries)

    # If the combined summary exceeds the target token count, truncate it
    summary_tokens = tokenizer.tokenize(combined_summary)
    if len(summary_tokens) > target_token_count:
        truncated_tokens = summary_tokens[:target_token_count]
        final_summary = tokenizer.convert_tokens_to_string(truncated_tokens)
    else:
        final_summary = combined_summary

    return final_summary

def main():
    # Define the directory containing the evidence files
    search_dir = "/scratch/jin7/datasets/XCommnunityNote/evidence"
    # Define the output directory for the summarized evidence
    output_dir = "/scratch/jin7/datasets/XCommnunityNote/summarized_evidence"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each file in the search directory with a progress bar
    for filename in tqdm(os.listdir(search_dir), desc="Summarizing evidence files"):
        if filename.endswith(".txt"):
            input_file = os.path.join(search_dir, filename)
            output_file = os.path.join(output_dir, filename)
            if os.path.exists(output_file):
                continue

            with open(input_file, "r") as f:
                evidence_text = f.read()

            summary = summarize_text_with_pegasus(evidence_text, summarize_tokenizer, summarize_model)
            with open(output_file, "w") as f:
                f.write(summary)

if __name__ == "__main__":
    main()