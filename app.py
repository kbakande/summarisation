import gradio as gr
from transformers import pipeline

def summarize_text(input_text):
    """
    Function to summarize the input text using a Hugging Face transformers pipeline.

    Args:
    input_text (str): Text to be summarized.

    Returns:
    str: Summarized text.
    """
    # Load the summarization pipeline using a specific model from Hugging Face
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # Summarize the text
    summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
    # Extract and return the summarized text
    return summary[0]['summary_text']

def main():
    """
    Main function to launch the Gradio interface.
    """
    # Define the Gradio interface
    interface = gr.Interface(
        fn=summarize_text,
        inputs=gr.Textbox(lines=10, placeholder="Enter text here to summarize..."),
        outputs="text",
        title="Text Summarizer",
        description="A simple text summarization app using Hugging Face's transformers. Enter your text and get a summarized version instantly!"
    )
    # Launch the app
    interface.launch()

if __name__ == "__main__":
    main()
