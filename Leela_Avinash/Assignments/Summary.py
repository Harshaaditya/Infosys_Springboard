from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=20, min_length=10):
    try:
        summary = summarizer(text, max_length=max_length,        min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    user_text = input()
    summary = summarize_text(user_text)
    print("Summary:", summary)
