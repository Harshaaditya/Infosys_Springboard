import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GOOGLE_GEMINI_API")
if not API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY environment variable or in .env file")
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-pro"

def generate_summary(transcription, audio_analysis):
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = f"""
        Analyze the following conversation and provide:

        1.  A concise, text-based summary of the call insights, highlighting key points, sentiment, and topics.
        2.  Structured analysis of the conversation in JSON format covering the following aspects:
            - Overall sentiment (positive, negative, neutral).
            - Key tones expressed (e.g., assertive, polite, hesitant).
            - List the key topics discussed.
            - Extract and list any actionable recommendations for follow-up.

        Conversation:
        {transcription}
        
        
        Call Summary:
        {audio_analysis}


        Output should be in the following format:

        Summary: [concise text summary]
        
        JSON Output:
        json
        {{
          "sentiment": "example",
          "tone": ["example", "example"],
          "key_topics": ["example", "example"],
          "recommendations": ["example", "example"]
        }}
        
    """

    try:
        response = model.generate_content(prompt)
        response.resolve()

        if response.text:
            summary_match = re.search(r"Summary:\s*(.*?)\s*JSON Output:", response.text, re.DOTALL)
            json_match = re.search(r"(?:json)?\s*({.+?})\s*", response.text, re.DOTALL)

            summary_text = summary_match.group(1).strip() if summary_match else "No summary extracted"

            if json_match:
                json_string = json_match.group(1)
                try:
                    analysis = json.loads(json_string)
                    return format_summary(analysis, summary_text)
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON response: {json_string}")
                    return None
            else:
                print("Error: Could not find valid JSON in response.")
                return None

        else:
            print("Error: No text in response")
            return None

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return None

def format_summary(analysis, summary_text):
    sentiment = analysis.get("sentiment", "Unknown")
    tone = analysis.get("tone", [])
    topics = analysis.get("key_topics", [])
    recommendations = analysis.get("recommendations", [])

    summary = f"""
    Post-Call Summary:
    ------------------
    Summary of call: {summary_text}

    Sentiment: {sentiment}
    Tone: {', '.join(tone)}
    Key Topics: {', '.join(topics)}
    Recommendations:
    - {chr(10).join(recommendations)}
    """
    return summary

summary = generate_summary("transcription", "udio_file")
