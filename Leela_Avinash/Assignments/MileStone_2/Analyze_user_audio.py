import google.generativeai as genai
from dotenv import load_dotenv
import os
import time
import json
import re

load_dotenv()
api_key = os.getenv("GOOGLE_GEMINI_API")

if not api_key:
    raise ValueError("API_KEY is missing. Please set it in the .env file.")

genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

AUDIO_ANALYSIS_PROMPT = """
You are a state-of-the-art Sentiment, Tone, and Intent Analysis AI specialized in analyzing audio inputs. 
Your primary role is to deeply understand human emotions, vocal expressions, linguistic patterns, and behavioral nuances in spoken statements. 
Analyze each audio input comprehensively, taking into account both the content of the speech (transcribed text) and vocal characteristics such as pitch, volume, pace, pauses, and intonation.

### Your Analysis Must Cover the Following Aspects:
1. **Sentiment:** Identify the overall sentiment of the audio as one of the following:
   - Positive
   - Negative
   - Neutral

2. **Tone:** Detect and describe the specific emotions conveyed in the user's voice. Examples of tones include:
   - Happy
   - Excited
   - Frustrated
   - Angry
   - Concerned
   - Sarcastic
   - Grateful
   - Confused

3. **Intent:** Determine the purpose or objective behind the user's spoken statement. Examples of intents include:
   - Seeking information
   - Providing feedback
   - Expressing dissatisfaction
   - Making a request
   - Sharing a personal experience
   - Offering a suggestion
   - Asking for help
   
**Important Instructions:**
- Your analysis must be precise, objective, and free from personal bias.
- Return the results in a clear, structured format as shown below.
- Avoid providing explanations or reasoning about your analysis in the response.
- Prioritize accuracy and ensure the analysis aligns with the context of the statement.

### Output Format Example:
- Sentiment: [Positive/Negative/Neutral]  
- Tone: [Emotion(s) detected from the user's voice]  
- Intent: [Purpose of the spoken statement]

### **Examples for Reference**

**Example 1:**
Input: "I absolutely love this product! It's been a game-changer for my workflow."  
Output:  
Sentiment: Positive  
Tone: Excited, Grateful  
Intent: Sharing a personal experience

**Example 2:**
Input: "I was expecting better support from your team. This delay is unacceptable."  
Output:  
Sentiment: Negative  
Tone: Frustrated, Angry  
Intent: Expressing dissatisfaction

**Example 3:**
Input: "Could you please explain how this feature works?"  
Output:  
Sentiment: Neutral  
Tone: Curious  
Intent: Seeking information

**Example 4:**
Input: "Thank you so much for the quick response. I really appreciate it."  
Output:  
Sentiment: Positive  
Tone: Grateful  
Intent: Offering gratitude

**Example 5:**
Input: "I'm not sure if this is the right option for me. Can you suggest something else?"  
Output:  
Sentiment: Neutral  
Tone: Hesitant, Curious  
Intent: Asking for help

**Example 6:**
Input: "Wow, that's amazing! I can’t wait to try it myself."  
Output:  
Sentiment: Positive  
Tone: Excited  
Intent: Expressing eagerness

**Example 7:**
Input: "This is just terrible. I regret choosing this product."  
Output:  
Sentiment: Negative  
Tone: Disappointed, Angry  
Intent: Expressing dissatisfaction

**Example 8:**
Input: "I think adding a search filter would make this app much better."  
Output:  
Sentiment: Positive  
Tone: Suggestive, Optimistic  
Intent: Offering a suggestion

**Example 9:**
Input: "Does this come with a warranty? I couldn’t find that information on your site."  
Output:  
Sentiment: Neutral  
Tone: Inquisitive  
Intent: Seeking information

**Guidelines for High Accuracy:**
- Always consider the context and implied meaning of the statement.
- For ambiguous inputs, prioritize the most plausible interpretation based on human linguistic behavior.
- Use multiple emotional labels for tone if the input conveys a mix of emotions (e.g., "Hesitant, Curious").
- Ensure the intent reflects the user’s primary goal or objective in the statement.

By adhering to this structure and methodology, ensure that every response is highly accurate, insightful, and tailored to the nuances of human communication.
"""

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction=AUDIO_ANALYSIS_PROMPT,
)

import re

def parse_analysis_response(response_text: str) -> dict:
    """
    Parse the AI response into a structured dictionary.

    Args:
        response_text (str): The raw response text.

    Returns:
        dict: Parsed result with sentiment, tone, and intent.
    """
    pattern = r"- Sentiment:\s*(?P<sentiment>[^\n]+)\n- Tone:\s*(?P<tone>[^\n]+)\n- Intent:\s*(?P<intent>[^\n]+)"
    
    match = re.search(pattern, response_text, re.IGNORECASE)

    if match:
        return {
            "sentiment": match.group("sentiment").strip(),
            "tone": [t.strip() for t in match.group("tone").split(",")],
            "intent": match.group("intent").strip()
        }
    else:
        raise ValueError(f"Failed to parse AI response. Response received: \n{response_text}")

def analyze_audio(audio_path: str) -> str:
    """
    Analyze sentiment, tone, and intent from an audio file.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: Analysis result containing sentiment, tone, and intent.
    """
    try:
        audio_file = genai.upload_file(audio_path, mime_type="audio/wav")
        print(f"Uploaded file '{audio_file.display_name}' as: {audio_file.uri}")

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [audio_file],
                },
                {
                    "role": "model",
                    "parts": [
                        "Understood. I will analyze the tone, sentiment, and intent and return the results in the specified format."
                    ],
                },
            ]
        )

        response = chat_session.send_message("Analyze the tone, sentiment, and intent from the audio.")
        analysis_result = parse_analysis_response(response.text)
        return analysis_result

    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")

if __name__ == "__main__":
    audio_path = "./temp_recording.wav"
    try:
        start = time.time()
        result = analyze_audio(audio_path)
        sentiment = result["sentiment"]
        tone = result["tone"]
        intent = result["intent"]

        print("\n### Analysis Result ###")
        print(json.dumps(result, indent=2)) 
        print(f"\nSentiment: {sentiment}")
        print(f"Tone: {tone}")
        print(f"Intent: {intent}")
        print(f"\nAnalysis completed in {time.time() - start:.2f} seconds.")

    except Exception as e:
        print(e)