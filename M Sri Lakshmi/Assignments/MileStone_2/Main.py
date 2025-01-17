from Assignment2 import record_and_transcribe, speak_text
from Milestone2 import Analyze_audio
from Assignment3 import generate_sales_response
import pyaudio
import time  # Import the time module for measuring execution time
import csv

audio = pyaudio.PyAudio()

if __name__ == "__main__":
    try:
        while True:
            print("Start speaking or press Ctrl+C to exit.")
            
            # Start timing here
            overall_start_time = time.time()

            query, audio_file = record_and_transcribe()
            print("You asked:", query)

            # Measure time for audio analysis
            analysis_start_time = time.time()
            summery = Analyze_audio(audio_file)
            analysis_time = time.time() - analysis_start_time
            print(f"Time taken for audio analysis: {analysis_time:.6f} seconds")

            print("Generating response...")
            
            # Measure time for response generation
            response_start_time = time.time()
            answer = generate_sales_response(query, summery)
            response_time = time.time() - response_start_time
            print(f"Time taken for response generation: {response_time:.6f} seconds")

            print("AI Response:", answer)
            
            # Measure time for speaking response
            speaking_start_time = time.time()
            print("Speaking response...")
            speak_text(answer)
            speaking_time = time.time() - speaking_start_time
            print(f"Time taken for speaking response: {speaking_time:.6f} seconds")

            # End timing
            overall_time = time.time() - overall_start_time
            print(f"Total Execution Time: {overall_time:.6f} seconds")
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print("Error:", e)
    finally:
        audio.terminate()
