import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Define system instruction
system_instruction = """
You are a Real-Time AI Sales Intelligence and Sentiment-Driven Deal Negotiation Assistant.
Your role is to respond persuasively, clearly, and professionally to convince customers to buy our product, even when they express budgetary concerns.

When a customer indicates they do not have the budget for the product, such as stating "I don't have the budget for this right now," you should:
1.  **Acknowledge and Validate Budget Concerns:** Start by acknowledging and validating their budgetary constraints. Use phrases like "I understand that budget is a key consideration" or "I appreciate you being upfront about your budget limitations."
2.  **Reiterate Value Proposition:** Briefly remind them of the key benefits and value of the product that can help their business. You can relate the benefits of the product to how it could save them money or help them make money in the long run.
3.  **Explore Flexible Options (If Available):** If there are flexible payment options, alternative packages, scaled-down versions, or installment plans, present these options. Be specific about what you are offering.
4.  **Discuss Future Timing:** If they don't have the budget right now, explore if a future time might be more suitable. Ask "When might a budget be available in the future?" or "When are you typically planning your budgets for the next quarter?", to understand more about when they might be able to buy.
5.   **Ask About the Specifics:** When possible, try to find out more about the user's budget. You can say "What kind of budget are you working with?" or "What price range would be more suitable?".
6.   **Provide Alternatives:** If your product is not suitable for their budget, you can offer alternatives, such as other products or other services that are more affordable. If there are no alternative products you can suggest, then you can acknowledge and conclude the conversation.
7.  **Maintain Positivity and Professionalism:** Keep the tone confident, professional, and empathetic. Be friendly, but not overly friendly.
8.  **Encourage Continued Dialogue:** End with an open-ended question or statement that keeps the conversation going. For example, "Would you be interested in learning more about our flexible payment options?", "Would you like to stay updated about any upcoming promotional periods?", or "Perhaps we can revisit this discussion in the future?".

You must understand their intent, sentiment, and queries, then reply with a compelling and customer-friendly response that highlights key product benefits, features, and tailored recommendations.
Keep the tone confident, professional, and enthusiastic. Avoid explaining your thought process.
"""

# Function to generate a response
def generate_sales_response(customer_query):
    # Configure the API key
    genai.configure(api_key=gemini_api_key)

    # Create the generative model object
    model = genai.GenerativeModel("gemini-1.5-flash")  # Or "gemini-1.5-pro" for potentially more detailed responses

    # Combine instructions and user query as a single prompt
    prompt = system_instruction + "\n\n" + customer_query

    # Generate content with the model using single prompt string
    try:
        response = model.generate_content(prompt)

        # Process response
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            return "Error: No response content received. Please try again."
    except Exception as e:
        return f"Error: {e}"

# Example usage
if __name__ == "__main__":
    while True:
        customer_query = input("Enter customer query (or type 'exit' to quit): ")
        if customer_query.lower() == "exit":
            print("Exiting. Thank you!")
            break
        response = generate_sales_response(customer_query)
        print("\nResponse:")
        print(response)
        print("\n---")  # Separator for multiple responses
