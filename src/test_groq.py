import os
from groq import Groq
import logging

def test_groq_connection():
    """Test Groq API connection"""
    try:
        # Initialize Groq client
        client = Groq(api_key=os.getenv('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'))
        
        # Test prompt
        test_prompt = """Given a calendar slot:
        - Task: Team Meeting
        - Duration: 1 hour
        - Available slot: 2pm-5pm
        
        Suggest the best time to schedule this meeting."""
        
        # Make API call
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a helpful calendar assistant."},
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.1
        )
        
        print("\nGroq API Test Results:")
        print("-" * 50)
        print("Connection: Success")
        print("Model: mixtral-8x7b-32768")
        print("\nSample Response:")
        print(completion.choices[0].message.content)
        
        return True
        
    except Exception as e:
        print(f"\nError testing Groq API: {str(e)}")
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("\nTesting Groq API Integration...")
    success = test_groq_connection()
    
    if success:
        print("\nGroq API test completed successfully!")
    else:
        print("\nGroq API test failed!")