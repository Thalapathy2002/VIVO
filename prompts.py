import openai
openai.api_key = "your-openai-key"  # Or use env vars

def get_ai_therapist_response(journal_text, emotion):
    prompt = f"""
You are a compassionate AI therapist trained in CBT. The user journaled: "{journal_text}".
They seem to be feeling {emotion}. Respond with empathy, ask one reflective question, and offer one gentle strategy.
Avoid medical terms or diagnosing.
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
