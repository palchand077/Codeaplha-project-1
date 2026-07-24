import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

def load_faqs(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def preprocess_text(text):
    # Simple cleaning and tokenization
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join([word for word in tokens if word.isalnum()])

class FAQBot:
    def __init__(self, faqs):
        self.faqs = faqs
        self.questions = [preprocess_text(faq['question']) for faq in faqs]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def get_answer(self, user_query):
        processed_query = preprocess_text(user_query)
        query_vec = self.vectorizer.transform([processed_query])
        
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        
        if similarities[best_idx] > 0.2:
            return self.faqs[best_idx]['answer']
        else:
            return "I'm sorry, I couldn't find a matching answer for your question."

if __name__ == "__main__":
    faqs = load_faqs('faqs.json')
    bot = FAQBot(faqs)
    
    print("Bot initialized. Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        print(f"Bot: {bot.get_answer(query)}")
