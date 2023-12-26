from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# nltk.download('punkt')
# nltk.download('stopwords')

def tickets():
    import json
    with open('dept_classifier_app/tickets.json', 'r') as f:
      data = json.load(f)
      tickets = data.get('tickets')
      return tickets

def predict_dept(description):
    
    data = tickets() # Sample data (replace this with your actual data)

    # Convert data to a DataFrame
    df = pd.DataFrame(data, columns=['Ticket', 'Department'])

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # Vectorize the text using TF-IDF or CountVectorizer
    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(train_data['Ticket'])
    X_test = vectorizer.transform(test_data['Ticket'])

    # Train a Support Vector Machine classifier with SVC or MultinomialNB
    classifier = SVC(kernel='linear')
    # classifier = MultinomialNB()
    classifier.fit(X_train, train_data['Department'])

    # Predict the departments for test data
    predictions = classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(test_data['Department'], predictions)
    print(f"Accuracy: {accuracy:.2f}")

    # Print classification report
    print("Classification Report:\n", classification_report(test_data['Department'], predictions))

    # Test the classifier with new tickets
    new_tickets = [description] # list of descriptions ["description", "description2"]
    new_tickets_tfidf = vectorizer.transform(new_tickets)
    predicted_departments = classifier.predict(new_tickets_tfidf)


    response = list()
    for ticket, department in zip(new_tickets, predicted_departments):
        response.append(f"Ticket: {ticket} | Predicted Department: {department}")
    return response