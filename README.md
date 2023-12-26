# dept_classifier Service


# Getting Started
In the project directory, follow the given steps to run dept_classifier service:

   1. Rename sample.env to .env and Configure .evn file
   2. Containerize the App with Docker
   3. Runs the app in the development mode > sudo docker-compose up
   4. Uncomment line in entrypoint.sh "# flask db upgrade" for migrate new changes in DB of migration file and run sudo docker-compose up again.
   5. Open [http://localhost:8080/predict/department/?description="ticket issue"] to Test it in your browser.

"""
   We are using Natural language processing (NLP) techniques combined with machine learning classification algorithms.
   Here's an example predict the department based on tickets and department dataset. We are using  text classification approach  using scikit-learn with a Support Vector Machine (SVM) classifier.

   The TF-IDF Vectorizer is used to convert the text data into numerical features, 
   and a Support Vector Machine (SVM) classifier is trained on this data.

   OR

   The CountVectorizer Vectorizer is used to convert the text data into numerical features, 
   and a MultinomialNB classifier is trained on this data.
"""

*
   SVCs can be faster to train, especially on smaller datasets and
   work well for linearly separable data.
   If the size of your dataset is larger or complex relationships in high-dimensional data like text. you can use pre-trained BERT model or similar transformer models.

