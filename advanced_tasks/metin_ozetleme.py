from transformers import pipeline


# ozetleme pipeline yukle
summarizer = pipeline("summarization")

text = """
Machine learning (ML) is a subfield of artificial intelligence (AI) that focuses on enabling machines to learn patterns and make decisions based on data, without being explicitly programmed. The term was coined by Arthur Samuel in 1959, and since then, it has evolved significantly, becoming a cornerstone of modern computing and data science.

At its core, machine learning involves algorithms that can identify complex patterns within large datasets. These patterns are then used to make predictions or decisions. For example, ML powers recommendation engines on platforms like Netflix and Amazon, spam detection in email systems, facial recognition in security systems, and even autonomous driving in self-driving cars.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning requires labeled data and is commonly used for classification and regression problems. Unsupervised learning works with unlabeled data and is often applied in clustering and dimensionality reduction. Reinforcement learning, on the other hand, involves an agent learning to make decisions by interacting with an environment and receiving rewards or penalties.

Machine learning models rely on training data to learn. During training, models adjust internal parameters to minimize prediction error. This process often involves optimization techniques such as gradient descent. Common ML algorithms include decision trees, support vector machines, neural networks, and ensemble methods like random forests and boosting.

Despite its advantages, ML also presents several challenges. These include overfitting, underfitting, data bias, and interpretability issues. Moreover, large-scale ML systems require significant computational power and high-quality datasets, which can be expensive or difficult to obtain.

In recent years, deep learning, a subfield of ML that uses neural networks with many layers, has gained popularity due to its impressive performance in tasks such as image and speech recognition. Frameworks like TensorFlow and PyTorch have made it easier for researchers and developers to implement complex ML systems.

In conclusion, machine learning is transforming how we interact with technology. Its applications span healthcare, finance, transportation, and beyond. As research continues and more data becomes available, ML is expected to become even more embedded in daily life, enabling smarter, more personalized, and efficient systems across all sectors.
"""

# metni ozetleme
summary = summarizer(
    text,
    max_length = 90,
    min_length = 45,
    do_sample = True
    )


print(summary[0]["summary_text"])