<H3>ENTER YOUR NAME: TIMMAPURAM YOGEESWAR</H3>
<H3>ENTER YOUR REGISTER NO: 212223230233</H3>
<H3>DATE: 07/09/2026</H3>
<H1 Align="center">Project Based Experiment<H1>
<H3>Objective:<H3>


* To perform sentiment analysis on Facebook feedback using Python.
* To classify the feedback into **Positive, Negative, and Neutral** sentiments.
* To filter and display only the **Positive feedback** from the dataset.
* To save the filtered positive feedback for further analysis.


<H3>Program:</H3>
  
```
# Facebook Sentiment Analysis

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Step 1: Load Facebook CSV
df = pd.read_csv("facebook_data.csv")

print("Facebook Data Loaded Successfully!")
print("Total Records:", len(df))

# Step 2: Perform sentiment analysis
df["polarity"] = df["feedback"].apply(
    lambda text: TextBlob(str(text)).sentiment.polarity
)

# Step 3: Classify sentiment
def classify_sentiment(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["polarity"].apply(classify_sentiment)

# Step 4: Display complete results
print("\nComplete Sentiment Analysis:")
display(df)

# Step 5: Count sentiments
print("\nSentiment Counts:")
print(df["sentiment"].value_counts())

# Step 6: Plot sentiment analysis
df["sentiment"].value_counts().plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Facebook Feedback Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Number of Feedback")
plt.xticks(rotation=0)
plt.show()

# Step 7: Filter only Positive Feedback
positive_feedback = df[df["sentiment"] == "Positive"]

print("\nPositive Feedback Only:")
display(positive_feedback)

# Step 8: Save positive feedback
positive_feedback.to_csv(
    "positive_feedback.csv",
    index=False
)

print("\nPositive feedback saved successfully!")
print("Total Positive Feedback:", len(positive_feedback))
```
  
<H3>Output:</H3>

<img width="973" height="477" alt="image" src="https://github.com/user-attachments/assets/3a3ef4b4-8b4f-4921-b61b-4b833e98470e" />


<H3>Inference:</H3>

```
### Inference

* I learned how to load and analyze Facebook data using Python.
* I learned how to use **Pandas** to handle CSV data.
* I learned how **sentiment analysis** works.
* I learned to classify feedback as **Positive, Negative, or Neutral**.
* I learned how to use **TextBlob** to find sentiment polarity.
* I learned how to filter only **Positive feedback** from the dataset.
* I learned how to save the filtered data into a new CSV file.
* I also learned to visualize sentiment results using a graph.
* Overall, this project improved my practical knowledge of **Python and Data Analysis**. 📊

```
