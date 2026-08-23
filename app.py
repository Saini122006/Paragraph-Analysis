import re
from collections import Counter

import gradio as gr
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob


# ============================================================
# SETUP
# ============================================================

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy English model is not installed."
    )


# ============================================================
# TOKENIZER
# ============================================================

def get_tokens(text):
    return re.findall(
        r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b",
        text.lower()
    )


# ============================================================
# NLP ANALYSIS
# ============================================================

def analyze_text(text):

    if not text or not text.strip():
        return (
            "⚠️ Please enter a paragraph.",
            [],
            "",
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            ""
        )

    text = text.strip()

    # --------------------------------------------------------
    # SPACY
    # --------------------------------------------------------

    doc = nlp(text)

    # --------------------------------------------------------
    # TOKENS
    # --------------------------------------------------------

    tokens = get_tokens(text)

    token_count = len(tokens)

    # --------------------------------------------------------
    # BASIC STATISTICS
    # --------------------------------------------------------

    sentences = list(doc.sents)

    sentence_count = len(sentences)

    characters = len(text)

    characters_no_spaces = len(
        text.replace(" ", "")
    )

    unique_words = set(tokens)

    vocabulary_size = len(unique_words)

    average_word_length = (
        sum(len(word) for word in tokens) / token_count
        if token_count
        else 0
    )

    words_per_sentence = (
        token_count / sentence_count
        if sentence_count
        else 0
    )

    vocabulary_ratio = (
        vocabulary_size / token_count * 100
        if token_count
        else 0
    )

    # --------------------------------------------------------
    # BAG OF WORDS
    # --------------------------------------------------------

    word_frequency = Counter(tokens)

    bow_rows = [
        [word, count]
        for word, count in word_frequency.most_common()
    ]

    # --------------------------------------------------------
    # STOP WORDS
    # --------------------------------------------------------

    stop_word_counter = Counter(
        word
        for word in tokens
        if word in STOP_WORDS
    )

    stopword_rows = [
        [word, count]
        for word, count in stop_word_counter.most_common()
    ]

    # --------------------------------------------------------
    # STEMMING
    # --------------------------------------------------------

    stem_rows = []

    for word in tokens:
        stem_rows.append([
            word,
            STEMMER.stem(word)
        ])

    # --------------------------------------------------------
    # LEMMATIZATION + POS
    # --------------------------------------------------------

    linguistic_rows = []

    for token in doc:

        if not token.is_alpha:
            continue

        linguistic_rows.append([
            token.text,
            token.lemma_,
            token.pos_,
            token.tag_,
            token.dep_
        ])

    # --------------------------------------------------------
    # NAMED ENTITY RECOGNITION
    # --------------------------------------------------------

    entity_rows = []

    for entity in doc.ents:

        description = spacy.explain(
            entity.label_
        ) or ""

        entity_rows.append([
            entity.text,
            entity.label_,
            description
        ])

    # --------------------------------------------------------
    # BIGRAMS
    # --------------------------------------------------------

    bigrams = list(
        ngrams(tokens, 2)
    )

    bigram_counter = Counter(bigrams)

    bigram_rows = [
        [" ".join(pair), count]
        for pair, count in bigram_counter.most_common()
    ]

    # --------------------------------------------------------
    # TRIGRAMS
    # --------------------------------------------------------

    trigrams = list(
        ngrams(tokens, 3)
    )

    trigram_counter = Counter(trigrams)

    trigram_rows = [
        [" ".join(triplet), count]
        for triplet, count in trigram_counter.most_common()
    ]

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    tfidf_rows = []

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )

        matrix = vectorizer.fit_transform([text])

        feature_names = vectorizer.get_feature_names_out()

        scores = matrix.toarray()[0]

        tfidf_data = list(
            zip(feature_names, scores)
        )

        tfidf_data.sort(
            key=lambda x: x[1],
            reverse=True
        )

        tfidf_rows = [
            [word, round(float(score), 4)]
            for word, score in tfidf_data
        ]

    except ValueError:

        tfidf_rows = []

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    blob = TextBlob(text)

    polarity = blob.sentiment.polarity

    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.05:
        sentiment = "Positive 😊"

    elif polarity < -0.05:
        sentiment = "Negative 😞"

    else:
        sentiment = "Neutral 😐"

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    overview = f"""
# 📊 NLP Analysis

## Basic Statistics

| Metric | Value |
|---|---:|
| Words | {token_count} |
| Unique Words | {vocabulary_size} |
| Sentences | {sentence_count} |
| Characters | {characters} |
| Characters without spaces | {characters_no_spaces} |
| Average Word Length | {average_word_length:.2f} |
| Words per Sentence | {words_per_sentence:.2f} |
| Vocabulary Ratio | {vocabulary_ratio:.2f}% |

---

## 😊 Sentiment

**{sentiment}**

**Polarity:** `{polarity:.3f}`

**Subjectivity:** `{subjectivity:.3f}`

---

## 🔝 Most Common Words

"""

    for word, count in word_frequency.most_common(10):

        overview += (
            f"- **{word}** → {count}\n"
        )

    # --------------------------------------------------------
    # TOKENS
    # --------------------------------------------------------

    token_output = """
# 🔤 Tokens

"""

    token_output += " • ".join(tokens)

    # --------------------------------------------------------
    # SENTIMENT OUTPUT
    # --------------------------------------------------------

    sentiment_output = f"""
# 😊 Sentiment Analysis

### Overall

## {sentiment}

### Polarity

`{polarity:.3f}`

**-1** = Very Negative  
**0** = Neutral  
**+1** = Very Positive

### Subjectivity

`{subjectivity:.3f}`

**0** = Objective  
**1** = Subjective
"""

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return (
        overview,
        bow_rows,
        token_output,
        stopword_rows,
        stem_rows,
        linguistic_rows,
        entity_rows,
        bigram_rows,
        trigram_rows,
        tfidf_rows,
        sentiment_output
    )


# ============================================================
# GRADIO UI
# ============================================================

DESCRIPTION = """
### Explore a paragraph using Natural Language Processing.

This tool analyzes:

🔤 Tokenization  
👜 Bag of Words  
🚫 Stop Words  
🌱 Stemming  
🧩 Lemmatization  
🏷️ POS Tagging  
🌍 Named Entity Recognition  
🔗 Bigrams  
🔗 Trigrams  
📈 TF-IDF  
😊 Sentiment Analysis  
📊 Text Statistics
"""


with gr.Blocks(
    title="NLP Text Analyzer",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        "# 🧠 NLP Text Analyzer"
    )

    gr.Markdown(DESCRIPTION)

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    with gr.Row():

        with gr.Column(scale=2):

            text_input = gr.Textbox(
                label="Enter your paragraph",
                placeholder=(
                    "Enter a paragraph here..."
                ),
                lines=12
            )

            with gr.Row():

                analyze_button = gr.Button(
                    "🔍 Analyze Text",
                    variant="primary"
                )

                clear_button = gr.ClearButton(
                    components=[text_input],
                    value="🗑️ Clear"
                )

        with gr.Column(scale=1):

            gr.Markdown(
                """
### 💡 Example

Natural language processing is a field
of artificial intelligence that helps
computers understand human language.

NLP is used in search engines,
chatbots, translation systems and many
other applications.
"""
            )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    gr.Markdown(
        "## 📊 Results"
    )

    with gr.Tabs():

        # OVERVIEW
        with gr.Tab("📊 Overview"):

            overview_output = gr.Markdown()

        # BAG OF WORDS
        with gr.Tab("👜 Bag of Words"):

            bow_output = gr.Dataframe(
                headers=[
                    "Word",
                    "Frequency"
                ],
                datatype=[
                    "str",
                    "number"
                ],
                interactive=False
            )

        # TOKENS
        with gr.Tab("🔤 Tokens"):

            token_output = gr.Markdown()

        # STOP WORDS
        with gr.Tab("🚫 Stop Words"):

            stopword_output = gr.Dataframe(
                headers=[
                    "Stop Word",
                    "Frequency"
                ],
                datatype=[
                    "str",
                    "number"
                ],
                interactive=False
            )

        # STEMMING
        with gr.Tab("🌱 Stemming"):

            stem_output = gr.Dataframe(
                headers=[
                    "Word",
                    "Stem"
                ],
                datatype=[
                    "str",
                    "str"
                ],
                interactive=False
            )

        # LINGUISTICS
        with gr.Tab("🧩 Linguistics"):

            linguistic_output = gr.Dataframe(
                headers=[
                    "Word",
                    "Lemma",
                    "POS",
                    "Detailed POS",
                    "Dependency"
                ],
                datatype=[
                    "str",
                    "str",
                    "str",
                    "str",
                    "str"
                ],
                interactive=False
            )

        # NER
        with gr.Tab("🏷️ Named Entities"):

            entity_output = gr.Dataframe(
                headers=[
                    "Entity",
                    "Type",
                    "Description"
                ],
                datatype=[
                    "str",
                    "str",
                    "str"
                ],
                interactive=False
            )

        # BIGRAMS
        with gr.Tab("🔗 Bigrams"):

            bigram_output = gr.Dataframe(
                headers=[
                    "Bigram",
                    "Frequency"
                ],
                datatype=[
                    "str",
                    "number"
                ],
                interactive=False
            )

        # TRIGRAMS
        with gr.Tab("🔗 Trigrams"):

            trigram_output = gr.Dataframe(
                headers=[
                    "Trigram",
                    "Frequency"
                ],
                datatype=[
                    "str",
                    "number"
                ],
                interactive=False
            )

        # TF-IDF
        with gr.Tab("📈 TF-IDF"):

            tfidf_output = gr.Dataframe(
                headers=[
                    "Word",
                    "TF-IDF Score"
                ],
                datatype=[
                    "str",
                    "number"
                ],
                interactive=False
            )

        # SENTIMENT
        with gr.Tab("😊 Sentiment"):

            sentiment_output = gr.Markdown()

    # --------------------------------------------------------
    # BUTTON
    # --------------------------------------------------------

    analyze_button.click(
        fn=analyze_text,
        inputs=text_input,
        outputs=[
            overview_output,
            bow_output,
            token_output,
            stopword_output,
            stem_output,
            linguistic_output,
            entity_output,
            bigram_output,
            trigram_output,
            tfidf_output,
            sentiment_output
        ]
    )


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
