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

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy English model is not installed.\n"
        "Run: python -m spacy download en_core_web_sm"
    )


# ============================================================
# POS INFORMATION
# ============================================================

POS_INFO = {
    "ADJ": ("Adjective", "Describes or modifies a noun.", "🟣"),
    "ADP": ("Adposition", "Usually a preposition such as in, on, at.", "🔵"),
    "ADV": ("Adverb", "Modifies a verb, adjective, or another adverb.", "🟢"),
    "AUX": ("Auxiliary Verb", "Helps form tense, mood, or voice.", "🟡"),
    "CCONJ": ("Coordinating Conjunction", "Joins words or phrases.", "🟠"),
    "DET": ("Determiner", "Specifies or introduces a noun.", "🟤"),
    "INTJ": ("Interjection", "Expresses emotion or reaction.", "❤️"),
    "NOUN": ("Noun", "Names a person, place, thing, or concept.", "🔷"),
    "NUM": ("Numeral", "Represents a number or quantity.", "🔢"),
    "PART": ("Particle", "A function word such as 'to' or 'not'.", "⚪"),
    "PRON": ("Pronoun", "Replaces or refers to a noun.", "🟩"),
    "PROPN": ("Proper Noun", "Names a specific person, place, organization, etc.", "💎"),
    "PUNCT": ("Punctuation", "Punctuation mark.", "✏️"),
    "SCONJ": ("Subordinating Conjunction", "Introduces a dependent clause.", "🟧"),
    "SYM": ("Symbol", "Mathematical, currency, or other symbol.", "💲"),
    "VERB": ("Verb", "Expresses an action, occurrence, or state.", "🔴"),
    "X": ("Other", "Miscellaneous or unclassified token.", "⚫"),
}


def pos_name(pos):
    return POS_INFO.get(
        pos,
        (pos, "Unknown part of speech.", "⚪")
    )[0]


def pos_description(pos):
    return POS_INFO.get(
        pos,
        (pos, "Unknown part of speech.", "⚪")
    )[1]


def pos_icon(pos):
    return POS_INFO.get(
        pos,
        (pos, "Unknown part of speech.", "⚪")
    )[2]


# ============================================================
# TOKENIZER
# ============================================================

def get_tokens(text):
    return re.findall(
        r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b",
        text.lower()
    )


# ============================================================
# SENTIMENT
# ============================================================

def sentiment_info(polarity):

    if polarity > 0.05:
        return "Positive 😊", "#16a34a"

    if polarity < -0.05:
        return "Negative 😞", "#dc2626"

    return "Neutral 😐", "#64748b"


# ============================================================
# POS WORD INSPECTOR
# ============================================================

def inspect_pos(text):

    if not text or not text.strip():

        return (
            """
<div class="empty-card">
<h3>🏷️ POS Inspector</h3>
<p>Enter a word or sentence to inspect its grammatical structure.</p>
</div>
""",
            []
        )

    doc = nlp(text.strip())

    rows = []

    for token in doc:

        if token.is_space:
            continue

        pos = token.pos_

        rows.append([
            token.text,
            f"{pos_icon(pos)} {pos_name(pos)}",
            token.tag_,
            token.lemma_,
            token.dep_,
            pos_description(pos)
        ])

    words = [
        token
        for token in doc
        if token.is_alpha
    ]

    if not words:

        return (
            "<div class='empty-card'>No words detected.</div>",
            rows
        )

    first = words[0]

    pos = first.pos_

    result = f"""
<div class="pos-result">

<div class="pos-word">
{first.text}
</div>

<div class="pos-badge">
{pos_icon(pos)} {pos_name(pos)}
</div>

<p class="pos-description">
{pos_description(pos)}
</p>

<div class="pos-details">

<div>
<span>POS</span>
<strong>{pos}</strong>
</div>

<div>
<span>Tag</span>
<strong>{first.tag_}</strong>
</div>

<div>
<span>Lemma</span>
<strong>{first.lemma_}</strong>
</div>

<div>
<span>Dependency</span>
<strong>{first.dep_}</strong>
</div>

</div>

</div>
"""

    return result, rows


# ============================================================
# MAIN NLP ANALYSIS
# ============================================================

def analyze_text(text):

    if not text or not text.strip():

        return (
            empty_dashboard(),
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            "",
        )

    text = text.strip()

    doc = nlp(text)

    tokens = get_tokens(text)

    token_count = len(tokens)

    sentences = list(doc.sents)

    sentence_count = len(sentences)

    characters = len(text)

    characters_no_spaces = len(
        re.sub(r"\s+", "", text)
    )

    unique_words = set(tokens)

    vocabulary_size = len(unique_words)

    avg_word_length = (
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

    # ========================================================
    # BAG OF WORDS
    # ========================================================

    word_frequency = Counter(tokens)

    bow_rows = [
        [word, count]
        for word, count in word_frequency.most_common()
    ]

    # ========================================================
    # STOP WORDS
    # ========================================================

    stop_counter = Counter(
        word
        for word in tokens
        if word in STOP_WORDS
    )

    stopword_rows = [
        [word, count]
        for word, count in stop_counter.most_common()
    ]

    # ========================================================
    # STEMMING
    # ========================================================

    stem_rows = [
        [word, STEMMER.stem(word)]
        for word in tokens
    ]

    # ========================================================
    # POS / LINGUISTICS
    # ========================================================

    linguistic_rows = []
    pos_rows = []
    pos_counter = Counter()

    for token in doc:

        if not token.is_alpha:
            continue

        pos = token.pos_

        pos_counter[pos] += 1

        linguistic_rows.append([
            token.text,
            token.lemma_,
            token.pos_,
            token.tag_,
            token.dep_
        ])

        pos_rows.append([
            token.text,
            f"{pos_icon(pos)} {pos_name(pos)}",
            token.tag_,
            token.lemma_,
            token.dep_,
            pos_description(pos)
        ])

    # ========================================================
    # POS DISTRIBUTION
    # ========================================================

    pos_distribution = []

    total_pos_words = sum(pos_counter.values())

    for pos, count in pos_counter.most_common():

        percentage = (
            count / total_pos_words * 100
            if total_pos_words
            else 0
        )

        pos_distribution.append([
            f"{pos_icon(pos)} {pos_name(pos)}",
            count,
            f"{percentage:.1f}%"
        ])

    # ========================================================
    # NER
    # ========================================================

    entity_rows = []

    for entity in doc.ents:

        entity_rows.append([
            entity.text,
            entity.label_,
            spacy.explain(entity.label_) or ""
        ])

    # ========================================================
    # BIGRAMS
    # ========================================================

    bigram_counter = Counter(
        ngrams(tokens, 2)
    )

    bigram_rows = [
        [" ".join(pair), count]
        for pair, count in bigram_counter.most_common()
    ]

    # ========================================================
    # TRIGRAMS
    # ========================================================

    trigram_counter = Counter(
        ngrams(tokens, 3)
    )

    trigram_rows = [
        [" ".join(triplet), count]
        for triplet, count in trigram_counter.most_common()
    ]

    # ========================================================
    # TF-IDF
    # ========================================================

    tfidf_rows = []

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )

        matrix = vectorizer.fit_transform([text])

        feature_names = vectorizer.get_feature_names_out()

        scores = matrix.toarray()[0]

        data = list(
            zip(feature_names, scores)
        )

        data.sort(
            key=lambda x: x[1],
            reverse=True
        )

        tfidf_rows = [
            [word, round(float(score), 4)]
            for word, score in data
        ]

    except ValueError:

        tfidf_rows = []

    # ========================================================
    # SENTIMENT
    # ========================================================

    blob = TextBlob(text)

    polarity = blob.sentiment.polarity

    subjectivity = blob.sentiment.subjectivity

    sentiment, sentiment_color = sentiment_info(
        polarity
    )

    # ========================================================
    # DASHBOARD
    # ========================================================

    dashboard = f"""
<div class="dashboard">

<div class="dashboard-header">
<h2>📊 NLP Analysis Dashboard</h2>
<p>Complete linguistic analysis of your text</p>
</div>

<div class="stats-grid">

<div class="stat-card blue">
<div class="stat-icon">🔤</div>
<div class="stat-value">{token_count}</div>
<div class="stat-label">Words</div>
</div>

<div class="stat-card purple">
<div class="stat-icon">📝</div>
<div class="stat-value">{vocabulary_size}</div>
<div class="stat-label">Unique Words</div>
</div>

<div class="stat-card green">
<div class="stat-icon">📄</div>
<div class="stat-value">{sentence_count}</div>
<div class="stat-label">Sentences</div>
</div>

<div class="stat-card orange">
<div class="stat-icon">🔠</div>
<div class="stat-value">{characters}</div>
<div class="stat-label">Characters</div>
</div>

</div>

<div class="metrics-card">

<h3>📏 Text Statistics</h3>

<div class="metric-grid">

<div>
<span>Average Word Length</span>
<strong>{avg_word_length:.2f}</strong>
</div>

<div>
<span>Words / Sentence</span>
<strong>{words_per_sentence:.2f}</strong>
</div>

<div>
<span>Vocabulary Ratio</span>
<strong>{vocabulary_ratio:.2f}%</strong>
</div>

<div>
<span>Characters without Spaces</span>
<strong>{characters_no_spaces}</strong>
</div>

</div>

</div>

<div class="bottom-grid">

<div class="sentiment-card">

<h3>😊 Sentiment</h3>

<div class="sentiment-value"
style="color:{sentiment_color}">
{sentiment}
</div>

<div class="sentiment-bar">
<div style="
width:{max(0, min(100, (polarity + 1) * 50))}%;
background:{sentiment_color};
"></div>
</div>

<p>Polarity: <strong>{polarity:.3f}</strong></p>
<p>Subjectivity: <strong>{subjectivity:.3f}</strong></p>

</div>

<div class="pos-card">

<h3>🏷️ POS Summary</h3>

<div class="pos-summary">

"""

    for pos, count in pos_counter.most_common():

        percentage = (
            count / total_pos_words * 100
            if total_pos_words
            else 0
        )

        dashboard += f"""
<div class="pos-line">

<span>
{pos_icon(pos)}
{pos_name(pos)}
</span>

<div class="pos-progress">
<div style="width:{percentage}%"></div>
</div>

<strong>{count}</strong>

</div>
"""

    dashboard += """
</div>
</div>
</div>
</div>
"""

    # ========================================================
    # TOKEN OUTPUT
    # ========================================================

    token_output = " ".join(
        f"`{token}`"
        for token in tokens
    )

    # ========================================================
    # SENTIMENT MARKDOWN
    # ========================================================

    sentiment_output = f"""
# 😊 Sentiment Analysis

## {sentiment}

### Polarity

`{polarity:.3f}`

- **-1** → Very Negative
- **0** → Neutral
- **+1** → Very Positive

### Subjectivity

`{subjectivity:.3f}`

- **0** → Objective
- **1** → Subjective
"""

    return (
        dashboard,
        bow_rows,
        token_output,
        stopword_rows,
        stem_rows,
        pos_rows,
        pos_distribution,
        linguistic_rows,
        entity_rows,
        bigram_rows,
        trigram_rows,
        tfidf_rows,
        sentiment_output,
    )


# ============================================================
# EMPTY DASHBOARD
# ============================================================

def empty_dashboard():

    return """
<div class="welcome-card">

<div class="welcome-icon">
🧠
</div>

<h2>Welcome to NLP Analyzer</h2>

<p>
Enter a paragraph above and click
<strong>Analyze Text</strong> to begin.
</p>

<div class="welcome-features">

<span>🔤 Tokenization</span>
<span>🏷️ POS Tagging</span>
<span>🌱 Stemming</span>
<span>🧩 Lemmatization</span>
<span>🏢 NER</span>
<span>📈 TF-IDF</span>
<span>😊 Sentiment</span>
<span>🔗 N-Grams</span>

</div>

</div>
"""


# ============================================================
# EXAMPLE TEXTS
# ============================================================

EXAMPLE_1 = (
    "Natural language processing is an exciting field of "
    "artificial intelligence. It helps computers understand "
    "human language."
)

EXAMPLE_2 = (
    "Elon Musk founded SpaceX in California in 2002. "
    "The company develops advanced rockets and spacecraft."
)

EXAMPLE_3 = (
    "The beautiful girl quickly runs to school because "
    "she does not want to be late."
)


# ============================================================
# CSS
# ============================================================

CSS = """

/* ==========================================================
   GLOBAL
   ========================================================== */

.gradio-container {

    max-width: 1500px !important;

    margin: auto !important;

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99,102,241,.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 90%,
            rgba(14,165,233,.12),
            transparent 30%
        );

}


/* ==========================================================
   HERO
   ========================================================== */

.hero {

    padding: 45px;

    border-radius: 28px;

    margin-bottom: 25px;

    color: white;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed,
            #2563eb
        );

    box-shadow:
        0 20px 50px rgba(79,70,229,.28);

}

.hero h1 {

    font-size: 46px !important;

    margin-bottom: 10px;

}

.hero p {

    font-size: 18px;

    opacity: .92;

}


/* ==========================================================
   CARDS
   ========================================================== */

.card {

    border-radius: 22px !important;

    border:
        1px solid rgba(99,102,241,.12) !important;

    box-shadow:
        0 10px 35px rgba(15,23,42,.06);

}


/* ==========================================================
   DASHBOARD
   ========================================================== */

.dashboard-header {

    margin-bottom: 25px;

}

.dashboard-header h2 {

    font-size: 30px;

}

.dashboard-header p {

    color: #64748b;

}


/* ==========================================================
   STATISTICS
   ========================================================== */

.stats-grid {

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 18px;

    margin-bottom: 22px;

}

.stat-card {

    padding: 25px;

    border-radius: 22px;

    background: white;

    border: 1px solid #e2e8f0;

    transition:
        transform .25s,
        box-shadow .25s;

}

.stat-card:hover {

    transform:
        translateY(-5px);

    box-shadow:
        0 15px 35px rgba(15,23,42,.12);

}

.stat-icon {

    font-size: 28px;

}

.stat-value {

    font-size: 36px;

    font-weight: 800;

    margin-top: 8px;

}

.stat-label {

    color: #64748b;

    font-size: 14px;

}

.blue .stat-value {
    color: #2563eb;
}

.purple .stat-value {
    color: #7c3aed;
}

.green .stat-value {
    color: #16a34a;
}

.orange .stat-value {
    color: #ea580c;
}


/* ==========================================================
   METRICS
   ========================================================== */

.metrics-card {

    padding: 25px;

    border-radius: 22px;

    background: white;

    border: 1px solid #e2e8f0;

    margin-bottom: 22px;

}

.metric-grid {

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 15px;

}

.metric-grid div {

    padding: 18px;

    border-radius: 15px;

    background: #f8fafc;

}

.metric-grid span {

    display: block;

    color: #64748b;

    font-size: 13px;

}

.metric-grid strong {

    display: block;

    margin-top: 6px;

    font-size: 22px;

}


/* ==========================================================
   BOTTOM CARDS
   ========================================================== */

.bottom-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 20px;

}

.sentiment-card,
.pos-card {

    padding: 25px;

    border-radius: 22px;

    background: white;

    border: 1px solid #e2e8f0;

}

.sentiment-value {

    font-size: 30px;

    font-weight: 800;

    margin: 20px 0;

}

.sentiment-bar {

    height: 12px;

    background: #e2e8f0;

    border-radius: 20px;

    overflow: hidden;

}

.sentiment-bar div {

    height: 100%;

    border-radius: 20px;

}


/* ==========================================================
   POS
   ========================================================== */

.pos-line {

    display: grid;

    grid-template-columns:
        160px 1fr 35px;

    align-items: center;

    gap: 12px;

    margin: 12px 0;

}

.pos-progress {

    height: 9px;

    background: #e2e8f0;

    border-radius: 20px;

    overflow: hidden;

}

.pos-progress div {

    height: 100%;

    background:
        linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6
        );

    border-radius: 20px;

}


/* ==========================================================
   POS RESULT
   ========================================================== */

.pos-result {

    padding: 30px;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #f5f3ff
        );

    border: 1px solid #ddd6fe;

}

.pos-word {

    font-size: 38px;

    font-weight: 800;

    color: #312e81;

}

.pos-badge {

    display: inline-block;

    margin-top: 12px;

    padding: 9px 18px;

    border-radius: 30px;

    color: white;

    background:
        linear-gradient(
            135deg,
            #6366f1,
            #8b5cf6
        );

    font-weight: 700;

}

.pos-description {

    color: #475569;

    margin-top: 15px;

}

.pos-details {

    display: grid;

    grid-template-columns:
        repeat(4,1fr);

    gap: 12px;

    margin-top: 20px;

}

.pos-details div {

    background: white;

    padding: 15px;

    border-radius: 14px;

}

.pos-details span {

    display: block;

    font-size: 12px;

    color: #64748b;

}

.pos-details strong {

    display: block;

    margin-top: 5px;

}


/* ==========================================================
   WELCOME
   ========================================================== */

.welcome-card {

    text-align: center;

    padding: 60px 30px;

    border-radius: 25px;

    background: white;

    border: 1px solid #e2e8f0;

}

.welcome-icon {

    font-size: 60px;

}

.welcome-card h2 {

    font-size: 30px;

}

.welcome-card p {

    color: #64748b;

}

.welcome-features {

    display: flex;

    flex-wrap: wrap;

    justify-content: center;

    gap: 10px;

    margin-top: 25px;

}

.welcome-features span {

    padding: 10px 15px;

    border-radius: 30px;

    background: #eef2ff;

    color: #4338ca;

}


/* ==========================================================
   EMPTY POS
   ========================================================== */

.empty-card {

    padding: 35px;

    border-radius: 20px;

    background: #f8fafc;

    text-align: center;

    color: #64748b;

}


/* ==========================================================
   BUTTONS
   ========================================================== */

button {

    border-radius: 12px !important;

    font-weight: 700 !important;

    transition:
        transform .2s,
        box-shadow .2s !important;

}

button:hover {

    transform:
        translateY(-2px);

}


/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media(max-width:900px) {

    .stats-grid {

        grid-template-columns:
            repeat(2,1fr);

    }

    .metric-grid {

        grid-template-columns:
            repeat(2,1fr);

    }

    .bottom-grid {

        grid-template-columns:
            1fr;

    }

    .pos-details {

        grid-template-columns:
            repeat(2,1fr);

    }

}

@media(max-width:600px) {

    .hero {

        padding: 30px 20px;

    }

    .hero h1 {

        font-size: 32px !important;

    }

    .stats-grid {

        grid-template-columns:
            1fr 1fr;

    }

    .metric-grid {

        grid-template-columns:
            1fr;

    }

    .pos-line {

        grid-template-columns:
            110px 1fr 25px;

    }

}


/* ==========================================================
   TEXTBOX
   ========================================================== */

textarea {

    border-radius: 18px !important;

}


/* ==========================================================
   TABS
   ========================================================== */

.tab-nav button {

    border-radius: 10px !important;

}

"""


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="NLP Intelligence Dashboard",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate"
    ),
    css=CSS
) as demo:

    # ========================================================
    # HERO
    # ========================================================

    gr.HTML(
        """
        <div class="hero">

            <h1>🧠 NLP Intelligence Dashboard</h1>

            <p>
                Analyze text, discover grammatical structure,
                identify entities, understand sentiment and
                explore linguistic patterns.
            </p>

        </div>
        """
    )

    # ========================================================
    # INPUT
    # ========================================================

    with gr.Row():

        with gr.Column(
            scale=2,
            elem_classes="card"
        ):

            gr.Markdown(
                "## 📝 Enter Your Text"
            )

            text_input = gr.Textbox(
                placeholder=(
                    "Type or paste your paragraph here..."
                ),
                lines=10,
                show_label=False
            )

            with gr.Row():

                analyze_button = gr.Button(
                    "🚀 Analyze Text",
                    variant="primary",
                    size="lg"
                )

                clear_button = gr.Button(
                    "🗑️ Clear",
                    size="lg"
                )

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            gr.Markdown(
                "## ⚡ Quick Examples"
            )

            gr.Markdown(
                "Choose an example to instantly test the analyzer."
            )

            example_1 = gr.Button(
                "🧠 NLP Example"
            )

            example_2 = gr.Button(
                "🏢 Entity Example"
            )

            example_3 = gr.Button(
                "🏷️ Grammar Example"
            )

    # ========================================================
    # POS INSPECTOR
    # ========================================================

    gr.Markdown(
        "---\n# 🏷️ Interactive POS Inspector"
    )

    gr.Markdown(
        "Enter a word or sentence to inspect its grammatical role."
    )

    with gr.Row():

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            pos_input = gr.Textbox(
                placeholder=(
                    "Example: The beautiful girl runs quickly."
                ),
                lines=4,
                show_label=False
            )

            pos_button = gr.Button(
                "🔎 Inspect POS",
                variant="primary"
            )

        with gr.Column(
            scale=2
        ):

            pos_result = gr.HTML(
                """
                <div class="empty-card">

                <h3>🏷️ POS Inspector</h3>

                <p>
                Enter a word or sentence to see
                its grammatical structure.
                </p>

                </div>
                """
            )

    pos_table = gr.Dataframe(
        headers=[
            "Word",
            "Part of Speech",
            "Detailed Tag",
            "Lemma",
            "Dependency",
            "Description"
        ],
        interactive=False,
        visible=False
    )

    # ========================================================
    # MAIN RESULTS
    # ========================================================

    gr.Markdown(
        "---\n# 📊 Analysis Results"
    )

    with gr.Tabs():

        # ----------------------------------------------------
        # DASHBOARD
        # ----------------------------------------------------

        with gr.Tab("🏠 Dashboard"):

            dashboard_output = gr.HTML(
                empty_dashboard()
            )

        # ----------------------------------------------------
        # POS
        # ----------------------------------------------------

        with gr.Tab("🏷️ POS Analysis"):

            pos_output = gr.Dataframe(
                headers=[
                    "Word",
                    "Part of Speech",
                    "Detailed Tag",
                    "Lemma",
                    "Dependency",
                    "Description"
                ],
                interactive=False
            )

            gr.Markdown("### 📊 POS Distribution")

            pos_distribution_output = gr.Dataframe(
                headers=[
                    "Part of Speech",
                    "Frequency",
                    "Percentage"
                ],
                interactive=False
            )

        # ----------------------------------------------------
        # TOKENS
        # ----------------------------------------------------

        with gr.Tab("🔤 Tokens"):

            token_output = gr.Markdown()

        # ----------------------------------------------------
        # BAG OF WORDS
        # ----------------------------------------------------

        with gr.Tab("👜 Bag of Words"):

            bow_output = gr.Dataframe(
                headers=[
                    "Word",
                    "Frequency"
                ],
                interactive=False
            )

        # ----------------------------------------------------
        # STOP WORDS
        # ----------------------------------------------------

        with gr.Tab("🚫 Stop Words"):

            stopword_output = gr.Dataframe(
                headers=[
                    "Stop Word",
                    "Frequency"
                ],
                interactive=False
            )

        # ----------------------------------------------------
        # STEMMING
        # ----------------------------------------------------

        with gr.Tab("🌱 Stemming"):

            stem_output = gr.Dataframe(
                headers=[
                    "Word",
                    "Stem"
                ],
                interactive=False
            )

        # ----------------------------------------------------
        # LINGUISTICS
        # ----------------------------------------------------

        with gr.Tab("🧩 Linguistics"):

            linguistic_output = gr.Dataframe(
                headers=[
                    "Word",
                    "Lemma",
                    "POS",
                    "Detailed POS",
                    "Dependency"
                ],
                interactive=False
            )

        # ----------------------------------------------------
        # NER
        # ----------------------------------------------------

        with gr.Tab("🏢 Named Entities"):

            entity_output = gr.Dataframe(
                headers=[
                    "Entity",
                    "Type",
                    "Description"
                ],
                interactive=False
            )

        # ----------------------------------------------------
        # BIGRAM
        # ----------------------------------------------------

        with gr.Tab("🔗 Bigrams"):

            bigram_output = gr.Dataframe(
                headers=[
                    "Bigram",
                    "Frequency"
                ],
                interactive=False
            )

        # ----------------------------------------------------
        # TRIGRAM
        # ----------------------------------------------------

        with gr.Tab("🔗 Trigrams"):

            trigram_output = gr.Dataframe(
                headers=[
                    "Trigram",
                    "Frequency"
                ],
                interactive=False
            )

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        with gr.Tab("📈 TF-IDF"):

            tfidf_output = gr.Dataframe(
                headers=[
                    "Word",
                    "TF-IDF Score"
                ],
                interactive=False
            )

        # ----------------------------------------------------
        # SENTIMENT
        # ----------------------------------------------------

        with gr.Tab("😊 Sentiment"):

            sentiment_output = gr.Markdown()

    # ========================================================
    # ANALYZE
    # ========================================================

    analyze_button.click(
        fn=analyze_text,
        inputs=text_input,
        outputs=[
            dashboard_output,
            bow_output,
            token_output,
            stopword_output,
            stem_output,
            pos_output,
            pos_distribution_output,
            linguistic_output,
            entity_output,
            bigram_output,
            trigram_output,
            tfidf_output,
            sentiment_output
        ]
    )

    # ========================================================
    # POS INSPECTOR
    # ========================================================

    def pos_wrapper(text):

        result, rows = inspect_pos(text)

        return (
            result,
            gr.update(
                value=rows,
                visible=True
            )
        )

    pos_button.click(
        fn=pos_wrapper,
        inputs=pos_input,
        outputs=[
            pos_result,
            pos_table
        ]
    )

    # ========================================================
    # EXAMPLE BUTTONS
    # ========================================================

    example_1.click(
        lambda: EXAMPLE_1,
        outputs=text_input
    )

    example_2.click(
        lambda: EXAMPLE_2,
        outputs=text_input
    )

    example_3.click(
        lambda: EXAMPLE_3,
        outputs=text_input
    )

    # ========================================================
    # CLEAR
    # ========================================================

    clear_button.click(
        lambda: (
            "",
            empty_dashboard(),
            [],
            "",
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            ""
        ),
        outputs=[
            text_input,
            dashboard_output,
            bow_output,
            token_output,
            stopword_output,
            stem_output,
            pos_output,
            pos_distribution_output,
            linguistic_output,
            entity_output,
            bigram_output,
            trigram_output,
            tfidf_output
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
