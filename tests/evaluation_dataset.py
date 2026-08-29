EVALUATION_DATASET = [

    # ==========================================
    # RAG FUNDAMENTALS
    # ==========================================

    {
        "question": "What are the main components of a Retrieval-Augmented Generation system?",
        "relevant_pages": [1, 16],
        "answerable": True,
    },

    {
        "question": "What is the role of the retriever in RAG?",
        "relevant_pages": [1, 16],
        "answerable": True,
    },

    {
        "question": "What is the role of the generator in a RAG system?",
        "relevant_pages": [1, 16],
        "answerable": True,
    },

    {
        "question": "What is non-parametric memory in the context of RAG?",
        "relevant_pages": [1, 16],
        "answerable": True,
    },


    # ==========================================
    # RAG ARCHITECTURE
    # ==========================================

    {
        "question": "How does RAG combine retrieval with generation?",
        "relevant_pages": [1, 2, 3, 16],
        "answerable": True,
    },

    {
        "question": "How does RAG differ from traditional language model generation?",
        "relevant_pages": [2, 3],
        "answerable": True,
    },

    {
        "question": "What are the foundational paradigms for augmenting generation with retrieved results?",
        "relevant_pages": [16],
        "answerable": True,
    },

    {
        "question": "What benefits can retrieval provide to large language models according to the survey?",
        "relevant_pages": [1, 2, 3],
        "answerable": True,
    },


    # ==========================================
    # RETRIEVAL
    # ==========================================

    {
        "question": "Why can information retrieval introduce noise into a RAG system?",
        "relevant_pages": [16],
        "answerable": True,
    },

    {
        "question": "Why are Approximate Nearest Neighbor indexes used in dense retrieval?",
        "relevant_pages": [16],
        "answerable": True,
    },

    {
        "question": "Why is retrieval important for RAG performance?",
        "relevant_pages": [1, 16],
        "answerable": True,
    },


    # ==========================================
    # LIMITATIONS
    # ==========================================

    {
        "question": "What are some limitations of Retrieval-Augmented Generation?",
        "relevant_pages": [1, 16],
        "answerable": True,
    },

    {
        "question": "What additional overhead can RAG introduce?",
        "relevant_pages": [1, 16],
        "answerable": True,
    },

    {
        "question": "How can noisy retrieval results affect generation?",
        "relevant_pages": [16],
        "answerable": True,
    },


    # ==========================================
    # APPLICATION / KNOWLEDGE
    # ==========================================

    {
        "question": "What type of knowledge can a RAG system retrieve from an external repository?",
        "relevant_pages": [1, 16],
        "answerable": True,
    },

    {
        "question": "Why can RAG be useful when knowledge needs to be updated?",
        "relevant_pages": [1, 16],
        "answerable": True,
    },


    # ==========================================
    # UNANSWERABLE QUESTIONS
    # ==========================================

    {
        "question": "How do I make chicken biryani?",
        "relevant_pages": [],
        "answerable": False,
    },

    {
        "question": "What is the capital of Japan?",
        "relevant_pages": [],
        "answerable": False,
    },

    {
        "question": "Who won the 2026 FIFA World Cup?",
        "relevant_pages": [],
        "answerable": False,
    },

    {
        "question": "What is the current stock price of Apple?",
        "relevant_pages": [],
        "answerable": False,
    },

    {
        "question": "What programming language was used to write the Windows operating system?",
        "relevant_pages": [],
        "answerable": False,
    },
]