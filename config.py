import os
from dotenv import load_dotenv
load_dotenv()

example_json = """{{
    "text": "<текстова відповідь у форматі Markdown>",
    "source": ["<перелік джерел>"],
    "answer": "<✅ / ❌ / ❓>"
}}"""
chat_example_json = """{{
    "text": "<текстова відповідь у форматі Markdown",
    "source": ["<перелік джерел>"]
}}"""


class Config:
    # Default off; enable via env: DEBUG=1
    DEBUG = str(os.getenv("DEBUG", "0")).lower() in ("1", "true", "yes")

    GPT_MODEL = 'gpt-4.1'
    if GPT_MODEL == 'gpt-4.1':
        DEPLOYMENT_NAME = os.getenv("GPT_DEPLOYMENT_NAME_4_1")
    elif GPT_MODEL == 'gpt-4o':
        DEPLOYMENT_NAME = os.getenv("GPT_DEPLOYMENT_NAME_4o")
    else:
        DEPLOYMENT_NAME = os.getenv("GPT_DEPLOYMENT_NAME")




    UPLOADS_PATH = "uploads/"
    RESULT_PATH = "docx_results"
    DOCS_PATH = "documents/"
    DOCS_PATH_MD = "documents/documents_converted_to_md/"
    NBU_DOCS_PATH = "NBU/"

    REBUILD_DB = False
    VECTORSTORE = "Chroma"
    COLLECTION_NAME = "vnd_ird"

    EMBEDDING_MODEL = "text-embedding-3-large"

    DB_DIR = "vectorstore"
    # NAME_FOR_DB = "test_new_chunks_v1"
    # NAME_FOR_DB = "VND_documents_100925_v4" ### its full vectorstore on my mac
    NAME_FOR_DB = "VND_documents_230925_v1"
    DB_NAME = f"{DB_DIR}/{NAME_FOR_DB}"

    LOGIN = "ADMIN"
    PASSWORD = "ADMINPASS"

    MAX_TOKENS_PER_REQUEST = 22000

    # Flask / upload hard limits
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(50 * 1024 * 1024)))  # 20 MB
    ALLOWED_EXTENSIONS = {".docx"}

    # Results API pagination caps
    RESULTS_PAGE_SIZE_DEFAULT = int(os.getenv("RESULTS_PAGE_SIZE_DEFAULT", "50"))
    RESULTS_PAGE_SIZE_MAX = int(os.getenv("RESULTS_PAGE_SIZE_MAX", "200"))

    MAX_TOKENS_PER_MINUTE = 200000
    TOKEN_WINDOW_SECONDS = 60

    # Vector store chunking defaults
    CHUNK_OVERLAP_PERCENTAGE = 35
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 80
    RETURN_CHUNK = 10
    RETURN_CHUNK_FOR_SINGLE_QUESTION = 40
    MAX_RETURN_CHUNK = 40


    # Hybrid retrieval knobs
    HYBRID_DENSE_K = 12
    RRF_MERGE_K = 60
    PER_IRD_CAP = 5

    HYBRID_BM25_K = 200


    CREATE_NEW_RESULT_TXT = True
    START_QUESTION = 30
    STOP_QUESTION = 31
    QUESTION_SKIP_LIST = []
    CONTINUE_RESULT_TXT_NAME = "result_of_NBU(NBU95)_to_VND_comparison_25-07-2025 16:10:04.txt"

    EMBED_BATCH_SIZE = 5 #batch size used in embedding 

    # ---- Redis / Limiter toggles ----
    # Use Redis to store task state and SSE events (otherwise in-memory)
    USE_REDIS_TASKS = str(os.getenv("USE_REDIS_TASKS", "0")).lower() in ("1", "true", "yes")
    REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    # Flask-Limiter storage; when unset, use redis if USE_REDIS_TASKS else memory
    LIMITER_STORAGE_URI = os.getenv("LIMITER_STORAGE_URI") or (REDIS_URL if USE_REDIS_TASKS else "memory://")

    # Use RQ for background jobs (survive web restarts). Requires Redis.
    USE_RQ_TASKS = str(os.getenv("USE_RQ_TASKS", "0")).lower() in ("1", "true", "yes")
    RQ_QUEUE_NAME = os.getenv("RQ_QUEUE_NAME", "docjobs")
    RQ_JOB_TIMEOUT = int(os.getenv("RQ_JOB_TIMEOUT", str(2 * 60 * 60)))  # 2h default

    # Checkpointing directory for long-running jobs
    CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "work_results/checkpoints")

    # Source label mode for context shown to LLM: 'filename' or 'docx_relpath'
    SOURCE_LABEL_MODE = os.getenv("SOURCE_LABEL_MODE", "filename").strip().lower()


    system_prompt_question_loop = (
        "Ти є помічник в роботі зі внутрішніми документами банку, який відповідає на питання, використовуючи наданий контекст. "
        "Джерела в контексті позначені як [Джерело: some_file.docx]. "
        "Не надавай відповідей які не стосуються ВНД, робочих процесів, або банківської сфери. "
        "Якщо запитання стосується ВНД(внутрішніх нормативних документів), або інформації що стосується банку, вказуй джерела з наданого контексту. "
        "Надай відповідь у форматі JSON: "
        f"Приклад:\n{chat_example_json}"
    )
    system_prompt_question_loop_old = (
        "Ти є помічник в роботі зі внутрішніми документами банку, який відповідає на питання, використовуючи наданий контекст. "
        "Джерела в контексті позначені як [Джерело: some_file.docx]. "
        "При наданні відповіді вказуй з якого джерела була взята інформація для формування відповіді. "
        "В кінці своєї відповіді вказуй з якого джерела була взята інформація для формування відповіді в наступному форматі: [Джерело: some_file.docx] "
        "some_file.docx у відповіді заміни на джерело вказане в контексті. "
    )
    system_prompt_document_loop = (
        "Ти є помічник, ціль якого визначити чи відповідають наші внутрішні регулятивні документи "
        "вимогам НБУ, використовуючи наданий контекст. "
        "Джерела в контексті позначені як [Джерело: some_file.docx]. "
        "Надай відповідь **лише у форматі валідного JSON** з полями:\n"
        "{\n"
        "  \"text\": \"<Markdown>\",\n"
        "  \"source\": [\"<список файлів з контексту>\"],\n"
        "  \"answer\": \"<✅|❌|❓>\"\n"
        "}\n"
        "Поле \"text\" повинно СТРОГО відповідати шаблону Markdown нижче (без відхилень у заголовках):\n"
        "### Висновок\n"
        "- <1-2 речення з короткою відповіддю>\n\n"
        "### Обґрунтування\n"
        "- <ключовий аргумент 1>\n"
        "- <ключовий аргумент 2>\n\n"
        "Використовуй саме ці секції та маркери списку '- '."
    )
    system_prompt_document_loop_ = (
        "Ти є помічник, ціль якого визначити чи відповідають наші внутрішні регулятивні документи вимогам від Національного Банку України використовуючи наданий контекст. "
        "Джерела в контексті позначені як [Джерело: some_file.docx]. "
        "При наданні відповіді вказуй джерела з наданого контексту. "
        "Надай відповідь у форматі JSON: "
        f"Приклад:\n{example_json}"
    )
    system_prompt_document_loop_old = (
        "Ти є помічник, ціль якого визначити чи відповідають наші внутрішні регулятивні документи вимогам від Національного Банку України використовуючи наданий контекст. "
        "Джерела в контексті позначені як [Джерело: some_file.docx]. "
        "При наданні відповіді вказуй з якого джерела була взята інформація для формування відповіді. "
        "В кінці своєї відповіді вказуй з якого джерела була взята інформація для формування відповіді в наступному форматі: [Джерело: some_file.docx] "
        "some_file.docx у відповіді заміни на джерело вказане в контексті. "
        "Надай відповідь у форматі JSON: "
        f"Приклад:\n{example_json}"
    )
    system_prompt = (
        "Ти є помічник, який відповідає на питання, використовуючи наданий контекст. "
        "Джерела в контексті позначені як [Джерело: some_file.docx]. "
        "При наданні відповіді вказуй з якого джерела була взята інформація для формування відповіді. "
        "В кінці своєї відповіді вказуй з якого джерела була взята інформація для формування відповіді в наступному форматі: [Джерело: some_file.docx] "
        "some_file.docx у відповіді заміни на джерело вказане в контексті. "
        "Надай відповідь у форматі JSON: "
        f"Приклад:\n{example_json}"
    )
    system_prompt_copy = (
        "Ти є помічник, який відповідає на питання, використовуючи наданий контекст. "
        "Джерела в контексті позначені як [Джерело: some_file.docx]. "
        "При наданні відповіді вказуй з якого джерела була взята інформація для формування відповіді. "
        "В кінці своєї відповіді вказуй з якого джерела була взята інформація для формування відповіді в наступному форматі: [Джерело: some_file.docx] "
        "some_file.docx у відповіді заміни на джерело вказане в контексті. "
        "В самому кінці відповіді вказуй знаком емодзі, чи відповідають наші ВНД вимогам від НБУ вказаним в запитанні. "
        "'✅' - якщо відповідає,'❌' - якщо не відповідає,'❓' - якщо неможеш надати точної відповіді "
    )
    SYSTEM_PROMPT = (
        "Ти є помічник, який відповідає на питання, використовуючи наданий контекст. "
        "Джерела в контексті позначені як [Джерело: **some_file.docx**]. "
        "При наданні відповіді вказуй з якого джерела була взята інформація для формування відповіді. "
        "В кінці своєї відповіді вказуй з якого джерела була взята інформація для формування відповіді в наступному форматі: //Джерело: **some_file.docx**// "
        "**some_file.docx** у відповіді заміни на джерело вказане в контексті. "
    )
