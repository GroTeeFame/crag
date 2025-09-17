import sys
import datetime
import threading

from langchain.prompts import PromptTemplate

from colorama import Fore, Style  # Terminal colors

from config import Config

from rag_functions import count_tokens, rerank_by_keyword_overlap, spinner

DEBUG = getattr(Config, "DEBUG", False)

def dprint(msg: str):
    if DEBUG:
        print(msg)

def question_logic(db, llm):
    dprint(f"{'='*33} def question_logic(db, llm): {'='*33}")
    
    system_prompt = Config.system_prompt_question_loop

    prompt_template = PromptTemplate(
        template=f"""
        {system_prompt}
        Подроблене запитання з контекстом:
        {{context}}

        Основне запитання:
        {{question}}

        Відповідь українською:
    """,
        input_variables=["context", "question"]
    )

    # Using manual retrieval + prompt assembly below; no RetrievalQA chain required


    while True:
        ct = datetime.datetime.now()
        fct = ct.strftime("%d-%m-%Y %H:%M:%S")
        # query = input(f"\n📝 ({fct}) Enter your question (or type 'exit'): ").strip()
        print(f" ({fct}) -> 📝 Paste your question (end with Ctrl+D):")
        try:
            query = sys.stdin.read().strip()
        except KeyboardInterrupt:
            print("\n❌ Input canceled by user.")
            break
        if not query:
            print("⚠️ Empty input. Try again.")
            continue
        if query.lower() in ["exit", "quit", "учше", "йгше"]:
            print("Exiting.")
            break

        splited_query = query.splitlines()

        dprint(Fore.YELLOW + f" Query was splitted to {len(splited_query)} pieces" + Fore.RESET)

        retrieved_chunks = Config.RETURN_CHUNK_FOR_SINGLE_QUESTION
            
        # retriever = db.as_retriever(search_kwargs={"k": retrieved_chunks})
        raw_retriever = db.as_retriever(search_kwargs={"k": retrieved_chunks})

        retrieved_docs_list = []
        retrieved_docs = []

        for s_query in splited_query:
            raw_docs = raw_retriever.invoke(s_query)
            context_data = rerank_by_keyword_overlap(s_query, raw_docs)
            retrieved_docs_list.append(context_data)

        # context_data = retriever.invoke(query)
        raw_docs = raw_retriever.invoke(query)
        context_data = rerank_by_keyword_overlap(query, raw_docs)

        retrieved_docs_list.append(context_data)

        calculated_chunk_buffer = 0
        max_adjust_loops = 5
        loops = 0
        while True:
            retrieved_docs = []
            # temp_retrived_docs = []
            
            for context_data in retrieved_docs_list:
                temp_retrived_docs = []
                for i, context in enumerate(context_data, start=1):
                    if i <= retrieved_chunks:
                        temp_retrived_docs.append(context)
                retrieved_docs.append(temp_retrived_docs)

            dprint(f"✅ Chunks used for this query: {sum(len(sublist) for sublist in retrieved_docs)}")


            retrieved_context = []
            questions = splited_query.copy()
            questions.append(query)
            for question, doc_for_question in zip(questions, retrieved_docs):
                question_str = f"""
    Підпитання : {question}
    Контекст до питання:
        """
                context_str = "\n\n".join([
                    f"===\n[Джерело: {doc.metadata.get('filename', 'невідомо')}]\n{doc.page_content}\n==="
                    for doc in doc_for_question
                ])
                formated_question = question_str + context_str

                retrieved_context.append(formated_question)


            retrieved_context = "\n".join(retrieved_context)

            formatted_prompt = prompt_template.format(context=retrieved_context, question=query)

            token_count = count_tokens(formatted_prompt, model_name="gpt-4o")
            dprint(Fore.CYAN + f"🔢 Estimated prompt tokens: {token_count}" + Fore.RESET)

            dprint(f"✅ Chunks used for this query: {sum(len(sublist) for sublist in retrieved_docs)}")
            dprint(f" Config.MAX_TOKENS_PER_REQUEST: {Config.MAX_TOKENS_PER_REQUEST}")
            dprint(f" token_count: {token_count}")
            dprint(f" (len(splited_query) + 1): {(len(splited_query) + 1)}")
            dprint(f" sum(len(sublist) for sublist in retrieved_docs): {sum(len(sublist) for sublist in retrieved_docs)}")
            dprint(f" len(retrieved_docs): {len(retrieved_docs)}")


            sum_chunks = sum(len(sublist) for sublist in retrieved_docs)
            if sum_chunks == 0:
                print(Fore.RED + "No documents retrieved for this question. Try adjusting the query." + Fore.RESET)
                break

            if token_count < Config.MAX_TOKENS_PER_REQUEST or calculated_chunk_buffer == 1:
                break
            else:
                # calculated_chunk = token_count/(len(splited_query)+1)
                # calculated_chunk = round((MAX_TOKENS / (token_count / len(retrieved_docs))) / (len(splited_query) + 1)) ### FIXME: old formula: don't know why its stop works
                calculated_chunk = round((Config.MAX_TOKENS_PER_REQUEST / (token_count / sum_chunks)) / (len(splited_query) + 1)) 

                if calculated_chunk == 0:
                    calculated_chunk = 1

                # calculated_chunk = round((Config.MAX_TOKENS_PER_REQUEST / (token_count / len(retrieved_docs))))
                print(Fore.LIGHTMAGENTA_EX + f"Calculated retrieved chunk amount : {calculated_chunk}" + Fore.RESET)
                if calculated_chunk == calculated_chunk_buffer: 
                    print(Fore.LIGHTCYAN_EX + f"Calculate same amount chunk as before!!! (calculated_chunk = {calculated_chunk})" + Fore.RESET)
                    calculated_chunk -= 1
                    print(Fore.LIGHTCYAN_EX + f"Making calculated_chunk smaller (calculated_chunk = {calculated_chunk})" + Fore.RESET)
                
                if calculated_chunk == 0:
                    calculated_chunk = 1
                
                calculated_chunk_buffer = calculated_chunk
                retrieved_chunks = calculated_chunk
                loops += 1
                if loops >= max_adjust_loops:
                    print(Fore.YELLOW + "Reached max adjustment loops; proceeding with current chunk size." + Fore.RESET)
                    break

        dprint(Fore.GREEN + Style.DIM + formatted_prompt + Fore.RESET + Style.RESET_ALL)

        if token_count > (Config.MAX_TOKENS_PER_REQUEST - 1000):  # 1000 is the max output tokens set in LLM config
            print(Fore.RED + f"⚠️ Warning: prompt might exceed allowed limit!" + Fore.RESET)

        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(target=spinner, args=("Thinking...", stop_spinner))
        spinner_thread.start()
        result = None
        try:
            if query.startswith("@raw"):
                raw_query = query.replace("@raw", "", 1).strip()
                docs = raw_retriever.invoke(raw_query)
                stop_spinner.set()
                spinner_thread.join()
                print("\n--- Retrieved Chunks Only (No LLM) ---")
                for idx, doc in enumerate(docs, 1):
                    print(f"\n--- Chunk {idx} ---")
                    print(doc.page_content)
                continue

            # result = qa_chain.invoke(query)


            result = llm.invoke(formatted_prompt)  # current working LLM call
            
        finally:
            stop_spinner.set()
            spinner_thread.join()

        print("\n--- Answer ---")
        print(Fore.GREEN)
        if result is None:
            print(Fore.RED + "LLM invocation failed." + Fore.RESET)
        elif hasattr(result, "content"):
            # Try to parse JSON, else print raw content
            try:
                import json
                parsed = json.loads(result.content)
                text = parsed.get("text", "")
                sources = parsed.get("source", [])
                print(text if text else result.content)
                if sources:
                    print(Fore.LIGHTBLACK_EX + f"Sources: {sources}" + Fore.RESET)
            except Exception:
                print(result.content)

            token_usage = result.response_metadata.get('token_usage', {})
            dprint(Fore.YELLOW + f"---------------------------")
            if DEBUG and token_usage:
                print(f"completion_tokens : {token_usage.get('completion_tokens')}")
                print(f"prompt_tokens : {token_usage.get('prompt_tokens')}")
                print(f"total_tokens : {token_usage.get('total_tokens')}")
            elif DEBUG:
                print("Token usage metadata is not available")
            dprint(f"---------------------------" + Fore.RESET)
            dprint(Fore.CYAN + f"---------------------------")
            dprint(f"question was splitted to {len(splited_query)} question")
            dprint(f"for each question was used {calculated_chunk_buffer} chunks")
            dprint(f"---------------------------" + Fore.RESET)
        
        else:
            print(result)
        print(Fore.RESET)

    return
