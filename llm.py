from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain.vectorstores.pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate
)
from example import answer_examples

### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_llm():
    llm = ChatUpstage(model = 'solar-pro')
    return llm

def get_dictionary_chain():
    llm = get_llm()
    dictionary = ['사람을 나타내는 표현 -> 거주자']

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 사전을 참고해서 질문을 기존 단어를 대체하여 변경해주세요.
        만약 변경할 필요가 없다면, 변경하지 않아도 됩니다.

        사전 : {dictionary}
        질문 : {{query}}
    """)
    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain

def get_retriever():
    embeddings = UpstageEmbeddings(model = 'embedding-query')
    load_vec_db = Pinecone.from_existing_index(
        index_name='upstage-tax-1500',
        embedding=embeddings
    )

    retriever = load_vec_db.as_retriever()
    return retriever

def get_history_chain():
    llm = get_llm()
    retriever = get_retriever()
    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def get_qa_chain():
    llm = get_llm()
    history_aware_retriever = get_history_chain()
    ### Answer question ###
    qa_system_prompt = """
    당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요.
    답변은 아래 [조건]을 따릅니다.

    [조건]
    - 아래 제공 된 내용(context)를 활용해서 답변해주세요.
    - 답변을 알 수 없다면 모른다고 답변해주세요.
    - 답변은 '소득세법 (XX조)에 따르면'이라고 시작해주세요.
    - 3문장 이내로 짧게 대답해주세요.

    {context}"""

    example_prompt = ChatPromptTemplate.from_messages(
    [('human', '{input}'), ('ai', '{answer}')]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=answer_examples,
        # This is a prompt template used to format each individual example.
        example_prompt=example_prompt,
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    return conversational_rag_chain

def get_llm_response(user_input):
    dictionary_chain = get_dictionary_chain()
    qa_retriever = get_qa_chain()

    tax_chain = {'input' : dictionary_chain} | qa_retriever

    llm_response = tax_chain.stream(
        {"query": user_input},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
    )

    return llm_response