import os
import PyPDF2
from docx import Document
from fastapi import UploadFile
from user import User
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
from langchain_openai import AzureOpenAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Milvus
from pymilvus import Collection,connections
import pickle
from datetime import datetime
import io
from dotenv import load_dotenv


USER_STORE = {}

is_loaded = load_dotenv()

# Milvus connection parameters
CONNECTION_URI = os.getenv("CONNECTION_URI")
connections.connect(uri=CONNECTION_URI)


async def upload_vectordb(user: User, files: list[UploadFile]) -> tuple[str, int]:
    text = await _extract_text_from_document(files)
    chunks = await _chunk_text(text)
    await _create_embeddings_and_save_vectordb(user, chunks)
    return "Document is uploaded successfully.", 200

async def upload_documents(user: User, files: list[UploadFile]) -> tuple[str, int]:
    text = await _extract_text_from_document(files)
    chunks = await _chunk_text(text)
    await _create_embeddings_and_save_local(user, chunks)
    return "Document is uploaded successfully.", 200


async def _extract_text_from_document(files: list[UploadFile]) -> str:
    text = ""
    for file in files:
        byte_object = await file.read()
        file_name = file.filename
        file_extension = os.path.splitext(file_name)[1]
        if file_extension == '.txt':
            text += byte_object.decode('utf-8')
        elif file_extension == '.pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(byte_object))
            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                text += page.extract_text()
        elif file_extension == '.docx':
            doc = Document(io.BytesIO(byte_object))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
    return text


async def _chunk_text(text: str) -> list[str]:
    chunks = None
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=10,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


async def _create_embeddings_and_save_vectordb(user: User, chunks: any) -> Collection:
    embedding_model = HuggingFaceEmbeddings(model_name=user.embedder)
    pkl_name = os.path.join("admin.pkl")
    vector_store = Milvus.from_texts(
        texts=chunks,
        embedding=embedding_model,
        metadatas=[{"source": f"{pkl_name}:{i}"} for i in range(len(chunks))],
        connection_args={
            "uri": CONNECTION_URI,
        },
        drop_old=True,
    )
    with open(pkl_name, "wb") as f:
        pickle.dump(vector_store, f)
    return vector_store


async def _create_embeddings_and_save_local(user: User, chunks: any) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=user.embedder)
    pkl_name = os.path.join(user.username + ".pkl")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=[{"source": f"{pkl_name}:{i}"} for i in range(len(chunks))])
    with open(pkl_name, "wb") as f:
        pickle.dump(vector_store, f)
    return vector_store


async def ask_question(user: User, question: str, api_key: str) -> tuple[str, int]: 
    
    user = await _get_saved_user(user)

    answer = await _rag(user, question, api_key)
    
    system_message = """
    Sen hukuk ile alakalı insanlara yardımcı olan bir hukuk asistanısın.
    Senin görevin kullanıcının sana sorduğu sorulara sana verilen bilgileri kullanarak cevap vermek.
    Cevaplarının tonu dostane, nötr ve hukuki jargona hakimdir.
    Yalnızca Türkçe cevap ver.
    
    <Örnek 1>:
    <Human>:
    <Soru>:
    Kanun çerçevesinde açık rızanın unsurları nelerdir?
    <Bilgi>:
    6698 sayılı Kişisel Verilerin Korunması Kanunu-MADDE 3
    (1)Bu Kanunun uygulanmasında;
    a)Açık rıza: Belirli bir konuya ilişkin, bilgilendirilmeye dayanan ve özgür iradeyle açıklanan rızayı,
    b)Anonim hâle getirme: Kişisel verilerin, başka verilerle eşleştirilerek dahi hiçbir surette kimliği belirli veya belirlenebilir bir gerçek kişiyle ilişkilendirilemeyecek hâle getirilmesini,
    c)Başkan: Kişisel Verileri Koruma Kurumu Başkanını,
    ç)İlgili kişi: Kişisel verisi işlenen gerçek kişiyi,
    d)Kişisel veri: Kimliği belirli veya belirlenebilir gerçek kişiye ilişkin her türlü bilgiyi,
    e)Kişisel verilerin işlenmesi: Kişisel verilerin tamamen veya kısmen otomatik olan ya da herhangi bir veri kayıt sisteminin parçası olmak kaydıyla otomatik olmayan yollarla elde edilmesi, kaydedilmesi, depolanması, muhafaza edilmesi, değiştirilmesi, yeniden düzenlenmesi, açıklanması, aktarılması, devralınması, elde edilebilir hâle getirilmesi, sınıflandırılması ya da kullanılmasının engellenmesi gibi veriler üzerinde gerçekleştirilen her türlü işlemi,
    f)Kurul: Kişisel Verileri Koruma Kurulunu,
    g)Kurum: Kişisel Verileri Koruma Kurumunu,
    ğ)Veri işleyen: Veri sorumlusunun verdiği yetkiye dayanarak onun adına kişisel verileri işleyen gerçek veya tüzel kişiyi,
    h)Veri kayıt sistemi: Kişisel verilerin belirli kriterlere göre yapılandırılarak işlendiği kayıt sistemini,
    ı)Veri sorumlusu: Kişisel verilerin işleme amaçlarını ve vasıtalarını belirleyen, veri kayıt sisteminin kurulmasından ve yönetilmesinden sorumlu olan gerçek veya tüzel kişiyi,
    ifade eder.
    <Hafıza>:
    <AI>:
    Kanun çerçevesinde açık rızanın unsurları, 6698 sayılı Kişisel Verilerin Korunması Kanunu'nun 3. maddesinde yer alan tanımda üç temel unsur olarak belirlenmiştir:
    Belirli bir konuya ilişkin olma: Açık rıza, belirli bir konuya ilişkin olmalıdır. Bu, verinin işlenmesini gerektiren belirli bir amaç veya durumun varlığını gösterir.
    İlgili kişinin bilgilendirilmesi: Açık rıza, ilgili kişinin bilgilendirilmesini de içerir. Bu, kişinin verilerinin işleneceği durumun ve veri sorumlusunun kimliği hakkında bilgi sahibi olmasını sağlar.
    İlgili kişinin onayının verilmesi: Açık rıza, ilgili kişinin onayını da içerir. Bu, kişinin verilerinin işlenmesine onay vermesi ve bu onayını geri çekme hakkına sahip olmasını sağlar.
    Bu unsurların varlığı, açık rızanın hukuki olarak geçerli olmasını sağlar.

    Şimdi sıra sende:
    """
    memory = user.memory.get_memory()

    llm = await _get_llm(model_name=user.llm)
    prompt = f" Sistem Mesajı: {system_message} <Soru>: {question} <Bilgi>: {answer} <Hafıza>: {memory}"
    print("[DEBUG] Memory: ",memory)
    final_answer = llm.invoke(prompt)
    final_answer = final_answer.content
    user.memory.save(question=question, answer=final_answer)
    await _log(user=user, memory=memory, question=question, system_message=system_message, answer= answer, final_answer = final_answer)

    return final_answer, 200

async def _get_saved_user(user: User) -> User:
    if user.username in USER_STORE:
        return USER_STORE[user.username]
    else:
        USER_STORE[user.username] = user
        return user


async def _rag(user: User, question: str, api_key:str = None) -> tuple[str, int]:
    
    if is_loaded == False:
        return "Environment file is not found.", 400
    
    llm = await _get_llm(model_name=user.llm, api_key=api_key)

    if llm is None:
        return "API key not found.", 400

    #local_vector_store = await _get_vector_file(user.username)
    #local_docs = local_vector_store.similarity_search(question, k=3)
    #print(local_docs)
    #local_retrieved_chunks = local_docs[0].page_content + local_docs[1].page_content + local_docs[2].page_content

    vectordb_vector_store = await _get_vector_file("admin")
    vectordb_docs = vectordb_vector_store.similarity_search(question, k=3)
    print(vectordb_docs)
    vectordb_retrieved_chunks = vectordb_docs[0].page_content + vectordb_docs[1].page_content + vectordb_docs[2].page_content

    system_message="Verilen en alakalı eşleşmeli metinlere göre sorunun doğru cevabını bul."
    context = "Lokal Dosyalardaki en alakalı eşleşmeler: " + "Vektör Veritabanındaki en alakalı eşleşmeler: " + vectordb_retrieved_chunks # + local_retrieved_chunks
    prompt = system_message + "Soru: " + question + context
    try:
        response = llm.invoke(prompt)
    except Exception:
        return "Wrong API key.", 400
    answer = response.content + " Bulunan en alakalı metin parçaları: " + context
    print(f"[DEBUG] RAG Results: {answer}")
    return answer, 200


async def _get_vector_file(username: str)-> any:
    with open(username+".pkl", "rb") as f:
        vector_store = pickle.load(f)
    return vector_store


async def _get_llm(model_name:str, api_key: str=None):
    try:
        if model_name == "openai":
            OPENAI_KEY = os.getenv("OPENAI_KEY")
            llm = OpenAI(api_key=OPENAI_KEY, model="gpt-3.5-turbo-instruct")
        elif model_name == "azure_openai":
            AZURE_AD_TOKEN = os.getenv("AZURE_AD_TOKEN")
            AZURE_AD_TOKEN_PROVIDER = os.getenv("AZURE_AD_TOKEN_PROVIDER")
            AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
            AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
            llm = AzureOpenAI(azure_ad_token=AZURE_AD_TOKEN, azure_ad_token_provider=AZURE_AD_TOKEN_PROVIDER, azure_deployment=AZURE_DEPLOYMENT, azure_endpoint=AZURE_ENDPOINT, model="gpt-3.5-turbo-instruct")
        elif model_name == "llama3":
            GROQ_API_KEY = os.getenv("GROQ_API_KEY")
            os.environ["GROQ_API_KEY"] = GROQ_API_KEY
            llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
        else:
            GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
            llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY,model="gemini-pro")
    except Exception:
        return None
    return llm


async def _log(user: User, memory:str, question: str, system_message: str, answer: str, final_answer: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    username = user.username
    llm = user.llm
    embedder = user.embedder

    log_message = f"\n{timestamp}, Username: {username}, Memory: {memory}, Question: {question}, LLM: {llm}, Embedder: {embedder}, System Message: {system_message},  Answer: {answer}, Final Answer: {final_answer}\n"
    with open("log.txt", "a", encoding="utf-8") as file:
        file.write("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        file.write(log_message)
        file.write("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")