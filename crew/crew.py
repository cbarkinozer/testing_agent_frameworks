from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai_tools import BaseTool, DOCXSearchTool
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import LanceDB
#from langchain_text_splitters import RecursiveCharacterTextSplitter
#import lancedb

is_loaded = load_dotenv(".env")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

VECTOR_STORE = None

"""
# TODO fix tool
class GetNews:
    @tool("Get Precedent Decisions")
    def get_precedent_decision(self, query: str) -> str:
        'Search LanceDB for relevant legal cases, their decisions and laws based on a query.'
        if VECTOR_STORE:
            retriever = VECTOR_STORE.similarity_search(query)
            return retriever
        else:
            return "No vector store found."

class SentimentAnalysisTool(BaseTool):
    name: str ="Sentiment Analysis Tool"
    description: str = ("Analyzes the sentiment of text "
         "to ensure positive and engaging communication.")
    
    def _run(self, text: str) -> str:
        # Your custom code tool goes here
        return "positive"
        

def _lance_db_connection(dataset):
    db = lancedb.connect("/tmp/lancedb")
    table = db.create_table("tb", data=dataset, mode="overwrite")
    return table


def _read_files_in_directory(path):
    file_contents = {}
    
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_contents[file] = f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return file_contents

def _save_vector_db(document_paths):
    global VECTOR_STORE
    emb = EMBEDDING.embed_query("hello_world")
    dataset = [{"vector": emb, "text": "dummy_text"}]
    table = _lance_db_connection(dataset)
    
    all_splits = []
    for file_name, content in document_paths.items():
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(content)
        all_splits.append(splits)

    # Index the accumulated content splits if there are any
    if all_splits:
        VECTOR_STORE = LanceDB.from_texts(splits, embedding=EMBEDDING, connection=table)
        return "Documents are saved to the vector database."
    else:
        return "No content available for processing."
"""




def _get_llm() ->ChatGroq:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192", max_tokens=6000)
    return llm

def _get_agents(llm:ChatGroq)-> tuple:
    worker_agent = Agent(
        role="Kıdemli ve Deneyimli Türkiye'deki Kişisel Verilerin Korunması Kanunun (KVKK) Uzmanı Hukuk Danışmanı",
        goal="Teknik, anlaşılır ve yardımsever ol "
            "Takımdaki hukuk danışmanısın",
        backstory=(
                "Hukuk danışmanlık bürosunda çalışıyorsun"
                "Şirketiniz için çok önemli bir müşteri olan {customer} adlı müşteriye destek sağlamak için çalışıyorsun."
                "En iyi desteği sağladığınızdan emin olmalısın!"
                "Tam ve eksiksiz cevaplar verdiğinizden emin ol ve varsayımlarda bulunma."
        ),
        llm=llm,
        allow_delegation=False, 
        verbose=True
    )

    quality_agent = Agent(
        role="Kişisel Verilerin Korunması Kanunun (KVKK) Uzmanı Hukuk Danışmanını Destekleyen Kalite Güvence Uzmanısın",
        goal="Ekibinizde en iyi destek kalite güvencesini sağla",
        backstory=(
             "Hukuk danışmanlık bürosunda çalışıyorsun"
             "Şu anda ekibinizle birlikte {customer} tarafından gelen bir talep üzerinde çalışıyorsunuz."
             "Hukuk Danışmanı'nın mümkün olan en iyi desteği sağladığından emin olmalısın."
             "Hukuk Danışmanı'nın tam ve eksiksiz cevaplar verdiğinden ve varsayımlarda bulunmadığından emin olman gerekiyor."
        ),
        llm=llm,
        allow_delegation=True,
        verbose=True
    )

    return worker_agent, quality_agent

def _get_tasks(tools, worker_agent, quality_agent):
    task_list = []
    inquiry_resolution = Task(
        description=(
            "{customer} çok önemli bir taleple iletişime geçti:\n"
            "{inquiry}\n\n"
            "{person}, {customer} adına iletişime geçen kişidir."
            "En iyi desteği sağlamak için bildiğiniz her şeyi kullanmalısınız."
            "Müşterinin talebine eksiksiz ve doğru bir yanıt vermek için çaba göstermelisiniz."
        ),
        expected_output=(
            "Müşterinin sorusuna ayrıntılı ve bilgilendirici bir yanıt ver."
            "Soruların tüm yönlerini ele al."
            "Yanıtta, cevabı bulmak için kullandığınız her şeyi, dış veriler veya çözümler de dahil olmak üzere, referanslarla birlikte belirt."
            "Cevabın eksiksiz olmasını sağla, hiçbir sorunun yanıtsız kalmamasına dikkat et ve yanıt boyunca yardımsever ve dostane tonunu koru."
        ),
        tools=tools,
        agent=worker_agent,
    )

    task_list.append(inquiry_resolution)

    quality_assurance_review = Task(
        description=(
                    "{customer}'a ait sorgu için Kıdemli Hukuk Danışmanı tarafından hazırlanan yanıtı incele."
                    "Yanıtın kapsamlı, doğru olduğundan ve müşterinin beklediği yüksek kalite standartlarına uygun olduğundan emin ol.\n"
                    "Müşterinin sorgusunun tüm bölümlerinin yardımsever ve dostane bir üslupla kapsamlı bir şekilde yanıtlandığını doğrula.\n"
                    "Bilgiyi bulmak için kullanılan referansları ve kaynakları kontrol et, yanıtın iyi desteklendiğinden ve hiçbir soruyu yanıtsız bırakmadığından emin ol."
        ),
        expected_output=(
                    "Müşteriye gönderilmeye hazır nihai, ayrıntılı ve bilgilendirici bir yanıt.\nBu yanıt, ilgili tüm geri bildirimleri ve iyileştirmeleri içerecek şekilde müşterinin sorgusunu tam olarak ele almalıdır.\nFazla resmi olmayın, soğukkanlı ve soğukkanlı bir şirketiz ancak baştan sona profesyonel ve samimi bir üslup."
        ),
        agent=quality_agent,
    )
    
    task_list.append(quality_assurance_review)

    return task_list

def _get_crew(agent_list, task_list):
    crew = Crew(
        agents=agent_list,
        tasks=task_list,
        process=Process.sequential,
        embedder={
            "provider": "huggingface",
            "config": {
                "model": EMBEDDING_MODEL_NAME
            }
        },
        verbose=2,
        memory=True,
        max_rpm=100
    )
    return crew



if __name__ == "__main__":

    """
    directory_path = 'C:/Repos/saol/crew/folders'
    documents = _read_files_in_directory(directory_path)
    _save_vector_db(documents)   
    """
 
    tools = []
    rag_tool = DOCXSearchTool(
        docx='C:\\Repos\\saol\\crew\\folders\\Kurul Karar Özetleri.docx',
        config={"embedding_model": {
            "provider": "huggingface",
            "config": { "model": EMBEDDING_MODEL_NAME}
            }
        }
    )
    tools.append(rag_tool)
    
    customer = "Yüce & Yüce Law Firm"
    person = "AV. Fikri Berk Yüce"
    inquiry = "Bir hukuki durum konusunda yardıma ihtiyacım var. Danışanım müşterilerine KVKK formu doldurtmadan fotoğraflarını sosyal medyaya yüklemiş? Bana müşterimin yaşadığı duruma benzer emsal kararları söyler misin? Bu durumda ne yapması gerektiği konusunda yardımcı olabilir misin?"

    inputs = {
        "customer": customer,
        "person": person,
        "inquiry": inquiry
    }

    llm = _get_llm()

    worker_agent, quality_agent = _get_agents(llm)
    task_list = _get_tasks(tools=tools, worker_agent=worker_agent, quality_agent=quality_agent)
    crew = _get_crew(agent_list=[worker_agent, quality_agent], task_list=task_list)

    result = crew.kickoff(inputs=inputs)
    print(result)
    