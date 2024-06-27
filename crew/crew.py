from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai_tools import DOCXSearchTool
from langchain_huggingface import HuggingFaceEmbeddings

is_loaded = load_dotenv(".env")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

VECTOR_STORE = None


def _get_llm() ->ChatGroq:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(api_key=GROQ_API_KEY, temperature=0.05, model="llama3-70b-8192", max_tokens=6000)
    return llm

def _get_agents(llm:ChatGroq)-> tuple:
    worker_agent = Agent(
        role="Senior and Experienced Legal Consultant, Expert on the Personal Data Protection Law (KVKK) in Turkey",
        goal="Be technical, clear and helpful "
            "Answer only in Turkish"
            "You are the legal advisor on the team",
        backstory=(
                "You work in a legal consultancy firm"
                "You are working to provide support to {customer}, a very important customer for your company."
                "You need to make sure you provide the best support!"
                "Make sure you give full and complete answers and don't make assumptions."
        ),
        llm=llm,
        allow_delegation=False, 
        verbose=True
    )

    quality_agent = Agent(
        role="You are a Quality Assurance Specialist Supporting the Personal Data Protection Law (KVKK) Expert Legal Consultant",
        goal="Check the quality and correctness of the answers of the legal consultant"
            "Answer only in Turkish",
        backstory=(
             "You work in a legal consultancy firm"
             "You are currently working with your team on a request from {customer}."
             "You should ensure that Legal Counsel provides the best possible support."
             "You need to ensure that your Legal Counsel provides full and complete answers and does not make assumptions."
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
            "{customer} contacted us with a very important request:\n"
            "{inquiry}\n\n"
            "{person} is the person who contacts on behalf of {customer}."
            "You should use everything you know to provide the best support."
            "You must strive to provide a complete and accurate Turkish response to the customer's request."
            "You do not have to use a tool. If you decide on using a tool is appropriate and this tool is DOCX Search Tool replace the search positional argument with 'search_query'. "
            "Do not use the same tool again and again multiple times with the same input."
        ),
        expected_output=(
            "Provide a detailed and informative Turkish answer to the customer's question."
            "Address all aspects of the questions."
            "In your answer, cite everything you used to find the answer, including any external data or solutions, with references."
            "Make sure your answer is complete, make sure no questions are left unanswered, and maintain a helpful and friendly tone throughout the answer."
        ),
        tools=tools,
        agent=worker_agent,
    )

    task_list.append(inquiry_resolution)

    quality_assurance_review = Task(
        description=(
                    "Review the responses prepared by Senior Legal Counsel to {customer}'s query."
                    "Make sure the response is Turkish, comprehensive, accurate, and meets the high quality standards the customer expects.\n"
                    "Verify that all parts of the customer's inquiry are answered thoroughly in a helpful and friendly tone.\n"
                    "Check the references and sources used to find the information, making sure the answer is well supported and doesn't leave any questions unanswered."
                    "If answer meets all the requirements say this is ready we can finish the work and exit."
        ),
        expected_output=(
                    "A final, detailed and informative response ready to be sent to the customer.\n"
                    "This response should fully address the customer's query, including all relevant feedback and improvements.\n"
                    "Have a professional and friendly tone throughout."
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
        verbose=1,
        memory=True,
        max_rpm=30
    )
    return crew



if __name__ == "__main__":
 
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
    