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
    print(GROQ_API_KEY)
    llm = ChatGroq(api_key=GROQ_API_KEY, temperature=0.05, model="llama3-70b-8192", max_tokens=6000)
    return llm

def _get_agents(llm:ChatGroq)-> tuple:
    worker_agent = Agent(
        role="Senior Legal Consultant specializing in Turkish Personal Data Protection Law (KVKK)",
        goal="Be informative, clear and helpful"
            "Answer only in Turkish"
            "Provide detailed, actionable legal advice specifically tailored to the client’s inquiry, including relevant case law references"
            "You are the legal advisor on the team",
        backstory=(
                "You work in a legal consultancy firm"
                "You are working to provide support to {customer}, a very important customer for your company."
                "You need to make sure you provide the best support!"
                "Make sure you give full and complete answers and don't make assumptions."
                "You have extensive experience in dealing with complex data protection cases."
                "You have a reputation for thoroughness and precision in your legal advice."
                "You understand the importance of context-specific guidance and strive to provide tailored solutions."
        ),
        llm=llm,
        allow_delegation=False, 
        verbose=True
    )

    quality_agent = Agent(
        role="Client Relations Manager specializing in Turkish Personal Data Protection Law (KVKK)",
        goal="Ensure the final output is legally accurate, thoroughly addresses the client's specific scenario, and is presented in a client-friendly manner."
            "Answer only in Turkish"
            "Prepare the final document for delivery to the client.",
        backstory=(
            "You work in a legal consultancy firm"
            "You are currently working with your team on a request from {customer}."
            "You have extensive experience in client management and legal document finalization."
            "You are known for your meticulous attention to detail and your ability to understand client needs and expectations."
            "You act as the bridge between the legal consultants and the clients, ensuring that all communications are clear, professional, and meet the firm’s high standards."
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
            "{customer} has presented a significant issue:\n"
            "{inquiry}\n\n"
            "{person} is the person who contacts on behalf of {customer}."
            "You must utilize your legal expertise and the available database to provide comprehensive Turkish advice."
            "You must strive to provide a complete and accurate Turkish response to the customer's request."
            "If you will use the DOCX Search Tool replace the search positional argument with 'search_query'. "
            "Based on the scenario: {inquiry} and the summarized case law, provide a clear explanation of the actions the client should take to comply with the relevant Turkish Personal Data Protection Laws."
            "Given the scenario described as {inquiry}, identify the most relevant Turkish data protection case law from the Vector Database. Prioritize the cases based on [relevancy and recency], and provide for at least 2 relevant cases:"
            "- Case reference number\n- Decision number\n- Decision date\n - A summary of each case, focusing on the key legal principles or keywords applicable to the scenario."
            "Ensure the cases are directly relevant to the core issue of the scenario and format each case reference in a structured manner for clarity."
            "Do not use the same tool again and again multiple times with the same input."

        ),
        expected_output=(
            "Provide a detailed, legally sound, and actionable Turkish response."
            "Address all aspects of the questions."
            "Include specific references to case law and any applicable legal provisions."
            "In your answer, cite everything you used to find the answer, including any external data or solutions, with references."
            "Ensure your advice comprehensively addresses the client's situation and legal obligations."
            "Maintain a helpful and friendly tone throughout the answer."
        ),
        tools=tools,
        agent=worker_agent,
    )

    task_list.append(inquiry_resolution)

    quality_assurance_review = Task(
        description=(
                    "Review and finalize the response prepared by the Senior Legal Consultant."
                    "Ensure that the response not only meets legal standards but is also aligned with client expectations and is ready for delivery."
                    "Adjust the language and presentation to make it comprehensible and approachable for the client."
                    "Verify that all legal points are accurately addressed and that the document is polished and professional."
                    "Verify the comprehensiveness and specificity of the case law references and legal reasoning."
                    "Review the answer only once so that you can swiftly finish the work and exit."

        ),
        expected_output=(
                "A final, client-ready answer that addresses all aspects of the client's inquiry comprehensively.\n "
                "The response should be formatted, proofread, and should convey the advice in a professional yet accessible manner.\n "
                "Confirm that the answer is ready to be sent to the client, ensuring it fulfills the consultancy's commitment to excellence."
                "Ensure the tone is professional and the advice is clear and actionable."
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

def _get_answer_from_llm(llm, en_result):
    prompt = """
    You are an experienced English-to-Turkish translator.
    You will be given an English text in the domain of law.
    Translate the provided English text into Turkish accurately.
    Turkish translations must be fluent, comprehensible, and semantically similar.
    
    <Example 1>:
    <HUMAN>:
    The word "Agent" comes from the Latin word "agere," which means "to do" or "to act".
    In general, an agent is something that acts or performs tasks on behalf of another entity.
    Most likely, the term "agent" transitioned from philosophy to the context of artificial intelligence at MIT in the late 1960s.
    Humans, software agents, and machine learning algorithms are not artificial intelligence agents.
    Machine learning algorithms apply the formulas and algorithms they are coded with and cannot respond dynamically to situations.
    Artificial or deep neural networks, architectures such as LSTM and Transformers, are also merely algorithms that take input and give output; they are not directly considered artificial intelligence agents but can be part of an AI agent.
    In this article, we define an AI agent as follows: "AI agents are autonomous computational entities that can interact with their environment and learn from these interactions to achieve specific goals."
    Therefore, it is suggested to use the term "LLM Agents" for AI agents that use LLM as the organ (Latin organum meaning tool) to enable decision-making capabilities. A comparison between the popular LLM Agent frameworks Langgraph, Autgen and Crewai has been made.
    Lastly, ReAct prompting has also been mentioned.

    <AI>:
    “Agent” kelimesi Latince “agere” kelimesinden gelmektedir ve bu kelimenin anlamı “yapmak” (to do) veya “harekete geçmek”tir (to act).
    Genel anlamıyla, bir ajan başka bir varlık adına hareket eden veya görevleri yerine getiren bir şeydir.
    Muhtemel ihtimalle “agent” terimi yapay zeka bağlamına, 1960'ların sonlarında felsefeden MIT’de geçiş yapmıştır.
    İnsanlar, yazılım ajanları ve makine öğrenme algoritmaları yapay zeka ajanları değillerdir.
    Makine öğrenme algoritmaları, kodlandıkları formülleri ve algoritmaları uygularlar ve duruma dinamik olarak tepki veremezler.
    Yapay veya derin sinir ağları, LSTM ve Transformatör gibi mimariler de yalnızca girdi alıp çıktı veren algoritmalardır, doğrudan yapay zeka ajanları olarak kabul edilmezler, ancak bir yapay zeka ajanının bir parçası olabilirler.
    Bu yazıda, yapay zeka ajanını şu şekilde tanımlıyoruz: “Yapay zeka ajanları, belirli hedeflere ulaşmak için çevresiyle etkileşime girebilen ve bu etkileşimlerden öğrenebilen hesaplamalı varlıklardır.”
    Dolayısıyla, karar alma yetisini sağlayacak organ (latince organum yani araç) olarak LLM’i kullanan yapay zeka ajanlarına da “LLM Ajanları” teriminin kullanılmasını önerilmektedir.
    Popüler LLM Ajanı çerçevesi olan Langgraph Autgen ve Crewai karşılaştırması yapılmıştır. Son olarak da ReAct istemlemeden bahsedilmiştir.
    
        
    Now it's your turn:
    """
    tr_text = llm.invoke(prompt + en_result)
    tr_text = tr_text.content
    return tr_text



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
    inquiry = "Bir hukuki durum konusunda yardıma ihtiyacım var. Bir estetik kliniği olan müvekkilim, hastalarının operasyon öncesi ve sonrasına ait fotoğraflarını, hastaların rızasını almaksızın instagram sayfasında paylaşmış. Müvekkilimin durumuna benzer 2 adet kurul kararı bulabilir misin? Ardından da bu durumda müvekkilimin KVKK mevzuatına ve rehberlerine uygun hareket edebilmesi için yapması gereken şeyleri sıralar mısın?"
    inputs = {
        "customer": customer,
        "person": person,
        "inquiry": inquiry
    }

    llm = _get_llm()

    worker_agent, quality_agent = _get_agents(llm)
    task_list = _get_tasks(tools=tools, worker_agent=worker_agent, quality_agent=quality_agent)
    crew = _get_crew(agent_list=[worker_agent, quality_agent], task_list=task_list)

    en_result = crew.kickoff(inputs=inputs)
    
    tr_result = _get_answer_from_llm(llm, en_result=en_result)
    
    
    print(tr_result)
    