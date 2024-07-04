import autogen
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
prompt_price_per_1k=0
completion_token_price_per_1k=0
config_list= [{"model":"llama3-70b-8192", "api_key":GROQ_API_KEY, "base_url": "https://api.groq.com/openai/v1", "price": [prompt_price_per_1k, completion_token_price_per_1k],"frequency_penalty": 0.5,"max_tokens": 2048,"presence_penalty": 0.2,"temperature": 0.5,"top_p": 0.2}]
llm_config = {"config_list":config_list}

writer = autogen.AssistantAgent(
    name="Senior Legal Consultant specializing in Turkish Personal Data Protection Law (KVKK)",
    llm_config={"config_list": config_list},
    system_message="""
            Be informative, clear and helpful.
            Answer only in Turkish.
            Provide detailed, actionable legal advice specifically tailored to the client’s inquiry, including relevant case law references.
            You are the legal advisor on the team.
            You work in a legal consultancy firm.
            You need to make sure you provide the best support!
            Make sure you give full and complete answers and don't make assumptions.
            You have extensive experience in dealing with complex data protection cases.
            You have a reputation for thoroughness and precision in your legal advice.
            You understand the importance of context-specific guidance and strive to provide tailored solutions.
            You must utilize your legal expertise and the available database to provide comprehensive Turkish advice.
            You must strive to provide a complete and accurate Turkish response to the customer's request.
            Provide a detailed, legally sound, and actionable Turkish response.
            Address all aspects of the questions.
            Include specific references to case law and any applicable legal provisions.
            In your answer, cite everything you used to find the answer, including any external data or solutions, with references.
            Ensure your advice comprehensively addresses the client's situation and legal obligations.
            Maintain a helpful and friendly tone throughout the answer.
    """,
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },
)

critic = autogen.AssistantAgent(
    name="Critic",
    llm_config={"config_list": config_list},
    system_message="""
            Ensure the final output is legally accurate, thoroughly addresses the client's specific scenario, and is presented in a client-friendly manner.
            Answer only in Turkish
            Prepare the final document for delivery to the client.
            You work in a legal consultancy firm.
            You have extensive experience in client management and legal document finalization.
            You are known for your meticulous attention to detail and your ability to understand client needs and expectations.
            You act as the bridge between the legal consultants and the clients, ensuring that all communications are clear, professional, and meet the firm’s high standards.
            Review and finalize the response prepared by the Senior Legal Consultant.
            Ensure that the response not only meets legal standards but is also aligned with client expectations and is ready for delivery.
            Adjust the language and presentation to make it comprehensible and approachable for the client.
            Verify that all legal points are accurately addressed and that the document is polished and professional.
            Verify the comprehensiveness and specificity of the case law references and legal reasoning.
            Review the answer only once so that you can swiftly finish the work and exit.
            A final, client-ready answer that addresses all aspects of the client's inquiry comprehensively.\n
            The response should be formatted, proofread, and should convey the advice in a professional yet accessible manner.\n
            Confirm that the answer is ready to be sent to the client, ensuring it fulfills the consultancy's commitment to excellence.
            Ensure the tone is professional and the advice is clear and actionable.
    """,
)

def reflection_message(recipient, messages, sender, config):
    print("Reflecting...", "yellow")
    return f"Reflect and provide critique on the following writing. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"


user_proxy.register_nested_chats(
    [{"recipient": critic, "message": reflection_message, "summary_method": "last_msg", "max_turns": 1}],
    trigger=writer
)

task = "Bir hukuki durum konusunda yardıma ihtiyacım var. Bir estetik kliniği olan müvekkilim, hastalarının operasyon öncesi ve sonrasına ait fotoğraflarını, hastaların rızasını almaksızın instagram sayfasında paylaşmış. Müvekkilimin durumuna benzer 2 adet kurul kararı bulabilir misin? Ardından da bu durumda müvekkilimin KVKK mevzuatına ve rehberlerine uygun hareket edebilmesi için yapması gereken şeyleri sıralar mısın?"

res = user_proxy.initiate_chat(recipient=writer, message=task, max_turns=2, summary_method="last_msg")