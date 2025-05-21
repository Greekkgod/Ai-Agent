from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_openai from ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import pydanticOutputParser
fron langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
class ResearchResponse(BaseModel):
    title: str
    abstract: str
    authors: list[str]
    journal: str
    year: int
    tools_used: list[str]

llm = ChatAnthropic(model ="claude-3-5-sonnet-20241022")
parser = pydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}{name}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[],
)
agent_executor = AgentExecutor(agent=agent, verbose=True , tools=[])
raw_response = agent_executor.invoke({})