from typing import Annotated, AsyncGenerator

from fastapi import FastAPI, Body
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain_community.tools.convert_to_openai import format_tool_to_openai_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from starlette.responses import StreamingResponse

MODEL_NAME = 'gpt-3.5-turbo'
OPENAI_API_KEY = '...'
TAVILY_API_KEY = '...'

prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'Your name is Robert Paulson. Your know all about fight clubs.'
        ),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', '{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad'),
    ]
)
llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    streaming=True
)


def _get_tools() -> list[BaseTool]:
    """ Return a list of tools. """

    return [
        TavilySearchResults(
            api_wrapper=TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)
        )
    ]


def _create_agent_with_tools() -> AgentExecutor:
    """Create an agent with custom tools."""

    tools = _get_tools()
    if tools:
        llm_with_tools = llm.bind(
            tools=[format_tool_to_openai_tool(tool) for tool in tools]
        )
    else:
        llm_with_tools = llm

    agent = (
            {
                'input': lambda x: x['input'],
                'agent_scratchpad': lambda x: format_to_openai_tool_messages(
                    x['intermediate_steps']
                ),
                'chat_history': lambda x: x['chat_history'],
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_config(
        {'run_name': 'agent'}
    )
    return agent_executor


app = FastAPI()


class Answer(BaseModel):
    token: str = ''


@app.post('/chat')
async def chat(message: Annotated[str, Body(embed=True)]):
    agent_executor = _create_agent_with_tools()

    chat_history = [
        HumanMessage('Hi, my name is Artem'),
        AIMessage('Hi, Artem! How are you?'),
    ]

    async def async_generator() -> AsyncGenerator[str, None]:
        """ Async generator for streaming response. """

        async for event in agent_executor.astream_events(
                {
                    'input': message,
                    'chat_history': chat_history,
                },
                version='v1',
        ):
            if event['event'] == 'on_chat_model_stream':
                content = event['data']['chunk'].content
                if content:
                    print(type(Answer(token=content).model_dump_json() + '\n'))
                    yield Answer(token=content).model_dump_json() + '\n'

    return StreamingResponse(
        async_generator(), media_type='application/x-ndjson'
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='localhost', port=8000)
