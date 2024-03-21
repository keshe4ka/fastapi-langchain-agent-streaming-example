import asyncio
import json

import requests

query = "What is the weather in San Francisco?"
url = 'http://localhost:8000/chat'


async def get_event():
    with requests.post(
            url,
            stream=True,
            json={'message': query},
    ) as stream:
        for chunk in stream.iter_lines():
            yield json.loads(chunk.decode('utf-8'))


async def main():
    final_response = ''
    async for event in get_event():
        final_response += event['token']
        print(event['token'])

    print(f'Final Answer: \n{final_response}')


asyncio.run(main())
