from agents import build_reader_agent,build_search_agent,writer_chain,critic_chain
from tools import scrape_url,web_search

def run_research_pipeline(topic :str)-> str:

    state = {}

    #search agent working
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [("user",f"Find recent, reliable and detailed information about: {topic}")]
    })

    state["search_results"] = search_result['messages'][-1].content

    print("\n search result",state['search_results'])

    #reader agent 
    print("\n"+" ="*50)
    print("step 2 - readeer agent is scraping top resources...")
    print("="*50)

    reader_agent = build_reader_agent()
    
    reader_results = reader_agent.invoke({
         "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })

    state['scraped_content'] = reader_results['messages'][-1].content
    print("\nscraped\n",state['scraped_content'])

    #writer chain 
    print("\n"+" ="*50)
    print("step 3 - writer is drafting the report...")
    print("="*50)

    research_combined = (
        f"SEARCH REULTS : \n {state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT : \n{state['scraped_content']}"
    )

    state['report'] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n",state['report'])

    print("\n"+" ="*50)
    print("step 4 - critic is judging...")
    print("="*50)

    state['feedback'] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\n critic report \n",state['feedback'])

    return state


if __name__ == '__main__':
    topic = input("\n enter a research topic : ")
    run_research_pipeline(topic)