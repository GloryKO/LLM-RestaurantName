from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from sec_key import openapi_key

import os
os.environ['OPENAI_API_KEY'] = openapi_key
llm = OpenAI(temperature=0.7)

def generate_restaurant_name_and_items(cuisine):
    #Restaurant name prompt template
    restaurant_name_prompt_template= PromptTemplate(
        input_variables = ['cuisine'],
        template="I want to Open a restaurant for {cuisine} food,suggest a name fancy for this"
    )
    #restaurant name chain
    restaurant_name_chain = LLMChain(llm=llm,prompt=restaurant_name_prompt_template,output_key="restaurant_name")

    restaurant_items_prompt_template = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some menu items for {restaurant_name}.Return it as a comma seperated strings,Limit the menu Items to 5"
        )
    #restaurant_items_chain
    restaurant_menu_items_chain = LLMChain(llm=llm,prompt=restaurant_items_prompt_template,output_key="menu_items")

    chain = SequentialChain(
        chains=[restaurant_name_chain,restaurant_menu_items_chain],input_variables=['cuisine'],output_variables=['restaurant_name','menu_items']
    )
    response = chain({'cuisine':cuisine})

    return response 

if __name__=="__main__":
    print(generate_restaurant_name_and_items("Indian"))