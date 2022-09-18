from math import prod
import os
import numpy as np
import openai
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

product_list = []
with open("./distinct_product_types.csv", encoding='utf-8-sig') as product_type_file: 
    for line in product_type_file.readlines(): 
        product_list.append(line.replace("_"," ").lower().strip("\n"))


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        animal = request.form["animal"].lower()
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=generate_prompt(animal),
            temperature=0.6,
        )
        generated_type_keywords = response.choices[0].text.split(",")
        product_list_for_embedding = product_list + generated_type_keywords
        index_for_keyword = product_list_for_embedding.index(animal)
        # print(product_list_for_embedding)

        embedding_vector = openai.Embedding.create(
            input = product_list_for_embedding,
            engine="text-similarity-davinci-001"
        )
        valid_results = []
        final_results =[]
        valid_index= []
        
        embedding_vector_for_generated = embedding_vector['data'][-5::1]
        embedding_vector_for_amazon = embedding_vector['data'][0:-5:1]
        
        # Get the closest amazon product type 
        for vector in embedding_vector_for_generated :
            print("starting loop for similarity")
            max_score = -10000
            for amzn_vector_num in range(len(embedding_vector_for_amazon)): 
                similarity_score = np.dot(embedding_vector_for_amazon[amzn_vector_num]['embedding'],vector['embedding'])
                print(f"similarity_score {similarity_score}")
                if similarity_score>max_score:
                    if product_list[amzn_vector_num]==animal: 
                        pass 
                    else: 
                        max_score=similarity_score
                        print(f"max_score{max_score}")
                        valid_results.append(product_list[amzn_vector_num])
                        valid_index.append(amzn_vector_num)
                        print(product_list[amzn_vector_num])

        # remove duplicates from list 
        valid_results_without_dup = [*set(valid_results)]
        valid_index_without_dup = [*set(valid_index)]

        # get the amazon product types that are closest to original keywords 
        candid_score_dict = {}
        for index in valid_index_without_dup: 
            similarity_score = np.dot(embedding_vector_for_amazon[index]['embedding'], embedding_vector['data'][index_for_keyword]['embedding'])
            candid_score_dict[index] = similarity_score
        
        sorted_candid_score_dict =dict(sorted(candid_score_dict.items(),reverse=True, key=lambda item: item[1]))
        print(sorted_candid_score_dict)
        for final_index in sorted_candid_score_dict.keys(): 
            final_results.append(product_list[final_index])
            print(f"final_index:{final_index} \n product: {product_list[final_index]}")
            
        response_text = response.choices[0].text + f"\n valid amazon types : {final_results}" 
        return redirect(url_for("index", result=response_text))

    result = request.args.get("result")
    return render_template("index.html", result=result)


def generate_prompt(animal):
    return """Suggest five product types that are complementary to the suggested product type. 

Product type:  bread 
Complementary product types: cheese, butter, jam, butterknife, toaster
Product type:  bed 
Complementary product types: pillow, blanket, bedsheet, mattress, bed frame 
Product type:  cups
Complementary product types: dishes, forks, knives, cutlery 
Product type: {}
Complementary product types: """.format(
        animal.capitalize()
    )
