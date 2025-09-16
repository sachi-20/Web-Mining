import sys, os, requests, spacy, time
import google.generativeai as genai

from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from spacy_help_functions import create_entity_pairs
from spanbert import SpanBERT
from collections import defaultdict
from google.api_core.exceptions import InternalServerError, ResourceExhausted

# X is set of extracted tuples
X = defaultdict(int)
seen_urls = set()


spacy2bert = { 
        "ORG": "ORGANIZATION",
        "PERSON": "PERSON",
        "GPE": "LOCATION", 
        "LOC": "LOCATION",
        "DATE": "DATE"
        }

bert2spacy = {
        "ORGANIZATION": "ORG",
        "PERSON": "PERSON",
        "LOCATION": "LOC",
        "CITY": "GPE",
        "COUNTRY": "GPE",
        "STATE_OR_PROVINCE": "GPE",
        "DATE": "DATE"
        }

def get_query():
    api_key = os.environ.get('API_KEY', None)
    engine_id = os.environ.get('ENGINE_ID', None)
    gemini_key = os.environ.get('GEMINI_API_KEY', None)

    if len(sys.argv) == 9:
        model, api_key, engine_id, gemini_key, r, t, q, k = sys.argv[1:9]
    elif len(sys.argv) == 6:
        model, r, t, q, k = sys.argv[1:6]
    else:
        print("Error: Please provide either command line arguments in the format '[-spanbert|gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>', or set the required environment variables and provide arguments in the format '[-spanbert|-gemini] <r> <t> <q> <k>'.")
        sys.exit(1)
    
    return [model[1:], api_key, engine_id, gemini_key, int(r), float(t), q, int(k)]

def search(api_key, engine_id, query):
    service = build("customsearch", "v1", developerKey=api_key)
    res = (
        service.cse()
        .list(
            q=query, 
            cx=engine_id,
        )
        .execute()
    )
    return res

def print_parameters(user_input, iter):
    model, api_key, engine_id, gemini_key, r, t, q, k = user_input
    print("Parameters:")
    print(f"Client key   = {api_key}")
    print(f"Engine key   = {engine_id}")
    print(f"Gemini key   = {gemini_key}")
    print(f"Method       = {model}")
    print(f"Relation     = {r}")
    print(f"Threshold    = {t}")
    print(f"Query        = {q}")
    print(f"# of Tuples  = {k}")
    print("Loading necessary libraries; This should take a minute or so ...)")
    print(f"=========== Iteration: {iter} - Query: {q} ===========")


def run_spanbert(entity_pairs, t, spanbert, r):
    predictions = spanbert.predict(entity_pairs)
    extracted = 0
    extraction_added = 0
    for extr, pred in list(zip(entity_pairs, predictions)):
        relation = pred[0]

        # Predicted relation must match user input r
        if (r == 1 and relation != 'per:schools_attended') or (r == 2 and relation != 'per:employee_of') or (r == 3 and (relation not in ('per:countries_of_residence', 'per:cities_of_residence', 'per:stateorprovinces_of_residence')) or (r == 4 and relation != 'org:top_members/employees')):
            continue
        print("\t\t=== Extracted Relation ===")
        print(f"\t\tInput tokens: {extr['tokens']}")
        extracted += 1
        subj = extr['subj'][0]
        obj = extr['obj'][0]
        confidence = pred[1]
        print(f"\t\tOutput Confidence: {confidence}; Subject: {subj}; Object: {obj};")
        if confidence > t:
            if X[(subj, relation, obj)] < confidence:
                X[(subj, relation, obj)] = confidence
                print("\t\tAdding to set of extracted relations")
                extraction_added += 1
            else:
                print("\t\tDuplicate with lower confidence than existing record, Ignoring this.")
        else:
            print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
        print("\t\t==========")
    return extracted, extraction_added


def get_gemini_response(prompt, model_name, max_tokens, temperature, top_p, top_k):

    # Initialize a generative model
    model = genai.GenerativeModel(model_name)

    # Configure the model with your desired parameters
    generation_config=genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    # Generate a response
    response = model.generate_content(prompt, generation_config=generation_config)

    return response.text


# To get a response from prompt
def safe_get_gemini_response(prompt_text, model_name, max_tokens, temperature, top_p, top_k, retries=3):
    for attempt in range(retries):
        try:
            return get_gemini_response(
                prompt_text,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        except (requests.exceptions.ConnectionError, InternalServerError, ResourceExhausted) as e:
            print(f"[Warning] Gemini API error on attempt {attempt+1}/{retries}: {e}")
            wait_time = 5 * (attempt + 1)
            print(f"Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
    print("[Error] All retries failed for Gemini API call.")
    return ""


def run_gemini(sentence, gemini_key, r):

    genai.configure(api_key=gemini_key)

    # Define relationship type
    if r == 1:
        relationship = "Schools_Attended"
        subj_type = "person"
        obj_type = "organization"
    elif r == 2:
        relationship = "Work_For"
        subj_type = "person"
        obj_type = "organization"
    elif r == 3:
        relationship = "Live_In"
        subj_type = "person"
        obj_type = "location, city, state, province, or country"
    else:
        relationship = "Top_Member_Employees"
        subj_type = "organization"
        obj_type = "person"

    # Prompt for Gemini
    prompt_text = f"""
Extract the subject and object with the relationship: {relationship}.
The subject is a {subj_type}, and the object is a {obj_type}.

Return only the following format:

subject: <subject text>
object: <object text>

Sentence: {sentence}
"""

    # Call Gemini with retries
    try:
        response_text = safe_get_gemini_response(
            prompt_text,
            model_name='gemini-2.0-flash',
            max_tokens=100,
            temperature=0.2,
            top_p=1,
            top_k=32
        )

    except Exception as e:
        print(f"Error from Gemini: {e}")
        time.sleep(5)
        return 0, 0

    # Parse Gemini output
    extracted_subj, extracted_obj = '', ''
    for line in response_text.split('\n'):
        if 'subject:' in line.lower():
            extracted_subj = line.split(':', 1)[1].strip()
        elif 'object:' in line.lower():
            extracted_obj = line.split(':', 1)[1].strip()

    if not extracted_subj or not extracted_obj:
        print(f"Failed to extract from response:\n{response_text}")
        return 0, 0

    # Output results
    print("\t\t=== Extracted Relation ===")
    print(f"\t\tSentence: {sentence}")
    print(f"\t\tSubject: {extracted_subj} ; Object: {extracted_obj} ;")

    extracted = 1
    extraction_added = 0
    if X.get((extracted_subj, extracted_obj), 0) < 1:
        X[(extracted_subj, extracted_obj)] = 1
        extraction_added = 1
        print(f"\t\tAdding to set of extracted relations")
    else:
        print(f"\t\tDuplicate. Ignoring this.")
    print("\t\t==========")

    return extracted, extraction_added



def process_url(url, url_num, r, model, t, gemini_key, spanbert):
    print(f"\n\nURL ( {url_num+1} / 10 ): {url}")
    # Check if we have processed this url before
    if url in seen_urls:
        print("We have seen this url before. Continuing...")
        return
    seen_urls.add(url)
    print("\tFetching text from url ...")
    # Retrieve the corresponding webpage
    response = requests.get(url)
    # Skip the webpage that cannot be retrieved
    if response.status_code != 200:
        print("Unable to fetch url. Continuing...")
        return
    # Extract plain text using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    plain_text = soup.get_text()
    # Truncate the text if too long
    if len(plain_text) > 10000:
        plain_text = plain_text[:10000]
    print(f"\tWebpage length (num characters): {len(plain_text)}")
    print("\tAnnotating the webpage using spacy...")
    # Split the text into sentences 
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(plain_text)
    num_sentences = len([s for s in doc.sents])
    print(f"\tExtracted {num_sentences} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
    # Find required named entities based on user input r
    entities_of_interest = ["PERSON"]
    live_in_obj = ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    if r in (1,2,4):
        entities_of_interest.append("ORGANIZATION")
    else:
        entities_of_interest.extend(live_in_obj)
    # sentence_extracted is the number of sentences that we extracted annotations from
    sentence_extracted = 0
    # website_extracted is the total number of extractions we had on current website
    website_extracted = 0
    # website_extraction_added is the number of extractions we added to X from current website
    website_extraction_added = 0
    for i, sentence in enumerate(doc.sents):
        entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        # examples are all entity pairs that have relationships we are interested in
        examples = []
        # extracted is the number of extractions we had on current sentence
        extracted = 0
        # extraction_added is the number of extractions we added to X from current sentence
        extraction_added = 0
        # check if the relationship match the input r
        for ep in entity_pairs:
            if r == 3:
                if "PERSON" not in [ep[1][1], ep[2][1]] or (ep[2][1] not in live_in_obj and ep[1][1] not in live_in_obj):
                    continue
                if ep[1][1] == "PERSON":
                    examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
                else:
                    examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
            else:
                if "PERSON" not in [ep[1][1], ep[2][1]] or "ORGANIZATION" not in [ep[1][1], ep[2][1]]:
                    continue
                if r == 4:
                    if ep[1][1] == "ORGANIZATION":
                        examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
                    else:
                        examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
                else:
                    if ep[1][1] == "PERSON":
                        examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
                    else:
                        examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
        if (i + 1) % 5 == 0:
            print(f"\tProcessed {i+1} / {num_sentences} sentences")
        if len(examples) == 0:
            continue
        if model == "spanbert":
            extracted, extraction_added = run_spanbert(examples, t, spanbert, r)
        else:
            extracted, extraction_added = run_gemini(sentence, gemini_key, r)
        if extracted > 0:
            sentence_extracted += 1
        website_extracted += extracted
        website_extraction_added += extraction_added
        
    print(f"\tExtracted annotations for {sentence_extracted} out of total {num_sentences} sentences")
    print(f"\tRelations extracted from this website: {website_extraction_added} (Overall: {website_extracted})")


def loop(user_input):
    # i is the current number of iterations
    i = 0
    model, api_key, engine_id, gemini_key, r, t, q, k = user_input
    # Keep track of the queries we have used so far in each iteration
    query_used = {q}
    spanbert = None
    if model == "spanbert":
        spanbert = SpanBERT("./pretrained_spanbert")
    
    while True:
        # Retrieve the top 10 results for the query from Google 
        res = search(api_key, engine_id, q)
        user_input[6] = q
        print_parameters(user_input, i)
        # j is the current number of url
        for j in range(10):
            url = res['items'][j]['formattedUrl']
            process_url(url, j, r, model, t, gemini_key, spanbert)
        i += 1
        if r == 1:
            relationship = "Schools_Attended"
        elif r == 2:
            relationship = "Work_For"
        elif r == 3:
            relationship = "Live_In"
        else:
            relationship = "Top_Member_Employees"
        if model == 'spanbert':
            sorted_X = sorted(X.items(), key=lambda item: item[1], reverse=True)
            if len(X) >= k:
                print(f"================== ALL RELATIONS for {relationship} ( {len(X)} ) ==================")
                for x in sorted_X:
                    print(f"Confidence: {x[1]}\t| Subject: {x[0][0]}\t| Object: {x[0][2]}")
                print(f"Total # of iterations = {i}")
                break
            else:
                query_found = False
                for x in sorted_X:
                    potential_query = x[0][0] + ' ' + x[0][2]
                    if potential_query not in query_used:
                        q = potential_query
                        query_found = True
                        query_used.add(potential_query)
                        break
                

        else:
            if len(X) >= k:
                print(f"================== ALL RELATIONS for {relationship} ( {len(X)} ) ==================")
                for x in X:
                    print(f"Subject: {x[0]}\t| Object: {x[1]}")
                print(f"Total # of iterations = {i}")
                break
            else:
                query_found = False
                for x in X:
                    potential_query = x[0] + ' ' + x[1]
                    if potential_query not in query_used:
                        q = potential_query
                        query_found = True
                        query_used.add(potential_query)
                        break
        if not query_found:
            print("ISE has stalled before retrieving k high-confidence tuples")
            break


def main():
    user_input = get_query()
    loop(user_input)


if __name__ == "__main__":
    main()
