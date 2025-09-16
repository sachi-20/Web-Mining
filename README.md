# Web-Mining

### How to Run the Program

Follow these steps to set up the project in a Google Cloud VM:

1. **Create a Virtual Environment**  
   First, create a virtual environment to isolate the project dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**  
   Install the required libraries:

   ```bash
   pip3 install --upgrade google-api-python-client
   pip3 install beautifulsoup4
   pip3 install -U pip setuptools wheel
   pip3 install -U spacy
   python3 -m spacy download en_core_web_lg
   pip install -q -U google-generativeai
   ```

3. **Run the Program**  
   Use the following command to run the program:

   ```bash
   python3 project2.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>
   ```

   - Replace `[-spanbert|-gemini]` with either -spanbert or -gemini, to indicate which relation extraction method you are requesting.
   - Replace `<google api key>` with Google Custom Search Engine API Key.
   - Replace `<google engine id>` with Google Custom Search Engine ID.
   - Replace `<google gemini api key>` with Google Gemini API Key.
   - Replace `<r>` with an integer between 1 and 4, indicating the relation to extract: 1 is for Schools_Attended, 2 is for Work_For, 3 is for Live_In, and 4 is for Top_Member_Employees.
   - Replace `<t>` with a real number between 0 and 1, indicating the "extraction confidence threshold," which is the minimum extraction confidence that you request for the tuples in the output; t is ignored if you are specifying -gemini.
   - Replace `<q>` with a "seed query," which is a list of words in double quotes corresponding to a plausible tuple for the relation to extract (e.g., "bill gates microsoft" for relation Work_For).
   - Replace `<k>` with an integer greater than 0, indicating the number of tuples that you request in the output.

### Internal Design of the Project

The project is structured as follows:

1. **Components**:

- **Search**: We perform a primary web search through Google's Custom Search API (`search()` method) initially.
- **Text Parsing**: Then we retrieve the first 10 pages' content and sanitize them with `BeautifulSoup` for text processing.
- **Entity Extraction**: From the cleaned text, spaCy is used for sentence splitting and named entity extraction.
- **Relation Extraction**: Depending on the model being employed (`-spanbert` or `-gemini`), we use either SpanBERT (via the `SpanBERT` class) or the Gemini API (via `google.generativeai`) to extract the relations from the entity pairs.
- **Tuple Collection**: Extracted relations are stored as tuples in the `X` set. Duplicates are removed on confidence scores.

2. **External Libraries**:

- `google-api-python-client`: For Google's Custom Search API interaction.
- `spacy`: For tokenization and named entity recognition.
- `beautifulsoup4`: Used to parse and extract text from HTML web pages.
- `google-generativeai`: Used to invoke the Google Gemini API to perform relation extraction.
- `spanbert`: A library for doing relation extraction using SpanBERT.

### A Detailed Description of How We Carried Out Relation Extraction

## **process_url() Function**

The `process_url()` function processes all URLs received from the Google Custom Search API:

1. **Check if the URL has been processed**: Skip previously processed URLs.
2. **Retrieve the webpage**: If the URL is new, retrieve the page and check that the status code is `200`.
3. **Plain text extraction**: Strip the HTML using **BeautifulSoup** and cut the text to 10,000 characters.
4. **Sentence segmentation**: Split the text into sentences with **SpaCy**.
5. **Entity extraction**: Extract entities of interest according to the user's relation type `r`.
6. **Entity pair generation**: For each sentence, create subject-object pairs and store useful pairs to the `examples` list.

## **run_spanbert() Function**

If `-spanbert` is selected, we:

1. **Extract relations**: Run SpanBERT on the `examples` list for relation extraction, filtering based on the confidence threshold `t`.
2. **Store valid relations**: Store relations with the right type and above-threshold confidence, skipping duplicates.

## **run_gemini() Function**

If `-gemini` is selected, we:

1. **Create a prompt**: Construct a prompt with the target relation, subject, object, and sentence.
2. **Run Gemini**: Utilize the Gemini API to get extractions of relations from the sentence.
3. **Store results**: Save non-empty extractions in output set `X`.

### Google Custom Search Engine API Key and Engine ID

Please replace the placeholders `<google api key>` and `<google engine id>` in the command above with own keys.

### References

- Wang, Richard C., and William W. Cohen. "Iterative set expansion of named entities using the web." 2008 eighth IEEE international conference on data mining. IEEE, 2008.
- Joshi, Mandar, et al. "Spanbert: Improving pre-training by representing and predicting spans." Transactions of the association for computational linguistics 8 (2020): 64-77.
