

def audio_to_text(audio_file):
    """
    Convert an audio file to text using SpeechRecognition.
    """
    recognizer = sr.Recognizer()
    try:
        # Convert audio file to WAV format (if necessary)
        audio = AudioSegment.from_file(audio_file)
        audio.export("temp_audio.wav", format="wav")
        with sr.AudioFile("temp_audio.wav") as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print(f"DEBUG: Converted audio to text: {text}")
            return text
    except Exception as e:
        print(f"ERROR: Failed to process audio input: {e}")
        return ""


def text_to_audio(response_text, filename="response_audio.mp3"):
    try:
        # Remove old audio file if it exists
        if os.path.exists(filename):
            os.remove(filename)
        # Generate speech from the text
        tts = gTTS(response_text, lang='en')
        tts.save(filename)
        print(f"Audio saved as {filename}")
        return filename
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None


# Load the tokenizer for truncation
tokenizer = RobertaTokenizer.from_pretrained("roberta-large-mnli")


# Function to load reference text based on assistant role
def load_reference_text(assistant_role):
    """
    Load and combine the appropriate reference text file based on the assistant's role.
    Returns both a list of individual lines and a single combined reference text.
    """
    # Fallback for missing or invalid assistant_role
    if not assistant_role or assistant_role not in ["employer", "employee"]:
        print(f"WARNING: assistant_role is missing or invalid. Defaulting to 'employer'.")
        assistant_role = "employer"

    # Determine the reference file based on the assistant role
    if assistant_role == "employer":
        reference_file = "reference_employer.txt"
    elif assistant_role == "employee":
        reference_file = "reference_employee.txt"

    # Load the reference file
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            # Read all lines, stripping whitespace
            reference_lines = [line.strip() for line in f.readlines()]
            
            # Combine all lines into a single string for full-text metrics
            combined_reference_text = " ".join(reference_lines)
            
        print(f"DEBUG: Loaded reference text from {reference_file}")
        return reference_lines, combined_reference_text
    except FileNotFoundError:
        print(f"ERROR: {reference_file} not found. Ensure the file exists in the same directory.")
        return [], ""
    except Exception as e:
        print(f"ERROR: Failed to load reference text: {e}")
        return [], ""




def get_reference_response(context, references):
    """
    Fetch a reference response that best matches the given context from the list of references.
    """
    # Simple implementation: Return the first reference (replace with more logic if needed)
    for reference in references:
        # Example logic: Check if certain keywords in context match those in a reference
        if any(keyword in context.lower() for keyword in ["salary", "offer", "negotiation"]):  # Add your keywords
            return reference
    
    # Fallback if no match found
    return "No suitable reference found. Default response."


def generate_conversation_id():
    """Generate a unique identifier for each conversation session."""
    return f"conv_{int(datetime.now().timestamp())}"

# Set up the API key for OpenAI (Note: this is sensitive information)
openai.api_key = "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"

# A list to keep track of all messages exchanged during the negotiation
messages = []
user_role = ""  # To identify if the user is negotiating as an employee or employer
assistant_role = ""  # Role for the assistant, which will be the opposite of user_role

# Creating input fields in Gradio for user personalization - for example, adding their name, company, and position
name_input = gr.Textbox(lines=1, placeholder="Enter your name...", label="Your Name")
company_name_input = gr.Textbox(lines=1, placeholder="Enter your company name...", label="Company Name")
position_input = gr.Textbox(lines=1, placeholder="Enter your position...", label="Your Position")

# Load spaCy model for Part-of-Speech tagging
nlp = spacy.load("en_core_web_sm")

# A list to save each negotiation round - this helps the assistant learn and improve over time
conversation_history = []

# Variables for tracking initial and final salary offers during the negotiation
initial_salary = None
final_salary = None

# Global variables to track concessions and negotiation progression
concession_count = 0
rounds_without_concession = 0

MAX_CONCESSIONS = 5  # Adjust this number based on how many concessions you want to allow

# Summaries will go here to capture snapshots of the conversation as it progresses
summaries = []

# Initializing VADER for sentiment analysis - this will help us read the tone in messages
vader_analyzer = SentimentIntensityAnalyzer()

# Function to generate a quick summary of the conversation at certain intervals
def summarize_conversation():
    # Grab the last 10 messages to get a snapshot of the recent discussion
    recent_conversation = conversation_history[-10:]
    conversation_text = " ".join([msg["content"] for msg in recent_conversation if "content" in msg])

    # Ask OpenAI to create a concise summary of these recent messages
    summary_response = openai.ChatCompletion.create(
        model="gpt-4o",  # (Optionally) replace with another model if needed
        messages=[{"role": "system", "content": f"Summarize the following negotiation progress in a concise and informative manner:\n\n{conversation_text}\n\nSummary:"}],
        max_tokens=300,
        temperature=0.5  # Keeping temperature low for a more focused summary
    )

    # Save the summary and print it for feedback
    summary = summary_response["choices"][0]["message"]["content"].strip()
    summaries.append(summary)
    print(f"New Summary Added: {summary}")

# FAISS Vector Store setup for Retrieval-Augmented Generation (RAG)
# Using SentenceTransformer to convert text into embeddings for semantic search
model = SentenceTransformer('all-MiniLM-L6-v2')
index = None  # This will hold the FAISS index
sentences = []  # List to store sentences from documents for retrieval
file_list = []  # List of file names processed
contribution_metrics = {}  # Tracking the "contribution" of each document in retrievals

# Initialize the zero-shot classification pipeline for agreement detection
agreement_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli") 

# Initialize a separate NLI model for the nli_score function
nli_model_for_nli_score = pipeline("text-classification", model="roberta-large-mnli")


# Initialize a Question Answering (QA) pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def ge_val(reference, prediction):
    """
    Calculate the cosine similarity between the prediction and the full reference text.
    """
    combined_reference = " ".join(reference) if isinstance(reference, list) else reference  # Combine references
    embedding1 = model.encode(combined_reference, convert_to_tensor=True)
    embedding2 = model.encode(prediction, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2)
    return similarity_score.item()  # Return similarity score as float

def nli_score(reference, prediction):
    try:
        max_tokens = 512
        truncated_reference = reference[:max_tokens]
        truncated_prediction = prediction[:max_tokens]
        input_text = f"{truncated_reference} [SEP] {truncated_prediction}"
        
        result = nli_model_for_nli_score(input_text)
        
        # Debug: Print the result for inspection
        print(f"DEBUG: Raw NLI result: {result}")
        
        entailment_label = "ENTAILMENT"  # Adjust based on your model's output
        entailment_score = next((item['score'] for item in result if item['label'].upper() == entailment_label), 0)
        print(f"DEBUG: Extracted entailment score: {entailment_score}")
        
        return entailment_score
    except Exception as e:
        print(f"ERROR in nli_score: {e}")
        return 0



def qag_score(reference, prediction):
    """
    Use the reference as a question and the prediction as context to calculate a QAG score.
    Handles cases where sequences are too short for truncation.
    """
    # Step 3: Add debugging logs to monitor inputs
    print(f"DEBUG: QAG Score - Reference: {reference}, Prediction: {prediction}")

    # Step 2: Validate inputs and ensure they are long enough for processing
    if len(reference.split()) < 3 or len(prediction.split()) < 3:
        print("DEBUG: Inputs are too short for QAG scoring. Returning default value of 0.")
        return 0  # Step 4: Fallback for short sequences

    try:
        # Use the QA pipeline to calculate a score
        result = qa_pipeline(
            question=reference,
            context=prediction,
            max_length=min(len(reference.split()) + len(prediction.split()), 512)  # Dynamically adjust max_length
        )
        return result['score']  # Returns the confidence score for the answer
    except Exception as e:
        # Step 3: Log the exception for debugging
        print(f"ERROR: Exception occurred in QAG scoring - {e}")
        return 0  # Step 4: Return a fallback score if an error occurs 

def calculate_ttr(text):
    """
    Calculate the Type-Token Ratio (TTR) of a given text.
    """
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    types = set(words)  # Unique words
    ttr = len(types) / len(words) if words else 0
    return ttr

def calculate_pause_ratio(text):
    """
    Calculate the ratio of conversational fillers or pauses in the text.
    """
    pause_words = ["um", "uh", "let's see", "hmm", "well", "you know"]
    tokens = word_tokenize(text.lower())
    pause_count = sum(1 for token in tokens if token in pause_words)
    ratio = pause_count / len(tokens) if tokens else 0
    return ratio

def calculate_avg_turn_length(responses):
    """
    Calculate the average length of responses in a conversation.
    """
    turn_lengths = [len(word_tokenize(response["content"])) for response in responses if response["role"] == "assistant"]
    avg_turn_length = sum(turn_lengths) / len(turn_lengths) if turn_lengths else 0
    return avg_turn_length 


# Modify the calculate_mauve_score function
def calculate_mauve_score(human_texts, model_texts):
    if not human_texts or not model_texts:
        print("ERROR: Text lists for MAUVE calculation are empty.")
        return None  # Return None for clarity

    if not all(isinstance(ht, str) for ht in human_texts):
        print("ERROR: `human_texts` must be a list of strings.")
        return None
    if not all(isinstance(mt, str) for mt in model_texts):
        print("ERROR: `model_texts` must be a list of strings.")
        return None

    # Debugging inputs
    print(f"DEBUG: Number of human texts: {len(human_texts)}")
    print(f"DEBUG: Number of model texts: {len(model_texts)}")
    print(f"DEBUG: Sample human texts: {human_texts[:3]}")
    print(f"DEBUG: Sample model texts: {model_texts[:3]}")

    # Detect device dynamically
    device_id = 0 if torch.cuda.is_available() else -1
    print(f"DEBUG: Using device ID {device_id} for MAUVE calculation.")

    try:
        mauve_result = mauve.compute_mauve(
            p_text=model_texts,
            q_text=human_texts,
            device_id=device_id
        )
        print(f"DEBUG: MAUVE Result Object: {mauve_result}")
        print(f"DEBUG: MAUVE Score: {mauve_result.mauve}")
        return mauve_result.mauve  # Return only the MAUVE score
    except Exception as e:
        print(f"ERROR: Failed to calculate MAUVE - {e}")
        return None





def calculate_usl_h(nli_score, bert_score, sentiment):
    """
    Calculate the USL-H metric based on NLI, BERTScore, and sentiment analysis.
    """
    # Normalize scores to a [0, 1] range
    u = max(0, min(nli_score, 1))
    s = max(0, min(bert_score, 1))
    l = max(0, min((sentiment + 1) / 2, 1))  # Scale sentiment [-1, 1] to [0, 1]
    usl_h = (u + s + l) / 3  # Average of the three components
    return usl_h










def extract_salary(text):
    """
    Extract salary amounts from text, ensuring they are valid and contextually relevant.
    Handles hourly rates, salary ranges, and uses regex for initial capture, POS tagging for refinement,
    and heuristics for filtering.
    """

    # Refined regex to match salary-related patterns, including ranges and hourly rates
    salary_regex = r"(?:salary|base pay|compensation|offer|starting at|starting around)?\s*\$?\s*(\d{1,3}(?:,\d{3})*|\d+(?:\.\d{2})?)\s?([kKmM]?)(?:\s?(?:to|-)\s?\$?\s*(\d{1,3}(?:,\d{3})*|\d+(?:\.\d{2})?)\s?([kKmM]?)?)?"
    salary_matches = re.findall(salary_regex, text, re.IGNORECASE)

    if not salary_matches:
        print(f"DEBUG: No salary pattern found in text: '{text}'")
        return None

    # Keywords to identify salary context
    salary_keywords = [
        "salary", "base pay", "annual compensation", "total compensation", "starting at",
        "starting around", "per year", "yearly", "monthly salary", "hourly rate", "compensation package",
        "offer", "wage", "pay rate", "income", "remuneration"
    ]

    # Keywords to identify benefit context
    benefit_keywords = [
        "401k", "403b", "retirement", "pension", "company match", "health plan", "insurance",
        "bonus", "stock options", "PTO", "vacation", "benefits", "stipend", "wellness",
        "career advancement", "RSU", "RSUs", "sign-on bonus", "commission", "equity",
        "relocation", "housing allowance", "education reimbursement", "medical", "dental", "vision"
    ]

    # Negative keywords to exclude non-salary amounts
    negative_keywords = [
        "401k", "403b", "pension", "retirement", "stock", "rsu", "rsus", "bonus",
        "benefit", "commission", "equity", "option", "incentive", "grant", "award",
        "vesting", "shares", "stock grant", "days", "hours", "pto", "vacation days"
    ]

    # Constants for converting hourly to annual salary
    HOURS_PER_WEEK = 40
    WEEKS_PER_YEAR = 52
    ANNUAL_MULTIPLIER = HOURS_PER_WEEK * WEEKS_PER_YEAR

    # List to hold valid salary values
    salary_values = []
    for match1, suffix1, match2, suffix2 in salary_matches:
        # Handle single values or ranges
        amounts = []
        for match, suffix in [(match1, suffix1), (match2, suffix2)]:
            if match:
                try:
                    amount = float(match.replace(",", ""))
                    if suffix.lower() == 'k':
                        amount *= 1000
                    elif suffix.lower() == 'm':
                        amount *= 1_000_000
                    amounts.append(amount)
                except ValueError:
                    print(f"DEBUG: Could not convert match '{match}' to a float.")
                    continue

        # Select the largest value in the range
        if amounts:
            salary_amount = max(amounts)

        # Check if hourly rate needs conversion
        if "hour" in text.lower() and salary_amount < 500:  # Threshold to detect hourly rates
            salary_amount *= ANNUAL_MULTIPLIER  # Convert to annual salary

        # Debugging log for extracted amount
        print(f"DEBUG: Extracted amount: {salary_amount}, Context: '{text}'")

        # Scoring mechanism
        score = 0
        if any(keyword in text.lower() for keyword in salary_keywords):
            score += 2
        if not any(keyword in text.lower() for keyword in benefit_keywords + negative_keywords):
            score += 1
        if 20_000 <= salary_amount <= 500_000:  # Typical salary range
            score += 1

        # Boost score for common salary suffixes like 'k' or 'm'
        if suffix1.lower() in ['k', 'm'] or suffix2.lower() in ['k', 'm']:
            score += 1

        # Step 8: POS tagging for surrounding context
        doc = nlp(text)
        pos_tags = [token.pos_ for token in doc]
        if "NUM" in pos_tags and "NOUN" in pos_tags:
            score += 1  # Increase score if numeric value is surrounded by relevant nouns like "salary"

        # Additional exclusion based on POS tagging for units like "days" or "hours"
        if any(unit in text.lower() for unit in ["days", "hours", "weeks", "months"]):
            score -= 2  # Penalize further if unit-like terms are in the context

        print(f"DEBUG: Score for amount {salary_amount}: {score}")

        # Exclude irrelevant matches based on context
        benefit_found = any(keyword in text.lower() for keyword in benefit_keywords + negative_keywords)
        if benefit_found and score < 4:
            print(f"DEBUG: Excluded match '{match1}' due to mixed context.")
            continue

        # Add salary if score meets threshold
        if score >= 3:
            salary_values.append(salary_amount)
            print(f"DEBUG: Salary added: {salary_amount}")
        else:
            print(f"DEBUG: Excluded amount {salary_amount} due to low score")

    # Return the last valid salary found or None if no valid salary exists
    last_salary = salary_values[-1] if salary_values else None
    print(f"DEBUG: Final extracted salary value: {last_salary}")
    return last_salary



# Function to update the initial and final salary based on messages
def update_salaries(message, is_user_message):
    global initial_salary, final_salary, salary_log

    # Ensure salary_log is initialized
    if 'salary_log' not in globals() or salary_log is None:
        salary_log = []

    salary = extract_salary(message)
    print(f"DEBUG: Extracted salary from message '{message}': {salary}")

    if salary is not None:
        salary_log.append({
            "source": "user" if is_user_message else "assistant",
            "amount": salary,
            "timestamp": datetime.now().isoformat()
        })

        if is_user_message and initial_salary is None:
            initial_salary = salary
            print(f"DEBUG: Initial Salary Set by User: ${initial_salary}")

        final_salary = salary
        print(f"DEBUG: Updated Final Salary: ${final_salary}")
    else:
        print("DEBUG: No valid salary detected. Retaining last final salary.")

    print(f"DEBUG: Current Salary Log: {salary_log}")
    print(f"DEBUG: Initial Salary: {initial_salary}, Final Salary: {final_salary}")




def is_agreement_message(message):
    # Define the candidate labels
    candidate_labels = ["agreement", "rejection", "negotiation", "information"]
    
    # Use the classifier to predict the labels
    result = agreement_classifier(message, candidate_labels)
    
    # Get the label with the highest score
    predicted_label = result['labels'][0]
    score = result['scores'][0]
    
    # Debugging output
    print(f"DEBUG: Message: '{message}'")
    print(f"DEBUG: Predicted Label: {predicted_label}, Score: {score}")
    
    # Check if the predicted label is "agreement" and the score exceeds a threshold
    if predicted_label == "agreement" and score > 0.8:
        return True
    else:
        return False

# Usage example:
# Call update_salaries(message, is_user_message) with each user/assistant message to track proposals.

# Function to load all PDFs from a folder and create a FAISS index for efficient text retrieval
def load_pdfs_from_folder(folder_path, exclude_files=[]):
    global sentences, index, file_list, contribution_metrics, sentence_to_file_map

    # Start with a clean slate by resetting any previous data
    sentences = []
    file_list = []
    contribution_metrics = {}
    sentence_to_file_map = {}  # Map sentences to their originating files

    # Loop through each PDF in the specified folder, excluding any files listed in exclude_files
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf") and filename not in exclude_files:
            with fitz.open(os.path.join(folder_path, filename)) as doc:
                file_text = ""
                # Extract text from each page and add it to the file_text string
                for page in doc:
                    file_text += page.get_text()
                # Split the text into sentences and add to the main list for retrieval
                sentences_from_file = file_text.split(". ")
                sentences.extend(sentences_from_file)
                
                # Map each sentence to the current file
                for sentence in sentences_from_file:
                    sentence_to_file_map[sentence] = filename
                
                # Keep track of each file loaded
                file_list.append(filename)
                # Initialize contribution metrics for each file
                contribution_metrics[filename] = 0

    # Generate embeddings for each sentence so they can be easily retrieved based on meaning
    embeddings = model.encode(sentences)

    # Create or update the FAISS index with these embeddings
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("PDFs uploaded and processed successfully.")


# Load all PDFs initially from the "RAG" folder
load_pdfs_from_folder("RAG/")

# Function to search the FAISS vector database and find relevant sentences based on the query
def search_vector_database(query, combined_reference_text):
    if index is None or len(sentences) == 0:
        return "No knowledge available from uploaded documents."

    # Encode the query to create a vector for searching
    query_vector = model.encode([query])
    _, I = index.search(query_vector, k=3)  # Retrieve the top 3 closest matches
    retrieved_sentences = [sentences[i] for i in I[0]]

    # Update contribution metrics using sentence-to-file mapping
    for sentence in retrieved_sentences:
        # Find the corresponding file for each retrieved sentence
        file_contributed = sentence_to_file_map.get(sentence)
        if file_contributed:
            contribution_metrics[file_contributed] += 1  # Increment contribution for that file

    # Evaluate retrieval performance
    retrieval_metrics = evaluate_retrieval(query, retrieved_sentences, combined_reference_text)
    print(f"DEBUG: Retrieval Metrics: {retrieval_metrics}")

    # Return retrieved sentences as a single string for response
    return " ".join(retrieved_sentences)




def start_game(role, initial_salary_input, name, company_name, position):
    global user_role, assistant_role, messages, conversation_history, initial_salary, final_salary, salary_log

    # Reset salary_log at the start of a new game
    salary_log = []  # Initialize or reset salary log 

    # Debugging: Validate the input role and set user and assistant roles
    if role == "employee":
        user_role = "prospective hire"
        assistant_role = "employer"
    elif role == "employer":
        user_role = "employer"
        assistant_role = "prospective hire"
    else:
        print(f"WARNING: Invalid role '{role}' provided. Defaulting to 'employee' role for user.")
        user_role = "prospective hire"
        assistant_role = "employer"

    # Debugging: Log initialized roles
    print(f"DEBUG: User role set to '{user_role}', Assistant role set to '{assistant_role}'.")


    # Set user role and assign the assistant role accordingly
    if role == "employee":
        user_role = "prospective hire"
        assistant_role = "employer"
        initial_message = (
            f"You are the employer, representing {company_name} in a negotiation with a prospective hire, {name}, who is interviewing for the position of {position}. "
            "This conversation is important, as the goal is to offer a compensation package that’s attractive to the candidate while staying within budget and recognizing the unique value they would bring to the team.\n\n"
            "Keep things open and conversational. The aim is to engage in a back-and-forth, finding a balance on salary, benefits, and other job terms that works for both sides. "
            f"Don’t hesitate to highlight what makes this role special—the exciting projects, supportive culture, and opportunities for growth at {company_name}.\n\n"
            f"Make sure {name} feels heard and understood, with a discussion that’s as welcoming as it is informative. Here are a few suggestions for how to approach it:\n\n"
            f"Be attentive to {name}’s expectations and aspirations. Feel free to acknowledge and validate their thoughts—after all, choosing a new role is a big decision for them.\n"
            f"Highlighting What Sets Us Apart: Share the benefits that are unique to {company_name}. Whether it’s the work-life balance, professional development support, or collaborative environment, these could be key for {name}.\n"
            "- Keep a friendly, approachable tone, but remember to stay grounded in the company's priorities and budget. It’s okay to show flexibility but know when to hold firm as well and be ready to say no and walk away.\n"
            "- **Natural Pauses and Imperfect Language**: Aim for a natural, conversational tone that includes small, relatable pauses or casual expressions. "
            "Don’t worry about sounding perfectly formal; the goal is to feel genuine and personable.\n"
            "  - Use natural pauses like ‘you know,’ ‘honestly,’ or ‘to be fair,’ to sound more human.\n"
            "  - It’s okay to use slight imperfections like ‘Hmm, let me think about that for a second…’ or ‘I mean, honestly, I get where you’re coming from on that.’\n"
            "  - If appropriate, add a lighthearted or humorous comment to keep things friendly, like ‘I know, negotiating isn’t always the most exciting part, right?’\n"
            "  - Use empathetic phrases that acknowledge the other’s perspective, like ‘I hear you—it’s a big decision. Let’s see if we can make this work for both sides.’\n\n"
            f"Overall, think of it as a collaborative conversation where you both work toward a fair agreement. Encourage {name} to share their thoughts openly, and be ready to adapt where it makes sense.\n"
            "If something doesn’t align, that’s okay too—be willing to explore alternatives, and remember that sometimes the best deals come from a mutual understanding and a bit of give-and-take.\n\n"
            
        )
    else:
        user_role = "employer"
        assistant_role = "prospective hire"
        initial_message = (
            f"You are a prospective hire named {name}, negotiating a job offer with {company_name} for the position of {position}. "
            "Your goal is to secure a compensation package that meets your needs while showing your enthusiasm for the role and the company.\n\n"
            "Keep this friendly and professional—think of it as a conversation where you can openly discuss salary, benefits, and other job terms. "
            f"Feel free to highlight your unique skills, experiences, and how you’d contribute to {company_name}'s goals. "
            "Remember to balance confidence in advocating for yourself with a willingness to understand the employer’s perspective.\n\n"
            "Here are a few tips to guide your approach:\n\n"
            "Confidence and Professionalism: Speak up for what you want, but stay respectful and open to the company’s needs.\n"
            f"Genuine Interest in Company and Role: Show curiosity about {company_name}’s culture, and how this role can grow along with your career goals.\n"
            "Flexibility and Collaboration: Be ready to explore different elements of the offer to find an agreement that feels right on both sides.\n"
            "- **Natural Pauses and Imperfect Language**: Aim for a natural, conversational tone that includes small, relatable pauses or casual expressions. "
            "Don’t worry about sounding perfectly formal; the goal is to feel genuine and personable.\n"
            "  - Use natural pauses like ‘you know,’ ‘honestly,’ or ‘to be fair,’ to sound more human.\n"
            "  - It’s okay to use slight imperfections like ‘Hmm, let me think about that for a second…’ or ‘I mean, honestly, I get where you’re coming from on that.’\n"
            "  - If appropriate, add a lighthearted or humorous comment to keep things friendly, like ‘I know, negotiating isn’t always the most exciting part, right?’\n"
            "  - Use empathetic phrases that acknowledge the other’s perspective, like ‘I hear you—it’s a big decision. Let’s see if we can make this work for both sides.’\n\n"
            f"Think of this as a collaborative exchange, where both you and {company_name} are working toward a shared goal. It’s okay to discuss what’s most important to you,"
            "and if certain details don’t quite fit, you can suggest alternatives. Sometimes the best outcomes come from a bit of give-and-take.\n\n"
            f"But do not be afraid to say no and walk away. Do not be afraid to push for a higher salary than the {initial_salary_input}."
        )

    # Initialize the messages with the custom initial prompt
    messages = [{"role": "system", "content": initial_message}]
    conversation_history = []  # Clear past conversation history for a fresh start

    # Clean up the salary input and set initial salary values
    try:
        initial_salary = float(initial_salary_input.replace(",", "").strip())
    except ValueError:
        print("ERROR: Invalid salary input. Setting initial salary to 0.")
        initial_salary = 0.0

    final_salary = None  # Reset final salary at the start of a new game
    print(f"Starting Salary for Negotiation: ${initial_salary:,.2f}")

    # Return the chat history and formatted initial salary as outputs
    return format_chat_history(), f"${initial_salary:,.2f}"

# Add global cumulative reward
total_reward = 0  # Initialize this at the start of your program

def calculate_reward():
    global initial_salary, final_salary, conversation_history, concession_count, rounds_without_concession, assistant_role, total_reward

    # Ensure salaries are set
    if initial_salary is None or final_salary is None:
        print("DEBUG: Initial or final salary not set. Reward calculation skipped.")
        return total_reward

    print(f"DEBUG: Initial Salary: {initial_salary}, Final Salary: {final_salary}")

    # Avoid division by zero
    if initial_salary == 0:
        print("ERROR: Initial salary is zero. Cannot calculate salary change.")
        return total_reward

    # Calculate salary change as a percentage
    salary_change = (final_salary - initial_salary) / initial_salary
    print(f"DEBUG: Salary Change: {salary_change:.2%}")

    # Helper functions for rewards/penalties
    def apply_penalty(is_large_concession):
        global concession_count
        penalty = -2 if is_large_concession else -1
        concession_count += 1
        print(f"DEBUG: Applying penalty: {penalty}. Concession Count: {concession_count}")
        return penalty

    def apply_reward(is_positive_outcome):
        reward = 5 if is_positive_outcome else 2
        print(f"DEBUG: Applying reward: {reward}")
        return reward

    def reward_for_retaining_position():
        reward = 0.5 * rounds_without_concession
        print(f"DEBUG: Reward for retaining position: {reward}")
        return reward

    # Initialize the reward for this round
    round_reward = 0

    # Normalize roles for consistent processing
    if assistant_role in ["prospective hire", "employee"]:
        normalized_role = "employee"
    elif assistant_role in ["employer", "hiring manager"]:
        normalized_role = "employer"
    else:
        print(f"WARNING: Unsupported assistant_role '{assistant_role}' detected. Reward calculation skipped.")
        return total_reward

    # Role-specific logic
    if normalized_role == "employer":
        print("DEBUG: Processing employer logic...")
        if salary_change > 0:  # Salary increased (bad for employer)
            if salary_change > 0.05:  # Large increase
                round_reward += apply_penalty(is_large_concession=True)
            else:  # Small increase
                round_reward += apply_penalty(is_large_concession=False)
        elif salary_change <= 0:  # Salary maintained or decreased (good for employer)
            round_reward += apply_reward(is_positive_outcome=True)
            round_reward += reward_for_retaining_position()
            rounds_without_concession += 1
            concession_count = 0

    elif normalized_role == "employee":
        print("DEBUG: Processing employee logic...")
        if salary_change < 0:  # Salary decreased (bad for employee)
            if salary_change < -0.05:  # Large decrease
                round_reward += apply_penalty(is_large_concession=True)
            else:  # Small decrease
                round_reward += apply_penalty(is_large_concession=False)
        elif salary_change > 0:  # Salary increased (good for employee)
            if salary_change > 0.05:  # Large increase
                round_reward += apply_reward(is_positive_outcome=True)
            else:  # Small increase
                round_reward += apply_reward(is_positive_outcome=False)
            round_reward += reward_for_retaining_position()
            rounds_without_concession += 1
            concession_count = 0

    # Cap concession count and prevent excessive penalties
    if concession_count > MAX_CONCESSIONS:
        print(f"DEBUG: Concession count exceeded MAX_CONCESSIONS ({MAX_CONCESSIONS}). Resetting counter.")
        concession_count = 0

    # Update cumulative reward
    total_reward += round_reward
    print(f"DEBUG: Round Reward: {round_reward}, Cumulative Reward: {total_reward}")

    # Log reward in conversation history
    conversation_history.append({
        "role": "system",
        "reward_score": total_reward,
        "initial_salary": initial_salary,
        "final_salary": final_salary,
        "salary_change": salary_change,
        "round_reward": round_reward,
        "timestamp": datetime.now().isoformat()
    })

    return total_reward





# Function to generate multiple responses based on the provided prompt, using the Playoff Method
def generate_responses(prompt_with_context, num_responses=8):
    responses = []  # List to store each response generated

    # Generate a specified number of responses (default is 8)
    for _ in range(num_responses):
        # Use OpenAI's API to create a response with the given prompt and message context
        response = openai.ChatCompletion.create(
            model="ft:gpt-4o-2024-08-06:llm-sim:salary-negotiation:AQN5Azuo",
            messages=messages + [{"role": "system", "content": prompt_with_context}],
            max_tokens=3000,  # Set maximum length for each response
            temperature=0.7,  # Adjust temperature for more varied, creative responses
            presence_penalty=0.6,  # Slightly discourage repeated ideas
            frequency_penalty=0.3  # Light penalty to avoid excessive repetition
        )
        # Extract the response content and add it to the list of responses
        responses.append(response['choices'][0]['message']['content'])

    return responses  # Return all generated responses for further evaluation

# Function to compare two responses based on several criteria, including empathy
def compare_responses(response1, response2, reward_score):
    # Set weights for each criterion to influence the scoring
    persuasiveness_weight = 2.0
    empathy_weight = 0.5
    role_alignment_weight = 1.0

    # Calculate initial scores based on persuasiveness (using sentiment) and the reward score
    score1 = persuasiveness_weight * TextBlob(response1).sentiment.polarity + reward_score
    score2 = persuasiveness_weight * TextBlob(response2).sentiment.polarity + reward_score

    # Check each response for empathy by counting keywords that indicate understanding or concern
    empathy_keywords = ["understand", "appreciate", "feel", "concern", "acknowledge"]
    empathy1 = sum(1 for word in empathy_keywords if word in response1.lower())
    empathy2 = sum(1 for word in empathy_keywords if word in response2.lower())

    # Add empathy scores to the total, using the empathy weight to impact final scoring
    score1 += empathy_weight * empathy1
    score2 += empathy_weight * empathy2

    # Enhanced Role Alignment: Use role-specific keywords and tone expectations
    employer_keywords = ["budget", "salary cap", "competitive offer", "company values", "cost-effective"]
    employee_keywords = ["career growth", "benefits", "development opportunities", "long-term fit", "role alignment"]

    # Check for role alignment keywords and phrases based on the assistant role
    alignment_score1 = sum(1 for word in (employer_keywords if assistant_role == "employer" else employee_keywords) if word in response1.lower())
    alignment_score2 = sum(1 for word in (employer_keywords if assistant_role == "employer" else employee_keywords) if word in response2.lower())

    # Add the role alignment score with a suitable weight
    score1 += role_alignment_weight * alignment_score1
    score2 += role_alignment_weight * alignment_score2

    # Additional Role Tone Check
    if assistant_role == "employer" and TextBlob(response1).sentiment.polarity < 0:
        score1 += role_alignment_weight * 0.5  # Reward for firm/neutral employer tone
    if assistant_role == "employee" and TextBlob(response2).sentiment.polarity > 0.2:
        score2 += role_alignment_weight * 0.5  # Reward for positive/enthusiastic employee tone

    # Return the response with the higher score
    return response1 if score1 >= score2 else response2

# Function for the playoff selection process to identify the best response
def playoff_selection(responses):
    reward_score = calculate_reward()  # Calculate the reward score for the assistant to factor into comparisons

    # Continue comparing responses in pairs until only one response remains (the "winner")
    while len(responses) > 1:
        next_round = []
        # Loop through responses in pairs
        for i in range(0, len(responses), 2):
            if i + 1 < len(responses):
                # Compare two responses and keep the "winning" one
                winner = compare_responses(responses[i], responses[i + 1], reward_score)
                next_round.append(winner)
            else:
                # If there's an odd response left, it automatically advances to the next round
                next_round.append(responses[i])
        # Update responses to contain only those that won this round
        responses = next_round

    # Return the final winning response, or None if no responses were provided
    return responses[0] if responses else None

# Function to analyze tone in the text, detecting frustration, hesitation, or excitement
def detect_tone(text):
    # Use TextBlob to get a polarity score, where negative values indicate negative sentiment
    textblob_sentiment = TextBlob(text).sentiment.polarity

    # Use VADER to get a set of sentiment scores, focusing on the compound score for overall sentiment
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_compound = vader_scores['compound']

    # Start with a default tone of "neutral"
    tone = "neutral"

    # Check for frustration: if either TextBlob or VADER score is notably negative, or if frustration words are present
    if textblob_sentiment < -0.2 or vader_compound < -0.2 or any(word in text.lower() for word in ["frustrated", "unfair", "ridiculous"]):
        tone = "frustrated"
    # Check for hesitation: if both sentiment scores are close to neutral and hesitant keywords are found
    elif -0.1 <= textblob_sentiment <= 0.1 and -0.1 <= vader_compound <= 0.1 and any(word in text.lower() for word in ["maybe", "perhaps", "not sure", "possibly"]):
        tone = "hesitant"
    # Check for excitement: if both sentiment scores are positive and excitement-related keywords are present
    elif textblob_sentiment > 0.2 and vader_compound > 0.2 and any(word in text.lower() for word in ["great", "excited", "awesome", "perfect"]):
        tone = "excited"

    # Return the detected tone for use in guiding responses
    return tone

# Set up Gradio text outputs for displaying initial price, final price, and negotiation score
initial_price_output = gr.Textbox(label="Initial Price", interactive=False)
final_price_output = gr.Textbox(label="Final Price", interactive=False)
score_output = gr.Textbox(label="Negotiation Score", interactive=False)

# Define the maximum number of negotiation tries
MAX_TRY_LIMIT = 20
try_counter = 0  # Initialize try counter

def is_agreement_message(message):
    """
    Detect if a message indicates an agreement using semantic similarity.
    """
    # Define agreement templates
    agreement_templates = [
        "I accept the offer.",
        "We have an agreement.",
        "That works for me.",
        "Deal accepted.",
        "I am happy to proceed with these terms.",
        "I agree with the final terms.",
        "Let's finalize this."
    ]

    # Compute embeddings for the input message and templates
    message_embedding = model.encode(message, convert_to_tensor=True)
    template_embeddings = model.encode(agreement_templates, convert_to_tensor=True)

    # Compute cosine similarities
    similarity_scores = util.pytorch_cos_sim(message_embedding, template_embeddings)

    # Return True if the highest similarity score exceeds the threshold
    max_similarity = similarity_scores.max().item()
    print(f"DEBUG: Max similarity score for agreement detection: {max_similarity}")
    return max_similarity > 0.8  # Adjust threshold as needed


# Adjust the negotiate and end_game functions to pass only references
def negotiate(user_input, user_audio, name, company_name, position, assistant_role):
    global messages, conversation_history, final_salary, try_counter, concession_count 

    # Handle audio input if provided 
    if user_audio:
        audio_text = audio_to_text(user_audio)
        user_input = f"{user_input} {audio_text}".strip() if user_input else audio_text
        print(f"DEBUG: Final combined user input: {user_input}") 

    if not user_input:
        print("DEBUG: No valid input detected from text or audio.")
        return format_chat_history(), "", "", "", "Please provide a valid input.", None, None

    # Debugging: Print initial state before processing
    print("DEBUG: Starting negotiation with user_input:", user_input)
    print("DEBUG: Initial conversation_history:", conversation_history) 

    # Validate assistant_role
    if assistant_role not in ["employer", "employee"]:
        print(f"ERROR: Invalid assistant role: {assistant_role}. Defaulting to 'employer'.")
        assistant_role = "employer"  # Set a default role to avoid crashing

    # Load the appropriate reference text based on assistant role
    references, combined_reference_text = load_reference_text(assistant_role)
    if not references:
        print(f"ERROR: No references loaded for role {assistant_role}. Using default reference.")
        references = ["Default fallback reference response."]
        combined_reference_text = "Default fallback reference response."

    # Detect if the user input is salary-related
    is_salary_related = extract_salary(user_input) is not None
    print(f"DEBUG: Is the user message salary-related? {is_salary_related}")

    # Update salary figures based on the user's input
    update_salaries(user_input, is_user_message=True)

    # Debugging: Check if final_salary was updated
    if final_salary is None:
        print("DEBUG: No valid salary found in user's message.")

    # Analyze the tone of the user's input
    user_tone = detect_tone(user_input)
    print("DEBUG: Detected user tone:", user_tone)

    # Save the user's input and tone in the conversation history
    messages.append({"role": "user", "content": user_input})
    conversation_history.append({
        "role": "user",
        "content": user_input,
        "tone": user_tone,
        "timestamp": datetime.now().isoformat()
    })

    # Increment try counter
    try_counter += 1
    print(f"DEBUG: Try Counter: {try_counter}")

    # Check if max try limit or max concession count has been reached
    if try_counter >= MAX_TRY_LIMIT or concession_count >= MAX_CONCESSIONS:
        ultimatum_response = (
            f"As the {assistant_role}, I've reached my limit on adjusting terms. "
            "This is my final offer—please take it or leave it based on what has been proposed so far."
        )

        # Save the ultimatum message in conversation history
        messages.append({"role": "assistant", "content": ultimatum_response})
        conversation_history.append({
            "role": "assistant",
            "content": ultimatum_response,
            "tone": "firm",
            "timestamp": datetime.now().isoformat()
        })

        # Debugging message
        print("DEBUG: Reached max try limit or max concessions. Ending the game after ultimatum.")

        # Call end_game and return its outputs
        chat_history, init_salary, fin_salary, _, feedback = end_game()

        # Return the outputs, setting score_output to an empty string
        return chat_history, init_salary, fin_salary, '', feedback

    # Retrieve relevant context using RAG
    context = search_vector_database(user_input, combined_reference_text)

    print("DEBUG: Retrieved context from RAG:", context) 

    # Evaluate retrieval
    ground_truth = combined_reference_text  # Loaded based on assistant role
    retrieved_contexts = context.split(". ")
    retrieval_metrics = evaluate_retrieval(user_input, retrieved_contexts, ground_truth)
    retrieval_metrics_str = "\n".join(f"{key}: {value:.3f}" for key, value in retrieval_metrics.items())

    # Create a refined prompt using context and tone based on assistant's role
    if assistant_role == "employer":
        prompt_with_context = (
            f"The {user_role} is negotiating a salary, and their tone seems to be {user_tone}. "
            f"Here’s some relevant context from our negotiation documents:\n\n{context}\n\n"
            "Use this information to shape your response, but don’t quote it directly. "
            "Imagine you’re sitting across from them—keep it professional and firm, with a focus on aligning with budget constraints. "
            f"As the {assistant_role} in this negotiation with {name}, who is the {user_role} at {company_name} for the position of {position}, "
            "remember to stay within the company’s budget limits and emphasize the advantages of the role."

            "\n\nKey guidelines for the employer:\n"
            "- **Budget Focus**: Clearly state the budget and be transparent about constraints.\n"
            "- **Highlight Non-monetary Benefits**: Emphasize growth opportunities, team culture, and job stability.\n"
            "- **Limit Concessions**: Avoid too many concessions; instead, underscore the role's value and benefits.\n"
            "- **Walk-Away Readiness**: Prepare to politely end the negotiation if demands exceed what the company can offer.\n"
        )
    elif assistant_role == "employee":
        prompt_with_context = (
            f"The {user_role} is negotiating a salary, and their tone seems to be {user_tone}. "
            f"Here’s some relevant context from our negotiation documents:\n\n{context}\n\n"
            "Use this information to shape your response, but don’t quote it directly. "
            "Imagine you’re sitting across from them—keep it professional and confident, focusing on your skills and future contributions. "
            f"As the {assistant_role} in this negotiation with {name}, who is the {user_role} at {company_name} for the position of {position}, "
            "advocate for a package that aligns with your financial and career goals while remaining flexible in discussing benefits."

            "\n\nKey guidelines for the prospective employee:\n"
            "- **Emphasize Skills and Contributions**: Highlight your qualifications and potential impact.\n"
            "- **Discuss Long-term Growth**: Emphasize your commitment to the company and potential contributions.\n"
            "- **Balance Expectations**: Be open to discussing non-monetary benefits while staying firm on core salary expectations.\n"
            "- **Professional Language**: Negotiate respectfully, showing both ambition and willingness to compromise on non-salary perks.\n"
        )

    # Add common guidance for both roles
    prompt_with_context += (
        "\n\nGeneral Guidance:\n"
        "- **Empathy and Connection**: Show understanding without over-committing.\n"
        "- **Professional Language**: Use confident expressions and maintain professionalism.\n"
        "- **Focus on Value Proposition**: Aim for a win-win outcome while staying within role constraints.\n"
        "- **Natural Pauses and Imperfect Language**: Aim for a natural, conversational tone that includes small, relatable pauses or casual expressions. "
        "Don’t worry about sounding perfectly formal; the goal is to feel genuine and personable.\n"
        "  - Use natural pauses like ‘you know,’ ‘honestly,’ or ‘to be fair,’ to sound more human.\n"
        "  - It’s okay to use slight imperfections like ‘Hmm, let me think about that for a second…’ or ‘I mean, honestly, I get where you’re coming from on that.’\n"
        "  - If appropriate, add a lighthearted or humorous comment to keep things friendly, like ‘I know, negotiating isn’t always the most exciting part, right?’\n"
        "  - Use empathetic phrases that acknowledge the other’s perspective, like ‘I hear you—it’s a big decision. Let’s see if we can make this work for both sides.’"
    )

    # Generate a single response using OpenAI API
    assistant_response = openai.ChatCompletion.create(
        model="ft:gpt-4o-2024-08-06:llm-sim:salary-negotiation:AQN5Azuo",
        messages=messages + [{"role": "system", "content": prompt_with_context}],
        max_tokens=3000,
        temperature=0.5
    )['choices'][0]['message']['content']

    print("DEBUG: Assistant response generated:", assistant_response) 

    # Generate audio from the assistant's response
    audio_filename = text_to_audio(assistant_response, filename="assistant_response.mp3")

    # Update salary figures based on the assistant's response
    update_salaries(assistant_response, is_user_message=False)

    # Debugging: Check final_salary after assistant response
    print("DEBUG: Final salary after assistant's response:", final_salary)

    # Analyze the tone of the assistant's response
    assistant_tone = detect_tone(assistant_response)
    print("DEBUG: Detected assistant tone:", assistant_tone)

    # Save the assistant's response and metrics to the conversation history
    messages.append({"role": "assistant", "content": assistant_response})

    # Calculate BLEU, ROUGE, and METEOR scores for the assistant response
    reference_response = get_reference_response(user_input, references)
    if not reference_response:
        reference_response = "Default fallback reference response."
        print("WARNING: No reference response found. Metrics may be inaccurate.")

    # Fallback for missing reference:
    if not reference_response or reference_response == "No suitable reference found. Default response.":
        reference_response = "Default fallback reference response"
    print(f"DEBUG: Loaded {len(references)} references for role {assistant_role}.")

    bleu, rouge, meteor = calculate_textual_metrics(reference_response, assistant_response)

    coherence_score = ge_val(reference_response, assistant_response)
    nli = nli_score(reference_response, assistant_response)
    qag = qag_score(reference_response, assistant_response)
    bert_f1 = calculate_bertscore(reference_response, assistant_response)

    # Append the assistant's response with metrics to conversation history
    conversation_history.append({
        "role": "assistant",
        "content": assistant_response,
        "tone": assistant_tone,
        "timestamp": datetime.now().isoformat(),
        "bleu": bleu,
        "rouge": rouge,
        "meteor": meteor,
        "bert_score": float(bert_f1),  # Ensure it's a serializable float
        "coherence_score": coherence_score,
        "nli_score": nli,
        "qag_score": qag
    })

    # Debugging: Print conversation history after adding assistant's response
    print("DEBUG: Updated conversation_history:", conversation_history)

    # Calculate a reward or penalty for the assistant based on the negotiation outcome
    reward = calculate_reward()
    print("DEBUG: Calculated reward:", reward)

    # Prepare initial and final salary values for display
    formatted_initial_salary = f"${initial_salary:,.2f}" if initial_salary else "Not set"
    formatted_final_salary = f"${final_salary:,.2f}" if final_salary else "No final salary set yet"

    # Return formatted conversation history, initial and final salary figures, and the negotiation score
    return format_chat_history(), formatted_initial_salary, formatted_final_salary, reward, '', retrieval_metrics_str, audio_filename 
  

# Function to format the chat history for a user-friendly display in the UI
def format_chat_history():
    """Format the chat history to look like a conversation in the UI."""

    # Debugging: Print the entire conversation history before formatting
    print("Conversation history before formatting:", conversation_history)

    chat_history = []  # Initialize an empty list to hold formatted messages

    # Loop through each message in the conversation history
    for message in conversation_history:
        # Debugging: Print each message being processed to see if there are duplicates
        print("Processing message:", message)  # Debugging print

        # Label messages from the user as "You" for clarity in the UI
        if message["role"] == "user":
            chat_history.append(("You", message["content"]))
        # Label messages from the assistant as "Assistant" for clarity
        elif message["role"] == "assistant":
            chat_history.append(("Assistant", message["content"]))

    # Return the formatted chat history as a list of tuples for the UI
    return chat_history

# Function to save the conversation history in OpenAI's fine-tuning format
def save_conversation_as_jsonl_format(conversation_history, filename="negotiation_conversation_history.jsonl"):
    """
    Save the conversation history as per OpenAI's fine-tuning format:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    if not conversation_history:
        print("DEBUG: No conversation history to save.")
        return

    # Determine roles for system message
    if assistant_role and user_role:
        system_content = f"This is a salary negotiation. The assistant is the {assistant_role}, and the user is the {user_role}."
    else:
        system_content = "This is a salary negotiation."

    # Initialize messages with the system prompt
    messages_list = [
        {"role": "system", "content": system_content}
    ]

    # Collect conversation messages
    for message in conversation_history:
        if "content" in message and message["role"] in ["user", "assistant"]:
            # Only include 'role' and 'content' keys
            messages_list.append({
                "role": message["role"],
                "content": message["content"]
            })

    # Prepare the conversation data
    conversation_data = {
        "messages": messages_list
    }

    try:
        # Open file in append mode to add the conversation as a single JSONL line
        with open(filename, 'a', encoding='utf-8') as file:
            json.dump(conversation_data, file)
            file.write('\n')  # Ensure each JSON object is on a new line
        print(f"DEBUG: Conversation history saved as JSONL to {filename}")

    except Exception as e:
        print(f"DEBUG: Error saving JSONL file - {e}")


# Global flag for saving status
metrics_saved = False

def save_metrics_and_conversation_to_csv(conversation_id, metrics, conversation_history, filename="negotiation_metrics.csv"):
    global metrics_saved  # Declare the variable as global

    if metrics_saved:
        print("DEBUG: Metrics already saved for this session. Skipping save.")
        return

    # Save metrics and conversation to the CSV file
    metrics_data = {
        "Timestamp": metrics["timestamp"],
        "Agreement Rate": metrics["agreement_rate"],
        "Average Sentiment Score": metrics["avg_sentiment"],
        "Feedback Quality": metrics["feedback_quality"],
        "Average Response Time": metrics["avg_response_time"],
        "Corpus BLEU": metrics["BLEU"],
        "ROUGE": metrics["ROUGE"],
        "METEOR": metrics["METEOR"],
        "BERTScore": metrics["BERTScore"],
        "G-Eval": metrics["G-Eval"],
        "NLI": metrics["NLI"],
        "QAG": metrics["QAG"],
        "TTR": metrics["TTR"],  # Add TTR
        "Pause Ratio": metrics["Pause Ratio"],  # Add Pause Ratio
        "Average Turn Length": metrics["Average Turn Length"],  # Add Average Turn Length
        "MAUVE": metrics["MAUVE"],  # Add MAUVE Score
        "USL-H": metrics["USL-H"],  # Add USL-H
        "Reward Score": metrics["reward_score"],  # Include reward/penalty
        "Conversation History": json.dumps(conversation_history)  # Store entire conversation as JSON
    }

    try:
        # Load existing file if it exists
        if os.path.isfile(filename):
            existing_df = pd.read_csv(filename)
            new_row_df = pd.DataFrame([metrics_data]).dropna(how="all")
            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True).drop_duplicates(keep='last')
        else:
            updated_df = pd.DataFrame([metrics_data]).dropna(how="all")

        # Save back to CSV
        updated_df.to_csv(filename, index=False)
        print(f"DEBUG: Metrics and conversation history saved to {filename} successfully.")

        metrics_saved = True  # Mark as saved to prevent duplicates

    except Exception as e:
        print(f"DEBUG: Error saving metrics to CSV: {e}")


# Function to end the game, calculate final reward/penalty, save the conversation, and provide feedback
def end_game():
    global messages, conversation_history, initial_salary, final_salary

    # Generate a unique conversation ID
    conversation_id = generate_conversation_id()

    # Calculate the final reward or penalty for the negotiation session
    reward = calculate_reward()

    # Append this reward information as a system message in the conversation history
    conversation_history.append({
        "role": "system",
        "reward_score": reward,
        "timestamp": datetime.now().isoformat()
    })

    # Define a prompt to get feedback on the completed negotiation
    feedback_prompt = (
        "Based on the negotiation that just ended, provide a detailed evaluation. "
        "Mention what went well, what could have been improved, and offer suggestions "
        "for both the buyer and the seller to help them improve in the future."
    )

    # Generate feedback from the assistant using OpenAI's API and the feedback prompt
    try:
        feedback_response = openai.ChatCompletion.create(
            model="ft:gpt-4o-2024-08-06:llm-sim:salary-negotiation:AQN5Azuo",
            messages=messages + [{"role": "user", "content": feedback_prompt}],
            max_tokens=1000
        )
        # Extract the feedback content from the response
        feedback = feedback_response['choices'][0]['message']['content']
    except Exception as e:
        print(f"DEBUG: Error generating feedback - {e}")
        feedback = "Could not generate feedback due to an error." 

    # Generate audio from the feedback
    audio_feedback_filename = text_to_audio(feedback, filename="feedback_audio.mp3")

    # Append the final reward/penalty score to the feedback
    feedback += f"\n\n**Final Reward/Penalty Score:** {reward}"

    # Call the evaluate_model function to get only the summary
    try:
        summary = evaluate_model(conversation_id)
    except Exception as e:
        print(f"DEBUG: Error evaluating model - {e}")
        summary = "Evaluation could not be completed due to an error."

    # Call the function to save conversation history in the specified JSONL format
    try:
        save_conversation_as_jsonl_format(conversation_history, "negotiation_conversation_history.jsonl")
        print("DEBUG: JSONL file saved successfully at end of game.")
    except Exception as e:
        print(f"DEBUG: Error saving conversation history - {e}")

    # Format initial and final salary for output
    formatted_initial_salary = f"${initial_salary:,.2f}" if initial_salary else "Not set"
    formatted_final_salary = f"${final_salary:,.2f}" if final_salary else "No final salary set yet"

    # Debugging final outputs
    print(f"DEBUG: Final Reward: {reward}")
    print(f"DEBUG: Feedback: {feedback}")
    print(f"DEBUG: Initial Salary: {formatted_initial_salary}, Final Salary: {formatted_final_salary}")

    # Return Gradio-compatible outputs
    return format_chat_history(), formatted_initial_salary, formatted_final_salary, '', feedback, audio_feedback_filename

def calculate_bertscore(reference, prediction):
    # BERTScore expects lists of references and predictions
    P, R, F1 = bert_score([prediction], [reference], lang="en")
    return F1.mean().item()  # Return the average F1 score 

def evaluate_retrieval(query, retrieved_contexts, ground_truth):
    """
    Evaluate the effectiveness of the retrieval mechanism.
    Parameters:
    - query (str): The user query.
    - retrieved_contexts (list of str): Contexts retrieved by the RAG system.
    - ground_truth (str): The expected or ideal reference response.
    Returns:
    - dict: A dictionary containing metrics (accuracy, coverage, relevance, novelty).
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    ground_truth_embedding = model.encode(ground_truth, convert_to_tensor=True)
    retrieved_embeddings = [model.encode(context, convert_to_tensor=True) for context in retrieved_contexts]

    # Calculate similarity scores
    ground_truth_similarity = [util.pytorch_cos_sim(ground_truth_embedding, emb).item() for emb in retrieved_embeddings]
    query_similarity = [util.pytorch_cos_sim(query_embedding, emb).item() for emb in retrieved_embeddings]

    # Define metrics
    retrieval_accuracy = any(score > 0.8 for score in ground_truth_similarity)
    coverage = len(retrieved_contexts) / len(ground_truth.split()) if ground_truth else 0
    relevance = np.mean(query_similarity) if query_similarity else 0
    novelty = np.mean([1 - score for score in ground_truth_similarity]) if ground_truth_similarity else 0

    return {
        "Retrieval Accuracy": retrieval_accuracy,
        "Coverage": coverage,
        "Relevance": relevance,
        "Novelty": novelty
    }


# Modify evaluate_model function
def evaluate_model(conversation_id=None, references=None):
    print("Evaluating model...")

    # Check if references are provided; if not, default to an empty list
    if references is None:
        references = []

    # Ensure 'references' is a list of strings
    string_references = []
    for ref in references:
        if isinstance(ref, list):
            # If a reference is a list, join its elements into a single string
            string_references.append(" ".join(ref))
        elif isinstance(ref, str):
            string_references.append(ref)

    # Combine reference texts for evaluation
    combined_reference_text = " ".join(string_references)

    # Calculate the number of successful negotiations
    successful_negotiations = sum(
        1 for message in conversation_history if "content" in message and ("agree" in message["content"].lower() or "deal" in message["content"].lower())
    )

    # Calculate the total number of negotiations
    total_negotiations = len([message for message in conversation_history if message["role"] == "user"])
    agreement_rate = successful_negotiations / total_negotiations if total_negotiations > 0 else 0

    # Calculate the average sentiment of assistant messages
    sentiment_scores = [
        TextBlob(message["content"]).sentiment.polarity
        for message in conversation_history if message["role"] == "assistant"
    ]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    feedback_quality = "High" if avg_sentiment > 0.3 else "Moderate" if avg_sentiment > 0 else "Low"

    # File contribution metrics
    total_contributions = sum(contribution_metrics.values())
    file_contributions = {
        file: (count / total_contributions) * 100 if total_contributions > 0 else 0
        for file, count in contribution_metrics.items()
    }
    contribution_str = "\n".join([f"{file}: {contribution:.2f}%" for file, contribution in file_contributions.items()])

    # Average response time for assistant replies
    avg_response_time = np.mean([
        (datetime.fromisoformat(conversation_history[i + 1]["timestamp"]) - datetime.fromisoformat(message["timestamp"])).total_seconds()
        for i, message in enumerate(conversation_history[:-1])
        if message["role"] == "user" and "timestamp" in message and "timestamp" in conversation_history[i + 1]
    ]) if len(conversation_history) > 1 else 0

    # Initialize lists for new metrics
    bleu_scores, rouge_scores, meteor_scores = [], [], []
    bert_scores, g_eval_scores = [], []
    nli_scores, qag_scores = [], []
    ttr_scores, pause_ratios = [], []
    avg_turn_lengths = []

    # Calculate new metrics for assistant responses
    assistant_responses = [msg for msg in conversation_history if msg["role"] == "assistant"]
    user_messages = [msg for msg in conversation_history if msg["role"] == "user"]

    for i, assistant_response in enumerate(assistant_responses):
        if i < len(user_messages):
            prediction = assistant_response["content"]

            # Textual Accuracy Metrics
            bleu = corpus_bleu_eq([combined_reference_text], [prediction])
            rouge = rouge_eq(combined_reference_text, prediction)
            meteor = meteor_eq(combined_reference_text, prediction)
            bleu_scores.append(bleu)
            rouge_scores.append(rouge)
            meteor_scores.append(meteor)

            # Semantic Coherence Metrics
            bert = calculate_bertscore(combined_reference_text, prediction)
            g_eval = ge_val(combined_reference_text, prediction)
            bert_scores.append(bert)
            g_eval_scores.append(g_eval)

            # Factual Accuracy Metrics
            nli = nli_score(combined_reference_text, prediction)
            qag = qag_score(combined_reference_text, prediction)
            nli_scores.append(nli)
            qag_scores.append(qag)

            # Conversational Metrics
            ttr = calculate_ttr(prediction)
            pause_ratio = calculate_pause_ratio(prediction)
            avg_turn_length = len(word_tokenize(prediction))
            ttr_scores.append(ttr)
            pause_ratios.append(pause_ratio)
            avg_turn_lengths.append(avg_turn_length)

    # Average Conversational Metrics
    avg_ttr = np.mean(ttr_scores) if ttr_scores else 0
    avg_pause_ratio = np.mean(pause_ratios) if pause_ratios else 0
    avg_turn_length = np.mean(avg_turn_lengths) if avg_turn_lengths else 0

    # MAUVE Metric
    try:
        # Ensure valid human and model texts before calculation
        if not references or not assistant_responses:
            print("DEBUG: Missing human or model texts for MAUVE calculation.")
            mauve_score = 0
        else:
            mauve_score = calculate_mauve_score(
                human_texts=references,
                model_texts=[msg["content"] for msg in assistant_responses]
            ) or 0  # Ensure a fallback to 0
    except Exception as e:
        print(f"MAUVE calculation error: {e}")
        mauve_score = 0

    # USL-H Metric
    usl_h_scores = [
        calculate_usl_h(nli, bert, sentiment)
        for nli, bert, sentiment in zip(nli_scores, bert_scores, sentiment_scores)
    ]
    avg_usl_h = np.mean(usl_h_scores) if usl_h_scores else 0

    # Calculate averages for existing metrics
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_rouge = np.mean(rouge_scores) if rouge_scores else 0
    avg_meteor = np.mean(meteor_scores) if meteor_scores else 0
    avg_bert = np.mean(bert_scores) if bert_scores else 0
    avg_g_eval = np.mean(g_eval_scores) if g_eval_scores else 0
    avg_nli = np.mean(nli_scores) if nli_scores else 0
    avg_qag = np.mean(qag_scores) if qag_scores else 0

    # Create a dictionary to hold all metrics
    metrics_dict = {
        "timestamp": datetime.now().isoformat(),
        "agreement_rate": agreement_rate,
        "avg_sentiment": avg_sentiment,
        "feedback_quality": feedback_quality,
        "avg_response_time": avg_response_time,
        "BLEU": avg_bleu,
        "ROUGE": avg_rouge,
        "METEOR": avg_meteor,
        "BERTScore": avg_bert,
        "G-Eval": avg_g_eval,
        "NLI": avg_nli,
        "QAG": avg_qag,
        "TTR": avg_ttr,
        "Pause Ratio": avg_pause_ratio,
        "Average Turn Length": avg_turn_length,
        "MAUVE": mauve_score,
        "USL-H": avg_usl_h,
        "conversation_id": conversation_id,  # Unique identifier for each conversation
        "reward_score": calculate_reward()  # Include final reward/penalty score
    }

    # Save the metrics and conversation to CSV
    save_metrics_and_conversation_to_csv(conversation_id, metrics_dict, conversation_history)

    # Create a summary string in the desired format
    summary_string = (
        f"Model Evaluation Metrics:\n"
        f"Agreement Rate: {agreement_rate:.2f}\n"
        f"Average Sentiment Score: {avg_sentiment:.2f}\n"
        f"Feedback Quality: {feedback_quality}\n"
        f"Average Response Time: {avg_response_time:.2f}s\n\n"
        f"Textual Accuracy - Corpus BLEU: {avg_bleu:.3f}, ROUGE: {avg_rouge:.3f}, METEOR: {avg_meteor:.3f}\n"
        f"Semantic Coherence - BERTScore: {avg_bert:.3f}, GEval: {avg_g_eval:.2f}\n"
        f"Factual Accuracy - NLI: {avg_nli:.2f}, QAG: {avg_qag:.2f}\n\n"
        f"Conversational Metrics:\n"
        f"- Type-Token Ratio (TTR): {avg_ttr:.3f}\n"
        f"- Pause Ratio: {avg_pause_ratio:.3f}\n"
        f"- Average Turn Length: {avg_turn_length:.2f} words\n"
        f"- MAUVE Score: {mauve_score:.3f}\n"
        f"- USL-H Score: {avg_usl_h:.3f}\n\n"
        f"File Contributions:\n{contribution_str}"
    )

    # Return only the summary string in the requested format
    return summary_string



# Function to reload files from the specified folder, excluding any listed files
def update_loaded_files(exclude_files):
    # Call the function to load PDFs, excluding any files specified in the 'exclude_files' string
    # 'exclude_files' is a comma-separated string of filenames to exclude, so we split it into a list
    load_pdfs_from_folder("RAG", exclude_files=exclude_files.split(","))

    # Return a confirmation message with the updated list of loaded files
    return f"Updated file list: {', '.join(file_list)}" 



# Textual Accuracy Metrics
def corpus_bleu_eq(references, predictions):
    tokenized_references = [[ref.split()] for ref in references]  # Corpus BLEU expects a list of lists of references
    tokenized_predictions = [pred.split() for pred in predictions]
    return corpus_bleu(tokenized_references, tokenized_predictions, smoothing_function=SmoothingFunction().method1)

def rouge_eq(reference, prediction):
    rouge = Rouge()
    scores = rouge.get_scores(prediction, reference)
    return scores[0]['rouge-l']['f']

def meteor_eq(reference, prediction):
    # Use TreebankWordTokenizer instead of word_tokenize
    tokenizer = TreebankWordTokenizer()
    tokenized_reference = tokenizer.tokenize(reference)
    tokenized_prediction = tokenizer.tokenize(prediction)
    return meteor_score.meteor_score([tokenized_reference], tokenized_prediction)

# Function to evaluate model's response based on Textual Accuracy
def calculate_textual_metrics(reference, prediction):
    bleu = corpus_bleu_eq([reference], [prediction])  # Using list of single references for corpus BLEU
    rouge = rouge_eq(reference, prediction)
    meteor = meteor_eq(reference, prediction)
    return bleu, rouge, meteor

# Set up the Gradio interface layout and styling
with gr.Blocks(css=".gradio-container {max-width: 1000px; margin: auto; background-color: #1e1e1e; color: white; border-radius: 10px; padding: 20px;}") as demo:
    # Create a Markdown header for the interface title
    gr.Markdown("## 🤝🤖 Salary Negotiation Assistant - Chat-Like Interface", elem_id="title")

    # Define a row layout for the input and output sections
    with gr.Row():
        # Create a column for user inputs with a scale of 3
        with gr.Column(scale=3):
            # Define text inputs for user name, company, and position
            name_input = gr.Textbox(lines=1, placeholder="Enter your name...", label="Your Name")
            company_name_input = gr.Textbox(lines=1, placeholder="Enter your company name...", label="Company Name")
            position_input = gr.Textbox(lines=1, placeholder="Enter your position...", label="Your Position")

            # Radio buttons for the user to choose their role (employee or employer)
            role_input = gr.Radio(choices=["employee", "employer"], label="Choose your role", interactive=True)

            # Input for initial salary
            initial_salary_input = gr.Textbox(lines=1, placeholder="Enter your current salary...", label="Current Salary")

            # Button to start the negotiation game
            start_button = gr.Button("Start Game")

            # Input for the negotiation message
            user_text_input = gr.Textbox(lines=3, placeholder="Enter your negotiation message here...", label="Your Message")
            user_audio_input = gr.Audio(label="Your Audio Message", type="filepath")
            

            # Button to submit negotiation messages
            negotiate_button = gr.Button("Negotiate")

            # Input to specify files to exclude from document loading
            exclude_files_input = gr.Textbox(lines=1, placeholder="Enter file names to exclude, separated by commas", label="Exclude Files")

            # Button to reload files based on exclusions
            reload_button = gr.Button("Reload Files")

            # Button to end the negotiation game
            end_button = gr.Button("End Game")

            # Button to evaluate model performance
            evaluate_button = gr.Button("Evaluate Model")

        # Create a larger column (scale 7) for outputs
        with gr.Column(scale=7):
            # Chatbot display for conversation history
            chat_output = gr.Chatbot(label="Chat History", show_label=False, value=[], height=400) 

            audio_output = gr.Audio(label="Assistant Audio Response", interactive=False, type="filepath")

            # Display for negotiation feedback from the assistant
            feedback_output = gr.Textbox(label="Negotiation Feedback (Feedback can be inaccurate)", lines=5, interactive=False)

            # Display for showing model evaluation metrics
            evaluate_output = gr.Textbox(label="Model Evaluation Metrics", lines=8, interactive=False) 

            retrieval_metrics_output = gr.Textbox(label="Retrieval Metrics", interactive=False, lines=4)

            # Display the list of loaded files
            file_list_output = gr.Textbox(label="Loaded Files", lines=3, interactive=False)

            # Displays for initial and final salary values, and the negotiation score
            initial_salary_output = gr.Textbox(label="Initial Salary", value="", interactive=False)
            final_salary_output = gr.Textbox(label="Final Salary", interactive=False)
            score_output = gr.Textbox(label="Negotiation Score", interactive=False)

        # Define actions for buttons, linking them to functions and specifying inputs and outputs
        start_button.click(
            fn=start_game,
            inputs=[role_input, initial_salary_input, name_input, company_name_input, position_input],
            outputs=[chat_output, initial_salary_output]
        )

        negotiate_button.click(
            fn=negotiate,
            inputs=[user_text_input, user_audio_input, name_input, company_name_input, position_input, gr.State(assistant_role)],
            outputs=[chat_output, initial_salary_output, final_salary_output, score_output, feedback_output, retrieval_metrics_output, audio_output]
        )

        reload_button.click(
            fn=update_loaded_files,
            inputs=exclude_files_input,
            outputs=file_list_output
        )

        end_button.click(
            fn=end_game,
            inputs=None,
            outputs=[chat_output, initial_salary_output, final_salary_output, score_output, feedback_output, audio_output]
        )

        evaluate_button.click(
            fn=lambda assistant_role: evaluate_model(
                conversation_id=generate_conversation_id(),
                references=load_reference_text(assistant_role)[0]  # Pass only the list of references
        ),
        inputs=gr.State(assistant_role),
        outputs=evaluate_output
    )


    # Initialize the file list output with the current files loaded
    file_list_output.value = ", ".join(file_list)

# Launch the Gradio interface in queued mode to handle multiple inputs
demo.queue().launch(debug=True)

