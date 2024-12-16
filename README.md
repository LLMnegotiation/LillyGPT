# :cherry_blossom: Welcome to LilyGPT
## :books: Product Summary
This project leverages GPT-4o Large Language Models (LLM), advanced prompt engineering, Retrieval-Augmented Generation (RAG), and fine-tuning techniques to revolutionize negotiation processes across industries. As a beta use case, we focused on salary negotiations, a critical area plagued by inefficiencies, to refine our approach and demonstrate its potential. Salary negotiations often involve time-consuming back-and-forths, frustrating employees and overburdening recruiters, which slows hiring and risks losing top talent. By addressing this specific challenge, the project aims to improve industry-wide negotiations and scale to other high-stakes scenarios.

The AI-powered negotiation agent developed in this project delivers human-like, emotionally resonant interactions. It leverages fine-tuned models to interpret nuanced human communication, such as emotional cues and strategic preferences, while RAG ensures unified access to relevant contextual knowledge, such as negotiation best practices and organizational constraints. Additionally, heavy prompt engineering optimizes the agent’s responses for complex, multidimensional dialogues, while comprehensive metrics are used to evaluate and continuously improve its performance.

To enhance accessibility, we integrated OpenAI Whisper for speech-to-text processing and Google TTS for in-house text-to-speech functionality, enabling seamless voice-based interactions. These features make the agent suitable for dynamic, real-time negotiation scenarios while maintaining a natural, conversational experience.

By automating much of the negotiation process while preserving empathy and personalization, the agent reduces salary confirmation times from weeks to hours, freeing recruiters to focus on strategic priorities. This improves the candidate experience, positions organizations as employee-centric, and enhances operational efficiency.

While salary negotiations serve as the pilot, this project lays the groundwork for AI-driven negotiation solutions across industries. Its methodologies—fine-tuning, prompt engineering, RAG for unified knowledge, and advanced metrics for continuous improvement—can extend to contexts such as vendor contract negotiations, customer pricing agreements, and dispute resolution. The integration of speech processing and emotional intelligence ensures the system adapts to unstructured inputs and fosters trust and collaboration, setting a new standard for negotiation tools in diverse applications.
### :briefcase: Get started
### :microphone: What is [OpenAI Whisper](https://openai.com/index/whisper/)
Getting Started Docs for [Speech to Text](https://platform.openai.com/docs/guides/speech-to-text)
OpenAI Whisper is an advanced speech-to-text model designed to convert spoken language into written text with high accuracy. It is capable of recognizing and transcribing multiple languages, handling diverse accents, and processing noisy audio environments. Whisper is built using a large-scale transformer architecture and trained on a vast dataset of multilingual and multitask supervised data. This enables it to perform not only transcription but also tasks like language detection and translation. Its robust design makes it suitable for applications such as voice assistants, transcription services, and real-time language processing.
![](https://images.ctfassets.net/kftzwdyauwt9/d9c13138-366f-49d3-a1a563abddc1/8acfb590df46923b021026207ff1a438/asr-summary-of-model-architecture-desktop.svg?w=1920&q=90)

### What is [OpenAI GPT4o](https://platform.openai.com/docs/models#gpt-4o)
GPT-4o (“o” for “omni”) is our most advanced GPT model. It is multimodal (accepting text or image inputs and outputting text), and it has the same high intelligence as GPT-4 Turbo but is much more efficient—it generates text 2x faster and is 50% cheaper. Additionally, GPT-4o has the best vision and performance across non-English languages of any of our models. GPT-4o is available in the OpenAI API to paying customers.Learn how to use GPT-4o in the [text generation guide](https://platform.openai.com/docs/guides/text-generation).

### What is [OpenAI Fine-Tune](https://platform.openai.com/docs/guides/fine-tuning)
Fine-tuning is a process that enhances pre-trained language models by training them on specific datasets to improve their performance for targeted applications. It offers higher-quality results than prompt engineering, accommodates more examples than a prompt can handle, reduces token usage with shorter prompts, and ensures faster response times. OpenAI’s fine-tuning supports models like `gpt-4o` and `gpt-4o-mini`, allowing customization for use cases such as tone setting, improving reliability, and addressing edge cases. The process involves preparing training data, training the model, and evaluating results. Techniques like Retrieval-Augmented Generation (RAG) for contextual knowledge and metrics-based evaluations further refine performance. Fine-tuning is ideal when prompt engineering alone cannot achieve desired outcomes, offering tailored, efficient, and cost-effective solutions for specific tasks.

## Some Basic Python Libraries needed to run LilyGPT and Self Play

> [!IMPORTANT]
> OpenAI instalation needs to be in openai==0.28 to work and Numpy Version ==1.24.4

### GCP Code to run on VertexAI Collab Enterprise for `GCP_LillyGPT.ipynb`
```
!pip install openai gradio pandas numpy==1.24.4 vaderSentiment textblob PyMuPDF transformers sentence-transformers faiss-cpu rouge bert-score nltk spacy torch gtts SpeechRecognition pydub
!pip install openai==0.28
!pip install mauve-text
!pip install google-cloud-bigquery google-cloud-storage
```
### Local Machine `LillyGPT.ipynb`
```
!pip install openai==0.28 gradio pandas numpy==1.24.4 vaderSentiment textblob PyMuPDF transformers sentence-transformers faiss-cpu rouge bert-score nltk spacy torch gtts SpeechRecognition pydub

```
### How to create an API KEY for OpenAI for the code
OpenAI Getting Started [Docs](https://platform.openai.com/docs/quickstart)

### API KEY for OpenAI
```
openai.api_key = "USE_YOUR_OWN_API_KEY"
```



