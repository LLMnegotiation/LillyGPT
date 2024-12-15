# Welcome to LilyGPT
### Product Summary

This project leverages GPT-4o Large Language Models (LLM), advanced prompt engineering, Retrieval-Augmented Generation (RAG), and fine-tuning techniques to revolutionize negotiation processes across industries. As a beta use case, we focused on salary negotiations, a critical area plagued by inefficiencies, to refine our approach and demonstrate its potential. Salary negotiations often involve time-consuming back-and-forths, frustrating employees and overburdening recruiters, which slows hiring and risks losing top talent. By addressing this specific challenge, the project aims to improve industry-wide negotiations and scale to other high-stakes scenarios.

The AI-powered negotiation agent developed in this project delivers human-like, emotionally resonant interactions. It leverages fine-tuned models to interpret nuanced human communication, such as emotional cues and strategic preferences, while RAG ensures unified access to relevant contextual knowledge, such as negotiation best practices and organizational constraints. Additionally, heavy prompt engineering optimizes the agent’s responses for complex, multidimensional dialogues, while comprehensive metrics are used to evaluate and continuously improve its performance.

To enhance accessibility, we integrated OpenAI Whisper for speech-to-text processing and Google TTS for in-house text-to-speech functionality, enabling seamless voice-based interactions. These features make the agent suitable for dynamic, real-time negotiation scenarios while maintaining a natural, conversational experience.

By automating much of the negotiation process while preserving empathy and personalization, the agent reduces salary confirmation times from weeks to hours, freeing recruiters to focus on strategic priorities. This improves the candidate experience, positions organizations as employee-centric, and enhances operational efficiency.

While salary negotiations serve as the pilot, this project lays the groundwork for AI-driven negotiation solutions across industries. Its methodologies—fine-tuning, prompt engineering, RAG for unified knowledge, and advanced metrics for continuous improvement—can extend to contexts such as vendor contract negotiations, customer pricing agreements, and dispute resolution. The integration of speech processing and emotional intelligence ensures the system adapts to unstructured inputs and fosters trust and collaboration, setting a new standard for negotiation tools in diverse applications.
### Get started
#### Some Basic Python Libraries needed to run LilyGPT and Self Play
##### GCP Code to run on VertexAI Collab Enterprise for `GCP_LillyGPT.ipynb`
```
!pip install openai gradio pandas numpy==1.24.4 vaderSentiment textblob PyMuPDF transformers sentence-transformers faiss-cpu rouge bert-score nltk spacy torch gtts SpeechRecognition pydub
!pip install openai==0.28
!pip install mauve-text
!pip install google-cloud-bigquery google-cloud-storage
```
##### Local Machine `LillyGPT.ipynb`
```
!pip install openai==0.28 gradio pandas numpy==1.24.4 vaderSentiment textblob PyMuPDF transformers sentence-transformers faiss-cpu rouge bert-score nltk spacy torch gtts SpeechRecognition pydub

```
#### How to create an API KEY for OpenAI for the code
OpenAI Getting Started [Docs](https://platform.openai.com/docs/quickstart)
#### API KEY for OpenAI
```
openai.api_key = "USE_YOUR_OWN_API_KEY"
```
