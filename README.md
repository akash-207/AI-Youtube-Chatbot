An AI-powered chatbot that lets you chat with YouTube videos. Instead of watching an entire video, you can ask natural language questions and get instant answers derived from the transcript. Built using LangChain, FAISS, and OpenAI LLMs.

✨ Features

Extracts transcripts from YouTube videos automatically.

Splits and embeds transcripts for efficient retrieval.

Semantic search using FAISS vector store.

Powered by OpenAI embeddings + ChatGPT models.

Optionally includes video metadata (title, channel, description).

Query videos for summaries, explanations, and specific answers.

🛠️ Tech Stack

LangChain – LLM orchestration

FAISS – Vector search

YouTube Transcript API / LangChain YoutubeLoader – Transcript extraction

OpenAI – Embeddings + LLMs

🚀 Getting Started
1️⃣ Clone the repository
git clone https://github.com/your-username/ai-youtube-chatbot.git
cd ai-youtube-chatbot

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Set up API keys

Create a .env file in the project root and add your OpenAI API key:

OPENAI_API_KEY=your_api_key_here

4️⃣ Run the chatbot
python youtube_chat.py

📌 Example Usage
question = "What is nuclear fusion?"
answer = chatbot.ask(question)
print(answer)


✅ Output:

"The speaker explains that nuclear fusion is the process of fusing atomic nuclei to release massive energy, similar to what happens in the sun."

🤝 Contribution

Pull requests are welcome! For major changes, please open an issue first to discuss.

⚠️ Disclaimer

This tool is for educational purposes. The chatbot may not always provide perfect or complete answers. Always verify information from original sources.

👉 I can also create a ready-to-use README.md file for you with badges, emoji icons, and sections formatted.
Do you want me to make that full file so you can just paste it into your repo?
