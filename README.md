# CA-ThinkFlow 💡

📊 CA-ThinkFlow is an AI-powered financial consulting application 🧠💰 designed to assist users with various financial queries 📈💬. 🛠️ Built using Streamlit 🖥️ and LangChain 🔗, this application leverages advanced language models 🤖📚 to provide accurate and context-aware responses to user questions related to finance 💹🔍.

---

https://github.com/user-attachments/assets/5d18bc95-dd45-4788-84b9-3202662e8034

---

## ✨ Features

- 🖥️ **Interactive UI**: A user-friendly interface that allows users to input their financial questions and receive instant responses.
- 🔍 **Predefined Queries**: Quick access buttons for common financial questions, making it easier for users to get information.
- 🕒 **Conversation History**: Keeps track of user interactions for reference and continuity.
- 🛠️ **Robust Error Handling**: Ensures smooth operation even in the case of unexpected issues.

## 🛠️ Technologies Used

- 🖥️ **Streamlit**: For building the web application interface.
- 🔗 **Langchain**: For managing the language model and retrieval QA chain.
- 💾 **FAISS**: For efficient similarity search and retrieval of relevant documents.
- 💡 **HuggingFace Embeddings**: For generating embeddings to enhance the retrieval process.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/thejatingupta7/CA-ThinkFlow
   cd CA-Thinkflow
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the necessary models and data files in the `vectorstore` directory.

## Usage

To start the application, run the following command:

```bash
streamlit run app.py
```

Open your web browser and navigate to `http://localhost:8501` to access the application.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
