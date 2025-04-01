# Medical ChatBot

#### 1. Create Environment

```bash
conda create -n chatbot_med python=3.11 -y
```
```bash
conda activate chatbot_med
```
#### 2. Install Requirements
```bash
pip install -r requirements.txt
```

#### 3. Download the LLama-2 Weights

#### Download llama-2-7b-chat.ggmlv3.q4_0.bin from hugging face and save it in Model folder.

#### 4. Create a Pinecone Account and get your API_KEY and store it as environment variable.

#### 5. Running the Chatbot

```bash
python store_index.py
```

```bash
python app.py
```

##### 6. Open up the localhost to chat with the chatbot