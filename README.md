# Medical ChatBot

#### Create Environment

```bash
conda create -n chatbot_med python=3.11 -y
```
```bash
conda activate chatbot_med
```
#### Install Requirements
```bash
pip install -r requirements.txt
```

### Download the LLama-2 Weights

#### Download llama-2-7b-chat.ggmlv3.q4_0.bin from hugging face.

### Create a Pinecone Account and get your API_KEY and store it as environment variable.

### Running the Chatbot

```bash
python store_index.py
```

##### Open up the localhost to chat with the chatbot