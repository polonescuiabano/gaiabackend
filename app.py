from flask import Flask, request, jsonify
import psycopg2
from psycopg2 import sql
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import logging
import warnings
import threading
import time
from examples import exemplos_respostas


warnings.filterwarnings("ignore")
print(torch.__version__)
print(torch.cuda.is_available())
logging.basicConfig(level=logging.DEBUG)
device = torch.device('cpu')



app = Flask(__name__)

model_name = "EleutherAI/gpt-neo-1.3B"
OUTPUT_DIR = "./fine_tuned_model"
tokenizer = GPT2Tokenizer.from_pretrained(OUTPUT_DIR)
model = GPTNeoForCausalLM.from_pretrained(OUTPUT_DIR)

model.eval()

DB_HOST = 'localhost'
DB_NAME = 'server'
DB_USER = 'postgres'
DB_PASS = 'FERRARI02'

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    return conn


@app.route('/check_diagnosis', methods=['GET'])
def check_diagnosis():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "Username is required"}), 400

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT diagnosis FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()

    if result and result[0]:
        return jsonify({"has_diagnosis": True, "diagnosis": result[0]}), 200
    else:
        return jsonify({"has_diagnosis": False}), 200

@app.route('/check_or_create_history', methods=['GET'])
def check_or_create_history():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "Username is required"}), 400

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT history FROM user_histories WHERE username = %s", (username,))
        result = cursor.fetchone()

        if result:
            return jsonify({"history": result[0]}), 200
        else:
            cursor.execute(
                "INSERT INTO user_histories (username, history) VALUES (%s, %s)",
                (username, "")
            )
            conn.commit()
            return jsonify({"message": "Novo histórico criado", "history": ""}), 201

@app.route('/diagnosis', methods=['POST'])
def store_diagnosis():
    data = request.get_json()
    username = data.get('username')
    diagnosis = data.get('diagnosis')

    if not username or not diagnosis:
        return jsonify({'error': 'Username and diagnosis are required'}), 400

    formatted_diagnosis = "{" + ",".join(diagnosis.replace("(", "").replace(")", "").split(', ')) + "}"
    print(f"Formatted Diagnosis: {formatted_diagnosis}")

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            sql.SQL(
                "INSERT INTO users (username, diagnosis) VALUES (%s, %s) ON CONFLICT (username) DO UPDATE SET diagnosis = EXCLUDED.diagnosis"),
            [username, formatted_diagnosis]
        )
        conn.commit()
        return jsonify({'message': 'Diagnosis stored successfully'}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Database error'}), 500
    finally:
        cursor.close()
        conn.close()


@app.route('/gaia', methods=['POST'])
def chat_with_gaia():
    user_message = request.json.get('message')
    username = request.json.get('username')

    logging.info(f"Received message from {username}: {user_message}")

    exemplos_formados = "\n".join([f"{ex['input']}\n{ex['output']}" for ex in exemplos_respostas])

    input_text = (
        "Você é uma assistente virtual chamada Gaia, imitando uma psicoterapeuta empática. Responda com empatia e variedade:\n"
        f"{exemplos_formados}\n"
        f"Usuário: {user_message}\nGaia:"
    )

    logging.info(f"Input text for model: {input_text}")

    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    try:
        logging.info("Generating response from model...")
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=300,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )

        ai_response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        if ai_response.startswith("Gaia:"):
            ai_response = ai_response[len("Gaia:"):].strip()

        if not ai_response or "não sei" in ai_response.lower():
            alternative_input = (
                "Você é uma assistente virtual chamada Gaia, imitando uma psicoterapeuta empática. Foque em responder de forma acolhedora e compreensiva:\n"
                f"Usuário: {user_message}\nGaia:"
            )
            alternative_input_ids = tokenizer.encode(alternative_input, return_tensors='pt', max_length=512, truncation=True)
            with torch.no_grad():
                alternative_output = model.generate(
                    alternative_input_ids,
                    max_new_tokens=300,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9
                )
            ai_response = tokenizer.decode(alternative_output[0], skip_special_tokens=True).strip()
            if ai_response.startswith("Gaia:"):
                ai_response = ai_response[len("Gaia:"):].strip()
            if not ai_response:
                ai_response = "Entendo que você está passando por algo. Como posso te ajudar melhor?"

        logging.info(f"Final AI Response: {ai_response}")
        update_conversation_history(username, user_message, ai_response)

        lines = ai_response.split('\n')
        if lines:
            ai_response = lines[-1].strip()

        return jsonify({'response': ai_response})
    except Exception as e:
        logging.error(f"Error during generation: {str(e)}")
        return jsonify({'response': "Erro ao gerar a resposta."}), 500




def get_conversation_history(username):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT history FROM user_histories WHERE username = %s", (username,))
        result = cursor.fetchone()
        return result[0] if result else ""

def update_conversation_history(username, user_message, ai_response):
    new_entry = f"Usuário: {user_message}\nGaia: {ai_response}"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT history FROM user_histories WHERE username = %s", (username,))
        result = cursor.fetchone()

        if result:
            existing_history = result[0]
            updated_history = existing_history + "\n" + new_entry
            cursor.execute(
                "UPDATE user_histories SET history = %s WHERE username = %s",
                (updated_history, username)
            )
        else:
            cursor.execute(
                "INSERT INTO user_histories (username, history) VALUES (%s, %s)",
                (username, new_entry)
            )

        conn.commit()

@app.route('/session_history', methods=['GET'])
def get_session_history():
    username = request.args.get('username')
    if not username:
        return jsonify({"error": "Username is required"}), 400

    history = fetch_user_history(username)

    if not history:
        create_new_history(username)
        return jsonify({"message": "Novo histórico criado", "history": []}), 201

    return jsonify({"history": history}), 200

def fetch_user_history(username):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT history FROM user_histories WHERE username = %s", (username,))
        result = cursor.fetchone()
        return result[0] if result else None

def create_new_history(username):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_histories (username, history) VALUES (%s, %s)",
            (username, "")
        )
        conn.commit()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
