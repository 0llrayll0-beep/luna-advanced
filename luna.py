import os
import json
import sys
import speech_recognition as sr
import threading
import pyttsx3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton,
    QTextEdit, QVBoxLayout, QWidget, QMessageBox,
    QInputDialog, QLineEdit
)
import requests

JSON_FILE = 'comandos.json'
DATASET_FILE = 'dataset_conversa.txt'

# Inicializa o motor de fala da Luna
engine = pyttsx3.init()

# Configura a voz da Luna
voices = engine.getProperty('voices')
for voice in voices:
    if "female" in voice.name.lower() or "feminina" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

# Comandos padrão da assistente
comandos_padrao = {
    'luna n': {
        'descricao': 'Abrindo o navegador...',
        'tipo': 'navegador',
        'caminho': 'chrome'
    },
    'luna kit': {
        'descricao': 'Abrindo o site do GitHub...',
        'tipo': 'navegador',
        'caminho': 'https://github.com'
    },
    'luna pasta': {
        'descricao': 'Abrindo a pasta padrão...',
        'tipo': 'pasta',
        'caminho': r"C:\Users\power\OneDrive\Área de Trabalho"
    },
    'luna olho': {
        'descricao': 'Ativando o controle de mouse pelos olhos...',
        'tipo': 'olhos'
    },
    'luna sair': {
        'descricao': 'Saindo do modo de controle de olhos...',
        'tipo': 'sair'
    },
    'luna hoje': {
        'descricao': 'Falando a previsão do tempo...',
        'tipo': 'previsao'
    },
    'luna oi': {
        'descricao': 'Entrando no modo de conversa...',
        'tipo': 'conversa'
    }
}


# Carrega os comandos de um arquivo JSON
def carregar_comandos():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            return json.load(f)
    return {}


# Salva os comandos em um arquivo JSON
def salvar_comandos(comandos):
    with open(JSON_FILE, 'w') as f:
        json.dump(comandos, f, indent=4)


# Carrega o dataset de um arquivo
def carregar_dataset():
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            return [linha.strip() for linha in f.readlines()]
    return []


# Função para fazer a assistente falar a Luna
def falar(resposta):
    engine.say(resposta)
    engine.runAndWait()


# Função para chamar a API GPT-4o via RapidAPI
def chamar_api_gpt4o(input_text):
    url = "https://cheapest-gpt-4-turbo-gpt-4-vision-chatgpt-openai-ai-api.p.rapidapi.com/v1/chat/completions"

    payload = {
        "messages": [
            {
                "role": "user",
                "content": input_text
            }
        ],
        "model": "gpt-4o",
        "max_tokens": 100,
        "temperature": 0.9
    }
    headers = {
        "x-rapidapi-key": "d953421c96msh2f33f3648cf80edp18717ejsn40697575553a",  # Insira sua chave API
        "x-rapidapi-host": "cheapest-gpt-4-turbo-gpt-4-vision-chatgpt-openai-ai-api.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]  # Retorna a resposta do modelo
    else:
        return "Desculpe, não consegui me conectar ao GPT-4."


# Classe para gerenciar a conversa
class Conversacao:
    def __init__(self, dataset):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(dataset)
        self.total_words = len(self.tokenizer.word_index) + 1
        self.model = self.criar_modelo()
        self.treinar_modelo(dataset)

    def criar_modelo(self):
        model = Sequential()
        model.add(Embedding(self.total_words, 100, input_length=20))
        model.add(LSTM(150, return_sequences=True))
        model.add(LSTM(150))
        model.add(Dense(150, activation='relu'))
        model.add(Dense(self.total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def treinar_modelo(self, dataset):
        sequences = []
        for line in dataset:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                sequences.append(n_gram_sequence)

        max_sequence_length = max(len(x) for x in sequences)
        sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')
        X, y = sequences[:, :-1], sequences[:, -1]

        y = tf.keras.utils.to_categorical(y, num_classes=self.total_words)
        self.model.fit(X, y, epochs=100, verbose=2)

    def gerar_resposta(self, input_text):
        # Tenta usar a API GPT-4o para gerar a resposta
        try:
            response = chamar_api_gpt4o(input_text)
            return response
        except Exception as e:
            return "Desculpe, não consegui me conectar ao GPT-4."


# Classe principal da Luna
class LunaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Assistente Luna")
        self.setGeometry(100, 100, 400, 300)

        # Carrega comandos padrão e personalizados
        self.comandos = {**comandos_padrao, **carregar_comandos()}
        self.escutando = False
        self.modo_conversa = False

        # Carrega o dataset do arquivo
        dataset = carregar_dataset()
        if not dataset:
            dataset = ["Olá", "Oi, como você está?", "Estou bem, e você?", "Que legal!",
                       "O que você gosta de fazer?", "Eu sou a Luna, sua assistente.",
                       "O que você quer saber?", "Como está o tempo?", "Me fale sobre você."]

        self.conversacao = Conversacao(dataset)

        # Layout
        layout = QVBoxLayout()

        # Botões
        self.button_adicionar = QPushButton("Adicionar Comando")
        self.button_adicionar.clicked.connect(self.adicionar_comando)
        layout.addWidget(self.button_adicionar)

        self.button_remover = QPushButton("Remover Comando")
        self.button_remover.clicked.connect(self.remover_comando)
        layout.addWidget(self.button_remover)

        self.button_mostrar = QPushButton("Mostrar Comandos")
        self.button_mostrar.clicked.connect(self.mostrar_comandos)
        layout.addWidget(self.button_mostrar)

        self.button_comecar_escutar = QPushButton("Começar a Escutar")
        self.button_comecar_escutar.clicked.connect(self.comecar_escutar)
        layout.addWidget(self.button_comecar_escutar)

        self.button_parar_escutar = QPushButton("Parar de Escutar")
        self.button_parar_escutar.clicked.connect(self.parar_escutar)
        layout.addWidget(self.button_parar_escutar)

        self.input_comando = QLineEdit(self)
        self.input_comando.setPlaceholderText("Digite seu comando...")
        layout.addWidget(self.input_comando)

        # Terminal da Luna
        self.console = QTextEdit(self)
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(150)
        layout.addWidget(self.console)

        # Widget central
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Saudação inicial
        self.console.append("Bem-vindo à Luna!")
        falar("Bem-vindo à Luna! Como posso ajudá-lo hoje?")

        # QSS
        self.aplicar_estilo()

        # Conectar o retorno pressionado do campo de texto ao método processar_comando_escrito
        self.input_comando.returnPressed.connect(self.processar_comando_escrito)

    def aplicar_estilo(self):
        qss = """
        QMainWindow {
            background-color: #282a36;
        }
        QPushButton {
            background-color: #6272a4;
            color: #ffffff;
            border: none;
            padding: 10px;
            border-radius: 5px
        }
        QPushButton:hover {
            background-color: #44475a;
        }
        QTextEdit {
            background-color: #44475a;
            color: #ffffff;
            border: none;
            padding: 10px;
            border-radius: 5px
        }
        QLineEdit {
            background-color: #D9D9D9;
            color: #000000;
            border: none;
            padding: 10px;
            border-radius: 5px
        }
        """
        self.setStyleSheet(qss)

    def processar_comando_escrito(self):
        comando = self.input_comando.text().strip().lower()
        self.input_comando.clear()
        self.processar_comando(comando)

    def processar_comando(self, comando):
        if comando in self.comandos:
            if self.comandos[comando]['tipo'] == 'navegador':
                os.system(f'start {self.comandos[comando]["caminho"]}')
                self.console.append(self.comandos[comando]['descricao'])
                falar(self.comandos[comando]['descricao'])
            elif self.comandos[comando]['tipo'] == 'pasta':
                os.startfile(self.comandos[comando]['caminho'])
                self.console.append(self.comandos[comando]['descricao'])
                falar(self.comandos[comando]['descricao'])
            elif self.comandos[comando]['tipo'] == 'olhos':
                # Aqui você adicionaria a funcionalidade de controle de mouse pelos olhos
                self.console.append(self.comandos[comando]['descricao'])
                falar(self.comandos[comando]['descricao'])
            elif self.comandos[comando]['tipo'] == 'sair':
                self.modo_conversa = False
                self.console.append(self.comandos[comando]['descricao'])
                falar(self.comandos[comando]['descricao'])
            elif self.comandos[comando]['tipo'] == 'previsao':
                # Aqui você adicionaria a funcionalidade de previsão do tempo
                self.console.append(self.comandos[comando]['descricao'])
                falar(self.comandos[comando]['descricao'])
            elif self.comandos[comando]['tipo'] == 'conversa':
                self.modo_conversa = True
                self.console.append(self.comandos[comando]['descricao'])
                falar(self.comandos[comando]['descricao'])
        else:
            if self.modo_conversa:
                resposta = self.conversacao.gerar_resposta(comando)
                self.console.append(f"Luna: {resposta}")
                falar(resposta)
            else:
                self.console.append("Comando não reconhecido.")
                falar("Desculpe, não entendi o comando.")

    def adicionar_comando(self):
        novo_comando, ok = QInputDialog.getText(self, 'Adicionar Comando', 'Digite o novo comando:')
        if ok and novo_comando:
            descricao, ok = QInputDialog.getText(self, 'Descrição do Comando', 'Digite a descrição do comando:')
            if ok and descricao:
                self.comandos[novo_comando] = {
                    'descricao': descricao,
                    'tipo': 'personalizado'
                }
                salvar_comandos(self.comandos)
                self.console.append(f"Comando '{novo_comando}' adicionado com sucesso.")

    def remover_comando(self):
        comando_remover, ok = QInputDialog.getText(self, 'Remover Comando', 'Digite o comando a ser removido:')
        if ok and comando_remover in self.comandos:
            del self.comandos[comando_remover]
            salvar_comandos(self.comandos)
            self.console.append(f"Comando '{comando_remover}' removido com sucesso.")
        else:
            self.console.append("Comando não encontrado.")

    def mostrar_comandos(self):
        comandos_list = "\n".join([f"{cmd}: {info['descricao']}" for cmd, info in self.comandos.items()])
        self.console.append(f"Comandos disponíveis:\n{comandos_list}")

    def comecar_escutar(self):
        self.escutando = True
        self.console.append("Escutando...")
        falar("Estou escutando.")
        self.iniciar_reconhecimento_voz()

    def parar_escutar(self):
        self.escutando = False
        self.console.append("Parando de escutar...")
        falar("Parando de escutar.")

    def iniciar_reconhecimento_voz(self):
        if not self.escutando:
            return
        recognizer = sr.Recognizer()
        r = sr.Recognizer()
        r.energy_threshold = 500
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, timeout=5)
                comando = recognizer.recognize_google(audio, language="pt-BR").lower()
                self.console.append(f"Comando reconhecido: {comando}")
                self.processar_comando(comando)
            except sr.UnknownValueError:
                self.console.append("Não consegui entender o que você disse.")
            except sr.RequestError as e:
                self.console.append("Erro na conexão com o serviço de reconhecimento de voz.")
            except Exception as e:
                self.console.append(f"Ocorreu um erro: {e}")

        threading.Timer(1.0, self.iniciar_reconhecimento_voz).start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    luna = LunaApp()
    luna.show()
    sys.exit(app.exec_())

