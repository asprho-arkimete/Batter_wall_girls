
🧱 Batter Wall Girls
Un gioco arcade in cui il tuo obiettivo è abbattere il muro di mattoncini colorati. Man mano che avanzi nel gioco, la ragazza sullo sfondo eseguirà azioni di cambio outfit.

🛠️ Installazione
Segui questi passaggi per configurare l'ambiente di gioco sul tuo computer.

1. Clona il Repository
Apri il terminale e scarica il progetto:

Bash

git clone https://github.com/asprho-arkimete/Batter_wall_girls.git
cd Batter_wall_girls
2. Crea e Attiva l'Ambiente Virtuale
È consigliato usare un ambiente virtuale per non creare conflitti con altre librerie Python.

Su Windows:

Bash

python -m venv vbatter
.\vbatter\Scripts\activate
(Se usi Linux/Mac, il comando di attivazione è: source vbatter/bin/activate)

3. Installa le Dipendenze
Assicurati di avere il file dei requisiti (se lo hai chiamato riquisiti.txt, usa quel nome, ma lo standard è requirements.txt).

Bash

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
(Nota: Il comando include l'URL extra per scaricare correttamente PyTorch con supporto CUDA, come visto nelle librerie precedenti).

📥 Download Modelli e LoRA
Prima di avviare il gioco, è necessario scaricare i file dei modelli. Consulta il file download.txt presente nella cartella principale per i link specifici.

Modelli: Scarica i file checkpoint indicati e inseriscili nella cartella: \models

LoRA: Scarica i file LoRA indicati e inseriscili nella cartella: \lora

🚀 Avvio del Gioco
Una volta installato tutto e scaricati i modelli, puoi avviare il gioco con:

Bash

python fill.py
