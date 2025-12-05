import torch
from diffusers import FluxFillPipeline
from diffusers.utils import make_image_grid
from PIL import Image
import numpy as np
from diffusers import FluxTransformer2DModel
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize
import os
from deep_translator import GoogleTranslator
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn




os.makedirs("output",exist_ok=True)
def Fill(prompt, image_path, steps, cfg):
    
    # Rileva vestiti e crea mask
    print("Caricamento modello di segmentazione...")
    processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
    
    print(f"Analisi immagine: {image_path}")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Inferenza
    print("Generazione maschera vestiti...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits.cpu()
    
    # Upsample alla dimensione originale
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )
    
    # Ottieni la segmentazione predetta
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # Seleziona SOLO i vestiti: upper-clothes, skirt, pants, dress
    clothes_classes = [4, 5, 6, 7]

    # Crea maschera binaria solo per queste classi
    clothes_mask = np.isin(pred_seg, clothes_classes).astype(np.uint8) * 255

    print(f"  Classi rilevate: {np.unique(pred_seg)}")
    print(f"  Pixel vestiti: {np.sum(clothes_mask > 0)}")

    # Espansione ADATTIVA in base al tipo di vestito
    from scipy.ndimage import binary_dilation, gaussian_filter

    has_skirt = 5 in pred_seg
    has_pants = 6 in pred_seg
    has_dress = 7 in pred_seg

    if has_skirt or has_dress:
        kernel_size = 25  # Gonne/vestiti: espandi molto
        print("  Rilevata gonna/vestito → espansione 25px")
    elif has_pants:
        kernel_size = 20  # Pantaloni: espansione maggiorata
        print("  Rilevati pantaloni → espansione 15px")
    else:
        kernel_size = 18  # Default
        print("  Altri vestiti → espansione 13px")

    # Dilatazione
    kernel = np.ones((kernel_size*2+1, kernel_size*2+1), dtype=np.uint8)
    expanded_mask = binary_dilation(clothes_mask > 0, structure=kernel).astype(np.uint8) * 255

    # Feathering per transizioni più smooth (opzionale ma consigliato)
    expanded_mask_float = expanded_mask.astype(float) / 255.0
    blurred_mask = gaussian_filter(expanded_mask_float, sigma=2.5)
    expanded_mask = (blurred_mask * 255).astype(np.uint8)

    # Converti in immagine PIL e salva
    mask_image = Image.fromarray(expanded_mask, mode='L')
    mask_image.save("mask.jpg")
    print("✓ Maschera salvata: mask.jpg")
    
    # Visualizza anche la mask sovrapposta (opzionale, per debug)
    try:
        # Crea versione colorata per debug
        overlay = image.copy()
        mask_colored = Image.fromarray(expanded_mask).convert('RGB')
        mask_colored = Image.eval(mask_colored, lambda x: 255 if x > 0 else 0)
        
        from PIL import ImageDraw
        overlay.paste(mask_colored, (0, 0), mask_image)
        overlay.save("mask_overlay_debug.jpg")
        print("  Debug overlay salvato: mask_overlay_debug.jpg")
    except Exception as e:
        print(f"  (Debug overlay non creato: {e})")
    # Configurazione
    repofbl = "black-forest-labs/FLUX.1-Fill-dev"
    Dtype = torch.bfloat16
    
    modelfillOnereward_fp8 = "flux1FillDevOnereward_fp8.safetensors"
    modelfill_f8 = "agfluxFillNSFWFp8_agfluxFillNSFWV17Fp8.safetensors"
    
    print("Caricamento transformer custom...")
    transformer = FluxTransformer2DModel.from_single_file(
        os.path.join('model', modelfill_f8),
        torch_dtype=Dtype,
    )
    
    print("Quantizzazione transformer...")
    quantize(transformer, weights=qfloat8)
    freeze(transformer)
    
    print("Caricamento text encoder...")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        repofbl, 
        subfolder="text_encoder_2", 
        torch_dtype=Dtype
    )
    quantize(text_encoder_2, weights=qfloat8)
    freeze(text_encoder_2)
    
    print("Inizializzazione pipeline...")
    pipe = FluxFillPipeline.from_pretrained(repofbl, torch_dtype=Dtype)
    pipe.transformer = transformer
    pipe.text_encoder_2 = text_encoder_2
    
    pipe.enable_model_cpu_offload()
    
    # Carica immagini
    print(f"Caricamento immagine: {image_path}")
    init_image = Image.open(image_path).convert("RGB")
    mask_image = Image.open("mask.jpg").convert("RGB")
    w, h = init_image.size
    
    # Ridimensiona se necessario
    if w > 1024 or h > 1024:
        if w >= h:
            new_w = 1024
            new_h = (1024 * h) // w
        else:
            new_w = (1024 * w) // h
            new_h = 1024
        
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        
        print(f"Ridimensionamento da {w}x{h} a {new_w}x{new_h}")
        
        init_image = init_image.resize((new_w, new_h), Image.BICUBIC)
        mask_image = mask_image.resize((new_w, new_h), Image.NEAREST)
    else:
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w != w or new_h != h:
            print(f"Allineamento a multipli di 8: da {w}x{h} a {new_w}x{new_h}")
            init_image = init_image.resize((new_w, new_h), Image.BICUBIC)
            mask_image = mask_image.resize((new_w, new_h), Image.NEAREST)
    
    new_width, new_height = init_image.size
    print(f"Dimensioni finali: {new_width}x{new_height}")
    
    # Traduci prompt
    print("Traduzione prompt...")
    prompt_en = GoogleTranslator(source='it', target='en').translate(prompt)
    print(f"Prompt ({len(prompt_en)} caratteri): {prompt_en}\n")

    print(f"steps:{steps},cfg:{cfg}")
    
    # Genera immagine
    print("Generazione in corso...")
    image = pipe(
        prompt=prompt_en,
        image=init_image,
        mask_image=mask_image,
        width=new_width,
        height=new_height,
        guidance_scale=cfg,
        num_inference_steps=steps,
        max_sequence_length=512,
        strength=1.0,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    
    # Salva risultato
    nome= os.path.basename(image_path)
    image.save(f"output/{nome}")
    print("✓ Generazione completata!")
    print(f"  Immagine salvata: output_Fill.png")
    
    # Crea griglia di confronto
    try:
        grid = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
        grid.save("output_Fill_grid.png")
        print(f"  Griglia salvata: output_Fill_grid.png")
    except Exception as e:
        print(f"  Nota: impossibile creare griglia ({e})")


# Esempio di utilizzo
if __name__ == "__main__":
    # Esempio di prompt
    prompt_pube_rasato = """
    girl standing, completely naked, natural beauty,
    (full and round breasts:1.4), (prominent nipples:1.5), 
    (pussy fully shaved:1.9), (shaved, hairless pubis:1.9)
    (pussy lips clearly defined:1.8), detailed, quality 8k
    """

    prompt_peli_pube = """
    (masterpiece:1.2), (photorealistic:1.3), stunning details, 
    girl standing, completely naked, natural beauty,
    (full and round breasts:1.4), (prominent nipples:1.5), 
    highly detailed pubic area, (natural pubic hair:1.6), 
    (well-groomed bush:1.5), (realistic hair texture:1.4),
    (pussy lips clearly defined:1.8), perfect lighting, 8k quality
    """
 
import tkinter as tk
from tkinter import ttk
import os # Importazione necessaria per accedere al filesystem

def loadlora(event=None):
    """
    Carica i nomi dei file con estensione .safetensors (o simili)
    dalla sottocartella 'lora' e li imposta come valori della Combobox.
    """
    # Usa un blocco try-except per gestire il caso in cui la cartella non esista
    try:
        # Filtra solo i file che potrebbero essere Lora (ad es. .safetensors o .pt)
        # Ho usato un'estensione comune come esempio. Modificala se necessario.
        lora_files = [f for f in os.listdir('./lora') if f.endswith(('.safetensors', '.pt'))]
        
        # Aggiungi l'opzione 'None' se non è presente
        if 'None' not in lora_files:
            lora_files.insert(0, 'None')
            
        # Correzione: Il metodo per impostare i valori è .config(values=...)
        lora.config(values=lora_files)
        
        # Imposta 'None' come valore selezionato di default
        lora.set('None')
        
    except FileNotFoundError:
        # Se la cartella 'lora' non esiste, imposta solo 'None'
        lora.config(values=['None'])
        lora.set('None')
        print("Attenzione: La cartella './lora' non è stata trovata.")


# Inizializzazione della finestra
window = tk.Tk()
window.title("Batter Wall Girls")
window.geometry("600x400") # Aumentato per una migliore visualizzazione dei widget grid

# --- Widget Principali ---

# Correzione: Aggiunta un'area di testo più grande, ridotte le dimensioni per adattarsi meglio
text = tk.Text(window, width=50, height=10)
# Usa row=0 per l'area di testo, che si estende su più colonne (columnspan)
text.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

# --- Sezione STEPS ---
labsteps = tk.Label(window, text="Steps:")
labsteps.grid(row=1, column=0, padx=10, pady=5, sticky='w')

# Correzione: from_ e to_ devono essere separati da una virgola, non sono usati per l'assegnazione
steps = tk.Scale(window, from_=1, to=80, orient=tk.HORIZONTAL) # Aggiunto orientamento orizzontale
steps.grid(row=2, column=0, padx=10, pady=5, sticky='ew')
steps.set(30)

# --- Sezione CFG ---
lab_cfg = tk.Label(window, text='CFG:')
lab_cfg.grid(row=1, column=1, padx=10, pady=5, sticky='w')

cfg = tk.Scale(window, from_=1, to=40, orient=tk.HORIZONTAL)
cfg.grid(row=2, column=1, padx=10, pady=5, sticky='ew')
cfg.set(30) # Valore di default più comune
# Ho notato che avevi impostato 30, ho messo 7 come è più comune in ML, ma puoi lasciare 30.

# --- Sezione LORA ---
lab_Lora = tk.Label(window, text='Lora:')
lab_Lora.grid(row=1, column=2, padx=10, pady=5, sticky='w')

# Inizializza la Combobox con valori predefiniti
lora = ttk.Combobox(window, values=['None'], state='readonly') # 'readonly' impedisce la scrittura libera
lora.grid(row=2, column=2, padx=10, pady=5, sticky='ew')
loadlora() 

import random
import threading
def F_genera(image_path):
    if text.get('1.0', tk.END).strip() == '':
        prompt = ''
        
        # ... (logica per la selezione casuale del prompt)
        n = random.randint(1, 100)
        
        if n >= 50: 
            prompt = prompt_peli_pube
        else:
            prompt = prompt_pube_rasato
            
        # Correzione: Rimosso il quinto argomento 'True' (che si riferiva a 'demon')
        thread_args = (
            prompt, 
            image_path, 
            int(steps.get()), 
            float(cfg.get()) 
        )
        
        # La funzione 'Fill' ora riceverà esattamente 4 argomenti
        threading.Thread(target=Fill, args=thread_args).start()

genera = tk.Button(text='Genera', command=F_genera)
genera.grid(row=3, column=0)


import pygame
import random
import os
from PIL import Image

# Inizializza Pygame
pygame.init()
pygame.mixer.init()  # Inizializza il mixer audio
sound_fire = None
sound_explosion = None

# Carica lista musiche
musica = []
if os.path.exists("musica"):
    musica = [os.path.join("musica", m) for m in os.listdir("musica") 
              if m.endswith(('.mp3', '.wav', '.ogg'))]

# Seleziona canzone casuale
ind_canzone = 0
if musica:
    ind_canzone = random.randint(0, len(musica) - 1)
    try:
        pygame.mixer.music.load(musica[ind_canzone])
        pygame.mixer.music.play()
        print(f"Riproduzione: {os.path.basename(musica[ind_canzone])}")
    except Exception as e:
        print(f"Errore caricamento musica: {e}")




# Costanti
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 1024

# Crea schermata
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Batter Wall Girls")
clock = pygame.time.Clock()

# PASSO 1: Carica tutte le immagini dalle cartelle
print("Caricamento immagini...")

# Inizializza liste vuote
photoBackground = []
cubics = []
esplosione_frames = []
photoBackground_paths = []

# Carica sfondi con ridimensionamento proporzionale
if os.path.exists("PhotoBackgroundInput"):
    for img_name in os.listdir("PhotoBackgroundInput"):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join("PhotoBackgroundInput", img_name)
            
            photoBackground_paths.append(img_path)
            
            img = Image.open(img_path)
            
            # Calcola ridimensionamento proporzionale
            img_width, img_height = img.size
            ratio = min(SCREEN_WIDTH / img_width, SCREEN_HEIGHT / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            # Ridimensiona mantenendo proporzioni
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Crea surface con sfondo nero
            background_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            background_surface.fill((0, 0, 0))
            
            # Converti immagine ridimensionata
            mode = img.mode
            size = img.size
            data = img.tobytes()
            py_img = pygame.image.fromstring(data, size, mode)
            
            # Centra l'immagine nello schermo
            x_offset = (SCREEN_WIDTH - new_width) // 2
            y_offset = (SCREEN_HEIGHT - new_height) // 2
            background_surface.blit(py_img, (x_offset, y_offset))
            
            photoBackground.append(background_surface)

print(f"Caricati {len(photoBackground)} sfondi")
print(f"Caricati {len(photoBackground_paths)} percorsi sfondi")

# Carica cubi
if os.path.exists("sprints/cubs"):
    for img_name in os.listdir("sprints/cubs"):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join("sprints/cubs", img_name)
            img = Image.open(img_path)
            img = img.resize((40, 40))
            mode = img.mode
            size = img.size
            data = img.tobytes()
            py_img = pygame.image.fromstring(data, size, mode)
            cubics.append(py_img)

# Carica astronave
astronave_img = None
astronave_turbo_img = None
if os.path.exists("sprints/space.png"):
    img = Image.open("sprints/space.png")
    img = img.resize((80, 60))
    mode = img.mode
    size = img.size
    data = img.tobytes()
    astronave_img = pygame.image.fromstring(data, size, mode)

if os.path.exists("sprints/space_turbo.png"):
    img = Image.open("sprints/space_turbo.png")
    img = img.resize((80, 60))
    mode = img.mode
    size = img.size
    data = img.tobytes()
    astronave_turbo_img = pygame.image.fromstring(data, size, mode)

# Carica fuoco
fire_img = None
if os.path.exists("sprints/fire.png"):
    img = Image.open("sprints/fire.png")
    img = img.resize((30, 60))
    mode = img.mode
    size = img.size
    data = img.tobytes()
    fire_img = pygame.image.fromstring(data, size, mode)

# Carica esplosione
if os.path.exists("sprints/espo/"):
    for img_name in sorted(os.listdir("sprints/espo/")):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join("sprints/espo/", img_name)
            img = Image.open(img_path)
            img = img.resize((50, 50))
            mode = img.mode
            size = img.size
            data = img.tobytes()
            py_img = pygame.image.fromstring(data, size, mode)
            esplosione_frames.append(py_img)

print(f"Caricati {len(esplosione_frames)} frame esplosione")
print(f"Caricati {len(cubics)} cubi")

# Carica moltiplicatori
x2_img = None
x4_img = None
if os.path.exists("sprints/2x.png"):
    img = Image.open("sprints/2x.png")
    img = img.resize((60, 60))
    mode = img.mode
    size = img.size
    data = img.tobytes()
    x2_img = pygame.image.fromstring(data, size, mode)

if os.path.exists("sprints/4x.png"):
    img = Image.open("sprints/4x.png")
    img = img.resize((60, 60))
    mode = img.mode
    size = img.size
    data = img.tobytes()
    x4_img = pygame.image.fromstring(data, size, mode)

# Loop principale
running = True

# Seleziona un indice casuale all'inizio
if photoBackground:
    indice_sfondo = random.randint(0, len(photoBackground) - 1)
else:
    indice_sfondo = -1

# CREA IL MURO UNA SOLA VOLTA prima del loop
muro_mattoni = []
muro_rect = []

if cubics:
    larghezza_mattone = 40
    altezza_mattone = 40
    
    for fila in range(10):
        mattone = random.choice(cubics)
        muro_mattoni.append(mattone)
        
        num_mattoni = SCREEN_WIDTH // larghezza_mattone + 1
        fila_rects = []
        for colonna in range(num_mattoni):
            x = colonna * larghezza_mattone
            y = fila * altezza_mattone
            rect = pygame.Rect(x, y, larghezza_mattone, altezza_mattone)
            fila_rects.append({
                'rect': rect, 
                'attivo': True, 
                'animazione': -1, 
                'frame': 0,
                'colore_rosso': 0
            })
        muro_rect.append(fila_rects)

# Variabili per la navicella
navicella_x = SCREEN_WIDTH // 2
navicella_y = SCREEN_HEIGHT // 2
navicella_precedente_x = navicella_x
navicella_precedente_y = navicella_y
navicella_in_movimento = False
navicella_moltiplicatore = 1  # Numero di navicelle (1, 2 o 4)

# Lista fiamme sparate
fiamme = []

# Lista bonus in caduta
bonus_caduta = []

# Timer per aggiungere nuove file di mattoni
timer_nuova_fila = 0
intervallo_nuova_fila = 1600   

# Timer per generare bonus
timer_bonus = 0
intervallo_bonus = 500  

# Variabili per gestione livello completato
livello_completato = False
generazione_in_corso = False
immagine_da_mostrare = None
indice_sfondo_successivo = 0
indice_sfondo_precedente = -1  # Per tracciare cambio sfondo

# Carica suono sparo
if os.path.exists("sparo_suono.wav"):
    sound_fire = pygame.mixer.Sound("sparo_suono.wav")
    sound_fire.set_volume(0.3)  # Volume al 30%

# Carica suono esplosione (opzionale)
if os.path.exists("esploasione.wav"):
    sound_explosion = pygame.mixer.Sound("esploasione.wav")
    sound_explosion.set_volume(0.4)


while running:
    # Genera immagine dello sfondo corrente all'avvio o quando cambia lo sfondo
    if indice_sfondo != indice_sfondo_precedente:
        indice_sfondo_precedente = indice_sfondo
        output_path = f"output/{os.path.basename(photoBackground_paths[indice_sfondo])}"
        
        if not os.path.exists(output_path):
            # Chiama F_genera (già gestisce threading internamente)
            try:
                print(f"Pre-generazione immagine per sfondo {indice_sfondo}...")
                F_genera(photoBackground_paths[indice_sfondo])
            except Exception as e:
                print(f"Errore generazione: {e}")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if fire_img:
                # Riproduci suono sparo
                if sound_fire:
                    sound_fire.play()
                
                # Crea fiamme in base al numero di navicelle
                if navicella_moltiplicatore == 1:
                    # Una sola navicella al centro
                    fiamme.append({
                        'x': navicella_x,
                        'y': navicella_y - 40,
                        'velocita': 10
                    })
                elif navicella_moltiplicatore == 2:
                    # Due navicelle affiancate
                    offset = 50
                    fiamme.append({
                        'x': navicella_x - offset,
                        'y': navicella_y - 40,
                        'velocita': 10
                    })
                    fiamme.append({
                        'x': navicella_x + offset,
                        'y': navicella_y - 40,
                        'velocita': 10
                    })
                elif navicella_moltiplicatore == 4:
                    # Quattro navicelle a quadrato
                    offset_x = 50
                    offset_y = 40
                    fiamme.append({
                        'x': navicella_x - offset_x,
                        'y': navicella_y - offset_y - 40,
                        'velocita': 10
                    })
                    fiamme.append({
                        'x': navicella_x + offset_x,
                        'y': navicella_y - offset_y - 40,
                        'velocita': 10
                    })
                    fiamme.append({
                        'x': navicella_x - offset_x,
                        'y': navicella_y + offset_y - 40,
                        'velocita': 10
                    })
                    fiamme.append({
                        'x': navicella_x + offset_x,
                        'y': navicella_y + offset_y - 40,
                        'velocita': 10
                    })
    
    # Genera bonus casuali
    timer_bonus += 1
    if timer_bonus >= intervallo_bonus and random.randint(1, 100) >= 80:
        timer_bonus = 0
        bonus_type = 'x2' if random.randint(1, 2) == 1 else 'x4'
        bonus_x = random.randint(50, SCREEN_WIDTH - 50)
        bonus_caduta.append({
            'x': bonus_x,
            'y': 0,
            'type': bonus_type,
            'velocita': 3
        })
    
    # Aggiorna bonus in caduta
    for bonus in bonus_caduta[:]:
        bonus['y'] += bonus['velocita']
        
        # Rimuovi se esce dallo schermo
        if bonus['y'] > SCREEN_HEIGHT:
            bonus_caduta.remove(bonus)
            continue
        
        # Controlla collisione con navicella
        bonus_rect = pygame.Rect(bonus['x'], bonus['y'], 60, 60)
        navicella_rect = pygame.Rect(navicella_x - 40, navicella_y - 30, 80, 60)
        
        if bonus_rect.colliderect(navicella_rect):
            if bonus['type'] == 'x2':
                navicella_moltiplicatore = 2
            elif bonus['type'] == 'x4':
                navicella_moltiplicatore = 4
            bonus_caduta.remove(bonus)
    
    # Ottieni posizione del mouse
    mouse_x, mouse_y = pygame.mouse.get_pos()
    
    # Controlla se la navicella si sta muovendo
    if mouse_x != navicella_precedente_x or mouse_y != navicella_precedente_y:
        navicella_in_movimento = True
    else:
        navicella_in_movimento = False
    
    # Aggiorna posizione navicella
    navicella_precedente_x = navicella_x
    navicella_precedente_y = navicella_y
    navicella_x = mouse_x
    navicella_y = mouse_y
    
    # CONTROLLA COLLISIONE NAVICELLA CON MURO - GAME OVER
    navicella_rect = pygame.Rect(navicella_x - 40, navicella_y - 30, 80, 60)
    game_over = False
    
    for fila_rects in muro_rect:
        for mattone_data in fila_rects:
            if mattone_data['attivo'] and mattone_data['animazione'] == -1:
                if navicella_rect.colliderect(mattone_data['rect']):
                    game_over = True
                    print("COLLISIONE RILEVATA!")  # Debug
                    break
        if game_over:
            break
    
    if game_over:
        print("GAME OVER ATTIVATO!")  # Debug
        # Animazione esplosione navicella
        if esplosione_frames:
            for frame_idx in range(len(esplosione_frames)):
                screen.blit(photoBackground[indice_sfondo], (0, 0))
                
                # Disegna muro
                if muro_mattoni:
                    for fila_idx, fila_rects in enumerate(muro_rect):
                        if fila_idx < len(muro_mattoni):
                            mattone = muro_mattoni[fila_idx]
                            for mattone_data in fila_rects:
                                if mattone_data['attivo']:
                                    screen.blit(mattone, mattone_data['rect'])
                
                # Disegna esplosione navicella
                screen.blit(esplosione_frames[frame_idx], (navicella_x - 25, navicella_y - 25))
                pygame.display.flip()
                clock.tick(10)  # Rallenta l'animazione
        
        # Mostra GAME OVER
        screen.fill((0, 0, 0))
        font = pygame.font.Font(None, 100)
        text = font.render("GAME OVER", True, (255, 0, 0))
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(text, text_rect)
        
        font_small = pygame.font.Font(None, 40)
        text_continue = font_small.render("Press any key to quit", True, (255, 255, 255))
        text_continue_rect = text_continue.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
        screen.blit(text_continue, text_continue_rect)
        pygame.display.flip()
        
        # Aspetta input
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False
        
        running = False
        continue
    
    # Timer per aggiungere nuova fila
    timer_nuova_fila += 1
    if timer_nuova_fila >= intervallo_nuova_fila and cubics:
        timer_nuova_fila = 0
        
        larghezza_mattone = 40
        altezza_mattone = 40
        
        for fila_rects in muro_rect:
            for mattone_data in fila_rects:
                mattone_data['rect'].y += altezza_mattone
        
        mattone_nuovo = random.choice(cubics)
        muro_mattoni.insert(0, mattone_nuovo)
        
        num_mattoni = SCREEN_WIDTH // larghezza_mattone + 1
        nuova_fila = []
        for colonna in range(num_mattoni):
            x = colonna * larghezza_mattone
            y = 0
            rect = pygame.Rect(x, y, larghezza_mattone, altezza_mattone)
            nuova_fila.append({
                'rect': rect,
                'attivo': True,
                'animazione': -1,
                'frame': 0,
                'colore_rosso': 0
            })
        muro_rect.insert(0, nuova_fila)
    
    # Aggiorna posizione fiamme
    for fiamma in fiamme[:]:
        fiamma['y'] -= fiamma['velocita']
        
        if fiamma['y'] < -60:
            fiamme.remove(fiamma)
            continue
        
        fiamma_rect = pygame.Rect(fiamma['x'] - 15, fiamma['y'], 30, 60)
        
        for fila_idx, fila_rects in enumerate(muro_rect):
            for mattone_data in fila_rects:
                if mattone_data['attivo'] and mattone_data['animazione'] == -1:
                    if fiamma_rect.colliderect(mattone_data['rect']):
                        mattone_data['colore_rosso'] = min(255, mattone_data['colore_rosso'] + 85)
                        
                        if mattone_data['colore_rosso'] >= 250:
                            mattone_data['animazione'] = 0
                            # Riproduci suono esplosione
                            if sound_explosion:
                                sound_explosion.play()
                        
                        if fiamma in fiamme:
                            fiamme.remove(fiamma)
                        break
    
    # Controlla se tutti i mattoni sono distrutti
    tutti_distrutti = True
    for fila_rects in muro_rect:
        for mattone_data in fila_rects:
            if mattone_data['attivo']:
                tutti_distrutti = False
                break
        if not tutti_distrutti:
            break
    
    # Gestione completamento livello
    if tutti_distrutti and len(muro_rect) > 0 and not generazione_in_corso and not livello_completato:
        print("LIVELLO COMPLETATO! Generazione in corso...")
        generazione_in_corso = True
        indice_sfondo_successivo = indice_sfondo
        
        # CONTROLLA WINNER PRIMA DI RESETTARE IL MURO
        if photoBackground_paths:
            try:
                files_output = [f for f in os.listdir("output") if f.endswith(('.png', '.jpg', '.jpeg'))]
                names_input = [os.path.basename(path) for path in photoBackground_paths]
                
                print(f"\n=== CHECK WINNER (Fine Livello) ===")
                print(f"Immagini da generare: {len(names_input)}")
                print(f"Immagini generate: {len(files_output)}")
                print(f"Lista input: {names_input}")
                print(f"Lista output: {files_output}")
                
                # Verifica se tutti i file input sono presenti in output
                tutti_generati = all(name in files_output for name in names_input)
                
                if tutti_generati and len(files_output) >= len(names_input):
                    print("TUTTE LE IMMAGINI GENERATE - YOU WIN!")
                    
                    # Mostra WINNER
                    screen.fill((0, 0, 0))
                    font = pygame.font.Font(None, 100)
                    text = font.render("YOU WIN!", True, (0, 255, 0))
                    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
                    screen.blit(text, text_rect)
                    
                    # Mostra statistiche
                    font_small = pygame.font.Font(None, 40)
                    stats = f"Images generated: {len(files_output)} / {len(names_input)}"
                    text_stats = font_small.render(stats, True, (255, 255, 0))
                    text_stats_rect = text_stats.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
                    screen.blit(text_stats, text_stats_rect)
                    
                    text_continue = font_small.render("Press any key to quit", True, (255, 255, 255))
                    text_continue_rect = text_continue.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
                    screen.blit(text_continue, text_continue_rect)
                    pygame.display.flip()
                    
                    # Aspetta input
                    waiting = True
                    while waiting:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                waiting = False
                            elif event.type == pygame.KEYDOWN:
                                waiting = False
                    
                    running = False
                    continue
                else:
                    print(f"Ancora {len(names_input) - len(files_output)} immagini da generare")
            except Exception as e:
                print(f"Errore controllo winner: {e}")
        
        # Chiama F_genera (già gestisce threading internamente)
        try:
            F_genera(photoBackground_paths[indice_sfondo])
        except Exception as e:
            print(f"Errore generazione: {e}")
        
        # Reset del muro per continuare a giocare
        muro_mattoni = []
        muro_rect = []
        if cubics:
            larghezza_mattone = 40
            altezza_mattone = 40
            
            for fila in range(10):
                mattone = random.choice(cubics)
                muro_mattoni.append(mattone)
                
                num_mattoni = SCREEN_WIDTH // larghezza_mattone + 1
                fila_rects = []
                for colonna in range(num_mattoni):
                    x = colonna * larghezza_mattone
                    y = fila * altezza_mattone
                    rect = pygame.Rect(x, y, larghezza_mattone, altezza_mattone)
                    fila_rects.append({
                        'rect': rect,
                        'attivo': True,
                        'animazione': -1,
                        'frame': 0,
                        'colore_rosso': 0
                    })
                muro_rect.append(fila_rects)
    
    # Controlla se l'immagine generata è pronta
    if generazione_in_corso and photoBackground_paths:
        output_path = f"output/{os.path.basename(photoBackground_paths[indice_sfondo_successivo])}"
        if os.path.exists(output_path):
            try:
                # Aspetta che il file sia completamente scritto
                import time
                file_size = -1
                stable_count = 0
                
                # Controlla se il file è stabile (dimensione non cambia)
                while stable_count < 3:
                    current_size = os.path.getsize(output_path)
                    if current_size == file_size and current_size > 0:
                        stable_count += 1
                    else:
                        stable_count = 0
                    file_size = current_size
                    time.sleep(0.1)
                
                print("Immagine generata! Caricamento...")
                generazione_in_corso = False
                
                # Carica l'immagine generata
                img = Image.open(output_path)
                img.load()  # Forza il caricamento completo
                img_width, img_height = img.size
                ratio = min(SCREEN_WIDTH / img_width, SCREEN_HEIGHT / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                background_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                background_surface.fill((0, 0, 0))
                
                mode = img.mode
                size = img.size
                data = img.tobytes()
                py_img = pygame.image.fromstring(data, size, mode)
                
                x_offset = (SCREEN_WIDTH - new_width) // 2
                y_offset = (SCREEN_HEIGHT - new_height) // 2
                background_surface.blit(py_img, (x_offset, y_offset))
                
                immagine_da_mostrare = background_surface
                livello_completato = True
            except Exception as e:
                print(f"Errore caricamento immagine: {e}")
                # Riprova al prossimo frame
                pass
    
    # Disegna lo sfondo
    if indice_sfondo >= 0:
        screen.blit(photoBackground[indice_sfondo], (0, 0))
    else:
        screen.fill((0, 0, 0))
    
    # Se il livello è completato, mostra l'immagine generata (senza mattoni)
    if livello_completato and immagine_da_mostrare:
        screen.blit(immagine_da_mostrare, (0, 0))
        pygame.display.flip()
        
        # Aspetta input utente
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
                    # Passa al prossimo sfondo
                    if photoBackground:
                        indice_sfondo = (indice_sfondo + 1) % len(photoBackground)
                    livello_completato = False
                    immagine_da_mostrare = None
        continue
    
    # Disegna il muro di mattoni (solo se non in pausa)
    if muro_mattoni:
        larghezza_mattone = 40
        altezza_mattone = 40
        
        for fila_idx, fila_rects in enumerate(muro_rect):
            if fila_idx < len(muro_mattoni):
                mattone = muro_mattoni[fila_idx]
                
                for mattone_data in fila_rects:
                    if mattone_data['attivo']:
                        if mattone_data['animazione'] >= 0:
                            if esplosione_frames and mattone_data['animazione'] < len(esplosione_frames):
                                frame = esplosione_frames[int(mattone_data['animazione'])]
                                screen.blit(frame, (mattone_data['rect'].x - 5, mattone_data['rect'].y - 5))
                                mattone_data['animazione'] += 0.5
                            else:
                                mattone_data['attivo'] = False
                        else:
                            screen.blit(mattone, mattone_data['rect'])
                            
                            if mattone_data['colore_rosso'] > 0:
                                overlay = pygame.Surface((larghezza_mattone, altezza_mattone))
                                overlay.set_alpha(mattone_data['colore_rosso'])
                                overlay.fill((255, 0, 0))
                                screen.blit(overlay, mattone_data['rect'])
    
    # Disegna bonus in caduta
    for bonus in bonus_caduta:
        if bonus['type'] == 'x2' and x2_img:
            screen.blit(x2_img, (bonus['x'], bonus['y']))
        elif bonus['type'] == 'x4' and x4_img:
            screen.blit(x4_img, (bonus['x'], bonus['y']))
    
    # Disegna le fiamme
    if fire_img:
        for fiamma in fiamme:
            screen.blit(fire_img, (fiamma['x'] - 15, fiamma['y']))
    
    # Disegna la/le navicelle
    if astronave_img and astronave_turbo_img:
        nave_img = astronave_turbo_img if navicella_in_movimento else astronave_img
        
        if navicella_moltiplicatore == 1:
            # Una sola navicella al centro
            screen.blit(nave_img, (navicella_x - 40, navicella_y - 30))
        elif navicella_moltiplicatore == 2:
            # Due navicelle affiancate
            offset = 50
            screen.blit(nave_img, (navicella_x - offset - 40, navicella_y - 30))
            screen.blit(nave_img, (navicella_x + offset - 40, navicella_y - 30))
        elif navicella_moltiplicatore == 4:
            # Quattro navicelle a quadrato
            offset_x = 50
            offset_y = 40
            screen.blit(nave_img, (navicella_x - offset_x - 40, navicella_y - offset_y - 30))
            screen.blit(nave_img, (navicella_x + offset_x - 40, navicella_y - offset_y - 30))
            screen.blit(nave_img, (navicella_x - offset_x - 40, navicella_y + offset_y - 30))
            screen.blit(nave_img, (navicella_x + offset_x - 40, navicella_y + offset_y - 30))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
# Avvia il loop principale
window.mainloop()
 

 