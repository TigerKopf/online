import scipy
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import time
from tqdm.auto import tqdm
import numpy as np

def generate_lofi_music_long(
    total_duration_seconds: int = 225,  # 3 Minuten 45 Sekunden
    segment_duration_seconds: int = 28, # Max. 30 Sekunden, hier etwas Puffer
    prompt: str = "lo-fi music with a soothing melody, chill vibes, relaxed atmosphere, calm, gentle",
    output_filename: str = "lofi_music_3m45s.wav"
):
    """
    Generiert einen längeren Lo-Fi-Musiktrack, indem er ihn in Segmente aufteilt und zusammenfügt,
    mit Fortschrittsbalken und geschätzter Restzeit.

    Args:
        total_duration_seconds (int): Die gewünschte Gesamtlänge des Musiktracks in Sekunden.
        segment_duration_seconds (int): Die Dauer jedes generierten Segments in Sekunden (max. ~29-30s).
        prompt (str): Die Textbeschreibung für die zu generierende Musik.
        output_filename (str): Der Dateiname für die Ausgabe-WAV-Datei.
    """
    if segment_duration_seconds > 29: # Sicherstellen, dass wir unter 30 Sekunden bleiben
        print("Warnung: segment_duration_seconds sollte 29 Sekunden nicht überschreiten, um 'IndexError' zu vermeiden. Setze auf 28 Sekunden.")
        segment_duration_seconds = 28

    # Definiere die Hauptphasen des Prozesses
    phases = [
        "Modell laden",
        "Gerät initialisieren",
        "Prompt verarbeiten",
        "Audio-Segmente generieren", # Dieser Schritt wird nun in Schleifen ausgeführt
        "Audio-Segmente zusammenfügen und speichern"
    ]
    total_phases = len(phases)

    # Initialisiere den Gesamtfortschrittsbalken
    phase_bar = tqdm(total=total_phases, desc="Gesamtfortschritt", unit="Schritt")

    # --- Phase 1: MusicGen Modell und Prozessor laden ---
    phase_bar.set_description(f"Phase 1/{total_phases}: {phases[0]}")
    print(f"\nStarte: {phases[0]}...")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    phase_bar.update(1) # Fortschritt aktualisieren

    # --- Phase 2: Prüfe GPU und verschiebe Modell ---
    phase_bar.set_description(f"Phase 2/{total_phases}: {phases[1]}")
    print(f"Starte: {phases[1]}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("    GPU gefunden. Verschiebe Modell auf GPU...")
        model = model.to(device)
    else:
        print("    Keine GPU gefunden. Generiere auf CPU (kann deutlich länger dauern).")
    phase_bar.update(1)

    print(f"\nMusikgenerierung gestartet für:")
    print(f"  Prompt: '{prompt}'")
    print(f"  Gewünschte Gesamtdauer: {total_duration_seconds // 60}:{(total_duration_seconds % 60):02d} Minuten")
    print(f"  Segmentlänge: {segment_duration_seconds} Sekunden pro Generierung")

    # MusicGen generiert 50 auto-regressive Schritte pro Sekunde Audio.
    tokens_per_second = 50
    max_new_tokens_per_segment = int(segment_duration_seconds * tokens_per_second)

    # --- Phase 3: Text-Prompt verarbeiten ---
    phase_bar.set_description(f"Phase 3/{total_phases}: {phases[2]}")
    print(f"Starte: {phases[2]}...")
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    if device == "cuda":
        inputs = {key: value.to(device) for key, value in inputs.items()}
    phase_bar.update(1)

    all_audio_segments = []
    num_segments = int(np.ceil(total_duration_seconds / segment_duration_seconds))
    sampling_rate = model.config.audio_encoder.sampling_rate

    # --- Phase 4: Audio-Segmente generieren ---
    phase_bar.set_description(f"Phase 4/{total_phases}: {phases[3]}")
    print(f"\nStarte: {phases[3]} (Generiere {num_segments} Segmente)...")
    
    # Fortschrittsbalken für die Segmentgenerierung
    segment_progress_bar = tqdm(total=num_segments, desc="Segmente generieren", unit="Segment", leave=False)

    for i in range(num_segments):
        segment_progress_bar.set_description(f"Segmente generieren ({i+1}/{num_segments})")
        start_time_segment = time.time()
        
        # Generiere das Audio-Segment.
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens_per_segment,
            do_sample=True,
            guidance_scale=3.0
        )
        
        if device == "cuda":
            audio_values = audio_values.cpu()
        
        all_audio_segments.append(audio_values[0, 0].numpy())
        
        end_time_segment = time.time()
        segment_duration = end_time_segment - start_time_segment
        # print(f"    Segment {i+1}/{num_segments} generiert in {segment_duration:.2f} Sekunden.") # Kann optional ausgegeben werden
        segment_progress_bar.update(1)
    
    segment_progress_bar.close()
    phase_bar.update(1) # Gesamtfortschritt aktualisieren

    # --- Phase 5: Audio-Segmente zusammenfügen und speichern ---
    phase_bar.set_description(f"Phase 5/{total_phases}: {phases[4]}")
    print(f"Starte: {phases[4]}...")

    # Füge alle numpy-Arrays zusammen
    final_audio = np.concatenate(all_audio_segments)

    # Schneide auf die exakte Gesamtlänge, falls nötig (letztes Segment könnte länger sein)
    desired_samples = int(total_duration_seconds * sampling_rate)
    if len(final_audio) > desired_samples:
        final_audio = final_audio[:desired_samples]
    elif len(final_audio) < desired_samples:
        # Dies sollte nicht passieren, wenn die Berechnungen stimmen, aber zur Sicherheit
        print(f"Warnung: Generiertes Audio ist kürzer als gewünscht. Erwartet: {desired_samples} Samples, Tatsächlich: {len(final_audio)} Samples.")


    # Speichere die generierte Musik als WAV-Datei
    print(f"Speichere die zusammengefügte Musik als '{output_filename}' mit einer Abtastrate von {sampling_rate} Hz.")
    scipy.io.wavfile.write(output_filename, rate=sampling_rate, data=final_audio)
    phase_bar.update(1) # Fortschritt aktualisieren

    # Schließe den Gesamtfortschrittsbalken ab
    phase_bar.close()
    print(f"\nGenerierung abgeschlossen! Die Musik wurde erfolgreich als '{output_filename}' gespeichert.")

if __name__ == "__main__":
    # Die gewünschte Länge ist 3 Minuten und 45 Sekunden = 225 Sekunden
    target_duration_seconds = 3 * 60 + 45
    generate_lofi_music_long(total_duration_seconds=target_duration_seconds)