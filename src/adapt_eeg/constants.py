from adapt_eeg import DATA_ROOT

REGULAR_LINES = {3, 4, 10, 15, 23, 27, 30}
IRREGULAR_LINES = {1, 2, 5, 6, 7, 8, 9}
ENGLISH_COMPETENCE_PARTICIPANTS = {
    5, 10, 12, 13, 17, 19, 21, 24, 26, 27, 31, 32, 34, 37, 39, 40,
    41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 56, 58,
}
FREQUENCIES = {"POEM": 2.0, "SPEAKING": 4.0}
CHANNELS_OF_INTEREST = ("Pz", "C4", "T7")
N_CHANNELS = 18
MIN_DURATION = 1.5  # 3 cycles
MIN_SHIFT = 0.5  # 1 cycle

POEMS_CONFIG = [
    {
        "audio_url": DATA_ROOT / "raw-audio/1. Sailing_to_byzantium_simona_final.wav",
        "textgrid_url": DATA_ROOT / "text-grid/01_Sailing_to_byzantium_simona_final1_Textgrid.TextGrid",
        "intro_end": 2500,
    }
]
