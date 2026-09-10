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
        "iambic_pentameter": True,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/2. The_tower_final.wav",
        "textgrid_url": DATA_ROOT / "text-grid/02.The Tower_FinalThetowerfinal-",
        "intro_end": 1700,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/3. Meditation I_VII.wav",
        "textgrid_url": DATA_ROOT
        / "text-grid/03_Meditation_I_VII_Final_textGrid.TextGrid",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/4. Nineteen Nineteen I_VI.wav",
        "textgrid_url": DATA_ROOT / "text-grid/04.NineteenNineteenfinaltextgrid1",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/5. The Wheel.wav",
        "textgrid_url": DATA_ROOT / "text-grid/05.The_Wheel_Final_tEXT_GRID.TextGrid",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/6. Youth_and_Age.wav",
        "textgrid_url": DATA_ROOT / "text-grid/06_Youth_and_Age_final textgrid",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/7. The new faces.wav",
        "textgrid_url": DATA_ROOT / "text-grid/07_The_new_faces-Textgrid Final",
        "intro_end": 3700,
        "iambic_pentameter": True,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/8. A prayer for my son.wav",
        "textgrid_url": DATA_ROOT / "text-grid/08_A prayer for my son_.final textgrid",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/9. Two songs from a play.wav",
        "textgrid_url": DATA_ROOT / "text-grid/09. Two songs from a playFinal textgrid",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/10. Wisdom.wav",
        "textgrid_url": DATA_ROOT / "text-grid/10. WisdomTextgrid finAL",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/11. Leda and the swan.wav",
        "textgrid_url": DATA_ROOT / "text-grid/11.Leda and the swanFinal_textgrid",
        "intro_end": 3200,
        "iambic_pentameter": True,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/12. A black Centaur.wav",
        "textgrid_url": DATA_ROOT / "text-grid/12. A black centaurtextgrid final",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/13. Among School Children.wav",
        "textgrid_url": DATA_ROOT / "text-grid/13. among school childrenFinaltextgrid",
        "intro_end": 1800,
        "iambic_pentameter": True,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/14. Colonus¨Praise.wav",
        "textgrid_url": DATA_ROOT / "text-grid/14.final_textgridColonusPraise",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/15. Hero, Girl. and Fool.wav",
        "textgrid_url": DATA_ROOT / "text-grid/15_Hero_Girl_and_Fool_Final_textgrid",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/16. Owen Ahern.wav",
        "textgrid_url": DATA_ROOT / "text-grid/16.OwenAhern-merge_textgrid final",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/17. A man young and old.wav",
        "textgrid_url": DATA_ROOT / "text-grid/17.  A man young and old_finaltextgrid",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/18. The three monuments.wav",
        "textgrid_url": DATA_ROOT
        / "text-grid/18.Final+Textgrid_The_three_monuments-merge",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/19. From Oedipus at Colonus.wav",
        "textgrid_url": DATA_ROOT
        / "text-grid/19.Finaltextgrid_From_Oedipus_at_Colonus",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/20. Gift of Harun Al rashid.wav",
        "textgrid_url": DATA_ROOT
        / "text-grid/20. the gift of harun al rashid_Final Textgrid",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
    {
        "audio_url": DATA_ROOT / "raw-audio/21. All Souls Night final.wav",
        "textgrid_url": DATA_ROOT / "text-grid/21. _All Souls_Night_Textgrid Final",
        "intro_end": 0,
        "iambic_pentameter": False,
    },
]
