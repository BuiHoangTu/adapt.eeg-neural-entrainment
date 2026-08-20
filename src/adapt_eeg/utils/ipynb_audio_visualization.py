import base64
import io
import json
import mimetypes
import uuid
from pathlib import Path

from IPython.display import HTML, display
from pydub import AudioSegment


def _audio_to_data_url(audio):
    if isinstance(audio, AudioSegment):
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        audio_bytes = buf.getvalue()
        mime_type = "audio/wav"

    elif isinstance(audio, (str, Path)):
        path = Path(audio)

        if not path.exists():
            raise FileNotFoundError(path)

        audio_bytes = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or "audio/mpeg"

    else:
        raise TypeError("audio must be a pydub.AudioSegment or a file path")

    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def waveform_player(audio, timestamps, height=160):
    """
    Parameters
    ----------
    audio:
        pydub.AudioSegment or path to an audio file.

    timestamps:
        Iterable of timestamps in seconds.

        Examples:
            [0.08, 0.37, 0.73, 1.33]

        or:
            [(0.08, "start"), (2.27, "word"), (4.53, "event")]

    height:
        Waveform height in pixels.
    """

    audio_url = _audio_to_data_url(audio)

    # Normalize timestamps
    markers = []

    for item in timestamps:
        if isinstance(item, (tuple, list)):
            time, label = item
        else:
            time = item
            label = f"{float(time):.2f}s"

        markers.append(
            {
                "time": float(time),
                "label": str(label),
            }
        )

    uid = uuid.uuid4().hex

    waveform_id = f"waveform_{uid}"
    wrapper_id = f"wrapper_{uid}"
    markers_id = f"markers_{uid}"
    button_id = f"button_{uid}"
    time_id = f"time_{uid}"

    markers_json = json.dumps(markers)

    html = f"""
    <style>
        #{wrapper_id} {{
            max-width: 1000px;
            font-family: sans-serif;
        }}

        #{waveform_id} {{
            width: 100%;
        }}

        .wave-area-{uid} {{
            position: relative;
            width: 100%;
        }}

        #{markers_id} {{
            position: absolute;
            inset: 0;
            pointer-events: none;
            z-index: 10;
        }}

        #{markers_id} .marker {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 1px;
            background: #e11d48;
            transform: translateX(-0.5px);
            pointer-events: auto;
            cursor: pointer;
        }}

        #{markers_id} .marker:hover {{
            width: 2px;
        }}

        #{markers_id} .tooltip {{
            position: absolute;
            top: 4px;
            left: 5px;

            display: none;

            white-space: nowrap;

            padding: 2px 5px;
            border-radius: 4px;

            background: rgba(0, 0, 0, 0.80);
            color: white;

            font-size: 11px;
            line-height: 16px;

            z-index: 20;
        }}

        #{markers_id} .marker:hover .tooltip {{
            display: block;
        }}

        .controls-{uid} {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 10px;
        }}

        #{button_id} {{
            padding: 5px 10px;
            cursor: pointer;
        }}

        #{time_id} {{
            font-family: monospace;
            font-size: 13px;
        }}
    </style>

    <div id="{wrapper_id}">

        <div class="wave-area-{uid}">
            <div id="{waveform_id}"></div>
            <div id="{markers_id}"></div>
        </div>

        <div class="controls-{uid}">
            <button id="{button_id}">
                ▶ Play
            </button>

            <span id="{time_id}">
                0.00 s
            </span>
        </div>

    </div>

    <script type="module">

        import WaveSurfer from
        "https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js";

        const markers = {markers_json};

        const wavesurfer = WaveSurfer.create({{
            container: "#{waveform_id}",

            url: "{audio_url}",

            height: {height},

            waveColor: "#444",
            progressColor: "#999",

            cursorColor: "#2563eb",
            cursorWidth: 2,

            normalize: true,
            barWidth: 1,
            barGap: 0,
        }});

        const markerLayer =
            document.getElementById("{markers_id}");

        const timeDisplay =
            document.getElementById("{time_id}");

        const button =
            document.getElementById("{button_id}");


        wavesurfer.on("ready", () => {{

            const duration = wavesurfer.getDuration();

            markerLayer.innerHTML = "";

            markers.forEach(marker => {{

                if (
                    marker.time < 0 ||
                    marker.time > duration
                ) {{
                    return;
                }}

                const el = document.createElement("div");

                el.className = "marker";

                el.style.left =
                    `${{marker.time / duration * 100}}%`;

                const tooltip =
                    document.createElement("span");

                tooltip.className = "tooltip";
                tooltip.textContent = marker.label;

                el.appendChild(tooltip);

                el.title =
                    `${{marker.label}} (${{marker.time.toFixed(2)}}s)`;

                el.addEventListener("click", event => {{
                    event.stopPropagation();

                    wavesurfer.setTime(marker.time);
                }});

                markerLayer.appendChild(el);
            }});
        }});


        button.addEventListener("click", () => {{
            wavesurfer.playPause();
        }});


        wavesurfer.on("play", () => {{
            button.textContent = "⏸ Pause";
        }});


        wavesurfer.on("pause", () => {{
            button.textContent = "▶ Play";
        }});


        wavesurfer.on("timeupdate", time => {{
            timeDisplay.textContent =
                `${{time.toFixed(2)}} s / ` +
                `${{wavesurfer.getDuration().toFixed(2)}} s`;
        }});

    </script>
    """

    display(HTML(html))
