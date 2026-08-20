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


def waveform_player(
    audio,
    timestamps,
    window_seconds=20,
    height=160,
):
    """
    Parameters
    ----------
    audio:
        pydub.AudioSegment or path to file

    timestamps:
        [1.2, 3.4, 8.1]

        or

        [
            (1.2, "marker A"),
            (3.4, "marker B"),
        ]

    window_seconds:
        Total visible time.
        20 means roughly +/- 10 seconds around playback.
    """

    audio_url = _audio_to_data_url(audio)

    markers = []

    for item in timestamps:
        if isinstance(item, (tuple, list)):
            t, label = item
        else:
            t = item
            label = f"{float(t):.2f}s"

        markers.append(
            {
                "time": float(t),
                "label": str(label),
            }
        )

    markers_json = json.dumps(markers)

    uid = uuid.uuid4().hex

    waveform_id = f"waveform_{uid}"
    button_id = f"button_{uid}"
    time_id = f"time_{uid}"

    html = f"""
    <style>
        #{waveform_id} {{
            width: 100%;
        }}

        .controls-{uid} {{
            margin-top: 10px;
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: sans-serif;
        }}

        #{time_id} {{
            font-family: monospace;
            font-size: 13px;
        }}
    </style>

    <div id="{waveform_id}"></div>

    <div class="controls-{uid}">
        <button id="{button_id}">
            ▶ Play
        </button>

        <span id="{time_id}">
            0.00 s
        </span>
    </div>

    <script type="module">

        import WaveSurfer from
        "https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js";

        import RegionsPlugin from
        "https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/regions.esm.js";


        const markers = {markers_json};

        const waveformContainer =
            document.getElementById("{waveform_id}");

        const button =
            document.getElementById("{button_id}");

        const timeDisplay =
            document.getElementById("{time_id}");


        const regions = RegionsPlugin.create();


        const wavesurfer = WaveSurfer.create({{
            container: waveformContainer,

            url: "{audio_url}",

            height: {height},

            waveColor: "#aaa",
            progressColor: "#444",

            cursorColor: "#2563eb",
            cursorWidth: 2,

            normalize: true,

            autoScroll: true,
            autoCenter: true,

            minPxPerSec: 50,

            plugins: [regions],
        }});


        let duration = 0;


        function updateZoom() {{
            const width =
                waveformContainer.clientWidth;

            if (!width) return;

            const pxPerSecond =
                width / {float(window_seconds)};

            wavesurfer.zoom(pxPerSecond);
        }}


        function addMarkers() {{

            regions.clearRegions();

            markers.forEach(marker => {{

                if (
                    marker.time < 0 ||
                    marker.time > duration
                ) {{
                    return;
                }}

                /*
                 * Very narrow region.
                 *
                 * Width is 0.01 sec, which appears
                 * as a vertical red marker when zoomed.
                 */
                const region = regions.addRegion({{
                    start: marker.time,
                    end: Math.min(
                        marker.time + 0.01,
                        duration
                    ),

                    color: "rgba(239, 68, 68, 0.95)",

                    drag: false,
                    resize: false,
                }});


                /*
                 * Tooltip
                 */
                region.element.title =
                    `${{marker.label}} (${{marker.time.toFixed(2)}}s)`;


                /*
                 * Make narrow regions easier to see.
                 */
                region.element.style.minWidth = "2px";
                region.element.style.width = "2px";
                region.element.style.background =
                    "rgb(239, 68, 68)";
                region.element.style.zIndex = "10";
                region.element.style.cursor = "pointer";


                /*
                 * Click marker to seek.
                 */
                region.element.addEventListener(
                    "click",
                    event => {{

                        event.stopPropagation();

                        wavesurfer.setTime(
                            marker.time
                        );
                    }}
                );
            }});
        }}


        wavesurfer.on("ready", () => {{

            duration =
                wavesurfer.getDuration();

            updateZoom();

            requestAnimationFrame(() => {{
                addMarkers();
            }});

            timeDisplay.textContent =
                `0.00 s / ${{duration.toFixed(2)}} s`;
        }});


        wavesurfer.on("timeupdate", time => {{

            timeDisplay.textContent =
                `${{time.toFixed(2)}} s / ` +
                `${{duration.toFixed(2)}} s`;
        }});


        wavesurfer.on("play", () => {{
            button.textContent = "⏸ Pause";
        }});


        wavesurfer.on("pause", () => {{
            button.textContent = "▶ Play";
        }});


        wavesurfer.on("finish", () => {{
            button.textContent = "▶ Play";
        }});


        button.addEventListener("click", () => {{
            wavesurfer.playPause();
        }});


        const resizeObserver =
            new ResizeObserver(() => {{

                if (!duration) return;

                const currentTime =
                    wavesurfer.getCurrentTime();

                updateZoom();

                requestAnimationFrame(() => {{
                    addMarkers();

                    wavesurfer.setTime(
                        currentTime
                    );
                }});
            }});


        resizeObserver.observe(
            waveformContainer
        );

    </script>
    """

    display(HTML(html))
