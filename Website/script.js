"use strict";

/* ─── ASL Sign Dictionary ─────────────────── */
// Emoji representations for each ASL letter
const ASL_SIGNS = {
  A: "🤜",
  B: "🖐️",
  C: "🤏",
  D: "☝️",
  E: "🤞",
  F: "👌",
  G: "👆",
  H: "🤙",
  I: "🤙",
  J: "✌️",
  K: "✌️",
  L: "👆",
  M: "✊",
  N: "✊",
  O: "⭕",
  P: "👇",
  Q: "👇",
  R: "🤞",
  S: "✊",
  T: "👊",
  U: "✌️",
  V: "✌️",
  W: "🖖",
  X: "☝️",
  Y: "🤙",
  Z: "☝️",
  " ": null,
};

// More descriptive emoji map for the visual display
const SIGN_EMOJI = {
  A: "🤜",
  B: "🖐",
  C: "🫰",
  D: "☝️",
  E: "🤌",
  F: "👌",
  G: "👆",
  H: "🫵",
  I: "🤙",
  J: "✌️",
  K: "✌️",
  L: "🤙",
  M: "✊",
  N: "✊",
  O: "👌",
  P: "👇",
  Q: "👈",
  R: "🤞",
  S: "✊",
  T: "👊",
  U: "✌️",
  V: "✌️",
  W: "🖖",
  X: "🫵",
  Y: "🤙",
  Z: "✍️",
};

// Descriptions for accessibility
const SIGN_DESC = {
  A: "Closed fist, thumb to side",
  B: "Flat open hand, fingers together",
  C: "Curved hand, C-shape",
  D: "Index finger up, other fingers curve to thumb",
  E: "Fingers bent, touch thumb",
  F: "OK sign — three fingers up, index-thumb circle",
  G: "Index and thumb point horizontally",
  H: "Two fingers point horizontal",
  I: "Pinky up",
  J: "Pinky traces J shape",
  K: "Two fingers up, thumb between",
  L: "L-shape — thumb and index up",
  M: "Three fingers over thumb",
  N: "Two fingers over thumb",
  O: "All fingers curved to meet thumb",
  P: "Index points down, thumb out",
  Q: "Index and thumb point down",
  R: "Index and middle crossed",
  S: "Fist, thumb over fingers",
  T: "Thumb between index and middle",
  U: "Index and middle up together",
  V: "V-sign — index and middle apart",
  W: "Three fingers spread",
  X: "Index finger hooked",
  Y: "Thumb and pinky out",
  Z: "Index traces Z in air",
};

/* ─── State ───────────────────────────────── */
let currentLetters = []; // letters in current word
let currentWord = ""; // assembled word
let fullSentence = ""; // full sentence
let history = JSON.parse(localStorage.getItem("signbridge_history") || "[]");
let cameraStream = null;
let isRecognizing = false;
let simInterval = null;
let playbackTimer = null;
let isVoiceActive = false;
let recognition = null;

/* ─── Navigation ──────────────────────────── */
function navigate(page) {
  document
    .querySelectorAll(".page")
    .forEach((p) => p.classList.remove("active"));
  document
    .querySelectorAll(".nav-link")
    .forEach((l) => l.classList.remove("active"));

  const el = document.getElementById("page-" + page);
  if (el) el.classList.add("active");

  const nl =
    document.getElementById("nl-" + page) ||
    document.getElementById(
      "nl-" +
        (page === "asl2text" ? "a2t" : page === "text2asl" ? "t2a" : page),
    );
  if (nl) nl.classList.add("active");

  // Update drawer active
  document
    .querySelectorAll("#nav-drawer .nav-link")
    .forEach((l) => l.classList.remove("active"));

  window.scrollTo({ top: 0, behavior: "smooth" });

  // Init page-specific content
  if (page === "asl2text") initA2TPage();
  if (page === "text2asl") initT2APage();
  if (page === "about") {
  }
}

// Override nav-link IDs mapping
document.getElementById("nl-a2t") &&
  (document.getElementById("nl-a2t").id = "nl-asl2text");
document.getElementById("nl-t2a") &&
  (document.getElementById("nl-t2a").id = "nl-text2asl");

function toggleDrawer() {
  const d = document.getElementById("nav-drawer");
  const b = document.getElementById("ham-btn");
  const open = d.classList.toggle("open");
  b.setAttribute("aria-expanded", open);
}

function closeDrawer() {
  document.getElementById("nav-drawer").classList.remove("open");
  document.getElementById("ham-btn").setAttribute("aria-expanded", "false");
}

/* ─── Theme ───────────────────────────────── */
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute("data-theme") === "dark";
  html.setAttribute("data-theme", isDark ? "light" : "dark");
  localStorage.setItem("signbridge_theme", isDark ? "light" : "dark");
}

// Load saved theme
const savedTheme = localStorage.getItem("signbridge_theme");
if (savedTheme) document.documentElement.setAttribute("data-theme", savedTheme);

/* ─── Toast ───────────────────────────────── */
function showToast(msg, type = "info", duration = 3000) {
  const icons = { success: "✅", error: "❌", info: "ℹ️" };
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type]}</span><span>${msg}</span>`;
  document.getElementById("toast-container").appendChild(el);
  setTimeout(() => el.remove(), duration);
}

/* ─── Hero animation ──────────────────────── */
const heroEmojis = ["🤟", "✌️", "🤙", "👌", "🖐", "🤜", "👆"];
let heroIdx = 0;
const heroLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

// Populate hero letter grid
const heroGrid = document.getElementById("heroLetterGrid");
if (heroGrid) {
  heroLetters.slice(0, 14).forEach((l) => {
    const s = document.createElement("div");
    s.className = "hs-letter";
    s.textContent = l;
    s.setAttribute("aria-label", `Letter ${l}`);
    heroGrid.appendChild(s);
  });
}

setInterval(() => {
  heroIdx = (heroIdx + 1) % heroEmojis.length;
  const el = document.getElementById("heroEmoji");
  if (el) el.textContent = heroEmojis[heroIdx];
}, 1600);

/* ─── A2T Page init ───────────────────────── */
function initA2TPage() {
  buildSimButtons();
  renderHistory();
  updateWordDisplay();
}

function buildSimButtons() {
  const container = document.getElementById("simLetters");
  if (!container || container.children.length > 0) return;
  container.innerHTML = "";

  // A-Z
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("").forEach((l) => {
    const btn = document.createElement("button");
    btn.className = "sim-btn";
    btn.textContent = l;
    btn.setAttribute("aria-label", `Simulate letter ${l}`);
    btn.onclick = () => simulateLetter(l);
    container.appendChild(btn);
  });

  // Space
  const sp = document.createElement("button");
  sp.className = "sim-btn space-btn";
  sp.textContent = "SPACE";
  sp.setAttribute("aria-label", "Add space");
  sp.onclick = () => {
    finalizeWord();
    showToast("Space added", "success", 1500);
  };
  container.appendChild(sp);

  // Backspace
  const bs = document.createElement("button");
  bs.className = "sim-btn space-btn";
  bs.textContent = "⌫ DEL";
  bs.setAttribute("aria-label", "Delete last letter");
  bs.onclick = deleteLastLetter;
  container.appendChild(bs);

  // Clear
  const cl = document.createElement("button");
  cl.className = "sim-btn clear-btn";
  cl.textContent = "✕ CLEAR";
  cl.setAttribute("aria-label", "Clear all");
  cl.onclick = clearAll;
  container.appendChild(cl);
}
async function captureAndPredict() {
  if (!isRecognizing) return;

  const video = document.getElementById("webcam");
  if (video.videoWidth === 0) return;

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
      const res = await fetch("http://127.0.0.1:8001/predict-image", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      const pred = data.prediction;
      const conf = data.confidence || 0;

      // 🔥 SHOW overlay
      document.getElementById("predOverlay").style.display = "block";

      document.getElementById("predLetter").textContent = pred;
      document.getElementById("predConf").textContent =
        `Conf: ${(conf * 100).toFixed(0)}%`;
      document.getElementById("predFill").style.width = conf * 100 + "%";

      if (conf < 0.4) return;
      if (pred.length !== 1) return;

      handlePrediction(pred);
      console.log(video.videoWidth, video.videoHeight);
    } catch (e) {
      console.error("Prediction error:", e);
    }
  }, "image/jpeg");
}

/* ─── A2T — Camera ────────────────────────── */
async function startCamera() {
  if (isRecognizing) return; // 🔥 prevent double start

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: 640, height: 480 },
    });

    cameraStream = stream;

    const video = document.getElementById("webcam");
    video.srcObject = stream;

    video.onloadedmetadata = () => {
      video.play();

      isRecognizing = true;

      if (!simInterval) {
        simInterval = setInterval(captureAndPredict, 700);
      }
    };

    document.getElementById("camPlaceholder").style.display = "none";
    document.getElementById("camViewport").classList.add("active");
    document.getElementById("statusDot").classList.add("live");
    document.getElementById("statusText").textContent = "Live";

    document.getElementById("camStartBtn").style.display = "none";
    document.getElementById("camStopBtn").style.display = "inline-flex";

    showToast("Camera started! Real-time detection active.", "success");
    video.style.display = "block";
  } catch (e) {
    showToast("Camera permission denied or unavailable.", "error");
    console.error(e);
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }

  isRecognizing = false;

  clearInterval(simInterval);
  simInterval = null; // 🔥 IMPORTANT

  const video = document.getElementById("webcam");
  video.style.display = "none";
  video.srcObject = null;

  document.getElementById("camPlaceholder").style.display = "flex";
  document.getElementById("camViewport").classList.remove("active");
  document.getElementById("statusDot").classList.remove("live", "processing");
  document.getElementById("statusText").textContent = "Offline";

  document.getElementById("camStartBtn").style.display = "inline-flex";
  document.getElementById("camStopBtn").style.display = "none";

  document.getElementById("predOverlay").style.display = "none";

  showToast("Camera stopped.", "info");
}

// Simulate ML detections while camera is live

/* ─── A2T — Text building ─────────────────── */
function addLetterToStream(letter, conf = 90) {
  currentLetters.push(letter);
  currentWord += letter;

  const stream = document.getElementById("letterStream");
  const el = document.createElement("div");
  el.className = "stream-letter";
  el.textContent = letter;
  el.title = `${letter} — ${conf.toFixed(0)}% confidence`;
  stream.appendChild(el);
  stream.scrollLeft = stream.scrollWidth;

  updateWordDisplay();
}

function simulateLetter(letter) {
  const conf = 82 + Math.random() * 17;
  addLetterToStream(letter, conf);

  // Flash prediction overlay if camera active
  if (isRecognizing) {
    document.getElementById("predLetter").textContent = letter;
    document.getElementById("predConf").textContent =
      `Conf: ${conf.toFixed(0)}%`;
    document.getElementById("predFill").style.width = conf + "%";
  }
}

function addSpace() {
  finalizeWord();
}

function deleteLastLetter() {
  if (currentLetters.length > 0) {
    currentLetters.pop();
    currentWord = currentLetters.join("");
    const stream = document.getElementById("letterStream");
    if (stream.lastChild) stream.removeChild(stream.lastChild);
    updateWordDisplay();
  } else if (fullSentence.length > 0) {
    // Remove last word
    const words = fullSentence.trimEnd().split(" ");
    words.pop();
    fullSentence = words.join(" ");
    document.getElementById("sentenceArea").value = fullSentence;
  }
}

function finalizeWord() {
  if (!currentWord) return;
  fullSentence = (fullSentence + " " + currentWord).trim();
  document.getElementById("sentenceArea").value = fullSentence;

  // Add space in stream
  const stream = document.getElementById("letterStream");
  const sp = document.createElement("div");
  sp.className = "stream-space";
  stream.appendChild(sp);

  currentLetters = [];
  currentWord = "";
  updateWordDisplay();
}

function updateWordDisplay() {
  const el = document.getElementById("wordDisplay");
  if (!el) return;
  if (currentWord) {
    el.innerHTML =
      currentWord + '<span class="word-cursor" aria-hidden="true"></span>';
  } else {
    el.innerHTML = '<span class="word-cursor" aria-hidden="true"></span>';
  }
}

function clearAll() {
  currentLetters = [];
  currentWord = "";
  fullSentence = "";
  document.getElementById("letterStream").innerHTML = "";
  document.getElementById("sentenceArea").value = "";
  document.getElementById("predLetter").textContent = "?";
  document.getElementById("predFill").style.width = "0%";
  updateWordDisplay();
  showToast("Cleared!", "info", 1500);
}

function copyText() {
  const text = document.getElementById("sentenceArea").value || currentWord;
  if (!text) {
    showToast("Nothing to copy.", "info", 1500);
    return;
  }
  navigator.clipboard
    .writeText(text)
    .then(() => showToast("Copied to clipboard!", "success"));
}

function saveToHistory() {
  const text = document.getElementById("sentenceArea").value || currentWord;
  if (!text.trim()) {
    showToast("Nothing to save.", "info", 1500);
    return;
  }

  const entry = {
    text: text.trim(),
    time: new Date().toLocaleTimeString(),
    id: Date.now(),
  };
  history.unshift(entry);
  if (history.length > 20) history.pop();
  localStorage.setItem("signbridge_history", JSON.stringify(history));
  renderHistory();
  showToast("Saved to history!", "success");
}

function clearHistory() {
  history = [];
  localStorage.removeItem("signbridge_history");
  renderHistory();
  showToast("History cleared.", "info", 1500);
}

function deleteHistoryItem(id) {
  history = history.filter((h) => h.id !== id);
  localStorage.setItem("signbridge_history", JSON.stringify(history));
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById("historyList");
  if (!list) return;

  if (history.length === 0) {
    list.innerHTML =
      '<div class="history-empty">No saved translations yet.</div>';
    return;
  }

  list.innerHTML = history
    .map(
      (h) => `
    <div class="history-item">
      <span class="history-text">${escHtml(h.text)}</span>
      <span class="history-time">${h.time}</span>
      <button class="history-del" onclick="deleteHistoryItem(${h.id})" title="Delete" aria-label="Delete translation">×</button>
    </div>
  `,
    )
    .join("");
}

function escHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ─── T2A Page ────────────────────────────── */
function initT2APage() {
  buildRefGrid();
}

function buildRefGrid() {
  const grid = document.getElementById("aslRefGrid");
  if (!grid || grid.children.length > 0) return;

  Object.entries(SIGN_EMOJI).forEach(([letter, emoji]) => {
    const card = document.createElement("div");
    card.className = "ref-sign";
    card.setAttribute("tabindex", "0");
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `${letter}: ${SIGN_DESC[letter]}`);
    card.innerHTML = `
      <span class="ref-sign-emoji" aria-hidden="true">${emoji}</span>
      <span class="ref-sign-letter">${letter}</span>
    `;
    card.onclick = () => {
      const inp = document.getElementById("t2aInput");
      inp.value = (inp.value + letter).slice(0, 120);
      onT2AInput();
      showToast(`Added "${letter}"`, "success", 1000);
    };
    card.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") card.click();
    };
    grid.appendChild(card);
  });
}

function onT2AInput() {
  const val = document.getElementById("t2aInput").value;
  document.getElementById("charCount").textContent = val.length;
}

function t2aKeydown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    translateText();
  }
}

/* ─── T2A Video Playback ──────────────────── */
let aslPlaylist   = [];   // array of { label, src } objects
let aslPlayIndex  = 0;
let aslPlaying    = false;

// Fetch database.json once and cache it
let _dbCache = null;
async function getDatabase() {
  if (_dbCache) return _dbCache;
  const res  = await fetch("database.json");
  _dbCache   = await res.json();
  return _dbCache;
}

async function translateText() {
  const raw = document.getElementById("t2aInput").value.trim();
  if (!raw) { showToast("Please enter some text first.", "info", 2000); return; }

  const db = await getDatabase();

  // Build playlist
  const playlist = [];
  const words = raw.split(/\s+/);

  for (const word of words) {
    // Try the whole word first (capitalised first letter, rest lower)
    const key = word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    if (db[key]) {
      playlist.push({ label: word, src: db[key] });
    } else {
      // Spell it letter by letter
      for (const ch of word.toUpperCase()) {
        if (db[ch]) {
          playlist.push({ label: ch, src: db[ch] });
        } else {
          playlist.push({ label: ch, src: null }); // will be skipped
        }
      }
    }
    // Add a tiny gap token for spaces between words (no video, just a visual)
    playlist.push({ label: " ", src: null, isSpace: true });
  }

  // Remove trailing space token
  if (playlist.length && playlist[playlist.length - 1].isSpace) playlist.pop();

  aslPlaylist  = playlist;
  aslPlayIndex = 0;
  aslPlaying   = false;

  // Render queue bar
  renderQueueBar(playlist);

  // Show progress bar
  document.getElementById("playProgress").classList.add("visible");
  document.getElementById("playFill").style.width = "0%";

  // Show counter
  const signCount = playlist.filter(t => t.src).length;
  document.getElementById("tokenCounter").textContent =
    signCount + " sign" + (signCount !== 1 ? "s" : "");

  // Show replay button
  document.getElementById("replayBtn").style.display = "inline-flex";

  showToast(`Playing ${signCount} signs…`, "success", 2000);

  playFromIndex(0);
}

function renderQueueBar(playlist) {
  const bar = document.getElementById("aslQueueBar");
  bar.innerHTML = "";
  playlist.forEach((token, i) => {
    if (token.isSpace) {
      const sp = document.createElement("span");
      sp.style.cssText = "width:12px;display:inline-block;";
      sp.dataset.qIdx = i;
      bar.appendChild(sp);
      return;
    }
    const el = document.createElement("span");
    el.className = "queue-token" + (token.src ? "" : " skipped");
    el.textContent = token.label;
    el.dataset.qIdx = i;
    bar.appendChild(el);
  });
}

function updateQueueBar(idx) {
  document.querySelectorAll(".queue-token").forEach(el => {
    const i = parseInt(el.dataset.qIdx);
    el.classList.remove("active", "done");
    if (i < idx) el.classList.add("done");
    else if (i === idx) el.classList.add("active");
  });
}

function playFromIndex(idx) {
  if (idx >= aslPlaylist.length) {
    // Finished
    aslPlaying = false;
    document.getElementById("aslNowPlaying").textContent = "✅ Done";
    document.getElementById("playFill").style.width = "100%";
    updateQueueBar(aslPlaylist.length); // mark all done
    const video = document.getElementById("aslVideo");
    // Keep last frame visible; show replay hint
    return;
  }

  aslPlayIndex = idx;
  aslPlaying   = true;

  const token = aslPlaylist[idx];

  // Update progress bar
  const total = aslPlaylist.length;
  document.getElementById("playFill").style.width = ((idx / total) * 100) + "%";

  // Update queue highlight
  updateQueueBar(idx);

  // Space token — pause briefly then move on
  if (token.isSpace || !token.src) {
    document.getElementById("aslNowPlaying").textContent =
      token.isSpace ? "[ space ]" : `⚠️ No video for "${token.label}"`;
    setTimeout(() => playFromIndex(idx + 1), token.isSpace ? 300 : 200);
    return;
  }

  // Play the video
  const video = document.getElementById("aslVideo");
  const empty = document.getElementById("aslEmpty");

  empty.style.display  = "none";
  video.style.display  = "block";

  document.getElementById("aslNowPlaying").textContent = `▶ ${token.label}`;

  video.src = token.src;
  video.load();
  video.play().catch(() => {});

  video.onended = () => playFromIndex(idx + 1);

  // Fallback: if video fails to load, skip after a timeout
  video.onerror = () => {
    console.warn("Video missing:", token.src);
    playFromIndex(idx + 1);
  };
}

function replayASL() {
  if (!aslPlaylist.length) return;
  playFromIndex(0);
}

function clearASLDisplay() {
  aslPlaylist  = [];
  aslPlayIndex = 0;
  aslPlaying   = false;

  const video = document.getElementById("aslVideo");
  video.pause();
  video.src        = "";
  video.style.display = "none";

  document.getElementById("aslEmpty").style.display  = "flex";
  document.getElementById("aslQueueBar").innerHTML   = "";
  document.getElementById("playProgress").classList.remove("visible");
  document.getElementById("playFill").style.width    = "0%";
  document.getElementById("aslNowPlaying").textContent = "—";
  document.getElementById("tokenCounter").textContent  = "";
  document.getElementById("replayBtn").style.display   = "none";

  document.getElementById("t2aInput").value = "";
  document.getElementById("charCount").textContent = "0";
}

function copyT2AText() {
  const text = document.getElementById("t2aInput").value;
  if (!text) { showToast("Nothing to copy.", "info", 1500); return; }
  navigator.clipboard.writeText(text).then(() => showToast("Copied!", "success"));
}

/* ─── Voice Input ─────────────────────────── */
function toggleVoice() {
  if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
    showToast("Voice input not supported in this browser.", "error");
    return;
  }

  const btn = document.getElementById("voiceBtn");

  if (isVoiceActive) {
    recognition && recognition.stop();
    isVoiceActive = false;
    btn.classList.remove("recording");
    btn.textContent = "🎙️ Voice";
    return;
  }

  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = "en-US";

  recognition.onstart = () => {
    isVoiceActive = true;
    btn.classList.add("recording");
    btn.textContent = "⏹ Stop";
    showToast("Listening…", "info", 2000);
  };

  recognition.onresult = (e) => {
    const transcript = e.results[0][0].transcript.toUpperCase();
    const inp = document.getElementById("t2aInput");
    inp.value = (inp.value + " " + transcript).trim().slice(0, 120);
    onT2AInput();
    showToast(`Heard: "${transcript}"`, "success");
  };

  recognition.onerror = () => {
    showToast("Voice recognition error.", "error");
  };

  recognition.onend = () => {
    isVoiceActive = false;
    btn.classList.remove("recording");
    btn.textContent = "🎙️ Voice";
  };

  recognition.start();
}

/* ─── Keyboard shortcuts ──────────────────── */
document.addEventListener("keydown", (e) => {
  const activePage = document.querySelector(".page.active")?.id;

  if (activePage === "page-asl2text") {
    if (
      e.key === " " &&
      e.target.tagName !== "INPUT" &&
      e.target.tagName !== "TEXTAREA"
    ) {
      e.preventDefault();
      finalizeWord();
    }
    if (
      e.key === "Enter" &&
      e.target.tagName !== "INPUT" &&
      e.target.tagName !== "TEXTAREA"
    ) {
      e.preventDefault();
      saveToHistory();
    }
    if (
      e.key === "Backspace" &&
      e.target.tagName !== "INPUT" &&
      e.target.tagName !== "TEXTAREA"
    ) {
      e.preventDefault();
      deleteLastLetter();
    }
    if (e.key === "c" && e.ctrlKey) {
    } // allow native
    else if (
      /^[a-zA-Z]$/.test(e.key) &&
      e.target.tagName !== "INPUT" &&
      e.target.tagName !== "TEXTAREA"
    ) {
      simulateLetter(e.key.toUpperCase());
    }
  }
});

/* ─── Intersection observer for scroll animations ── */
const io = new IntersectionObserver(
  (entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.style.opacity = "1";
        e.target.style.transform = "translateY(0)";
      }
    });
  },
  { threshold: 0.1 },
);

document
  .querySelectorAll(".feature-card, .hiw-step, .fact-card")
  .forEach((el) => {
    el.style.opacity = "0";
    el.style.transform = "translateY(20px)";
    el.style.transition = "opacity 0.5s ease, transform 0.5s ease";
    io.observe(el);
  });

/* ─── Init ────────────────────────────────── */
// Fix nav link IDs
document.querySelectorAll(".nav-link").forEach((l) => {
  const fn = l.getAttribute("onclick");
  if (fn && fn.includes("'asl2text'")) l.id = "nl-asl2text";
  if (fn && fn.includes("'text2asl'")) l.id = "nl-text2asl";
  if (fn && fn.includes("'about'")) l.id = "nl-about";
  if (fn && fn.includes("'home'")) l.id = "nl-home";
});

// Load history
renderHistory();
buildRefGrid();
//added by taqi

let historyBuffer = [];
const MAX_HISTORY = 5;

let lastStable = "";
let holdStart = null;
const HOLD_TIME = 1200; // ms

function handlePrediction(pred) {
  historyBuffer.push(pred);
  if (historyBuffer.length > MAX_HISTORY) historyBuffer.shift();

  const stable = mostFrequent(historyBuffer);

  if (stable === lastStable) {
    if (!holdStart) holdStart = Date.now();

    if (Date.now() - holdStart > HOLD_TIME) {
      addLetterToStream(stable, 90);

      lastStable = stable + "_done";
      holdStart = null;
      historyBuffer = [];
    }
  } else {
    lastStable = stable;
    holdStart = Date.now();
  }
}

function mostFrequent(arr) {
  return arr
    .sort(
      (a, b) =>
        arr.filter((v) => v === a).length - arr.filter((v) => v === b).length,
    )
    .pop();
}
