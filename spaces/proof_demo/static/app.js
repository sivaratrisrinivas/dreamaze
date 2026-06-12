const solveButton = document.querySelector("#solve-button");
const result = document.querySelector("#result");
const statusLine = document.querySelector("#status");

const CELL_SIZE = 15;
const THUMB_CELL_SIZE = 3;
const DEFAULT_FRAME_MS = 360;
const COLORS = {
  start: "#2dd4bf",
  goal: "#f59e0b",
  wall: "#111827",
  open: "#f8fafc",
  solution: "#38bdf8",
  solutionSoft: "#0ea5e9",
};

let activeSource = null;
let frames = [];
let playbackTimer = null;
let isPlaying = false;
let userPaused = false;
let currentMaze = null;
let currentStart = null;
let currentGoal = null;
let selectedFrameIndex = -1;
let pendingDoneEvent = null;
let streamMeta = null;

function setLoading(isLoading) {
  solveButton.disabled = isLoading;
  solveButton.classList.toggle("is-loading", isLoading);
  solveButton.querySelector(".button-label").textContent = isLoading
    ? "Solving..."
    : "Solve New Maze";
}

function setStatus(message, isError = false) {
  statusLine.classList.toggle("is-error", isError);
  statusLine.textContent = message;
}

function showError(message) {
  stopStream();
  setStatus(message, true);
  setLoading(false);
}

function stopStream() {
  if (activeSource) {
    activeSource.close();
    activeSource = null;
  }
  if (playbackTimer) {
    clearTimeout(playbackTimer);
    playbackTimer = null;
  }
  isPlaying = false;
  pendingDoneEvent = null;
}

function cellId(row, column) {
  return `maze-cell-${row}-${column}`;
}

function isSameCell(cell, row, column) {
  return cell && cell[0] === row && cell[1] === column;
}

function colorForCell(row, column, mask) {
  if (isSameCell(currentStart, row, column)) return COLORS.start;
  if (isSameCell(currentGoal, row, column)) return COLORS.goal;
  if (mask && mask[row] && mask[row][column]) return COLORS.solution;
  if (currentMaze && currentMaze[row] && currentMaze[row][column]) {
    return COLORS.open;
  }
  return COLORS.wall;
}

function thumbnailColorForCell(row, column, mask) {
  if (isSameCell(currentStart, row, column)) return COLORS.start;
  if (isSameCell(currentGoal, row, column)) return COLORS.goal;
  if (mask && mask[row] && mask[row][column]) return COLORS.solutionSoft;
  if (currentMaze && currentMaze[row] && currentMaze[row][column]) return "#dbeafe";
  return "#020617";
}

function renderMazeShell(event) {
  currentMaze = event.renderedMaze;
  currentStart = event.startCell;
  currentGoal = event.goalCell;
  frames = [];
  selectedFrameIndex = -1;
  pendingDoneEvent = null;
  streamMeta = event;
  userPaused = false;
  isPlaying = true;

  const rows = currentMaze.length;
  const columns = rows > 0 ? currentMaze[0].length : 0;
  const width = columns * CELL_SIZE;
  const height = rows * CELL_SIZE;
  const rects = [];

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      rects.push(
        `<rect id="${cellId(row, column)}" x="${column * CELL_SIZE}" y="${
          row * CELL_SIZE
        }" width="${CELL_SIZE}" height="${CELL_SIZE}" fill="${colorForCell(
          row,
          column,
          null,
        )}" />`,
      );
    }
  }

  const family = event.mazeFamily || "maze";
  const totalFrames = event.totalFrames || event.totalSteps + 1;
  result.innerHTML = `
    <div class="lab-grid">
      <section class="maze-stage" aria-label="Rendered Maze">
        <div class="stage-topline">
          <span>${family}</span>
          <span>seed ${event.mazeSeed}</span>
        </div>
        <svg
          class="maze-svg"
          viewBox="0 0 ${width} ${height}"
          width="${width}"
          height="${height}"
          xmlns="http://www.w3.org/2000/svg"
          role="img"
          aria-label="Maze with live Conditional Diffusion Solver trajectory"
        >${rects.join("")}</svg>
      </section>

      <aside class="telemetry" aria-label="Solver telemetry">
        <div class="metric">
          <span>Step</span>
          <strong><span id="stream-step">0</span> / ${event.totalSteps}</strong>
        </div>
        <div class="metric">
          <span>Frames</span>
          <strong><span id="frame-count">0</span> / ${totalFrames}</strong>
        </div>
        <div class="metric">
          <span>Score</span>
          <strong>Single-Sample</strong>
        </div>
        <div id="stream-verdict" class="stream-verdict" hidden></div>
      </aside>
    </div>

    <section class="timeline-panel" aria-label="Diffusion Trajectory Timeline">
      <div class="transport">
        <button id="play-toggle" class="transport-button" type="button">Pause</button>
        <button id="prev-frame" class="transport-button" type="button">Prev</button>
        <button id="next-frame" class="transport-button" type="button">Next</button>
        <label class="speed-control">
          <span>Frame</span>
          <select id="frame-speed">
            <option value="180">180ms</option>
            <option value="360" selected>360ms</option>
            <option value="750">750ms</option>
            <option value="1200">1200ms</option>
          </select>
        </label>
      </div>
      <div id="timeline-strip" class="timeline-strip"></div>
    </section>
  `;

  document.querySelector("#play-toggle").addEventListener("click", togglePlayback);
  document.querySelector("#prev-frame").addEventListener("click", showPreviousFrame);
  document.querySelector("#next-frame").addEventListener("click", showNextFrame);
}

function applyFrame(event, index) {
  const mask = event.mask;
  for (let row = 0; row < currentMaze.length; row += 1) {
    for (let column = 0; column < currentMaze[row].length; column += 1) {
      const cell = document.getElementById(cellId(row, column));
      if (cell) {
        cell.setAttribute("fill", colorForCell(row, column, mask));
      }
    }
  }

  selectedFrameIndex = index;
  const step = document.querySelector("#stream-step");
  if (step) step.textContent = String(event.step);

  document.querySelectorAll(".timeline-frame").forEach((button) => {
    button.classList.toggle("is-active", Number(button.dataset.index) === index);
  });
}

function renderFrameThumbnail(event, index) {
  const rows = currentMaze.length;
  const columns = rows > 0 ? currentMaze[0].length : 0;
  const width = columns * THUMB_CELL_SIZE;
  const height = rows * THUMB_CELL_SIZE;
  const rects = [];

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      rects.push(
        `<rect x="${column * THUMB_CELL_SIZE}" y="${row * THUMB_CELL_SIZE}" width="${THUMB_CELL_SIZE}" height="${THUMB_CELL_SIZE}" fill="${thumbnailColorForCell(
          row,
          column,
          event.mask,
        )}" />`,
      );
    }
  }

  const button = document.createElement("button");
  button.type = "button";
  button.className = "timeline-frame";
  button.dataset.index = String(index);
  button.setAttribute("aria-label", `Diffusion step ${event.step}`);
  button.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" aria-hidden="true">${rects.join("")}</svg>
    <span>${event.step}</span>
  `;
  button.addEventListener("click", () => {
    pausePlayback();
    applyFrame(frames[index], index);
  });
  document.querySelector("#timeline-strip").appendChild(button);
}

function addFrame(event) {
  const index = frames.length;
  frames.push(event);
  renderFrameThumbnail(event, index);

  const frameCount = document.querySelector("#frame-count");
  if (frameCount) frameCount.textContent = String(frames.length);

  if (index === 0 || (isPlaying && !userPaused && selectedFrameIndex === index - 1)) {
    schedulePlayback(index);
  }
}

function frameDelay() {
  const speed = document.querySelector("#frame-speed");
  return speed ? Number(speed.value) : DEFAULT_FRAME_MS;
}

function schedulePlayback(index) {
  if (playbackTimer) clearTimeout(playbackTimer);
  playbackTimer = setTimeout(() => {
    if (!isPlaying || userPaused || !frames[index]) return;
    applyFrame(frames[index], index);
    const nextIndex = index + 1;
    if (frames[nextIndex]) {
      schedulePlayback(nextIndex);
    } else if (pendingDoneEvent) {
      finishSolve(pendingDoneEvent);
      pendingDoneEvent = null;
    }
  }, index === 0 ? 0 : frameDelay());
}

function pausePlayback() {
  userPaused = true;
  isPlaying = false;
  if (playbackTimer) {
    clearTimeout(playbackTimer);
    playbackTimer = null;
  }
  const toggle = document.querySelector("#play-toggle");
  if (toggle) toggle.textContent = "Play";
}

function resumePlayback() {
  if (!frames.length) return;
  userPaused = false;
  isPlaying = true;
  const toggle = document.querySelector("#play-toggle");
  if (toggle) toggle.textContent = "Pause";
  const nextIndex = Math.min(selectedFrameIndex + 1, frames.length - 1);
  schedulePlayback(nextIndex);
}

function togglePlayback() {
  if (isPlaying && !userPaused) {
    pausePlayback();
  } else {
    resumePlayback();
  }
}

function showPreviousFrame() {
  if (!frames.length) return;
  pausePlayback();
  applyFrame(frames[Math.max(0, selectedFrameIndex - 1)], Math.max(0, selectedFrameIndex - 1));
}

function showNextFrame() {
  if (!frames.length) return;
  pausePlayback();
  const index = Math.min(frames.length - 1, selectedFrameIndex + 1);
  applyFrame(frames[index], index);
}

function finishSolve(event) {
  if (frames.length && selectedFrameIndex < frames.length - 1 && !userPaused) {
    applyFrame(frames[frames.length - 1], frames.length - 1);
  }

  const verdict = document.querySelector("#stream-verdict");
  if (verdict) {
    verdict.hidden = false;
    verdict.className = event.validationStatus.includes("Valid")
      ? "stream-verdict valid"
      : "stream-verdict invalid";
    verdict.textContent = event.validationStatus;
  }

  setStatus(event.validationReason || "Graph Validation complete");
  setLoading(false);
  isPlaying = false;
  const toggle = document.querySelector("#play-toggle");
  if (toggle) toggle.textContent = "Play";
  if (activeSource) {
    activeSource.close();
    activeSource = null;
  }
}

function handleStreamEvent(event) {
  if (event.type === "init") {
    renderMazeShell(event);
    setStatus("Runtime Solving started");
    return;
  }
  if (event.type === "frame") {
    addFrame(event);
    return;
  }
  if (event.type === "done") {
    if (!userPaused && selectedFrameIndex < frames.length - 1) {
      pendingDoneEvent = event;
    } else {
      finishSolve(event);
    }
    return;
  }
  if (event.type === "error") {
    showError(event.message || "Solve request failed");
  }
}

function solveNewMaze() {
  stopStream();
  setLoading(true);
  setStatus("Preparing Runtime Solving stream");
  result.innerHTML = "";
  currentMaze = null;
  currentStart = null;
  currentGoal = null;
  streamMeta = null;
  frames = [];
  selectedFrameIndex = -1;

  activeSource = new EventSource(`/solve_new_maze_stream?t=${Date.now()}`);
  activeSource.onmessage = (message) => {
    try {
      handleStreamEvent(JSON.parse(message.data));
    } catch (_error) {
      showError("Solve stream returned invalid data");
    }
  };
  activeSource.onerror = () => {
    if (!activeSource) return;
    showError("Solve stream disconnected");
  };
}

solveButton.addEventListener("click", solveNewMaze);
